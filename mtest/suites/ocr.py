"""ocr 专项套件（multimodal，设计 §5.4）。

- 请求：``/v1/chat/completions`` + ``image_url``（本地图片 base64）
- 性能：单图延迟分布、吞吐 img/s（含并发档）、按分辨率分档延迟
- 质量：有 ground_truth.json 时算 CER（编辑距离/参考长度，文本先归一化）；
  无标注仅冒烟（输出非空、无乱码启发式）
- 容错：损坏图片、超大图片 → 期望 4xx/优雅报错而非服务崩溃，随后正常请求
  确认服务仍健康
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import struct
import time
import unicodedata
from pathlib import Path

from ..metrics import dist_summary, safe_div, write_csv
from ..paths import data_dir, resolve_data
from .base import SUITE_FAILED, SUITE_PASSED, Suite, SuiteResult

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_MIME = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "bmp": "bmp", "webp": "webp"}

# 分辨率分档（百万像素）
_BUCKETS = [(0.3, "≤0.3MP"), (1.0, "0.3-1MP"), (2.1, "1-2MP"), (9.0, "2-9MP"),
            (float("inf"), ">9MP")]


def _bucket_for(w: int | None, h: int | None) -> str:
    if not w or not h:
        return "unknown"
    mp = w * h / 1e6
    for bound, label in _BUCKETS:
        if mp <= bound:
            return label
    return "unknown"


def png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return w, h
    return None


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > len(data):
            break
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 <= len(data):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
        i += 2 + seg_len
    return None


def image_dimensions(data: bytes, ext: str) -> tuple[int, int] | None:
    if ext == "png":
        return png_dimensions(data)
    if ext in ("jpg", "jpeg"):
        return jpeg_dimensions(data)
    return None


# --------------------------------------------------------------------------- #
# 文本归一化与 CER
# --------------------------------------------------------------------------- #

def normalize_for_cer(text: str) -> str:
    """去空白、全半角统一（NFKC）、大小写统一。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(text.split())
    return text.lower()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float | None:
    """CER = 编辑距离 / 参考长度；参考为空返回 None。"""
    hyp_n, ref_n = normalize_for_cer(hyp), normalize_for_cer(ref)
    if not ref_n:
        return None
    return levenshtein(hyp_n, ref_n) / len(ref_n)


# --------------------------------------------------------------------------- #
# 套件
# --------------------------------------------------------------------------- #

class OcrSuite(Suite):
    name = "ocr"
    applies_to = ("multimodal",)

    @staticmethod
    def _data_url(path: Path) -> str:
        ext = path.suffix.lstrip(".").lower()
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/{_MIME.get(ext, 'png')};base64,{b64}"

    def load_images(self) -> list[dict]:
        directory = resolve_data(self.cfg.tests.ocr.image_dir)
        images: list[dict] = []
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in _IMAGE_EXTS or not path.is_file():
                continue
            data = path.read_bytes()
            ext = path.suffix.lstrip(".").lower()
            w, h = image_dimensions(data, ext) or (None, None)
            images.append({
                "name": path.name, "path": str(path),
                "bytes": len(data), "w": w, "h": h,
                "bucket": _bucket_for(w, h),
                "data_url": self._data_url(path),
            })
        return images

    def _load_ground_truth(self) -> dict[str, dict]:
        spec = self.cfg.tests.ocr.ground_truth
        if not spec:
            # 缺省取图片目录同级 ground_truth.json
            source = resolve_data(self.cfg.tests.ocr.image_dir).parent / "ground_truth.json"
            if not source.is_file():
                return {}
        else:
            source = resolve_data(spec)
        if not source.is_file():
            return {}
        raw = json.loads(source.read_text(encoding="utf-8"))
        out: dict[str, dict] = {}
        for name, val in raw.items():
            out[name] = val if isinstance(val, dict) else {"text": val}
        return out

    # ------------------------------------------------------------------ #
    async def _perf_sweep(self, images: list[dict]) -> tuple[dict, list[dict]]:
        o = self.cfg.tests.ocr
        client = self.ctx.client
        model = self.ctx.served_model
        matrix: dict[str, dict] = {}
        rows: list[dict] = []
        img_iter = itertools.cycle(images)

        for cc in o.concurrency:
            label = f"ocr[cc={cc}]"
            with self.ctx.phase(label):
                for _ in range(max(0, o.warmup)):
                    img = next(img_iter)
                    await client.chat(model, [self._message(img)], max_tokens=512)
                lat: list[float] = []
                errors: dict[str, int] = {}
                bucket_lat: dict[str, list[float]] = {}
                ok = total = 0
                seq = itertools.count(1)
                stop_at = time.monotonic() + o.duration
                started_at: list[float] = []

                async def worker() -> None:
                    nonlocal ok, total
                    while True:
                        i = next(seq)
                        if i > o.num_requests:
                            return
                        if time.monotonic() >= stop_at:
                            return
                        if not started_at:
                            started_at.append(time.monotonic())
                        img = next(img_iter)
                        res = await client.chat(model, [self._message(img)], max_tokens=512)
                        total += 1
                        if res.ok:
                            ok += 1
                            lat.append(res.e2e or 0.0)
                            bucket_lat.setdefault(img["bucket"], []).append(res.e2e or 0.0)
                        else:
                            key = (res.error or f"http:{res.status}").split(":", 1)[0]
                            errors[key] = errors.get(key, 0) + 1
                        rows.append({
                            "concurrency": cc, "seq": i, "image": img["name"],
                            "bucket": img["bucket"], "ok": res.ok,
                            "status": res.status, "error": res.error,
                            "e2e_s": res.e2e, "output_len": len(res.output or ""),
                        })

                await asyncio.gather(*[worker() for _ in range(max(1, cc))])
                wall = (time.monotonic() - started_at[0]) if started_at else 0.0
            matrix[f"cc{cc}"] = {
                "concurrency": cc,
                "requests": total,
                "ok": ok,
                "success_rate": safe_div(ok, total),
                "wall_s": wall,
                "img_per_s": safe_div(ok, wall),
                "latency_s": dist_summary(lat),
                "bucket_latency_s": {b: dist_summary(v) for b, v in sorted(bucket_lat.items())},
                "errors": errors,
            }
            cell = matrix[f"cc{cc}"]
            self.ctx.say(
                f"  ocr[cc{cc:<3}] ok {ok}/{total}  {cell['img_per_s'] or 0:6.2f} img/s  "
                f"lat p50/p99 {(cell['latency_s'] or {}).get('p50') or 0:6.3f}/"
                f"{(cell['latency_s'] or {}).get('p99') or 0:6.3f}s")
        return matrix, rows

    @staticmethod
    def _message(img: dict) -> dict:
        return {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img["data_url"]}},
            {"type": "text", "text": "识别图中所有文字，按原样输出。"},
        ]}

    # ------------------------------------------------------------------ #
    async def _quality(self, images: list[dict]) -> tuple[dict, list[str]]:
        """单图逐张请求（并发 1），有 GT 算 CER，无 GT 仅冒烟。"""
        client = self.ctx.client
        model = self.ctx.served_model
        gt = self._load_ground_truth()
        failures: list[str] = []
        per_file: list[dict] = []

        for img in images:
            res = await client.chat(model, [self._message(img)], max_tokens=1024,
                                    temperature=0.0)
            row = {"image": img["name"], "bucket": img["bucket"], "ok": res.ok,
                   "e2e_s": res.e2e, "output_len": len(res.output or "")}
            if not res.ok:
                failures.append(f"{img['name']}: 请求失败 {res.error}")
            else:
                if not (res.output or "").strip():
                    failures.append(f"{img['name']}: 输出为空")
                from .functional import mojibake_ratio
                if mojibake_ratio(res.output or "") >= 0.05:
                    failures.append(f"{img['name']}: 输出疑似乱码")
                if img["name"] in gt and gt[img["name"]].get("text"):
                    row["cer"] = cer(res.output or "", gt[img["name"]]["text"])
            per_file.append(row)
        cers = [r["cer"] for r in per_file if r.get("cer") is not None]
        quality = {
            "ground_truth_available": bool(gt),
            "files": per_file,
            "cer_summary": dist_summary(cers) if cers else None,
        }
        return quality, failures

    # ------------------------------------------------------------------ #
    async def _fault_cases(self) -> list[dict]:
        """损坏/超大图片容错。"""
        client = self.ctx.client
        model = self.ctx.served_model
        rows: list[dict] = []
        fault_dir = data_dir() / "images" / "ocr_fault"
        for path in sorted(fault_dir.glob("*")) if fault_dir.is_dir() else []:
            if path.suffix.lower() not in _IMAGE_EXTS:
                continue
            data_url = self._data_url(path)
            message = {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "识别图中文字"},
            ]}
            res = await client.chat(model, [message], max_tokens=256)
            graceful = res.ok or (res.status is not None and 400 <= res.status < 500)
            healthy, _ = await client.request_health()
            rows.append({
                "case": path.name, "graceful": graceful,
                "status": res.status, "error": res.error,
                "service_alive_after": healthy,
            })
            if not graceful:
                rows[-1]["problem"] = "期望 4xx/优雅报错"
            if not healthy:
                rows[-1]["problem"] = "服务疑似崩溃"
        return rows

    # ------------------------------------------------------------------ #
    async def run(self) -> SuiteResult:
        result = self.new_result()
        o = self.cfg.tests.ocr
        images = self.load_images()
        if not images:
            result.error = f"图片目录为空: {o.image_dir}"
            return self.finish(result, SUITE_FAILED)

        matrix, rows = await self._perf_sweep(images)
        csv_path = write_csv(Path(self.ctx.run_dir) / "ocr_details.csv", rows)
        quality, failures = await self._quality(images)
        result.metrics = {
            "matrix": matrix,
            "quality": quality,
            "images": [{"name": i["name"], "bucket": i["bucket"],
                        "w": i["w"], "h": i["h"], "bytes": i["bytes"]} for i in images],
        }
        result.artifacts["details"] = str(csv_path)

        if o.fault_cases:
            fault_rows = await self._fault_cases()
            result.metrics["fault_cases"] = fault_rows
            failures.extend(f"{r['case']}: {r.get('problem')}" for r in fault_rows
                            if r.get("problem"))

        result.warnings.extend(failures)
        any_ok = any(c.get("ok") for c in matrix.values())
        if not any_ok:
            result.error = "全部 OCR 请求失败"
            return self.finish(result, SUITE_FAILED)
        return self.finish(result, SUITE_PASSED if not failures else SUITE_FAILED)
