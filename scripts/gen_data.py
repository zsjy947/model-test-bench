#!/usr/bin/env python3
"""内置数据集生成器：prompt 池、长文本素材、OCR 样例图与 ground truth。

在开发机执行一次，产物提交入库（目标运行环境不需要 Pillow / 字体）：

    python scripts/gen_data.py

生成内容：
- data/prompts/{zh,en}.jsonl     按长度分桶的中英 prompt 池
- data/longdoc/{zh,en}.txt       长序列填充素材
- data/images/ocr/*.png          OCR 样例图（印刷中文/英文/表格/多分辨率）
- data/images/ocr/ground_truth.json
- data/images/ocr_fault/{corrupt.png,huge.png}   容错用例图
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# --------------------------------------------------------------------------- #
# 字体查找
# --------------------------------------------------------------------------- #

_FONT_CANDIDATES_ZH = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_FONT_CANDIDATES_EN = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(candidates: list[str]) -> str:
    for path in candidates:
        if Path(path).is_file():
            return path
    raise SystemExit(f"未找到可用字体: {candidates}")


ZH_FONT = find_font(_FONT_CANDIDATES_ZH)
EN_FONT = find_font(_FONT_CANDIDATES_EN)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


# --------------------------------------------------------------------------- #
# prompt 池
# --------------------------------------------------------------------------- #

ZH_SHORT = [
    "用一句话解释什么是大语言模型。",
    "把下面这句话翻译成英文：今天天气很好。",
    "列举三种常见的排序算法。",
    "中国的首都是哪里？",
    "写一个关于春天的五言绝句。",
]

ZH_MID = [
    "请对比 TCP 与 UDP 协议的主要区别，分别说明它们的可靠性、连接性、开销与典型应用场景，"
    "并用一个表格总结。",
    "介绍一下 Transformer 架构中自注意力机制的原理，说明 Q、K、V 的作用以及为什么要除以 "
    "根号 dk，最后谈谈多头注意力带来的好处。",
    "假设你是一名后端工程师，请给出设计一个高并发短链接服务的技术方案，包括存储选型、"
    "缓存策略、幂等性与容量估算。",
    "请总结《中华人民共和国国民经济和社会发展第十四个五年规划》中关于数字经济的主要方向，"
    "并谈谈对普通企业数字化转型的启示。",
]

ZH_LONG_SEED = (
    "大语言模型的推理服务部署涉及多个相互制约的环节。首先是显存与算力的权衡：模型权重、"
    "键值缓存与激活值共同占据设备显存，批大小越大吞吐越高，但显存压力也随之上升。"
    "其次是预填充与解码两阶段的性能差异：预填充属于计算密集型，受益于算力；解码属于"
    "访存密集型，受益于显存带宽。调度器需要在两者之间取得平衡。"
    "再者是量化与精度：低比特量化可以显著降低显存占用并提升吞吐，但可能带来精度损失，"
    "需要针对业务场景评估。此外，长上下文支持会放大键值缓存的显存开销，通常需要配合"
    "分组查询注意力、滑动窗口注意力或前缀缓存等优化手段。"
    "最后，工程层面还需要考虑动态批处理、请求排队策略、多实例负载均衡与灰度发布，"
    "以及完善的监控告警体系，覆盖延迟分位数、吞吐、错误率与资源利用率。"
)

EN_SHORT = [
    "Explain what an LLM is in one sentence.",
    "List three common sorting algorithms.",
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
]

EN_MID = [
    "Compare TCP and UDP: reliability, connection state, overhead, and typical use cases. "
    "Summarize the differences in a short table and explain when you would pick each one.",
    "Explain the self-attention mechanism in the Transformer architecture. Describe the roles "
    "of Q, K, V, why the scores are scaled by sqrt(dk), and what multi-head attention adds.",
    "As a backend engineer, design a high-throughput URL shortener service: storage choice, "
    "caching strategy, idempotency, and capacity estimation for 100k QPS.",
    "Summarize the key ideas behind continuous batching in LLM inference servers and why it "
    "improves GPU utilization compared with static batching.",
]

EN_LONG_SEED = (
    "Deploying large language model inference services involves many interlocking trade-offs. "
    "First, memory and compute must be balanced: model weights, key-value caches, and "
    "activations all occupy device memory; larger batches improve throughput but increase "
    "memory pressure. Second, the prefill and decode phases behave differently: prefill is "
    "compute-bound while decode is memory-bandwidth-bound, and the scheduler must balance "
    "both. Quantization reduces memory and improves throughput at the cost of accuracy, so "
    "each workload needs evaluation. Long contexts amplify key-value cache growth and usually "
    "require grouped-query attention, sliding-window attention, or prefix caching. "
    "Operationally, engineers must also consider dynamic batching, admission control, "
    "multi-instance load balancing, canary releases, and monitoring that covers latency "
    "percentiles, throughput, error rates, and hardware utilization. "
)


def repeat_to(text: str, target_chars: int) -> str:
    reps = target_chars // len(text) + 1
    return (text + "\n\n") * reps


def write_prompts() -> None:
    out_zh = DATA / "prompts" / "zh.jsonl"
    out_en = DATA / "prompts" / "en.jsonl"
    out_zh.parent.mkdir(parents=True, exist_ok=True)

    entries_zh: list[dict] = []
    for t in ZH_SHORT:
        entries_zh.append({"lang": "zh", "text": t})
    for t in ZH_MID:
        entries_zh.append({"lang": "zh", "text": t})
    for n in (600, 1200, 2400, 4800):
        entries_zh.append({
            "lang": "zh",
            "text": "以下是一段关于推理服务部署的技术材料，请阅读后回答问题。\n\n"
                    + repeat_to(ZH_LONG_SEED, n).strip()
                    + "\n\n请用三点总结上述材料的核心观点。",
        })
    out_zh.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries_zh) + "\n",
                      encoding="utf-8")

    entries_en: list[dict] = []
    for t in EN_SHORT:
        entries_en.append({"lang": "en", "text": t})
    for t in EN_MID:
        entries_en.append({"lang": "en", "text": t})
    for n in (600, 1200, 2400, 4800):
        entries_en.append({
            "lang": "en",
            "text": "Read the following technical material and answer the question.\n\n"
                    + repeat_to(EN_LONG_SEED, n).strip()
                    + "\n\nSummarize the key points in three bullets.",
        })
    out_en.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries_en) + "\n",
                      encoding="utf-8")
    print(f"prompts: zh={len(entries_zh)} en={len(entries_en)}")


def write_longdoc() -> None:
    out = DATA / "longdoc"
    out.mkdir(parents=True, exist_ok=True)
    (out / "zh.txt").write_text(repeat_to(ZH_LONG_SEED, 20000).strip() + "\n", encoding="utf-8")
    (out / "en.txt").write_text(repeat_to(EN_LONG_SEED, 20000).strip() + "\n", encoding="utf-8")
    print(f"longdoc: zh={20000} chars en={20000} chars")


# --------------------------------------------------------------------------- #
# OCR 样例图
# --------------------------------------------------------------------------- #

def new_canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), "white")
    return img, ImageDraw.Draw(img)


def save(img: Image.Image, name: str) -> None:
    directory = DATA / "images" / "ocr"
    directory.mkdir(parents=True, exist_ok=True)
    img.save(directory / name)
    print(f"image: {name} ({img.width}x{img.height})")


def gen_print_zh() -> tuple[str, str]:
    lines = [
        "人工智能推理平台测试样张",
        "模型一键测试平台使用说明",
        "第一，支持一键起停服务与自动化测试。",
        "第二，覆盖性能、长序列与功能冒烟。",
        "第三，采集昇腾芯片利用率与显存峰值。",
        "第四，自动生成 Markdown 分析报告。",
    ]
    img, draw = new_canvas(640, 480)
    f = font(ZH_FONT, 28)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=f)
        y += 56
    save(img, "print_zh_640x480.png")
    return "print_zh_640x480.png", "\n".join(lines)


def gen_print_en() -> tuple[str, str]:
    lines = [
        "Model Test Bench Sample Sheet",
        "One-command benchmark for LLM inference.",
        "1. Automatic service start, health check and teardown.",
        "2. Performance, long-context and functional suites.",
        "3. NPU utilization and HBM peak collection.",
        "4. Markdown report and metrics JSON generation.",
    ]
    img, draw = new_canvas(1280, 720)
    f = font(EN_FONT, 36)
    y = 80
    for line in lines:
        draw.text((60, y), line, fill="black", font=f)
        y += 80
    save(img, "print_en_1280x720.png")
    return "print_en_1280x720.png", "\n".join(lines)


def gen_table() -> tuple[str, str]:
    headers = ["套件", "指标", "单位"]
    rows = [
        ["perf", "RPS / TTFT", "req/s, s"],
        ["longctx", "prefill tps", "tok/s"],
        ["embedding", "sent/s", "sent/s"],
        ["ocr", "img/s, CER", "img/s, -"],
    ]
    img, draw = new_canvas(1920, 1080)
    f_head = font(ZH_FONT, 44)
    f_cell = font(ZH_FONT, 38)
    x0, y0, col_w, row_h = 160, 160, 500, 90
    # 标题
    draw.text((x0, 60), "模型测试套件一览表", fill="black", font=font(ZH_FONT, 52))
    # 表格线
    for i in range(len(rows) + 2):
        y = y0 + i * row_h
        draw.line((x0, y, x0 + 3 * col_w, y), fill="black", width=3)
    for i in range(4):
        x = x0 + i * col_w
        draw.line((x, y0, x, y0 + (len(rows) + 1) * row_h), fill="black", width=3)
    # 表头与单元格
    for j, h in enumerate(headers):
        draw.text((x0 + j * col_w + 24, y0 + 20), h, fill="black", font=f_head)
    for r, row in enumerate(rows):
        for j, cell in enumerate(row):
            draw.text((x0 + j * col_w + 24, y0 + (r + 1) * row_h + 22), cell,
                      fill="black", font=f_cell)
    save(img, "table_zh_1920x1080.png")
    text = "\n".join(["\t".join(headers)] + ["\t".join(r) for r in rows])
    return "table_zh_1920x1080.png", text


def gen_mixed_small() -> tuple[str, str]:
    lines = ["订单号：20260924-0001", "金额：壹佰贰拾叁元伍角", "状态：已发货 ABC-123"]
    img, draw = new_canvas(800, 600)
    f = font(ZH_FONT, 40)
    y = 140
    for line in lines:
        draw.text((80, y), line, fill="black", font=f)
        y += 110
    save(img, "mixed_zh_num_800x600.png")
    return "mixed_zh_num_800x600.png", "\n".join(lines)


def gen_tiny() -> tuple[str, str]:
    text = "小字号样例 A7X-99"
    img, draw = new_canvas(384, 288)
    draw.text((48, 110), text, fill="black", font=font(ZH_FONT, 32))
    save(img, "small_zh_384x288.png")
    return "small_zh_384x288.png", text


def write_minimal_png(width: int, height: int, path: Path) -> None:
    """纯 stdlib 写全白 PNG（不加载整幅位图，行级压缩）。"""
    raw = b"\x00" + b"\xff\xff\xff" * width

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (len(data).to_bytes(4, "big") + tag + data
                + zlib.crc32(tag + data).to_bytes(4, "big"))

    ihdr = (width.to_bytes(4, "big") + height.to_bytes(4, "big")
            + b"\x08\x02\x00\x00\x00")
    idat = zlib.compress(raw * height, 6)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", idat) + chunk(b"IEND", b""))
    print(f"image: {path.name} ({width}x{height}, {path.stat().st_size} bytes)")


def write_fault_images() -> None:
    fault = DATA / "images" / "ocr_fault"
    fault.mkdir(parents=True, exist_ok=True)
    # 损坏图片：合法 PNG 头 + 乱码体
    (fault / "corrupt.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00\xff\x10this is not a valid png payload" * 64)
    print(f"image: corrupt.png ({(fault / 'corrupt.png').stat().st_size} bytes)")
    # 超大图片：6000x6000 全白（文件极小，解码后占内存大，用于触发服务端优雅拒绝）
    write_minimal_png(6000, 6000, fault / "huge.png")


def write_ground_truth(gt: dict[str, str]) -> None:
    path = DATA / "images" / "ocr" / "ground_truth.json"
    path.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ground_truth.json: {len(gt)} 项")


def main() -> None:
    write_prompts()
    write_longdoc()
    gt: dict[str, str] = {}
    for gen in (gen_print_zh, gen_print_en, gen_table, gen_mixed_small, gen_tiny):
        name, text = gen()
        gt[name] = text
    write_fault_images()
    write_ground_truth(gt)
    print("done.")


if __name__ == "__main__":
    main()
