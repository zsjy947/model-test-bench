"""OCR 数学与图像头解析测试：CER / 归一化 / PNG-JPEG 尺寸 / 乱码比。"""

import struct

import pytest

from mtest.suites.functional import mojibake_ratio
from mtest.suites.ocr import (_bucket_for, cer, image_dimensions, jpeg_dimensions,
                              levenshtein, normalize_for_cer, png_dimensions)


def test_normalize():
    assert normalize_for_cer("你好 世界　ＡＢＣ") == "你好世界abc"  # 全角→半角+去空白
    assert normalize_for_cer("A B\nC") == "abc"


def test_levenshtein():
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("kitten", "sitting") == 3


def test_cer():
    assert cer("你好世界", "你好世界") == 0.0
    # 归一化后 "你好世界" vs "你好,世界!" → 2 个插入 / 6 参考长度
    assert abs(cer("你好世界", "你好，世界！") - (2 / 6)) < 1e-9
    assert cer("x", "") is None  # 空参考


def test_cer_ref_length_normalization():
    # 编辑距离对称，但 CER 按参考长度归一
    assert abs(cer("abc", "ab") - 0.5) < 1e-9  # 1 edit / |ref|=2


def _png_bytes(w: int, h: int) -> bytes:
    # 标准 PNG 帧：签名 + IHDR chunk（长度 13 + 类型 + 数据 + CRC 占位）
    ihdr_data = struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00"
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + b"\x00\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + ihdr


def test_png_dimensions():
    w, h = png_dimensions(_png_bytes(640, 480))
    assert (w, h) == (640, 480)
    assert png_dimensions(b"not png") is None


def _jpeg_bytes(w: int, h: int) -> bytes:
    # SOI + APP0(skip) + SOF0 段
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x01\x01\x00" + b"\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", h, w) + b"\x03" + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def test_jpeg_dimensions():
    w, h = jpeg_dimensions(_jpeg_bytes(1920, 1080))
    assert (w, h) == (1920, 1080)
    assert jpeg_dimensions(b"\xff\xd8garbage") is None


def test_image_dimensions_dispatch():
    assert image_dimensions(_png_bytes(10, 20), "png") == (10, 20)
    assert image_dimensions(b"xx", "bmp") is None


def test_bucket():
    assert _bucket_for(640, 480) == "0.3-1MP"
    assert _bucket_for(1920, 1080) == "1-2MP"
    assert _bucket_for(None, None) == "unknown"
    assert _bucket_for(6000, 6000) == ">9MP"


def test_mojibake_ratio():
    assert mojibake_ratio("正常文本") == 0.0
    assert mojibake_ratio("正常\ufffd\ufffd文本") == pytest.approx(2 / 6)
    assert mojibake_ratio("") == 0.0
