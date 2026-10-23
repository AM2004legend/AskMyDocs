"""OCR on a scanned (image-only) PDF and on an image upload, with the real local RapidOCR model.

Known limit (measured): RapidOCR drops the spaces in ALL-CAPS lines ("RENTISPAIDMONTHLY"),
while normal mixed-case text comes out perfectly, so these tests use realistic sentences.
"""

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ingest.parsers import parse_file

pytestmark = pytest.mark.slow

LINES = ["The notice period is thirty days.", "Rent is paid on the first day of each month."]


def scanned_page() -> Image.Image:
    image = Image.new("RGB", (1240, 600), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 40)  # a typical document font (Windows)
    except OSError:
        font = ImageFont.load_default(size=40)
    for i, line in enumerate(LINES):
        draw.text((80, 120 + i * 140), line, fill="black", font=font)
    return image


def test_scanned_pdf_is_ocrd(tmp_path):
    path = tmp_path / "scan.pdf"
    scanned_page().save(path, "PDF", resolution=150)  # image only, no text layer
    parsed = parse_file(path, "pdf")
    text = " ".join(b.text for b in parsed.blocks).lower()
    assert parsed.ocr_pages == 1
    assert "notice period is thirty days" in text and "first day of each month" in text
    assert {b.page for b in parsed.blocks} == {1}


def test_image_upload_is_ocrd(tmp_path):
    path = tmp_path / "photo.png"
    scanned_page().save(path)
    text = " ".join(b.text for b in parse_file(path, "png").blocks).lower()
    assert "notice period is thirty days" in text
