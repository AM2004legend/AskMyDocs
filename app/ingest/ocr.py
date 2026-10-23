"""Local OCR with RapidOCR (ONNX models bundled in the pip package; nothing is uploaded).

Used for scanned PDF pages (no text layer) and for image uploads.
"""

import threading

import numpy as np

_engine = None
_lock = threading.Lock()


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR  # loads ~15 MB of models, so lazily

            _engine = RapidOCR()
    return _engine


def _group_into_lines(boxes: list) -> list[str]:
    """RapidOCR returns text boxes; sort them top-to-bottom, left-to-right and join into lines."""
    items = []
    for box, text, _score in boxes:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), max(ys), min(xs), text))
    items.sort(key=lambda it: (it[0], it[2]))

    lines: list[list[tuple]] = []
    for item in items:
        top, bottom = item[0], item[1]
        if lines:
            last = lines[-1]
            last_top = min(i[0] for i in last)
            last_bottom = max(i[1] for i in last)
            overlap = min(bottom, last_bottom) - max(top, last_top)
            if overlap > 0.5 * min(bottom - top, last_bottom - last_top):  # same visual line
                last.append(item)
                continue
        lines.append([item])
    return [" ".join(i[3] for i in sorted(line, key=lambda i: i[2])) for line in lines]


def ocr_image(image) -> str:
    """OCR a PIL image or numpy array and return plain text, one line per visual line."""
    array = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else image
    result, _elapsed = _get_engine()(array)
    if not result:
        return ""
    return "\n".join(_group_into_lines(result))
