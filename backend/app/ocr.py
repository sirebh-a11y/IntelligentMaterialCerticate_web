import io
import os
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import pytesseract

from .settings import PDF_FOLDER, TEMP_IMAGE_FOLDER, TESSERACT_CMD, ensure_dirs


def _norm_bbox(b: Tuple[float, float, float, float], width: int, height: int):
    x0, y0, x1, y1 = b
    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]


def list_pdfs() -> List[str]:
    ensure_dirs()
    return [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]


def render_page_image(pdf_name: str, page_idx: int, zoom: float = 2.0) -> Image.Image:
    ensure_dirs()
    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    finally:
        doc.close()


def save_page_image(pdf_name: str, page_idx: int) -> str:
    ensure_dirs()
    img = render_page_image(pdf_name, page_idx)
    out_path = os.path.join(
        TEMP_IMAGE_FOLDER,
        f"{os.path.splitext(pdf_name)[0]}_page_{page_idx + 1}.png",
    )
    img.save(out_path, "PNG")
    return out_path


def extract_vector_tokens(pdf_name: str, page_idx: int, zoom: float = 2.0) -> List[Dict]:
    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        text = page.get_text("words")  # x0,y0,x1,y1,"word",block,line,word
        if not text:
            return []

        # Render to get width/height for normalization
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        width, height = pix.width, pix.height

        tokens = []
        for w in text:
            x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
            tokens.append(
                {
                    "text": word,
                    "bbox": [x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom],
                    "bbox_norm": _norm_bbox((x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom), width, height),
                    "source": "vector",
                }
            )
        return tokens
    finally:
        doc.close()


def extract_ocr_tokens(img: Image.Image) -> List[Dict]:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    width, height = img.size
    tokens = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        bbox = [x, y, x + w, y + h]
        tokens.append(
            {
                "text": txt,
                "bbox": bbox,
                "bbox_norm": _norm_bbox((x, y, x + w, y + h), width, height),
                "source": "ocr",
            }
        )
    return tokens


def _bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, (ax1 - ax0)) * max(0, (ay1 - ay0))
    b_area = max(0, (bx1 - bx0)) * max(0, (by1 - by0))
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _center_dist(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    acx, acy = (ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0
    bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def merge_vector_and_ocr(vector_tokens: List[Dict], ocr_tokens: List[Dict], iou_thr=0.18, dist_thr=18):
    merged = []
    used_ocr = set()

    for vt in vector_tokens:
        vb = vt["bbox"]
        best = None
        best_iou = 0.0
        for i, ot in enumerate(ocr_tokens):
            if i in used_ocr:
                continue
            iou = _bbox_iou(vb, ot["bbox"])
            if iou > best_iou:
                best_iou = iou
                best = i
        if best is not None and best_iou >= iou_thr:
            used_ocr.add(best)
            merged.append(vt)
        else:
            merged.append(vt)

    for i, ot in enumerate(ocr_tokens):
        if i in used_ocr:
            continue
        merged.append(ot)

    return merged


def get_tokens_for_page(pdf_name: str, page_idx: int) -> Dict:
    img = render_page_image(pdf_name, page_idx)
    ocr_tokens = extract_ocr_tokens(img)
    vector_tokens = extract_vector_tokens(pdf_name, page_idx)
    tokens = merge_vector_and_ocr(vector_tokens, ocr_tokens)
    for i, t in enumerate(tokens):
        t["id"] = i
    return {
        "image_size": {"width": img.size[0], "height": img.size[1]},
        "tokens": tokens,
    }
