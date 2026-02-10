import os
import re
from typing import Dict, List, Tuple, Any

import fitz
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForTokenClassification
import demjson3
import pytesseract

from .settings import (
    PDF_FOLDER,
    TESSERACT_CMD,
)


MODEL_NAME = "microsoft/layoutlmv3-base"
FINETUNED_MODEL_NAME = "HYPJUDY/layoutlmv3-base-finetuned-funsd"


_processor = None
_base_model = None
_finetuned_model = None
_id2label = None
_ft_id2label = None


def _load_models():
    global _processor, _base_model, _finetuned_model, _id2label, _ft_id2label
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)
    if _base_model is None:
        _base_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        _base_model.eval()
        _id2label = _base_model.config.id2label
    if _finetuned_model is None:
        _finetuned_model = AutoModelForTokenClassification.from_pretrained(FINETUNED_MODEL_NAME)
        _finetuned_model.eval()
        _ft_id2label = _finetuned_model.config.id2label


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]


def normalize_text(s):
    s = (s or "").lower()
    s = s.replace(",", ".")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\.\-/ ]", "", s)
    return s.strip()


def union_bbox(boxes):
    xs0 = [b[0] for b in boxes]
    ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]
    ys1 = [b[3] for b in boxes]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def find_multitoken_sequence(tokens, target_text, max_window=50):
    if not target_text or not target_text.strip():
        return None
    target_norm = normalize_text(target_text)
    token_texts = [normalize_text(t["text"]) for t in tokens]
    n = len(tokens)
    for i in range(n):
        acc = ""
        for j in range(i, min(i + max_window, n)):
            if acc:
                acc += " "
            acc += token_texts[j]
            if acc == target_norm:
                return tokens[i : j + 1]
    return None


def find_single_token(tokens, target_text):
    if not target_text or not target_text.strip():
        return None
    tnorm = normalize_text(target_text)
    for t in tokens:
        tok_norm = normalize_text(t["text"])
        if tok_norm == tnorm:
            return [t]
        if tnorm and tnorm in tok_norm:
            return [t]
    return None


def _extract_jsonish_block(text: str) -> str:
    if not text:
        return text
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]
    return text


def robust_json_parse(text: str):
    def extract_first_balanced_object(s: str):
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
        return None

    snippet = extract_first_balanced_object(text)
    if snippet:
        obj = demjson3.decode(snippet)
    else:
        obj = demjson3.decode(text)
    if isinstance(obj, dict):
        return obj
    raise Exception("Decoded object is not a dict")


def clean_text_for_ai(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"[^\w\.\-/%\+\=\(\)\[\]\:\;,\<\> ]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def coerce_ids_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for it in x:
            try:
                out.append(int(it))
            except Exception:
                if isinstance(it, str):
                    nums = re.findall(r"\d+", it)
                    out.extend([int(n) for n in nums])
        return out
    if isinstance(x, (int, float)):
        try:
            return [int(x)]
        except Exception:
            return []
    if isinstance(x, str):
        nums = re.findall(r"\d+", x)
        return [int(n) for n in nums]
    return []


def render_token_items_to_ascii(token_items, max_width=120):
    if not token_items:
        return ""
    max_x = max(t["x"] for t in token_items)
    scale_x = max_x / max_width if max_x > max_width else 1.0
    scale_y = 15
    grid = {}
    for t in token_items:
        col = int(t["x"] / scale_x)
        row = int(t["y"] / scale_y)
        if row not in grid:
            grid[row] = []
        grid[row].append((col, t["text"]))
    lines = []
    for row in sorted(grid.keys()):
        items = sorted(grid[row], key=lambda x: x[0])
        line = ""
        cursor = 0
        for col, text in items:
            if col > cursor:
                line += " " * (col - cursor)
            line += text + " "
            cursor = col + len(text) + 1
        lines.append(line.rstrip())
    return "\n".join(lines)


def _normalize_key_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace(",", ".")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\.\-%/ ]", "", s)
    return s.strip()


FIELD_KEYWORDS = {
    "azienda": [
        "azienda", "company", "supplier", "producer", "manufacturer", "vendor", "mill", "fornitore", "produttore"
    ],
    "data_certificato": [
        "data", "date", "data certificato", "certificate date", "issue date", "emissione", "certificato"
    ],
    "materiale": [
        "materiale", "material", "alloy", "lega", "grade", "spec", "specification"
    ],
    "trattamento_termico": [
        "trattamento", "heat treatment", "temper", "tempera", "treatment", "t6", "t6511", "t4", "t5", "o", "f"
    ],
    "composizione_chimica": [
        "composizione chimica", "chemical composition", "chemistry", "analisi chimica"
    ],
    "proprieta_meccaniche": [
        "proprieta meccaniche", "mechanical properties", "tensile", "yield", "hardness",
        "rp0.2", "rp0,2", "rm", "hb", "hv", "mpa", "n/mm2", "elongation", "a%"
    ],
}

CHEM_HEADERS = {"si", "fe", "cu", "mn", "mg", "zn", "cr", "ni", "ti", "pb", "sn", "al"}
MECH_HEADERS = {"rp0.2", "rp0,2", "rm", "a", "a%", "hb", "hv", "mpa", "n/mm2", "yield", "tensile"}


def _match_field_from_question(q_text: str):
    nq = _normalize_key_text(q_text)
    if not nq:
        return None
    for field, keys in FIELD_KEYWORDS.items():
        for k in keys:
            nk = _normalize_key_text(k)
            if nk and nk in nq:
                return field
    return None


def _cluster_lines(tokens, y_tol=10):
    rows = []
    for t in sorted(tokens, key=lambda x: (x["bbox"][1], x["bbox"][0])):
        placed = False
        for row in rows:
            if abs(t["bbox"][1] - row[0]["bbox"][1]) <= y_tol:
                row.append(t)
                placed = True
                break
        if not placed:
            rows.append([t])
    for row in rows:
        row.sort(key=lambda x: x["bbox"][0])
    return rows


def _label_tokens_finetuned(img, tokens):
    _load_models()
    words = [t["text"] for t in tokens]
    boxes_norm = [t["bbox_norm"] for t in tokens]
    if not words:
        return ["O"] * 0
    enc = _processor(images=img, text=words, boxes=boxes_norm, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = _finetuned_model(**enc)
    pred_ids = out.logits.argmax(-1)[0].tolist()
    word_ids = enc.word_ids(batch_index=0)
    word_labels = ["O"] * len(words)
    for idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        if word_labels[w_id] == "O":
            word_labels[w_id] = _ft_id2label.get(pred_ids[idx], "O")
    return word_labels


def _label_tokens_base(img, tokens):
    _load_models()
    words = [t["text"] for t in tokens]
    boxes_norm = [t["bbox_norm"] for t in tokens]
    if not words:
        return []
    enc = _processor(images=img, text=words, boxes=boxes_norm, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = _base_model(**enc)
    preds = out.logits.argmax(-1).squeeze().tolist()
    if isinstance(preds, int):
        preds = [preds]
    labels = []
    for p in preds[: len(tokens)]:
        labels.append(_id2label.get(p, "O"))
    return labels


def _heuristic_pick_company(tokens, page_height):
    header_limit = page_height * 0.15
    header_tokens = [t for t in tokens if t["bbox"][1] <= header_limit]
    if not header_tokens:
        return []
    rows = _cluster_lines(header_tokens, y_tol=10)
    best_row = None
    best_len = 0
    for row in rows:
        txt = " ".join(t["text"] for t in row)
        ntxt = _normalize_key_text(txt)
        if "forgialluminio" in ntxt or "cliente" in ntxt:
            continue
        if len(ntxt) > best_len:
            best_len = len(ntxt)
            best_row = row
    return best_row or []


def _heuristic_pick_date(tokens):
    date_pattern = re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b")
    out = []
    for t in tokens:
        if date_pattern.search(t.get("text") or ""):
            out.append(t)
    return out


def _heuristic_pick_trattamento(tokens):
    allowed = {"t6", "t6511", "t4", "t5", "o", "f"}
    out = []
    for t in tokens:
        nt = _normalize_key_text(t.get("text") or "")
        if nt in allowed:
            out.append(t)
    return out


def _heuristic_pick_materiale(tokens):
    out = []
    for t in tokens:
        nt = _normalize_key_text(t.get("text") or "")
        if "en aw" in nt or "enaw" in nt:
            out.append(t)
            continue
        if re.fullmatch(r"\d{4}", nt):
            out.append(t)
    return out


def _table_tokens_by_headers(tokens, headers, page_width, page_height):
    header_tokens = []
    for t in tokens:
        nt = _normalize_key_text(t["text"]).replace(" ", "")
        if nt in headers:
            header_tokens.append(t)
    if not header_tokens:
        return []
    bbox = union_bbox([t["bbox"] for t in header_tokens])
    avg_h = sum((t["bbox"][3] - t["bbox"][1]) for t in header_tokens) / max(1, len(header_tokens))
    y0 = max(0, bbox[1] - 10)
    y1 = min(page_height, bbox[3] + max(140, int(avg_h * 14)))
    x0 = 0
    x1 = page_width
    out = []
    for t in tokens:
        bx0, by0, bx1, by1 = t["bbox"]
        if not (bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1):
            out.append(t)
    return out


pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def remove_table_lines_bgr(cv_img_bgr):
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    h, w = bw.shape[:2]
    hor_len = max(20, w // 30)
    ver_len = max(20, h // 30)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_len, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_len))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_kernel, iterations=1)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_kernel, iterations=1)
    lines = cv2.bitwise_or(hor, ver)
    bw_no_lines = cv2.bitwise_and(bw, cv2.bitwise_not(lines))
    clean = cv2.bitwise_not(bw_no_lines)
    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)


def _preprocess_for_ocr(pil_img: Image.Image):
    cv_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv_bgr = remove_table_lines_bgr(cv_bgr)
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray2 = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )
    return gray2, thr


def _ocr_data(cv_img, lang="eng+ita", psm=6):
    config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    return pytesseract.image_to_data(
        cv_img, output_type=pytesseract.Output.DICT, lang=lang, config=config
    )


def _best_ocr(pil_img: Image.Image, lang="eng+ita"):
    gray2, thr = _preprocess_for_ocr(pil_img)
    candidates = [
        ("gray_psm6", _ocr_data(gray2, lang=lang, psm=6)),
        ("thr_psm6", _ocr_data(thr, lang=lang, psm=6)),
        ("gray_psm11", _ocr_data(gray2, lang=lang, psm=11)),
        ("thr_psm11", _ocr_data(thr, lang=lang, psm=11)),
    ]

    def score(d):
        s = 0.0
        for txt, conf in zip(d.get("text", []), d.get("conf", [])):
            t = (txt or "").strip()
            if not t:
                continue
            try:
                c = float(conf)
            except Exception:
                c = -1
            if c <= 0:
                continue
            s += c * max(1, len(t))
        return s

    best_tag, best_data = None, None
    best_score = -1
    for tag, data in candidates:
        sc = score(data)
        if sc > best_score:
            best_score = sc
            best_tag, best_data = tag, data
    return best_data, best_tag


def _bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / float(area_a + area_b - inter)


def _norm_token_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _bbox_center(b):
    x0, y0, x1, y1 = b
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _center_dist(a, b):
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _ocr_tokens_from_pil(img: Image.Image, width, height):
    ocr_data, _ = _best_ocr(img, lang="eng+ita")
    if not ocr_data:
        return []
    o_tokens = []
    n_boxes = len(ocr_data.get("text", []))
    for i in range(n_boxes):
        text = (ocr_data["text"][i] or "").strip()
        try:
            conf = float(ocr_data["conf"][i])
        except Exception:
            conf = -1
        if not text:
            continue
        if conf < 30:
            if not re.search(r"[0-9A-Za-z]", text):
                continue
        x, y, w, h = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        o_tokens.append(
            {
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "bbox_norm": normalize_bbox([x0, y0, x1, y1], width, height),
                "source": "ocr",
            }
        )
    return o_tokens


def _merge_vector_and_ocr(v_tokens, o_tokens, iou_thr=0.18, dist_thr=18):
    v_bboxes = [t["bbox"] for t in v_tokens]
    v_text_map = [(_norm_token_text(t["text"]), t["bbox"]) for t in v_tokens]
    merged = list(v_tokens)

    def is_noise(txt: str) -> bool:
        t = (txt or "").strip()
        if not t:
            return True
        if len(t) == 1:
            if re.fullmatch(r"[A-Za-z0-9]", t):
                return False
            if t in {"%", "-", "+", "/", ".", ","}:
                return False
            return True
        if re.fullmatch(r"[\.\,\-]+", t):
            return True
        return False

    for t in o_tokens:
        txt = t["text"]
        if is_noise(txt):
            continue
        bb = t["bbox"]
        covered = any(_bbox_iou(bb, vbb) >= iou_thr for vbb in v_bboxes)
        if covered:
            continue
        nt = _norm_token_text(txt)
        if nt:
            for vt, vbb in v_text_map:
                if vt == nt and _center_dist(bb, vbb) <= dist_thr:
                    covered = True
                    break
        if covered:
            continue
        merged.append(t)

    out = []
    for t in merged:
        out.append(
            {
                "id": len(out),
                "text": t["text"],
                "bbox": t["bbox"],
                "bbox_norm": t["bbox_norm"],
                "label_suggested": None,
                "used": False,
                "source": t.get("source"),
            }
        )
    return out


def extract_vector_tokens_from_pdf(pdf_path, page_idx, zoom=2.0):
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        width, height = img.size
        words = page.get_text("words")
        tokens = []
        for w in words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            text = (text or "").strip()
            if not text:
                continue
            x0z, y0z, x1z, y1z = x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom
            tokens.append(
                {
                    "id": len(tokens),
                    "text": text,
                    "bbox": [int(x0z), int(y0z), int(x1z), int(y1z)],
                    "bbox_norm": normalize_bbox([x0z, y0z, x1z, y1z], width, height),
                    "label_suggested": None,
                    "used": False,
                }
            )
        return img, tokens
    finally:
        doc.close()


def analyze_pdf_page(pdf_name: str, page_idx: int, zoom: float = 2.0):
    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
    img = None
    tokens = None
    if os.path.exists(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            try:
                page = doc[page_idx]
                is_vector_page = bool(page.get_text("words"))
                if is_vector_page:
                    img_vec, v_tokens = extract_vector_tokens_from_pdf(pdf_path, page_idx, zoom=zoom)
                    w, h = img_vec.size
                    o_tokens = _ocr_tokens_from_pil(img_vec, w, h)
                    tokens = _merge_vector_and_ocr(v_tokens, o_tokens, iou_thr=0.20)
                    labels = _label_tokens_base(img_vec, tokens)
                    for t, lbl in zip(tokens, labels):
                        t["label_suggested"] = lbl
                    img = img_vec
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            finally:
                doc.close()
        except Exception:
            img = None

    if img is None:
        raise Exception("Cannot open PDF page")

    if tokens is None:
        width, height = img.size
        ocr_data, _ = _best_ocr(img, lang="eng+ita")
        tokens = []
        words = []
        boxes_norm = []
        n_boxes = len(ocr_data.get("text", []))
        for i in range(n_boxes):
            text = ocr_data["text"][i].strip()
            try:
                conf = int(ocr_data["conf"][i])
            except Exception:
                conf = -1
            if not text or conf < 0:
                continue
            x, y, w, h = (
                ocr_data["left"][i],
                ocr_data["top"][i],
                ocr_data["width"][i],
                ocr_data["height"][i],
            )
            x0, y0, x1, y1 = x, y, x + w, y + h
            if (w >= 120 and h <= 3) or (h >= 120 and w <= 3):
                continue
            tokens.append(
                {
                    "id": len(tokens),
                    "text": text,
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "bbox_norm": normalize_bbox([x0, y0, x1, y1], width, height),
                    "label_suggested": None,
                    "used": False,
                }
            )
            words.append(text)
            boxes_norm.append(tokens[-1]["bbox_norm"])

        labels = _label_tokens_base(img, tokens)
        for t, lbl in zip(tokens, labels):
            t["label_suggested"] = lbl

    return img, tokens


def render_tokens_to_ascii(pages_data, max_width=120):
    all_tokens = []
    y_offset = 0
    for page_idx, pd in pages_data:
        for t in pd["tokens"]:
            x0, y0, x1, y1 = t["bbox"]
            all_tokens.append({"x": x0, "y": y0 + y_offset, "text": t["text"]})
        y_offset += pd["image"].height
    if not all_tokens:
        return ""
    max_x = max(t["x"] for t in all_tokens)
    scale_x = max_x / max_width if max_x > max_width else 1.0
    scale_y = 15
    grid = {}
    for t in all_tokens:
        col = int(t["x"] / scale_x)
        row = int(t["y"] / scale_y)
        if row not in grid:
            grid[row] = []
        grid[row].append((col, t["text"]))
    lines = []
    for row in sorted(grid.keys()):
        items = sorted(grid[row], key=lambda x: x[0])
        line = ""
        cursor = 0
        for col, text in items:
            if col > cursor:
                line += " " * (col - cursor)
            line += text + " "
            cursor = col + len(text) + 1
        lines.append(line.rstrip())
    return "\n".join(lines)
