import os
import json
from typing import Dict, Any, List, Tuple

from openai import OpenAI
from datetime import datetime


def _log(logs: List[Dict[str, Any]], msg: str):
    logs.append({"ts": datetime.utcnow().isoformat() + "Z", "msg": msg})

from .settings import PDF_FOLDER, OPENAI_MODEL
from .pipeline import (
    analyze_pdf_page,
    clean_text_for_ai,
    render_token_items_to_ascii,
    robust_json_parse,
    coerce_ids_list,
    normalize_text,
    union_bbox,
    find_multitoken_sequence,
    find_single_token,
    _table_tokens_by_headers,
    _heuristic_pick_company,
    _heuristic_pick_date,
    _heuristic_pick_trattamento,
    _heuristic_pick_materiale,
    CHEM_HEADERS,
    MECH_HEADERS,
    _label_tokens_finetuned,
    _cluster_lines,
    _match_field_from_question,
)


def build_pages_data(pdf_name: str, logs: List[Dict[str, Any]] = None, cache: Dict[str, Any] = None):
    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    if cache is None:
        cache = {}
    if logs is None:
        logs = []
    import fitz

    doc = fitz.open(pdf_path)
    try:
        pages = []
        for page_idx in range(doc.page_count):
            cache_key = f"{pdf_name}:{page_idx}"
            if cache_key in cache:
                pd = cache[cache_key]
            else:
                img, tokens = analyze_pdf_page(pdf_name, page_idx, zoom=2.0)
                pd = {"image": img, "tokens": tokens}
                cache[cache_key] = pd
            pages.append((page_idx, pd))
        return pages
    finally:
        doc.close()


def _reading_order_key(t):
    x0, y0, x1, y1 = t["bbox"]
    line = int(y0 / 12)
    return (line, y0, x0)


def _build_global_tokens(pages_data):
    page_offsets = {}
    y_offset = 0
    for page_idx, pd in pages_data:
        page_offsets[page_idx] = y_offset
        y_offset += pd["image"].height

    global_tokens = []
    gid_to_ref = {}
    gid = 0
    for page_idx, pd in pages_data:
        tokens_sorted = sorted(pd["tokens"], key=_reading_order_key)
        for t in tokens_sorted:
            x0, y0, x1, y1 = t["bbox"]
            abs_y = y0 + page_offsets[page_idx]
            text_orig = t["text"]
            text_clean = clean_text_for_ai(text_orig)
            global_tokens.append(
                {
                    "gid": gid,
                    "page": page_idx,
                    "local_id": t["id"],
                    "bbox": t["bbox"],
                    "abs_x": x0,
                    "abs_y": abs_y,
                    "text_orig": text_orig,
                    "text_clean": text_clean,
                }
            )
            gid_to_ref[gid] = (page_idx, t["id"])
            gid += 1
    return global_tokens, gid_to_ref


def _bbox_intersects(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1)


def _table_region_gids(global_tokens, hint_gids, pad=6, pad_y_down=120, pad_y_up=20):
    if not hint_gids:
        return set()
    per_page_boxes = {}
    for g in hint_gids:
        t = global_tokens[g]
        x0, y0, x1, y1 = t["bbox"]
        page = t["page"]
        if page not in per_page_boxes:
            per_page_boxes[page] = [x0, y0, x1, y1]
        else:
            bb = per_page_boxes[page]
            per_page_boxes[page] = [
                min(bb[0], x0),
                min(bb[1], y0),
                max(bb[2], x1),
                max(bb[3], y1),
            ]

    expanded = {}
    for page, bb in per_page_boxes.items():
        expanded[page] = [bb[0] - pad, bb[1] - pad_y_up, bb[2] + pad, bb[3] + pad_y_down]

    region_ids = set(hint_gids)
    for t in global_tokens:
        page = t["page"]
        if page not in expanded:
            continue
        if _bbox_intersects(t["bbox"], expanded[page]):
            region_ids.add(t["gid"])
    return region_ids


def _build_prompt(global_tokens, include_ascii=True, table_hints=None):
    lines = []
    items_for_ascii = []
    for t in global_tokens:
        if t["text_clean"]:
            lines.append(f'{t["gid"]}|p{t["page"]}|{t["abs_x"]}|{t["abs_y"]}|{t["text_clean"]}')
            items_for_ascii.append({"x": t["abs_x"], "y": t["abs_y"], "text": t["text_clean"]})

    tokens_block = "\n".join(lines)
    ascii_text = render_token_items_to_ascii(items_for_ascii) if include_ascii and items_for_ascii else ""

    table_hint_text = ""
    if table_hints:
        chem_ids = table_hints.get("chem_table_gids") or []
        mech_ids = table_hints.get("mech_table_gids") or []
        table_hint_text = (
            "TABLES (step 1):\n"
            f"- chem_gids: {chem_ids}\n"
            f"- mech_gids: {mech_ids}\n"
            "Use these IDs as primary constraint for chemical/mechanical tables.\n"
        )

    prompt = f"""
Extract fields from certificates.
RULES:
- Use ONLY provided tokens.
- Do NOT invent.
- Output JSON only.
- For each field return a list of integer token_ids (gid).
- If missing: [].
- "azienda" is Producer/Supplier/Manufacturer/Mill (not Customer).
- Ignore "Forgialluminio" or "Cliente" for "azienda".
- "composizione_chimica" and "proprieta_meccaniche" are tables: pick contiguous token blocks with headers.
- Chem headers: Si, Fe, Cu, Mn, Mg, Zn, Cr, Ni, Ti, Pb, Sn.
- Mech headers: Rp0.2, Rp0,2, Rm, A, A%, HB, HV, MPa, N/mm2, Yield, Tensile.
- Text fields must not include table values.

JSON output:
{{
  "azienda": [],
  "data_certificato": [],
  "materiale": [],
  "trattamento_termico": [],
  "composizione_chimica": [],
  "proprieta_meccaniche": []
}}

ASCII (layout):
{ascii_text}

{table_hint_text}

TOKEN (gid|page|x|y|text_clean):
{tokens_block}
""".strip()
    return prompt


def openai_full(pdf_name: str, cache: Dict[str, Any] = None, api_key: str = None):
    logs: List[Dict[str, Any]] = []
    _log(logs, "Caricamento pagine")
    pages_data = build_pages_data(pdf_name, logs=logs, cache=cache)
    _log(logs, "Token globali")
    global_tokens, gid_to_ref = _build_global_tokens(pages_data)
    n = len(global_tokens)
    if n == 0:
        raise Exception("No tokens found")

    table_hints = {}
    try:
        _log(logs, "OpenAI pass1 tabelle")
        pass1_prompt = f"""
Identify tables in the certificate using ONLY provided tokens.
Rules:
- Identify chemical composition table (headers like Si, Fe, Cu, Mg, Mn, Zn, Ti).
- Identify mechanical properties table (Rp0.2/Rp0,2, Rm, A%, HB/HV, MPa, N/mm2).
- Output JSON only with gid lists.

JSON output:
{{
  "chem_table_gids": [],
  "mech_table_gids": []
}}

TOKEN (gid|page|x|y|text_clean):
{chr(10).join([f'{t["gid"]}|p{t["page"]}|{t["abs_x"]}|{t["abs_y"]}|{t["text_clean"]}' for t in global_tokens if t["text_clean"]])}
""".strip()
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        pass1_response = client.chat.completions.create(
            model=OPENAI_MODEL, messages=[{"role": "user", "content": pass1_prompt}], temperature=0
        )
        pass1_raw = pass1_response.choices[0].message.content
        pass1_parsed = robust_json_parse(pass1_raw)
        chem_seed = [x for x in coerce_ids_list(pass1_parsed.get("chem_table_gids")) if 0 <= x < n]
        mech_seed = [x for x in coerce_ids_list(pass1_parsed.get("mech_table_gids")) if 0 <= x < n]
        table_hints = {
            "chem_table_gids": sorted(_table_region_gids(global_tokens, chem_seed)),
            "mech_table_gids": sorted(_table_region_gids(global_tokens, mech_seed)),
        }
    except Exception:
        table_hints = {}

    _log(logs, "OpenAI full prompt")
    prompt = _build_prompt(global_tokens, include_ascii=True, table_hints=table_hints)
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    raw_txt = response.choices[0].message.content
    parsed = robust_json_parse(raw_txt)

    supplier_keywords = {"producer", "supplier", "manufacturer", "mill", "issued", "azienda", "emesso"}

    def expand_supplier_tokens():
        anchor_gids = []
        for t in global_tokens:
            if normalize_text(t["text_orig"]) in supplier_keywords:
                anchor_gids.append(t["gid"])
        expanded = set()
        for g in anchor_gids:
            anchor = global_tokens[g]
            ay = anchor["bbox"][1]
            ax = anchor["bbox"][0]
            page = anchor["page"]
            same_line = [
                t
                for t in global_tokens
                if t["page"] == page and abs(t["bbox"][1] - ay) <= 6 and t["bbox"][0] >= ax
            ]
            same_line_sorted = sorted(same_line, key=lambda t: t["bbox"][0])[:12]
            expanded.update(t["gid"] for t in same_line_sorted)
        return expanded

    supplier_expanded = expand_supplier_tokens()
    fields_list = [
        "azienda",
        "data_certificato",
        "materiale",
        "trattamento_termico",
        "composizione_chimica",
        "proprieta_meccaniche",
    ]

    collected = {k: set() for k in fields_list}
    for k in fields_list:
        ids = coerce_ids_list(parsed.get(k))
        ids = [x for x in ids if 0 <= x < n]
        if k == "azienda":
            ids = list(set(ids) | supplier_expanded)
            ids = [
                x
                for x in ids
                if "forgialluminio" not in normalize_text(global_tokens[x]["text_orig"])
                and "cliente" not in normalize_text(global_tokens[x]["text_orig"])
            ]
        if k == "composizione_chimica" and table_hints.get("chem_table_gids"):
            allowed = set(table_hints["chem_table_gids"])
            ids = list(set(ids) | allowed)
        if k == "proprieta_meccaniche" and table_hints.get("mech_table_gids"):
            allowed = set(table_hints["mech_table_gids"])
            ids = list(set(ids) | allowed)
        for x in ids:
            collected[k].add(x)

    def sort_key_gid(g):
        t = global_tokens[g]
        x0, y0, x1, y1 = t["bbox"]
        return (t["page"], y0, x0)

    final_fields = {}
    pages_data_map = dict(pages_data)

    for k in fields_list:
        gids = sorted(collected[k], key=sort_key_gid)
        if not gids:
            final_fields[k] = None
            continue
        tokens_by_page = {}
        parts_in_order = []
        for g in gids:
            page_idx, local_id = gid_to_ref[g]
            tokens_by_page.setdefault(page_idx, set()).add(local_id)
            parts_in_order.append(global_tokens[g]["text_orig"])

        tokens_by_page_list = {}
        for p, idset in tokens_by_page.items():
            pd = pages_data_map.get(p)
            if not pd:
                continue
            tok_map = {t["id"]: t for t in pd["tokens"]}
            tokens_by_page_list[p] = sorted(
                list(idset), key=lambda tid: (tok_map[tid]["bbox"][1], tok_map[tid]["bbox"][0])
            )

        text_value = " ".join(parts_in_order)
        first_page = min(tokens_by_page_list.keys())
        pd0 = pages_data_map.get(first_page)
        if not pd0:
            continue
        toks0 = pd0["tokens"]
        idset0 = set(tokens_by_page_list[first_page])
        bboxes0 = [t["bbox"] for t in toks0 if t["id"] in idset0]
        bbox0 = union_bbox(bboxes0) if bboxes0 else [0, 0, 0, 0]

        final_fields[k] = {
            "text": text_value,
            "page": first_page,
            "bbox": bbox0,
            "tokens": tokens_by_page_list[first_page],
            "tokens_by_page": tokens_by_page_list,
        }

    _log(logs, "Post-processing 1-to-1")
    return {
        "fields": final_fields,
        "table_hints": table_hints,
        "prompt": prompt,
        "logs": logs,
    }


def openai_pdf(pdf_name: str, cache: Dict[str, Any] = None, api_key: str = None):
    logs: List[Dict[str, Any]] = []
    _log(logs, "Caricamento pagine")
    pages_data = build_pages_data(pdf_name, logs=logs, cache=cache)
    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
    if not os.path.exists(pdf_path):
        raise Exception("PDF not found")

    prompt = """
You are a field extractor for certificates (PDF attached). Do not invent.
Rules:
- Use ONLY info present in the PDF.
- Output JSON only.
- If a field does not exist, use null.
- Do not reformat or normalize values.

JSON output:
{
  "azienda": null,
  "data_certificato": null,
  "materiale": null,
  "trattamento_termico": null,
  "composizione_chimica": null,
  "proprieta_meccaniche": null
}
""".strip()

    _log(logs, "OpenAI PDF upload")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    used_fallback = False
    if hasattr(client, "responses"):
        with open(pdf_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="user_data")
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_file", "file_id": uploaded.id},
                    ],
                }
            ],
            temperature=0,
        )
        _log(logs, "OpenAI PDF parse")
        raw_txt = response.output_text
    else:
        used_fallback = True
        _log(logs, "OpenAI PDF fallback (text)")
        import fitz
        doc = fitz.open(pdf_path)
        try:
            pages_text = []
            for i in range(doc.page_count):
                pages_text.append(doc[i].get_text("text"))
            pdf_text = "\n\n".join(pages_text)
        finally:
            doc.close()
        fallback_prompt = f"{prompt}\n\nPDF_TEXT:\n{pdf_text}"
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": fallback_prompt}],
            temperature=0,
        )
        raw_txt = response.choices[0].message.content
    parsed = robust_json_parse(raw_txt)

    def _is_empty_value(v):
        if v is None:
            return True
        if isinstance(v, str):
            return not v.strip()
        if isinstance(v, (list, tuple, set)):
            return len(v) == 0
        if isinstance(v, dict):
            if not v:
                return True
            return all(_is_empty_value(x) for x in v.values())
        return False

    fields_list = [
        "azienda",
        "data_certificato",
        "materiale",
        "trattamento_termico",
        "composizione_chimica",
        "proprieta_meccaniche",
    ]

    if not used_fallback and all(_is_empty_value(parsed.get(k)) for k in fields_list):
        _log(logs, "OpenAI PDF empty result, retry text fallback")
        import fitz
        doc = fitz.open(pdf_path)
        try:
            pages_text = []
            for i in range(doc.page_count):
                pages_text.append(doc[i].get_text("text"))
            pdf_text = "\n\n".join(pages_text)
        finally:
            doc.close()
        fallback_prompt = f"{prompt}\n\nPDF_TEXT:\n{pdf_text}"
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": fallback_prompt}],
            temperature=0,
        )
        raw_txt = response.choices[0].message.content
        parsed = robust_json_parse(raw_txt)

    def _flatten_value(v):
        out = []
        if v is None:
            return out
        if isinstance(v, str):
            if v.strip():
                out.append(v)
            return out
        if isinstance(v, (int, float, bool)):
            out.append(str(v))
            return out
        if isinstance(v, dict):
            for kk, vv in v.items():
                out.extend(_flatten_value(kk))
                out.extend(_flatten_value(vv))
            return out
        if isinstance(v, (list, tuple)):
            for it in v:
                out.extend(_flatten_value(it))
            return out
        try:
            s = str(v)
            if s.strip():
                out.append(s)
        except Exception:
            pass
        return out

    def _value_to_text(v):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    pages_data_map = dict(pages_data)
    page_tokens_sorted = {
        page_idx: sorted(pd["tokens"], key=_reading_order_key) for page_idx, pd in pages_data
    }

    def _select_table_tokens(field_name):
        headers = CHEM_HEADERS if field_name == "composizione_chimica" else MECH_HEADERS
        selected = {}
        for page_idx, pd in pages_data:
            tks = _table_tokens_by_headers(pd["tokens"], headers, pd["image"].width, pd["image"].height)
            if tks:
                selected[page_idx] = tks
        return selected

    def _token_eq(token_text, target_word):
        tn = normalize_text(token_text)
        if not tn or not target_word:
            return False
        if tn == target_word:
            return True
        if target_word in tn or tn in target_word:
            return True
        return False

    def _match_tokens_fuzzy(tokens_sorted, target_words, max_window=120):
        if not target_words:
            return None
        n = len(tokens_sorted)
        for i in range(n):
            if not _token_eq(tokens_sorted[i]["text"], target_words[0]):
                continue
            matched = [tokens_sorted[i]]
            j = i + 1
            k = 1
            while j < n and k < len(target_words) and (j - i) <= max_window:
                if _token_eq(tokens_sorted[j]["text"], target_words[k]):
                    matched.append(tokens_sorted[j])
                    k += 1
                j += 1
            if k == len(target_words):
                return matched
        return None

    def _match_value_tokens(value_text, match_text=None):
        raw = value_text or ""
        mtext = match_text or value_text or ""
        if not raw.strip() and not mtext.strip():
            return {}
        for page_idx, tokens_sorted in page_tokens_sorted.items():
            mt = find_multitoken_sequence(tokens_sorted, mtext, max_window=120)
            if mt:
                return {page_idx: mt}
        target_words = normalize_text(mtext).split()
        for page_idx, tokens_sorted in page_tokens_sorted.items():
            ft = _match_tokens_fuzzy(tokens_sorted, target_words, max_window=160)
            if ft:
                return {page_idx: ft}
        for page_idx, tokens_sorted in page_tokens_sorted.items():
            st = find_single_token(tokens_sorted, mtext)
            if st:
                return {page_idx: st}
        value_norm = normalize_text(mtext)
        if not value_norm:
            return {}
        val_words = set(value_norm.split())
        selected = {}
        for page_idx, tokens_sorted in page_tokens_sorted.items():
            hits = []
            for t in tokens_sorted:
                tn = normalize_text(t["text"])
                if not tn:
                    continue
                if tn in val_words or tn in value_norm or value_norm in tn:
                    hits.append(t)
            if hits:
                selected[page_idx] = hits
        return selected

    def _heuristic_tokens(field_name):
        for page_idx, pd in pages_data:
            if field_name == "azienda":
                tks = _heuristic_pick_company(pd["tokens"], pd["image"].height)
            elif field_name == "data_certificato":
                tks = _heuristic_pick_date(pd["tokens"])
            elif field_name == "trattamento_termico":
                tks = _heuristic_pick_trattamento(pd["tokens"])
            elif field_name == "materiale":
                tks = _heuristic_pick_materiale(pd["tokens"])
            else:
                tks = []
            if tks:
                return {page_idx: tks}
        return {}

    final_fields = {}
    for k in fields_list:
        value = parsed.get(k)
        text_value = _value_to_text(value)
        match_text = " ".join(_flatten_value(value))
        if text_value is None or text_value == "":
            final_fields[k] = None
            continue

        if k in {"composizione_chimica", "proprieta_meccaniche"}:
            selected_by_page = _select_table_tokens(k)
            if not selected_by_page:
                selected_by_page = _match_value_tokens(text_value, match_text=match_text)
        else:
            selected_by_page = _match_value_tokens(text_value, match_text=match_text)
            if not selected_by_page:
                selected_by_page = _heuristic_tokens(k)

        tokens_by_page_list = {}
        for page_idx, toks in selected_by_page.items():
            pd = pages_data_map.get(page_idx)
            if not pd:
                continue
            tok_map = {t["id"]: t for t in pd["tokens"]}
            idset = {t["id"] for t in toks}
            tokens_by_page_list[page_idx] = sorted(
                list(idset), key=lambda tid: (tok_map[tid]["bbox"][1], tok_map[tid]["bbox"][0])
            )

        if tokens_by_page_list:
            first_page = min(tokens_by_page_list.keys())
            pd0 = pages_data_map.get(first_page)
            toks0 = pd0["tokens"] if pd0 else []
            idset0 = set(tokens_by_page_list[first_page])
            bboxes0 = [t["bbox"] for t in toks0 if t["id"] in idset0]
            bbox0 = union_bbox(bboxes0) if bboxes0 else [0, 0, 0, 0]
            tokens_main = tokens_by_page_list[first_page]
        else:
            first_page = 0
            bbox0 = [0, 0, 0, 0]
            tokens_main = []

        final_fields[k] = {
            "text": text_value,
            "page": first_page,
            "bbox": bbox0,
            "tokens": tokens_main,
            "tokens_by_page": tokens_by_page_list,
        }

    return {"fields": final_fields, "prompt": prompt, "logs": logs}


def openai_refine(pdf_name: str, current_fields: Dict[str, Any], table_hints: Dict[str, Any], cache: Dict[str, Any] = None, api_key: str = None):
    logs: List[Dict[str, Any]] = []
    _log(logs, "Caricamento pagine")
    pages_data = build_pages_data(pdf_name, logs=logs, cache=cache)
    global_tokens, gid_to_ref = _build_global_tokens(pages_data)
    n = len(global_tokens)
    if n == 0:
        raise Exception("No tokens found")

    local_to_gid = {(p, tid): g for g, (p, tid) in gid_to_ref.items()}
    fields_list = [
        "azienda",
        "data_certificato",
        "materiale",
        "trattamento_termico",
        "composizione_chimica",
        "proprieta_meccaniche",
    ]

    def table_region_gids_local(hint_gids, pad=6):
        if not hint_gids:
            return set()
        per_page_boxes = {}
        for g in hint_gids:
            t = global_tokens[g]
            x0, y0, x1, y1 = t["bbox"]
            page = t["page"]
            if page not in per_page_boxes:
                per_page_boxes[page] = [x0, y0, x1, y1]
            else:
                bb = per_page_boxes[page]
                per_page_boxes[page] = [
                    min(bb[0], x0),
                    min(bb[1], y0),
                    max(bb[2], x1),
                    max(bb[3], y1),
                ]
        expanded = {}
        for page, bb in per_page_boxes.items():
            expanded[page] = [bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad]
        region_ids = set(hint_gids)
        for t in global_tokens:
            page = t["page"]
            if page not in expanded:
                continue
            if _bbox_intersects(t["bbox"], expanded[page]):
                region_ids.add(t["gid"])
        return region_ids

    def current_field_gids(field_name):
        v = current_fields.get(field_name)
        if not v:
            return []
        gids = []
        if isinstance(v, dict) and "tokens_by_page" in v:
            for p, token_ids in v["tokens_by_page"].items():
                for tid in token_ids:
                    gid_val = local_to_gid.get((p, tid))
                    if gid_val is not None:
                        gids.append(gid_val)
        else:
            p = v.get("page")
            for tid in v.get("tokens", []):
                gid_val = local_to_gid.get((p, tid))
                if gid_val is not None:
                    gids.append(gid_val)
        return sorted(set(gids))

    if table_hints.get("chem_table_gids"):
        table_hints["chem_table_gids"] = sorted(
            table_region_gids_local(table_hints["chem_table_gids"])
        )
    if table_hints.get("mech_table_gids"):
        table_hints["mech_table_gids"] = sorted(
            table_region_gids_local(table_hints["mech_table_gids"])
        )

    suggestion_lines = []
    for k in fields_list:
        gids = current_field_gids(k)
        if gids:
            suggestion_lines.append(f"- {k}: {gids}")
    suggestions_block = "\n".join(suggestion_lines) if suggestion_lines else "(no suggestions)"

    tokens_block = "\n".join(
        f'{t["gid"]}|p{t["page"]}|{t["abs_x"]}|{t["abs_y"]}|{t["text_clean"]}'
        for t in global_tokens
        if t["text_clean"]
    )

    prompt = f"""
You are in refinement mode. Fields are already filled and must stay 1-to-1.
Rules:
- Use ONLY provided tokens.
- Do NOT invent.
- Output JSON only.
- "azienda" is Producer/Supplier/Manufacturer/Mill (not Customer).
- Ignore "Forgialluminio" or "Cliente" for "azienda".
- "composizione_chimica" and "proprieta_meccaniche" are tables with typical headers.
- Text fields must not include table values.

CURRENT SUGGESTIONS (gid):
{suggestions_block}

TABLES (base):
- chem_gids: {table_hints.get("chem_table_gids") or []}
- mech_gids: {table_hints.get("mech_table_gids") or []}

JSON output:
{{
  "azienda": [],
  "data_certificato": [],
  "materiale": [],
  "trattamento_termico": [],
  "composizione_chimica": [],
  "proprieta_meccaniche": []
}}

TOKEN (gid|page|x|y|text_clean):
{tokens_block}
""".strip()

    _log(logs, "OpenAI refine prompt")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    raw_txt = response.choices[0].message.content
    parsed = robust_json_parse(raw_txt)

    collected = {k: set() for k in fields_list}
    for k in fields_list:
        ids = coerce_ids_list(parsed.get(k))
        ids = [x for x in ids if 0 <= x < n]
        if k == "azienda":
            ids = [x for x in ids if "forgialluminio" not in normalize_text(global_tokens[x]["text_orig"])]
        if k == "composizione_chimica" and table_hints.get("chem_table_gids"):
            allowed = set(table_hints["chem_table_gids"])
            ids = list(set(ids) | allowed)
        if k == "proprieta_meccaniche" and table_hints.get("mech_table_gids"):
            allowed = set(table_hints["mech_table_gids"])
            ids = list(set(ids) | allowed)
        for x in ids:
            collected[k].add(x)

    def sort_key_gid(g):
        t = global_tokens[g]
        x0, y0, x1, y1 = t["bbox"]
        return (t["page"], y0, x0)

    final_fields = {}
    pages_data_map = dict(pages_data)

    for k in fields_list:
        gids = sorted(collected[k], key=sort_key_gid)
        if not gids:
            final_fields[k] = None
            continue
        tokens_by_page = {}
        parts_in_order = []
        for g in gids:
            page_idx, local_id = gid_to_ref[g]
            tokens_by_page.setdefault(page_idx, set()).add(local_id)
            parts_in_order.append(global_tokens[g]["text_orig"])

        tokens_by_page_list = {}
        for p, idset in tokens_by_page.items():
            pd = pages_data_map.get(p)
            if not pd:
                continue
            tok_map = {t["id"]: t for t in pd["tokens"]}
            tokens_by_page_list[p] = sorted(
                list(idset), key=lambda tid: (tok_map[tid]["bbox"][1], tok_map[tid]["bbox"][0])
            )

        text_value = " ".join(parts_in_order)
        first_page = min(tokens_by_page_list.keys())
        pd0 = pages_data_map.get(first_page)
        if not pd0:
            continue
        toks0 = pd0["tokens"]
        idset0 = set(tokens_by_page_list[first_page])
        bboxes0 = [t["bbox"] for t in toks0 if t["id"] in idset0]
        bbox0 = union_bbox(bboxes0) if bboxes0 else [0, 0, 0, 0]

        final_fields[k] = {
            "text": text_value,
            "page": first_page,
            "bbox": bbox0,
            "tokens": tokens_by_page_list[first_page],
            "tokens_by_page": tokens_by_page_list,
        }

    return {"fields": final_fields, "prompt": prompt, "logs": logs}


def ai_trained(pdf_name: str, cache: Dict[str, Any] = None, api_key: str = None):
    logs: List[Dict[str, Any]] = []
    _log(logs, "Caricamento pagine")
    pages_data = build_pages_data(pdf_name, logs=logs, cache=cache)
    page_offsets = {}
    y_offset = 0
    for page_idx, pd in pages_data:
        page_offsets[page_idx] = y_offset
        y_offset += pd["image"].height

    global_tokens = []
    gid_to_ref = {}
    local_to_gid = {}
    gid = 0
    for page_idx, pd in pages_data:
        tokens_sorted = sorted(pd["tokens"], key=_reading_order_key)
        for t in tokens_sorted:
            x0, y0, x1, y1 = t["bbox"]
            abs_y = y0 + page_offsets[page_idx]
            text_orig = t["text"]
            text_clean = clean_text_for_ai(text_orig)
            global_tokens.append(
                {
                    "gid": gid,
                    "page": page_idx,
                    "local_id": t["id"],
                    "bbox": t["bbox"],
                    "abs_x": x0,
                    "abs_y": abs_y,
                    "text_orig": text_orig,
                    "text_clean": text_clean,
                }
            )
            gid_to_ref[gid] = (page_idx, t["id"])
            local_to_gid[(page_idx, t["id"])] = gid
            gid += 1

    n = len(global_tokens)
    if n == 0:
        raise Exception("No tokens")

    fields_list = [
        "azienda",
        "data_certificato",
        "materiale",
        "trattamento_termico",
        "composizione_chimica",
        "proprieta_meccaniche",
    ]
    collected = {k: set() for k in fields_list}

    question_labels = {"B-QUESTION", "I-QUESTION", "B-HEADER", "I-HEADER"}
    answer_labels = {"B-ANSWER", "I-ANSWER"}

    for page_idx, pd in pages_data:
        img = pd["image"]
        tokens = pd["tokens"]
        labels = _label_tokens_finetuned(img, tokens)
        for t, lbl in zip(tokens, labels):
            t["_ft_label"] = lbl

        rows = _cluster_lines(tokens, y_tol=10)
        best_for_page = {}

        for row in rows:
            q_tokens = [t for t in row if t.get("_ft_label") in question_labels]
            if not q_tokens:
                continue
            q_text = " ".join(t["text"] for t in q_tokens)
            field = _match_field_from_question(q_text)
            if not field:
                continue
            q_bbox = union_bbox([t["bbox"] for t in q_tokens])
            ans_tokens = [
                t for t in row if t.get("_ft_label") in answer_labels and t["bbox"][0] >= q_bbox[2] - 3
            ]
            if not ans_tokens:
                for row2 in rows:
                    if row2 is row:
                        continue
                    y0 = row2[0]["bbox"][1]
                    if y0 < q_bbox[1] - 5 or y0 > q_bbox[3] + 60:
                        continue
                    for t in row2:
                        if t.get("_ft_label") in answer_labels and t["bbox"][0] >= q_bbox[0] - 5:
                            ans_tokens.append(t)
            if ans_tokens:
                if field not in best_for_page or len(ans_tokens) > len(best_for_page[field]):
                    best_for_page[field] = ans_tokens

        page_w, page_h = img.size
        chem_tokens = _table_tokens_by_headers(tokens, CHEM_HEADERS, page_w, page_h)
        mech_tokens = _table_tokens_by_headers(tokens, MECH_HEADERS, page_w, page_h)
        if chem_tokens and "composizione_chimica" not in best_for_page:
            best_for_page["composizione_chimica"] = chem_tokens
        if mech_tokens and "proprieta_meccaniche" not in best_for_page:
            best_for_page["proprieta_meccaniche"] = mech_tokens

        for field, toks in best_for_page.items():
            for t in toks:
                if field == "azienda":
                    norm_t = normalize_text(t.get("text") or "")
                    if "forgialluminio" in norm_t or "cliente" in norm_t:
                        continue
                gid_val = local_to_gid.get((page_idx, t["id"]))
                if gid_val is not None:
                    collected[field].add(gid_val)

        if not collected["azienda"]:
            for t in _heuristic_pick_company(tokens, img.size[1]):
                gid_val = local_to_gid.get((page_idx, t["id"]))
                if gid_val is not None:
                    collected["azienda"].add(gid_val)
        if not collected["data_certificato"]:
            for t in _heuristic_pick_date(tokens):
                gid_val = local_to_gid.get((page_idx, t["id"]))
                if gid_val is not None:
                    collected["data_certificato"].add(gid_val)
        if not collected["trattamento_termico"]:
            for t in _heuristic_pick_trattamento(tokens):
                gid_val = local_to_gid.get((page_idx, t["id"]))
                if gid_val is not None:
                    collected["trattamento_termico"].add(gid_val)
        if not collected["materiale"]:
            for t in _heuristic_pick_materiale(tokens):
                gid_val = local_to_gid.get((page_idx, t["id"]))
                if gid_val is not None:
                    collected["materiale"].add(gid_val)

    if all(len(s) == 0 for s in collected.values()):
        raise Exception("Trained model produced no valid token ids")

    def sort_key_gid(g):
        t = global_tokens[g]
        x0, y0, x1, y1 = t["bbox"]
        return (t["page"], y0, x0)

    final_fields = {}
    pages_data_map = dict(pages_data)
    for k in fields_list:
        gids = sorted(collected[k], key=sort_key_gid)
        if not gids:
            final_fields[k] = None
            continue
        tokens_by_page = {}
        parts_in_order = []
        for g in gids:
            page_idx, local_id = gid_to_ref[g]
            tokens_by_page.setdefault(page_idx, set()).add(local_id)
            parts_in_order.append(global_tokens[g]["text_orig"])
        tokens_by_page_list = {}
        for p, idset in tokens_by_page.items():
            pd = pages_data_map.get(p)
            if not pd:
                continue
            tok_map = {t["id"]: t for t in pd["tokens"]}
            tokens_by_page_list[p] = sorted(
                list(idset), key=lambda tid: (tok_map[tid]["bbox"][1], tok_map[tid]["bbox"][0])
            )
        text_value = " ".join(parts_in_order)
        first_page = min(tokens_by_page_list.keys())
        pd0 = pages_data_map.get(first_page)
        if not pd0:
            continue
        toks0 = pd0["tokens"]
        idset0 = set(tokens_by_page_list[first_page])
        bboxes0 = [t["bbox"] for t in toks0 if t["id"] in idset0]
        bbox0 = union_bbox(bboxes0) if bboxes0 else [0, 0, 0, 0]
        final_fields[k] = {
            "text": text_value,
            "page": first_page,
            "bbox": bbox0,
            "tokens": tokens_by_page_list[first_page],
            "tokens_by_page": tokens_by_page_list,
        }

    return {"fields": final_fields, "logs": logs}
