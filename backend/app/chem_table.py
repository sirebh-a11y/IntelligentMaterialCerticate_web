# chem_table.py
# Ported extraction logic (no GUI)

# chem_table_editor.py
# Editor tabellare (tipo Excel) per Composizione Chimica
# Input: tokens (con bbox) selezionati da un crop/catch
# Output:
#   - table: lista righe [{"elemento": "...", "valore": "..."}]
#   - used_token_ids: set di token-id "usati" (VERDI) per Bibbia e crop
#   - debug_dump: testo debug (mini pannello)
#
# RIPRISTINATO: motore score/layout (vertical/horizontal)
# MODIFICATO: interazioni GUI non ricalcolano score durante modalit? click
#             Elimina: rosso + rimuove righe collegate
#             Cattura: blu overlay + riempie entry
#             Inserisci riga: valida -> verdi (binding 1-to-1)

import os
import re
import statistics


# =======================
# Regex / Normalizzazioni
# =======================

_ELEMENT_RE = re.compile(r"^(?:[A-Z][a-z]?|[A-Z]{1,2})$")
_SPECIAL_ELEMENTS = {"Ti+Zr", "Zr+Ti"}

_DECOR_STRIP_RE = re.compile(r"^[^A-Za-z0-9+]+|[^A-Za-z0-9+]+$")

_NUM_RE = re.compile(r"^[\+\-]?\d+(?:[\,\.]\d+)?$")
_RANGE_RE = re.compile(r"^\s*([\+\-]?\d+(?:[\,\.]\d+)?)\s*-\s*([\+\-]?\d+(?:[\,\.]\d+)?)\s*$")

_PERCENT_ANYWHERE_RE = re.compile(r"%")
_PERCENT_ONLYISH_RE = re.compile(r"^\s*[\[\(\{]?\s*%+\s*[\]\)\}]?\s*$")


def _strip_decor(txt: str) -> str:
    s = (txt or "").strip()
    if not s:
        return ""
    prev = None
    while prev != s:
        prev = s
        s = _DECOR_STRIP_RE.sub("", s).strip()
    return s


def _token_center_x(t):
    x0, y0, x1, y1 = t["bbox"]
    return (x0 + x1) / 2.0


def _token_center_y(t):
    x0, y0, x1, y1 = t["bbox"]
    return (y0 + y1) / 2.0


def _canon_element(txt: str) -> str:
    raw = _strip_decor(txt)
    if not raw:
        return ""
    raw = raw.replace("%", "").strip()
    if not raw:
        return ""
    if raw in _SPECIAL_ELEMENTS:
        return raw
    if len(raw) == 1:
        return raw.upper()
    if len(raw) == 2:
        return raw[0].upper() + raw[1].lower()
    return raw


def _is_element_token(txt: str) -> bool:
    if not txt:
        return False
    canon = _canon_element(txt)
    if not canon:
        return False
    if canon in _SPECIAL_ELEMENTS:
        return True
    return bool(_ELEMENT_RE.match(canon))


def _is_percent_unitish_token(txt: str) -> bool:
    s = (txt or "").strip()
    if not s:
        return False
    return bool(_PERCENT_ONLYISH_RE.match(s))


def _clean_for_number_parse(txt: str) -> str:
    s = (txt or "").strip()
    if not s:
        return ""
    s = s.replace(" ", "")
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("{", "").replace("}", "")
    s = s.replace("%", "")
    return s.strip()


def _parse_range_endpoints(txt: str):
    s = _clean_for_number_parse(txt)
    if not s:
        return None
    m = _RANGE_RE.match(s.replace(",", "."))
    if not m:
        return None
    try:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", "."))
        return (a, b)
    except Exception:
        return None


def _parse_number_like(txt: str):
    s = _clean_for_number_parse(txt)
    if not s:
        return None

    re_end = _parse_range_endpoints(s)
    if re_end:
        a, b = re_end
        return (a + b) / 2.0

    if _NUM_RE.match(s):
        try:
            return float(s.replace(",", "."))
        except Exception:
            return None

    return None


def _has_percent_anywhere(txt: str) -> bool:
    return bool(_PERCENT_ANYWHERE_RE.search((txt or "")))


# =======================
# Clustering (BBox-based)
# =======================

def _cluster_by_x(tokens, x_tol=18):
    pts = []
    for t in tokens:
        if not t.get("bbox"):
            continue
        cx = _token_center_x(t)
        pts.append((cx, t))
    pts.sort(key=lambda a: a[0])

    cols = []
    cur = []
    cur_x = None

    for cx, t in pts:
        if cur_x is None:
            cur_x = cx
            cur = [t]
            continue
        if abs(cx - cur_x) <= x_tol:
            cur.append(t)
            cur_x = sum(_token_center_x(tt) for tt in cur) / max(1, len(cur))
        else:
            cols.append(cur)
            cur = [t]
            cur_x = cx

    if cur:
        cols.append(cur)

    for c in cols:
        c.sort(key=lambda tt: (_token_center_y(tt), _token_center_x(tt)))

    cols.sort(key=lambda c: sum(_token_center_x(tt) for tt in c) / max(1, len(c)))
    return cols


def _cluster_rows(tokens, y_tol=10):
    items = []
    for t in tokens:
        x0, y0, x1, y1 = t["bbox"]
        items.append((y0, x0, t))
    items.sort(key=lambda a: (a[0], a[1]))

    rows = []
    cur = []
    cur_y = None

    for y0, x0, t in items:
        if cur_y is None:
            cur_y = y0
            cur = [t]
            continue
        if abs(y0 - cur_y) <= y_tol:
            cur.append(t)
        else:
            cur.sort(key=lambda tt: tt["bbox"][0])
            rows.append(cur)
            cur_y = y0
            cur = [t]

    if cur:
        cur.sort(key=lambda tt: tt["bbox"][0])
        rows.append(cur)

    return rows


# =======================
# Matching per riga/colonna
# =======================

def _match_by_y(element_tokens, col_tokens, y_tol=12):
    out = []
    if not element_tokens:
        return out

    col_y = [(_token_center_y(t), t) for t in col_tokens]
    col_y.sort(key=lambda a: a[0])

    for el in element_tokens:
        y = _token_center_y(el)
        best = None
        best_d = None
        for yy, tt in col_y:
            d = abs(yy - y)
            if best_d is None or d < best_d:
                best_d = d
                best = tt
        if best_d is not None and best_d <= y_tol:
            out.append((el, best))
        else:
            out.append((el, None))

    return out


def _build_columns_from_header(header_tokens):
    cols = []
    for t in header_tokens:
        name = _canon_element((t.get("text") or "").strip())
        if not _is_element_token(name):
            continue
        cols.append({"name": name, "cx": _token_center_x(t), "t": t})

    cols.sort(key=lambda c: c["cx"])
    if not cols:
        return []

    centers = [c["cx"] for c in cols]
    bounds = []
    for i in range(len(centers) - 1):
        bounds.append((centers[i] + centers[i + 1]) / 2.0)

    for i, c in enumerate(cols):
        c["x0"] = float("-inf") if i == 0 else bounds[i - 1]
        c["x1"] = float("inf") if i == len(cols) - 1 else bounds[i]

    return cols


def _tokens_in_col(row_tokens, col_def):
    out = []
    for t in row_tokens:
        cx = _token_center_x(t)
        if col_def["x0"] <= cx < col_def["x1"]:
            out.append(t)
    out.sort(key=lambda tt: tt["bbox"][0])
    return out


def _cell_text(tokens):
    s = " ".join((t.get("text") or "").strip() for t in tokens if (t.get("text") or "").strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =======================
# Stats / Scoring (MIN/MAX/VAL)
# =======================

def _row_stats_for_horizontal(data_row_tokens, cols):
    nums = []
    n_elem = len(cols)
    matched = 0
    range_cnt = 0
    unitish_cnt = 0
    per_el = {}

    for c in cols:
        cell_toks = _tokens_in_col(data_row_tokens, c)
        txt = _cell_text(cell_toks)

        if txt and _is_percent_unitish_token(txt):
            unitish_cnt += 1

        v = _parse_number_like(txt) if txt else None
        if v is not None:
            matched += 1
            nums.append(v)
            if _parse_range_endpoints(txt) is not None:
                range_cnt += 1

        per_el[c["name"]] = (v, cell_toks, txt)

    median = statistics.median(nums) if nums else None
    return {
        "coverage": matched / max(1, n_elem),
        "range_rate": range_cnt / max(1, n_elem),
        "unitish_rate": unitish_cnt / max(1, n_elem),
        "median": median,
        "nums": nums,
        "per_el": per_el,
    }


def _extreme_penalty(horizontal_row_stats_list, cols):
    ridx_to_counts = {rs["ridx"]: {"max": 0, "min": 0, "n": 0} for rs in horizontal_row_stats_list}

    for c in cols:
        el = c["name"]
        values = []
        for rs in horizontal_row_stats_list:
            v = rs["stats"]["per_el"].get(el, (None, [], ""))[0]
            if v is not None:
                values.append((rs["ridx"], v))
        if len(values) < 2:
            continue

        min_v = min(v for _, v in values)
        max_v = max(v for _, v in values)
        if abs(max_v - min_v) < 1e-12:
            continue

        for ridx, v in values:
            ridx_to_counts[ridx]["n"] += 1
            if abs(v - max_v) < 1e-12:
                ridx_to_counts[ridx]["max"] += 1
            if abs(v - min_v) < 1e-12:
                ridx_to_counts[ridx]["min"] += 1

    out = {}
    for ridx, cc in ridx_to_counts.items():
        n = max(1, cc["n"])
        out[ridx] = {
            "max_rate": cc["max"] / n,
            "min_rate": cc["min"] / n,
        }
    return out


def _find_header_row(rows):
    best_idx = None
    best_score = 0
    for i, r in enumerate(rows):
        texts = [(t.get("text") or "").strip() for t in r if (t.get("text") or "").strip()]
        elems = [x for x in texts if _is_element_token(x)]
        score = len(elems)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score >= 3:
        return best_idx
    return None


# =======================
# Layout verticale
# =======================

def _col_stats(col_tokens):
    if not col_tokens:
        return {
            "n": 0,
            "unit_ratio": 0.0,
            "elem_ratio": 0.0,
            "num_ratio": 0.0,
            "range_ratio": 0.0,
            "median_num": None,
        }

    n = 0
    unit = 0
    elem = 0
    num = 0
    rng = 0
    nums = []

    for t in col_tokens:
        txt = (t.get("text") or "").strip()
        if not txt:
            continue
        n += 1

        if _is_percent_unitish_token(txt):
            unit += 1

        if _is_element_token(txt):
            elem += 1

        if _parse_range_endpoints(txt) is not None:
            rng += 1

        v = _parse_number_like(txt)
        if v is not None:
            num += 1
            nums.append(v)

    median_num = None
    if nums:
        try:
            median_num = statistics.median(nums)
        except Exception:
            median_num = None

    return {
        "n": n,
        "unit_ratio": (unit / n) if n else 0.0,
        "elem_ratio": (elem / n) if n else 0.0,
        "num_ratio": (num / n) if n else 0.0,
        "range_ratio": (rng / n) if n else 0.0,
        "median_num": median_num,
    }


def _choose_vertical_layout_table(selected_tokens, debug):
    if not selected_tokens:
        debug["vertical"] = {"ok": False, "reason": "no tokens", "n_cols": 0}
        return None, set(), []

    cols = _cluster_by_x(selected_tokens, x_tol=18)
    debug["vertical"] = {"n_cols": len(cols), "ok": False}

    if len(cols) < 2:
        debug["vertical"]["reason"] = "too few columns"
        return None, set(), []

    col_info = []
    for idx, c in enumerate(cols):
        st = _col_stats(c)
        col_info.append({"idx": idx, "tokens": c, "stats": st})

    elem_col = None
    best_score = 0.0
    for ci in col_info:
        st = ci["stats"]
        score = st["elem_ratio"] * max(1, st["n"])
        if score > best_score:
            best_score = score
            elem_col = ci

    if not elem_col or elem_col["stats"]["elem_ratio"] < 0.40:
        debug["vertical"]["reason"] = "no strong element column"
        debug["vertical"]["elem_col"] = None
        debug["vertical"]["col_stats"] = [{"idx": ci["idx"], **ci["stats"]} for ci in col_info]
        return None, set(), []

    elem_tokens_raw = [t for t in elem_col["tokens"] if _is_element_token((t.get("text") or "").strip())]
    if len(elem_tokens_raw) < 3:
        debug["vertical"]["reason"] = "too few element tokens"
        return None, set(), []

    # numeric candidates
    unit_cols = []
    for ci in col_info:
        if ci is elem_col:
            continue
        st = ci["stats"]
        if st["unit_ratio"] >= 0.50 and st["n"] >= 3:
            unit_cols.append(ci)
    unit_idx_set = {c["idx"] for c in unit_cols}

    numeric_candidates = []
    for ci in col_info:
        if ci["idx"] == elem_col["idx"]:
            continue
        if ci["idx"] in unit_idx_set:
            continue
        st = ci["stats"]
        if st["num_ratio"] >= 0.40 and st["n"] >= 3:
            numeric_candidates.append(ci)

    if not numeric_candidates:
        debug["vertical"]["reason"] = "no numeric candidates"
        return None, set(), []

    elem_tokens = sorted(elem_tokens_raw, key=lambda t: _token_center_y(t))

    # coverage + extreme penalty (come nel tuo codice precedente)
    for ci in numeric_candidates:
        pairs = _match_by_y(elem_tokens, ci["tokens"], y_tol=14)
        matched = 0
        matched_nums = []
        for el, mt in pairs:
            if mt is None:
                continue
            v = _parse_number_like((mt.get("text") or "").strip())
            if v is not None:
                matched += 1
                matched_nums.append(v)
        ci["coverage"] = matched / max(1, len(elem_tokens))
        ci["median_matched"] = statistics.median(matched_nums) if matched_nums else None

    good = [c for c in numeric_candidates if c.get("coverage", 0.0) >= 0.50]
    if not good:
        good = sorted(numeric_candidates, key=lambda c: c.get("coverage", 0.0), reverse=True)[:1]

    col_extreme = {c["idx"]: {"max": 0, "min": 0, "n": 0} for c in good}
    for el in elem_tokens:
        vals = []
        for ci in good:
            pairs = _match_by_y([el], ci["tokens"], y_tol=14)
            mt = pairs[0][1] if pairs else None
            if mt is None:
                continue
            v = _parse_number_like((mt.get("text") or "").strip())
            if v is None:
                continue
            vals.append((ci["idx"], v))
        if len(vals) < 2:
            continue
        min_v = min(v for _, v in vals)
        max_v = max(v for _, v in vals)
        if abs(max_v - min_v) < 1e-12:
            continue
        for idx, v in vals:
            col_extreme[idx]["n"] += 1
            if abs(v - max_v) < 1e-12:
                col_extreme[idx]["max"] += 1
            if abs(v - min_v) < 1e-12:
                col_extreme[idx]["min"] += 1

    best = None
    best_sc = None
    for ci in good:
        cc = col_extreme.get(ci["idx"], {"max": 0, "min": 0, "n": 0})
        denom = max(1, cc["n"])
        max_rate = cc["max"] / denom
        min_rate = cc["min"] / denom
        sc = (
            2.0 * ci.get("coverage", 0.0)
            - 1.2 * max_rate
            - 0.6 * min_rate
            - 0.2 * ci["stats"].get("range_ratio", 0.0)
        )
        if best_sc is None or sc > best_sc:
            best_sc = sc
            best = ci

    if not best:
        debug["vertical"]["reason"] = "no chosen column"
        return None, set(), []

    pairs = _match_by_y(elem_tokens, best["tokens"], y_tol=14)

    table = []
    used_ids = set()
    bindings = []  # per riga: el_ids, val_ids
    for el, mt in pairs:
        el_name = _canon_element((el.get("text") or "").strip())
        if not el_name:
            continue
        val = ""
        el_id = el.get("id", None)
        val_id = None
        if mt is not None:
            val = (mt.get("text") or "").strip()
            val_id = mt.get("id", None)

        row_el_ids = [el_id] if isinstance(el_id, int) else []
        row_val_ids = [val_id] if isinstance(val_id, int) else []

        table.append({"elemento": el_name, "valore": val})
        bindings.append({"elemento": el_name, "valore": val, "el_ids": row_el_ids, "val_ids": row_val_ids})

        for tid in row_el_ids + row_val_ids:
            used_ids.add(tid)

    non_empty = sum(1 for r in table if (r.get("valore") or "").strip())
    if non_empty < 2:
        debug["vertical"]["reason"] = "too few values after build"
        return None, set(), []

    debug["vertical"]["ok"] = True
    debug["vertical"]["elem_col"] = elem_col["idx"]
    debug["vertical"]["chosen_numeric_col"] = best["idx"]
    debug["vertical"]["col_stats"] = [{"idx": ci["idx"], **ci["stats"]} for ci in col_info]
    return table, used_ids, bindings


# =======================
# Layout orizzontale
# =======================

def _choose_horizontal_layout_table(selected_tokens, debug):
    rows = _cluster_rows(selected_tokens, y_tol=10)
    debug["horizontal"] = {"n_rows": len(rows), "ok": False}

    if not rows:
        debug["horizontal"]["reason"] = "no rows"
        return None, set(), []

    header_idx = _find_header_row(rows)
    if header_idx is None:
        debug["horizontal"]["reason"] = "no header row"
        return None, set(), []

    header_row = rows[header_idx]
    cols = _build_columns_from_header(header_row)
    if not cols:
        debug["horizontal"]["reason"] = "no columns from header"
        return None, set(), []

    data_rows = rows[header_idx + 1 :]
    if not data_rows:
        debug["horizontal"]["reason"] = "no data rows"
        return None, set(), []

    row_stats_list = []
    for ridx, r in enumerate(data_rows, start=header_idx + 1):
        st = _row_stats_for_horizontal(r, cols)
        row_stats_list.append({"ridx": ridx, "stats": st})

    numeric_rows = [rs for rs in row_stats_list if rs["stats"]["coverage"] >= 0.40]
    if not numeric_rows:
        numeric_rows = sorted(row_stats_list, key=lambda rs: rs["stats"]["coverage"], reverse=True)[:1]

    extreme = _extreme_penalty(numeric_rows, cols)

    best = None
    best_sc = None
    candidates_dbg = []
    for rs in numeric_rows:
        ridx = rs["ridx"]
        st = rs["stats"]
        ex = extreme.get(ridx, {"max_rate": 0.0, "min_rate": 0.0})

        sc = (
            2.2 * st["coverage"]
            - 1.3 * ex["max_rate"]
            - 0.7 * ex["min_rate"]
            - 1.8 * st["unitish_rate"]
            - 0.15 * st["range_rate"]
        )

        candidates_dbg.append({
            "ridx": ridx,
            "coverage": round(st["coverage"], 3),
            "median": st["median"],
            "range_rate": round(st["range_rate"], 3),
            "unitish_rate": round(st["unitish_rate"], 3),
            "max_rate": round(ex["max_rate"], 3),
            "min_rate": round(ex["min_rate"], 3),
            "score": round(sc, 4),
        })

        if best_sc is None or sc > best_sc:
            best_sc = sc
            best = rs

    if not best:
        debug["horizontal"]["reason"] = "no chosen row"
        debug["horizontal"]["candidates"] = candidates_dbg
        return None, set(), []

    chosen_ridx = best["ridx"]
    debug["horizontal"]["header_idx"] = header_idx
    debug["horizontal"]["chosen_ridx"] = chosen_ridx
    debug["horizontal"]["candidates"] = candidates_dbg
    debug["horizontal"]["ok"] = True

    chosen_row_tokens = rows[chosen_ridx]

    used_ids = set()
    table = []
    bindings = []

    # Mappa elemento->token id header
    header_el_ids = {}
    for c in cols:
        ht = c.get("t")
        hid = ht.get("id", None) if isinstance(ht, dict) else None
        if isinstance(hid, int):
            header_el_ids[c["name"]] = hid

    for c in cols:
        cell_toks = _tokens_in_col(chosen_row_tokens, c)
        cell_txt = _cell_text(cell_toks)

        el_name = c["name"]
        el_id = header_el_ids.get(el_name, None)
        val_ids = [tt.get("id") for tt in cell_toks if isinstance(tt.get("id"), int)]

        table.append({"elemento": el_name, "valore": cell_txt})
        bindings.append({
            "elemento": el_name,
            "valore": cell_txt,
            "el_ids": [el_id] if isinstance(el_id, int) else [],
            "val_ids": val_ids,
        })

        if isinstance(el_id, int):
            used_ids.add(el_id)
        for vid in val_ids:
            used_ids.add(vid)

    non_empty = sum(1 for r in table if (r.get("valore") or "").strip())
    if non_empty < 2:
        debug["horizontal"]["ok"] = False
        debug["horizontal"]["reason"] = "too few values in chosen row"
        return None, set(), []

    return table, used_ids, bindings


# =======================
# API principale estrazione
# =======================

def extract_chem_table_from_tokens(selected_tokens):
    """
    Ritorna:
      table: [{"elemento":..,"valore":..}, ...]
      used_token_ids: set(int)
      debug_dump: string
    """
    debug = {"layout": None}

    table_v, used_v, _ = _choose_vertical_layout_table(selected_tokens, debug)
    if table_v:
        debug["layout"] = "vertical"
        return table_v, used_v, _format_debug(debug)

    table_h, used_h, _ = _choose_horizontal_layout_table(selected_tokens, debug)
    if table_h:
        debug["layout"] = "horizontal"
        return table_h, used_h, _format_debug(debug)

    debug["layout"] = "none"
    return [], set(), _format_debug(debug)


def _extract_bindings_from_tokens(selected_tokens):
    """
    Interno GUI: come extract_chem_table_from_tokens ma ritorna direttamente bindings con ids.
    """
    debug = {"layout": None}

    table_v, used_v, bindings_v = _choose_vertical_layout_table(selected_tokens, debug)
    if table_v:
        debug["layout"] = "vertical"
        return bindings_v, used_v, _format_debug(debug)

    table_h, used_h, bindings_h = _choose_horizontal_layout_table(selected_tokens, debug)
    if table_h:
        debug["layout"] = "horizontal"
        return bindings_h, used_h, _format_debug(debug)

    debug["layout"] = "none"
    return [], set(), _format_debug(debug)


def _format_debug(debug: dict) -> str:
    lines = []
    layout = debug.get("layout")
    lines.append(f"layout: {layout!r}")

    if "vertical" in debug:
        v = debug["vertical"]
        lines.append(f"vertical: n_cols={v.get('n_cols')}")
        lines.append(f"  ok: {v.get('ok')}")
        if v.get("reason"):
            lines.append(f"  reason: {v.get('reason')!r}")
        if v.get("elem_col") is not None:
            lines.append(f"  elem_col: {v.get('elem_col')}")
        if v.get("chosen_numeric_col") is not None:
            lines.append(f"  chosen_numeric_col: {v.get('chosen_numeric_col')}")
        if v.get("col_stats"):
            lines.append("  col_stats:")
            for cs in v["col_stats"]:
                lines.append(
                    f"    - idx:{cs.get('idx')} n:{cs.get('n')} "
                    f"elem_ratio:{round(cs.get('elem_ratio',0.0),3)} "
                    f"unit_ratio:{round(cs.get('unit_ratio',0.0),3)} "
                    f"num_ratio:{round(cs.get('num_ratio',0.0),3)} "
                    f"range_ratio:{round(cs.get('range_ratio',0.0),3)} "
                    f"median_num:{cs.get('median_num')}"
                )

    if "horizontal" in debug:
        h = debug["horizontal"]
        lines.append(f"horizontal: n_rows={h.get('n_rows')}")
        lines.append(f"  ok: {h.get('ok')}")
        if h.get("reason"):
            lines.append(f"  reason: {h.get('reason')!r}")
        if h.get("header_idx") is not None:
            lines.append(f"  header_idx: {h.get('header_idx')}")
        if h.get("chosen_ridx") is not None:
            lines.append(f"  chosen_ridx: {h.get('chosen_ridx')}")
        if h.get("candidates"):
            lines.append("  candidates:")
            for c in h["candidates"]:
                lines.append(
                    f"    - ridx:{c.get('ridx')} cov:{c.get('coverage')} "
                    f"median:{c.get('median')} range_rate:{c.get('range_rate')} "
                    f"unitish:{c.get('unitish_rate')} max_rate:{c.get('max_rate')} "
                    f"min_rate:{c.get('min_rate')} score:{c.get('score')}"
                )

    return "\n".join(lines)


# =======================
# GUI
# =======================

