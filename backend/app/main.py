import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.responses import StreamingResponse
import io

from .settings import PDF_FOLDER, ensure_dirs
from .ocr import list_pdfs, render_page_image, get_tokens_for_page
from .ai import extract_fields_from_tokens
from .chem_table import extract_chem_table_from_tokens, _extract_bindings_from_tokens
from .pipeline import analyze_pdf_page
from .openai_flow import openai_full, openai_pdf, openai_refine, ai_trained

# simple in-memory cache for pdf pages/tokens
PDF_CACHE = {}
TOKENS_CACHE = {}


def _purge_pdf_cache(pdf_name: str):
    if not pdf_name:
        return
    prefix = f"{pdf_name}::"
    for key in list(TOKENS_CACHE.keys()):
        if key.startswith(prefix):
            TOKENS_CACHE.pop(key, None)

app = FastAPI()

# CORS is not needed when using Vite's dev proxy.
# Uncomment if you call the API directly from the browser on a different origin.
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/pdfs")
def get_pdfs():
    return {"pdfs": list_pdfs(), "pdf_folder": PDF_FOLDER}


@app.post("/api/pdfs/upload")
def upload_pdf(file: UploadFile = File(...)):
    try:
        ensure_dirs()
        filename = os.path.basename(file.filename)
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        _purge_pdf_cache(filename)
        out_path = os.path.join(PDF_FOLDER, filename)
        if os.path.exists(out_path):
            base, ext = os.path.splitext(filename)
            i = 1
            while True:
                candidate = f"{base} ({i}){ext}"
                out_path = os.path.join(PDF_FOLDER, candidate)
                if not os.path.exists(out_path):
                    filename = candidate
                    break
                i += 1
        with open(out_path, "wb") as f:
            f.write(file.file.read())
        return {"ok": True, "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.delete("/api/pdfs/{pdf_name}")
def delete_pdf(pdf_name: str):
    try:
        ensure_dirs()
        import time
        import gc
        safe_name = os.path.basename(pdf_name)
        target = os.path.join(PDF_FOLDER, safe_name)
        if not os.path.exists(target):
            raise HTTPException(status_code=404, detail="PDF not found")
        last_err = None
        for _ in range(8):
            try:
                os.remove(target)
                last_err = None
                break
            except PermissionError as e:
                last_err = e
                gc.collect()
                time.sleep(0.35)
        if last_err:
            raise last_err
        _purge_pdf_cache(safe_name)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


@app.get("/api/pdfs/{pdf_name}/pages")
def get_pdf_pages(pdf_name: str):
    try:
        import fitz
        doc = fitz.open(f"{PDF_FOLDER}\\{pdf_name}")
        try:
            return {"pages": doc.page_count}
        finally:
            doc.close()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Cannot open PDF: {e}")


@app.get("/api/pdfs/{pdf_name}/pages/{page_idx}/image")
def get_pdf_page_image(pdf_name: str, page_idx: int):
    try:
        img = render_page_image(pdf_name, page_idx)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Cannot render page: {e}")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/pdfs/{pdf_name}/pages/{page_idx}/tokens")
def get_pdf_page_tokens(pdf_name: str, page_idx: int):
    try:
        cache_key = f"{pdf_name}::{page_idx}"
        cached = TOKENS_CACHE.get(cache_key)
        if cached:
            return cached
        img, tokens = analyze_pdf_page(pdf_name, page_idx, zoom=2.0)
        payload = {"image_size": {"width": img.size[0], "height": img.size[1]}, "tokens": tokens}
        TOKENS_CACHE[cache_key] = payload
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token extraction failed: {e}")


@app.post("/api/ai/extract")
def ai_extract(payload: dict, x_openai_key: str = Header(default=None, alias="X-OpenAI-Key")):
    pdf_name = payload.get("pdf_name")
    page_idx = payload.get("page_idx")
    tokens = payload.get("tokens")
    if tokens is None:
        if pdf_name is None:
            raise HTTPException(status_code=400, detail="Provide tokens or pdf_name")
        if page_idx is None:
            # aggregate tokens from all pages with unique ids
            try:
                import fitz
                doc = fitz.open(f"{PDF_FOLDER}\\{pdf_name}")
                try:
                    all_tokens = []
                    next_id = 0
                    for p in range(doc.page_count):
                        page_tokens = get_tokens_for_page(pdf_name, int(p))["tokens"]
                        for t in page_tokens:
                            t = dict(t)
                            t["id"] = next_id
                            next_id += 1
                            all_tokens.append(t)
                    tokens = all_tokens
                finally:
                    doc.close()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Token extraction failed: {e}")
        else:
            tokens = get_tokens_for_page(pdf_name, int(page_idx))["tokens"]
    try:
        result = extract_fields_from_tokens(tokens, api_key=x_openai_key)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI extraction failed: {e}")


@app.post("/api/ai/full")
def ai_full(payload: dict, x_openai_key: str = Header(default=None, alias="X-OpenAI-Key")):
    pdf_name = payload.get("pdf_name")
    if not pdf_name:
        raise HTTPException(status_code=400, detail="pdf_name required")
    try:
        return openai_full(pdf_name, cache=PDF_CACHE, api_key=x_openai_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI full failed: {e}")


@app.post("/api/ai/pdf")
def ai_pdf(payload: dict, x_openai_key: str = Header(default=None, alias="X-OpenAI-Key")):
    pdf_name = payload.get("pdf_name")
    if not pdf_name:
        raise HTTPException(status_code=400, detail="pdf_name required")
    try:
        return openai_pdf(pdf_name, cache=PDF_CACHE, api_key=x_openai_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI pdf failed: {e}")


@app.post("/api/ai/refine")
def ai_refine(payload: dict, x_openai_key: str = Header(default=None, alias="X-OpenAI-Key")):
    pdf_name = payload.get("pdf_name")
    current_fields = payload.get("fields") or {}
    table_hints = payload.get("table_hints") or {}
    if not pdf_name:
        raise HTTPException(status_code=400, detail="pdf_name required")
    try:
        return openai_refine(pdf_name, current_fields, table_hints, cache=PDF_CACHE, api_key=x_openai_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI refine failed: {e}")


@app.post("/api/ai/trained")
def ai_trained_endpoint(payload: dict, x_openai_key: str = Header(default=None, alias="X-OpenAI-Key")):
    pdf_name = payload.get("pdf_name")
    if not pdf_name:
        raise HTTPException(status_code=400, detail="pdf_name required")
    try:
        return ai_trained(pdf_name, cache=PDF_CACHE, api_key=x_openai_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI trained failed: {e}")


@app.post("/api/chem/extract")
def chem_extract(payload: dict):
    tokens = payload.get("tokens")
    if not tokens:
        raise HTTPException(status_code=400, detail="Provide tokens")
    try:
        bindings, used_ids, debug = _extract_bindings_from_tokens(tokens)
        table = [{"elemento": b.get("elemento", ""), "valore": b.get("valore", "")} for b in bindings]
        return {
            "table": table,
            "bindings": bindings,
            "used_token_ids": list(used_ids),
            "debug": debug,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chem extraction failed: {e}")


@app.post("/api/pdfs/{pdf_name}/save")
def save_result(pdf_name: str, payload: dict):
    try:
        import json
        out_path = os.path.join(PDF_FOLDER, f"{pdf_name}_ai.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {"ok": True, "path": out_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {e}")
