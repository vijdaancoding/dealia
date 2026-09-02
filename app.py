"""
Attachment extraction sidecar for the HubSpot ticket indexer.

Why this exists as a service rather than more n8n nodes
-------------------------------------------------------
n8n's Extract From File node covers PDF, CSV and XLSX but has no DOCX support,
and its XLSX mode emits one item per row - which destroys the 1:1 item
alignment the attachment lane depends on. Working around that inside n8n means
unzipping OOXML with the Compression node and reassembling sheets from raw XML,
which is a lot of fragile machinery to maintain.

One HTTP endpoint that always returns exactly one JSON object per file keeps
the workflow linear and puts the format-specific mess where libraries already
solve it.

POST /extract
  {"extract": true, "url": "...", "kind": "pdf", "name": "x.pdf",
   "extension": "pdf", "asset_id": "123__att_001", "max_chars": 40000}

Always responds 200 with {"status": "ok"|"skipped"|"failed", ...} so a bad file
never fails the n8n node and stalls a crawl.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger("extractor")
app = FastAPI(title="ticket-attachment-extractor")

MAX_BYTES = int(os.getenv("MAX_BYTES", 40 * 1024 * 1024))
DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT", 90))
SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", 8))

# Vision captioning is optional. Without it images still contribute OCR text,
# which on support screenshots is usually the highest-value string on the
# ticket - the literal error message.
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT", "")
VISION_KEY = os.getenv("VISION_KEY", "")
VISION_PROMPT = (
    "This is an attachment from a customer support ticket. In 2-3 sentences, "
    "describe what it shows: the product screen or document type, any error or "
    "warning message (quote error text exactly), what is highlighted or circled, "
    "and any visible identifiers such as record IDs or dates. If it is a photo "
    "rather than a screenshot, say so. Do not speculate beyond what is visible."
)


class ExtractRequest(BaseModel):
    extract: bool = True
    url: str = ""
    kind: str = "file"
    name: str = ""
    extension: str = ""
    asset_id: str = ""
    max_chars: int = 40000


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
#  Format handlers
# ---------------------------------------------------------------------------
def extract_pdf(data: bytes, req: ExtractRequest) -> dict[str, Any]:
    import pdfplumber

    pages, tables_found = [], 0
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            body = page.extract_text() or ""
            # Tables extracted as pipe rows survive chunking far better than
            # the whitespace-aligned text pdfplumber returns by default.
            for tbl in page.extract_tables() or []:
                tables_found += 1
                rows = [
                    " | ".join((c or "").strip() for c in row)
                    for row in tbl[:40]
                ]
                body += "\n\n" + "\n".join(rows)
            if body.strip():
                # The page marker lets a retrieved chunk tell the reader where
                # in a 40-page statement the text came from.
                pages.append(f"[page {i}]\n{body.strip()}")
            if sum(len(p) for p in pages) > req.max_chars:
                break

    text = _clean("\n\n".join(pages))
    scanned = len(text) < 40 * max(1, n_pages) // 10

    # A PDF with almost no extractable text is a scan. OCR it instead.
    if scanned:
        ocr = ocr_pdf(data, req)
        if ocr:
            return {"status": "ok", "text": _clean(ocr)[: req.max_chars],
                    "pages": n_pages, "ocr_chars": len(ocr), "scanned": True}

    return {"status": "ok", "text": text[: req.max_chars], "pages": n_pages,
            "tables_found": tables_found, "scanned": False}


def ocr_pdf(data: bytes, req: ExtractRequest) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        out = []
        for img in convert_from_bytes(data, dpi=200, fmt="png")[:20]:
            out.append(pytesseract.image_to_string(img))
            if sum(len(o) for o in out) > req.max_chars:
                break
        return "\n\n".join(out)
    except Exception as exc:                      # noqa: BLE001
        log.warning("pdf ocr failed for %s: %s", req.asset_id, exc)
        return ""


def extract_docx(data: bytes, req: ExtractRequest) -> dict[str, Any]:
    import docx

    d = docx.Document(io.BytesIO(data))
    parts = []
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        # Heading style is preserved as markdown so the chunker's
        # heading-awareness works on document text too.
        style = (p.style.name or "").lower()
        if style.startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "2"
            parts.append(f"{'#' * min(int(level) + 1, 6)} {t}")
        elif style.startswith("list"):
            parts.append(f"- {t}")
        else:
            parts.append(t)

    for tbl in d.tables:
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in tbl.rows[:60]]
        if rows:
            parts.append("\n".join(rows))

    return {"status": "ok", "text": _clean("\n\n".join(parts))[: req.max_chars],
            "paragraphs": len(d.paragraphs), "tables_found": len(d.tables)}


def extract_spreadsheet(data: bytes, req: ExtractRequest) -> dict[str, Any]:
    """Summarise structure; never dump every cell.

    A 400-row export embedded cell by cell produces vectors dominated by
    numbers and retrieves for nothing. What people actually search for is the
    shape: which sheet, which columns, roughly what magnitude.
    """
    import pandas as pd

    ext = (req.extension or "").lower()
    if ext in ("csv", "tsv"):
        sep = "\t" if ext == "tsv" else ","
        sheets = {"Sheet1": pd.read_csv(io.BytesIO(data), sep=sep,
                                        dtype=str, on_bad_lines="skip",
                                        nrows=100_000)}
    else:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=object)

    tables, blurbs = [], []
    for sheet_name, df in list(sheets.items())[:12]:
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        headers = [str(c) for c in df.columns][:40]

        col_types, numeric = {}, {}
        for col in df.columns[:40]:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().sum() >= max(3, 0.6 * len(df)):
                col_types[str(col)] = "number"
                numeric[str(col)] = {
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                    "sum": round(float(series.sum()), 4),
                }
            else:
                col_types[str(col)] = "text"

        sample = df.head(SAMPLE_ROWS).astype(str)
        md = ["| " + " | ".join(headers) + " |",
              "| " + " | ".join("---" for _ in headers) + " |"]
        for _, row in sample.iterrows():
            md.append("| " + " | ".join(str(v)[:60] for v in row.tolist()[:40]) + " |")

        tables.append({
            "sheet": str(sheet_name), "rows": int(len(df)), "cols": int(df.shape[1]),
            "headers": headers, "column_types": col_types,
            "numeric_summary": numeric, "sample_markdown": "\n".join(md),
        })
        blurbs.append(f'Sheet "{sheet_name}": {len(df)} rows x {df.shape[1]} columns. '
                      f'Columns: {", ".join(headers)}.')

    return {"status": "ok", "text": _clean("\n".join(blurbs))[: req.max_chars],
            "tables": tables, "sheets": len(tables)}


def extract_pptx(data: bytes, req: ExtractRequest) -> dict[str, Any]:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        bits = [sh.text.strip() for sh in slide.shapes
                if getattr(sh, "has_text_frame", False) and sh.text.strip()]
        notes = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        if bits or notes:
            parts.append(f"[slide {i}]\n" + "\n".join(bits)
                         + (f"\nSpeaker notes: {notes}" if notes else ""))
    return {"status": "ok", "text": _clean("\n\n".join(parts))[: req.max_chars],
            "pages": len(prs.slides)}


def extract_image(data: bytes, req: ExtractRequest) -> dict[str, Any]:
    ocr_text, vision = "", ""

    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        # Upscaling small screenshots measurably improves OCR on UI fonts.
        if max(img.size) < 1000:
            img = img.resize((img.width * 2, img.height * 2))
        ocr_text = _clean(pytesseract.image_to_string(img))
        # Tesseract returns punctuation soup on photos and gradients; drop
        # anything without a run of real words rather than indexing noise.
        if len(re.findall(r"[A-Za-z]{3,}", ocr_text)) < 3:
            ocr_text = ""
    except Exception as exc:                      # noqa: BLE001
        log.warning("ocr failed for %s: %s", req.asset_id, exc)

    if VISION_ENDPOINT and VISION_KEY:
        try:
            vision = caption_image(data, req)
        except Exception as exc:                  # noqa: BLE001
            log.warning("vision failed for %s: %s", req.asset_id, exc)

    return {"status": "ok", "text": "", "ocr_text": ocr_text,
            "ocr_chars": len(ocr_text), "vision_description": vision}


def caption_image(data: bytes, req: ExtractRequest) -> str:
    mime = {"png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(
        (req.extension or "").lower(), "image/jpeg")
    b64 = base64.b64encode(data).decode()
    payload = {
        "temperature": 0,
        "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
    }
    with httpx.Client(timeout=120) as client:
        r = client.post(VISION_ENDPOINT, json=payload,
                        headers={"api-key": VISION_KEY,
                                 "content-type": "application/json"})
        r.raise_for_status()
        return _clean(r.json()["choices"][0]["message"]["content"])


def extract_text(data: bytes, req: ExtractRequest) -> dict[str, Any]:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return {"status": "ok",
                    "text": _clean(data.decode(enc))[: req.max_chars]}
        except UnicodeDecodeError:
            continue
    return {"status": "failed", "error": "undecodable text file"}


HANDLERS = {
    "pdf": extract_pdf,
    "document": extract_docx,
    "spreadsheet": extract_spreadsheet,
    "presentation": extract_pptx,
    "image": extract_image,
    "text": extract_text,
}


# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "handlers": sorted(HANDLERS), "vision": bool(VISION_ENDPOINT)}


@app.post("/extract")
def extract(req: ExtractRequest) -> dict[str, Any]:
    if not req.extract or not req.url:
        return {"status": "skipped", "asset_id": req.asset_id}

    handler = HANDLERS.get(req.kind)
    if handler is None:
        return {"status": "skipped", "asset_id": req.asset_id,
                "error": f"no handler for kind={req.kind}"}

    # Every failure below returns 200. The n8n node has onError set to continue,
    # but a non-2xx would still burn its retries and slow the crawl for a file
    # that is never going to parse.
    try:
        with httpx.Client(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            with client.stream("GET", req.url) as resp:
                resp.raise_for_status()
                buf = io.BytesIO()
                for chunk in resp.iter_bytes(65536):
                    buf.write(chunk)
                    if buf.tell() > MAX_BYTES:
                        return {"status": "skipped", "asset_id": req.asset_id,
                                "error": "exceeds MAX_BYTES"}
                data = buf.getvalue()
    except Exception as exc:                      # noqa: BLE001
        return {"status": "failed", "asset_id": req.asset_id,
                "error": f"download: {exc}"[:300]}

    if not data:
        return {"status": "failed", "asset_id": req.asset_id, "error": "empty file"}

    try:
        result = handler(data, req)
    except Exception as exc:                      # noqa: BLE001
        log.exception("extract failed for %s", req.asset_id)
        return {"status": "failed", "asset_id": req.asset_id,
                "error": f"{type(exc).__name__}: {exc}"[:300]}

    result.setdefault("status", "ok")
    result["asset_id"] = req.asset_id
    result["size_bytes"] = len(data)
    # Same physical file can arrive under different HubSpot file IDs - most
    # commonly a screenshot pasted into an email that gets re-uploaded fresh
    # every time the thread is quoted forward. A byte-exact hash is what
    # Assemble Ticket uses to collapse those back into one indexed asset.
    result["content_hash"] = hashlib.sha256(data).hexdigest()
    return result
