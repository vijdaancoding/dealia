from __future__ import annotations

import base64
import csv as csv_module
import gc
import hashlib
import hmac
import io
import logging
import os
import re
from typing import Any, Iterable

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

log = logging.getLogger("extractor")
app = FastAPI(title="ticket-attachment-extractor")

MAX_BYTES = int(os.getenv("MAX_BYTES", 40 * 1024 * 1024))
DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT", 90))
SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", 8))

# Bounds that exist purely to keep peak memory predictable on a small box.
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 100))     # text extraction
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", 20))      # rasterised OCR
MAX_SHEETS = int(os.getenv("MAX_SHEETS", 12))
MAX_COLS = int(os.getenv("MAX_COLS", 40))
MAX_ROWS_SCANNED = int(os.getenv("MAX_ROWS_SCANNED", 20000))
# tesseract's footprint tracks pixel count, not file size: a 12 MP phone photo
# costs far more than a 1 MP screenshot of the same byte length.
MAX_OCR_PIXELS = int(os.getenv("MAX_OCR_PIXELS", 4_000_000))
# What gets base64'd and sent to the vision endpoint. Downscaling first keeps
# the encoded copy small - base64 inflates by ~33% and the whole string is
# held in memory for the duration of the request.
MAX_VISION_PIXELS = int(os.getenv("MAX_VISION_PIXELS", 1_200_000))

EXTRACTOR_SECRET = os.getenv("EXTRACTOR_SECRET", "")
if not EXTRACTOR_SECRET:
    log.warning("EXTRACTOR_SECRET is not set - /extract is running with NO "
                "authentication. Fine for local-only testing, never for a "
                "publicly reachable deployment.")


def require_secret(x_extractor_secret: str = Header(default="")) -> None:
    # Constant-time compare so response timing can't leak the secret.
    if EXTRACTOR_SECRET and not hmac.compare_digest(x_extractor_secret, EXTRACTOR_SECRET):
        raise HTTPException(status_code=401,
                            detail="invalid or missing X-Extractor-Secret header")


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


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
#  PDF
# ---------------------------------------------------------------------------
def extract_pdf(buf: io.BytesIO, req: ExtractRequest) -> dict[str, Any]:
    import pdfplumber

    pages: list[str] = []
    tables_found = 0
    chars = 0

    with pdfplumber.open(buf) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages[:MAX_PDF_PAGES], 1):
            try:
                body = page.extract_text() or ""
                # Pipe rows survive chunking far better than the
                # whitespace-aligned text pdfplumber returns by default.
                for tbl in page.extract_tables() or []:
                    tables_found += 1
                    body += "\n\n" + "\n".join(
                        " | ".join((c or "").strip() for c in row) for row in tbl[:40]
                    )
            finally:
                # pdfplumber caches parsed objects per page; without this a
                # long PDF holds every page it has touched in memory at once.
                page.flush_cache()

            if body.strip():
                # The page marker lets a retrieved chunk tell the reader where
                # in a 40-page statement the text came from.
                pages.append(f"[page {i}]\n{body.strip()}")
                chars += len(body)
            if chars > req.max_chars:
                break

    text = _clean("\n\n".join(pages))
    del pages

    # A real scan has essentially NO text layer - not merely sparse text. Keep
    # this threshold low: a false positive here silently costs a full OCR pass
    # (the most expensive thing this service does) on a PDF that already had
    # perfectly good text, e.g. a short cover letter or a title page.
    scanned = chars < 20 * max(1, min(n_pages, MAX_PDF_PAGES))
    if scanned:
        ocr = _ocr_pdf(buf, req)
        if ocr:
            return {"status": "ok", "text": _clean(ocr)[: req.max_chars],
                    "pages": n_pages, "ocr_chars": len(ocr), "scanned": True}

    return {"status": "ok", "text": text[: req.max_chars], "pages": n_pages,
            "tables_found": tables_found, "scanned": False}


def _ocr_pdf(buf: io.BytesIO, req: ExtractRequest) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        buf.seek(0)
        data = buf.read()
        out: list[str] = []
        total = 0
        # first_page/last_page make poppler itself stop early. Rasterising the
        # whole document and then slicing in Python would hold every page in
        # memory at once, which is enough on its own to OOM a small container.
        for start in range(1, MAX_OCR_PAGES + 1):
            imgs = convert_from_bytes(data, dpi=150, fmt="png",
                                      first_page=start, last_page=start)
            if not imgs:
                break
            img = imgs[0]
            try:
                page_text = pytesseract.image_to_string(img)
            finally:
                img.close()
                del imgs, img
            out.append(page_text)
            total += len(page_text)
            if total > req.max_chars:
                break
        del data
        return "\n\n".join(out)
    except Exception as exc:                      # noqa: BLE001
        log.warning("pdf ocr failed for %s: %s", req.asset_id, exc)
        return ""


# ---------------------------------------------------------------------------
#  Word
# ---------------------------------------------------------------------------
def extract_docx(buf: io.BytesIO, req: ExtractRequest) -> dict[str, Any]:
    import docx

    d = docx.Document(buf)
    parts: list[str] = []
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

    n_tables = len(d.tables)
    for tbl in d.tables:
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in tbl.rows[:60]]
        if rows:
            parts.append("\n".join(rows))

    n_paras = len(d.paragraphs)
    del d
    return {"status": "ok", "text": _clean("\n\n".join(parts))[: req.max_chars],
            "paragraphs": n_paras, "tables_found": n_tables}


# ---------------------------------------------------------------------------
#  Spreadsheets - structure, not cells
# ---------------------------------------------------------------------------
# A 400-row export embedded cell by cell produces vectors dominated by numbers
# and retrieves for nothing. What people search for is the shape: which sheet,
# which columns, roughly what magnitude.
def _as_number(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _summarise_rows(rows: Iterable[Any], sheet_name: str) -> dict[str, Any] | None:
    """Single streaming pass: headers, sample, per-column stats. Never holds
    more than SAMPLE_ROWS rows at a time regardless of sheet size."""
    headers: list[str] = []
    sample: list[list[str]] = []
    stats: dict[str, dict[str, float]] = {}
    numeric_n: dict[str, int] = {}
    n_rows = 0

    for idx, row in enumerate(rows):
        vals = list(row)[:MAX_COLS]
        if idx == 0:
            headers = [str(v).strip() if v is not None and str(v).strip() else f"col{i+1}"
                       for i, v in enumerate(vals)]
            continue
        if not headers:
            break
        if all(v is None or not str(v).strip() for v in vals):
            continue

        n_rows += 1
        if len(sample) < SAMPLE_ROWS:
            sample.append([("" if v is None else str(v))[:60] for v in vals])

        for i, v in enumerate(vals[: len(headers)]):
            num = _as_number(v)
            if num is None:
                continue
            key = headers[i]
            s = stats.get(key)
            if s is None:
                stats[key] = {"min": num, "max": num, "sum": num}
            else:
                if num < s["min"]:
                    s["min"] = num
                if num > s["max"]:
                    s["max"] = num
                s["sum"] += num
            numeric_n[key] = numeric_n.get(key, 0) + 1

        if n_rows >= MAX_ROWS_SCANNED:
            break

    if not headers or not n_rows:
        return None

    # A column counts as numeric only if most of its values parsed as numbers -
    # otherwise a text column with a few stray digits would report a bogus sum.
    threshold = max(3, int(0.6 * n_rows))
    col_types = {h: ("number" if numeric_n.get(h, 0) >= threshold else "text")
                 for h in headers}
    numeric_summary = {h: {"min": round(s["min"], 4),
                           "max": round(s["max"], 4),
                           "sum": round(s["sum"], 4)}
                       for h, s in stats.items() if col_types.get(h) == "number"}

    md = ["| " + " | ".join(headers) + " |",
          "| " + " | ".join("---" for _ in headers) + " |"]
    for r in sample:
        cells = (r + [""] * len(headers))[: len(headers)]
        md.append("| " + " | ".join(cells) + " |")

    return {"sheet": str(sheet_name), "rows": n_rows, "cols": len(headers),
            "headers": headers, "column_types": col_types,
            "numeric_summary": numeric_summary, "sample_markdown": "\n".join(md)}


def extract_spreadsheet(buf: io.BytesIO, req: ExtractRequest) -> dict[str, Any]:
    ext = (req.extension or "").lower()
    tables: list[dict[str, Any]] = []

    if ext in ("csv", "tsv"):
        buf.seek(0)
        text = _decode(buf.read())
        reader = csv_module.reader(io.StringIO(text),
                                   delimiter="\t" if ext == "tsv" else ",")
        summary = _summarise_rows(reader, "Sheet1")
        del text
        if summary:
            tables.append(summary)
    else:
        import openpyxl

        # read_only streams rows off disk instead of building a full in-memory
        # object graph; data_only skips formula ASTs and returns cached values.
        wb = openpyxl.load_workbook(buf, read_only=True, data_only=True)
        try:
            for ws in list(wb.worksheets)[:MAX_SHEETS]:
                summary = _summarise_rows(ws.iter_rows(values_only=True), ws.title)
                if summary:
                    tables.append(summary)
        finally:
            wb.close()

    blurbs = [f'Sheet "{t["sheet"]}": {t["rows"]} rows x {t["cols"]} columns. '
              f'Columns: {", ".join(t["headers"])}.' for t in tables]

    return {"status": "ok", "text": _clean("\n".join(blurbs))[: req.max_chars],
            "tables": tables, "sheets": len(tables)}


# ---------------------------------------------------------------------------
#  Slides
# ---------------------------------------------------------------------------
def extract_pptx(buf: io.BytesIO, req: ExtractRequest) -> dict[str, Any]:
    from pptx import Presentation

    prs = Presentation(buf)
    parts: list[str] = []
    for i, slide in enumerate(prs.slides, 1):
        bits = [sh.text.strip() for sh in slide.shapes
                if getattr(sh, "has_text_frame", False) and sh.text.strip()]
        notes = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        if bits or notes:
            parts.append(f"[slide {i}]\n" + "\n".join(bits)
                         + (f"\nSpeaker notes: {notes}" if notes else ""))
    n_slides = len(prs.slides)
    del prs
    return {"status": "ok", "text": _clean("\n\n".join(parts))[: req.max_chars],
            "pages": n_slides}


# ---------------------------------------------------------------------------
#  Images
# ---------------------------------------------------------------------------
def extract_image(buf: io.BytesIO, req: ExtractRequest) -> dict[str, Any]:
    ocr_text, vision = "", ""

    try:
        import pytesseract
        from PIL import Image

        # Refuse absurd dimensions outright rather than letting PIL allocate
        # first and fail second (decompression-bomb guard).
        Image.MAX_IMAGE_PIXELS = 64_000_000

        buf.seek(0)
        with Image.open(buf) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            w, h = img.size
            if w * h > MAX_OCR_PIXELS:
                # thumbnail() resizes in place and never enlarges, so it is
                # both the cheap and the safe way to land under the budget.
                img.thumbnail((int(MAX_OCR_PIXELS ** 0.5), int(MAX_OCR_PIXELS ** 0.5)))
            elif max(w, h) < 1000:
                # Upscaling small screenshots measurably improves OCR on UI
                # fonts - but only when the result stays inside the budget.
                if (w * 2) * (h * 2) <= MAX_OCR_PIXELS:
                    img = img.resize((w * 2, h * 2))

            ocr_text = _clean(pytesseract.image_to_string(img))

        # Tesseract returns punctuation soup on photos and gradients; drop
        # anything without a run of real words rather than indexing noise.
        if len(re.findall(r"[A-Za-z]{3,}", ocr_text)) < 3:
            ocr_text = ""
    except Exception as exc:                      # noqa: BLE001
        log.warning("ocr failed for %s: %s", req.asset_id, exc)

    if VISION_ENDPOINT and VISION_KEY:
        try:
            vision = _caption_image(buf, req)
        except Exception as exc:                  # noqa: BLE001
            log.warning("vision failed for %s: %s", req.asset_id, exc)

    return {"status": "ok", "text": "", "ocr_text": ocr_text,
            "ocr_chars": len(ocr_text), "vision_description": vision}


def _caption_image(buf: io.BytesIO, req: ExtractRequest) -> str:
    from PIL import Image

    # Downscale and re-encode as JPEG before base64. Sending the original would
    # hold the raw bytes, a base64 string ~33% larger, and the JSON payload
    # containing it all at once - on a large PNG that is the single biggest
    # allocation this service makes.
    buf.seek(0)
    small = io.BytesIO()
    with Image.open(buf) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if img.size[0] * img.size[1] > MAX_VISION_PIXELS:
            side = int(MAX_VISION_PIXELS ** 0.5)
            img.thumbnail((side, side))
        img.convert("RGB").save(small, format="JPEG", quality=80, optimize=True)

    b64 = base64.b64encode(small.getbuffer()).decode()
    del small

    payload = {
        "temperature": 0,
        "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    }
    del b64

    with httpx.Client(timeout=120) as client:
        r = client.post(VISION_ENDPOINT, json=payload,
                        headers={"api-key": VISION_KEY,
                                 "content-type": "application/json"})
        r.raise_for_status()
        return _clean(r.json()["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
#  Plain text
# ---------------------------------------------------------------------------
def extract_text(buf: io.BytesIO, req: ExtractRequest) -> dict[str, Any]:
    buf.seek(0)
    return {"status": "ok", "text": _clean(_decode(buf.read()))[: req.max_chars]}


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


class _TooBig(Exception):
    pass


def _download(url: str) -> tuple[io.BytesIO, str, int]:
    """Stream to one buffer, hashing as we go.

    The hash is computed from the chunks in flight rather than from the
    finished bytes, so the file never needs to exist twice in memory - the
    previous version's buf.getvalue() silently doubled peak usage for every
    single attachment.
    """
    digest = hashlib.sha256()
    buf = io.BytesIO()
    size = 0
    with httpx.Client(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(65536):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise _TooBig
                digest.update(chunk)
                buf.write(chunk)
    buf.seek(0)
    return buf, digest.hexdigest(), size


@app.post("/extract", dependencies=[Depends(require_secret)])
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
        buf, content_hash, size = _download(req.url)
    except _TooBig:
        return {"status": "skipped", "asset_id": req.asset_id,
                "error": "exceeds MAX_BYTES"}
    except Exception as exc:                      # noqa: BLE001
        return {"status": "failed", "asset_id": req.asset_id,
                "error": f"download: {exc}"[:300]}

    if not size:
        return {"status": "failed", "asset_id": req.asset_id, "error": "empty file"}

    try:
        result = handler(buf, req)
    except Exception as exc:                      # noqa: BLE001
        log.exception("extract failed for %s", req.asset_id)
        return {"status": "failed", "asset_id": req.asset_id,
                "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        buf.close()
        del buf
        # One worker serving one ticket at a time means peak RSS is what
        # matters, not throughput. Returning pages promptly keeps the next
        # attachment from starting on top of this one's leftovers.
        gc.collect()

    result.setdefault("status", "ok")
    result["asset_id"] = req.asset_id
    result["size_bytes"] = size
    # The same physical file arrives under different HubSpot file IDs when a
    # pasted screenshot gets re-uploaded on each re-quote; Assemble Ticket uses
    # this hash to collapse those into one indexed asset.
    result["content_hash"] = content_hash
    return result
