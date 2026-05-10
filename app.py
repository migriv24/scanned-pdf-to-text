import json
import threading
import traceback
from pathlib import Path

import cv2
import fitz  # pymupdf
import easyocr
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, Response, stream_with_context

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

cancel_event = threading.Event()

BASE = Path(__file__).resolve().parent
SAM3_CKPT_DIR = BASE / "models" / "sam3"

# ---------------------------------------------------------------------------
# Model startup
# ---------------------------------------------------------------------------

print("Loading EasyOCR model (first run may download weights)…")
reader = easyocr.Reader(["en"], gpu=True)
print("EasyOCR ready.")

_sam3_processor = None

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3 model (first run downloads ~2 GB from HuggingFace)…")
    _sam3_model = build_sam3_image_model(
        checkpoint_path=str(SAM3_CKPT_DIR) if SAM3_CKPT_DIR.exists() else None,
        load_from_HF=True,
    )
    _sam3_processor = Sam3Processor(_sam3_model)
    print("SAM3 ready.")
except Exception as exc:
    print(f"SAM3 unavailable ({exc}). Falling back to CV figure detection.")

SAM3_AVAILABLE = _sam3_processor is not None

# Figure concepts SAM3 will search for (prompt, human-readable label)
_FIGURE_PROMPTS = [
    ("photograph",   "a photograph"),
    ("diagram",      "a diagram"),
    ("chart",        "a chart or graph"),
    ("map",          "a map"),
    ("table",        "a data table"),
    ("illustration", "an illustration"),
]
_FIG_SCORE_THRESH  = 0.50
_FIG_MIN_AREA_FRAC = 0.02   # ignore tiny blobs
_FIG_MAX_AREA_FRAC = 0.85   # ignore near-full-page masks
_FIG_TEXT_OVERLAP  = 0.25   # skip if >25 % of mask is OCR text
_SAM3_SKIP_THRESH  = 0.88   # skip SAM3 if page is >88 % OCR text


# ---------------------------------------------------------------------------
# Rotation detection
# ---------------------------------------------------------------------------

def _projection_score(img: np.ndarray) -> float:
    """Variance of per-row dark-pixel counts — high = aligned horizontal text lines."""
    binary = (img < 128).astype(np.float32)
    return float(binary.sum(axis=1).var())


def _ocr_confidence_score(img: np.ndarray) -> float:
    """Weighted character count (len × confidence) on a small crop — higher = more readable."""
    h, w = img.shape[:2]
    scale = min(1.0, 400 / max(h, w, 1))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else img
    results = reader.readtext(small, detail=1, paragraph=False)
    return sum(len(r[1]) * r[2] for r in results)


def detect_rotation(img: np.ndarray) -> int:
    """Return k for np.rot90(img, k) that makes text lines horizontal and right-side-up."""
    scores = [_projection_score(np.rot90(img, k=k)) for k in range(4)]
    ranked = sorted(range(4), key=lambda k: scores[k], reverse=True)
    best = scores[ranked[0]]

    if best > scores[ranked[1]] * 1.3:
        return ranked[0]

    h, w = img.shape[:2]
    best_k, best_ocr = ranked[0], -1.0
    for k in ranked[:2]:
        rotated = np.rot90(img, k=k)
        rh, rw = rotated.shape[:2]
        crop = rotated[rh // 3 : 2 * rh // 3, rw // 4 : 3 * rw // 4]
        score = _ocr_confidence_score(crop)
        if score > best_ocr:
            best_ocr = score
            best_k = k
    return best_k


def detect_doc_rotation(doc) -> int:
    """Detect orientation once for the whole document (consistent for a scanned book).
    Tries up to 3 pages to find one with enough content for a reliable signal."""
    mat = fitz.Matrix(200 / 72, 200 / 72)
    for i in range(min(3, len(doc))):
        pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        if img.std() < 5:
            continue
        return detect_rotation(img)
    return 0


# ---------------------------------------------------------------------------
# Two-page spread detection
# ---------------------------------------------------------------------------

def detect_spread(img: np.ndarray) -> tuple[bool, int]:
    """Decide whether this image is a two-page book spread or a single page."""
    h, w = img.shape[:2]

    if w <= h * 1.2:
        return False, w // 2

    binary = (img < 128).astype(np.float32)
    col_density = binary.sum(axis=0) / max(h, 1)

    ksz = max(5, w // 40)
    smooth = np.convolve(col_density, np.ones(ksz) / ksz, mode='same')

    lo, hi = w // 3, 2 * w // 3
    gutter_col = lo + int(smooth[lo:hi].argmin())
    gutter_val = float(smooth[gutter_col])

    left_ref  = float(np.percentile(smooth[w // 6 : lo], 75))
    right_ref = float(np.percentile(smooth[hi : 5 * w // 6], 75))
    ref = (left_ref + right_ref) / 2.0

    if ref < 0.01:
        return False, w // 2

    depth = 1.0 - (gutter_val / ref)
    if depth <= 0.55:
        return False, w // 2

    half_ref = ref * 0.5
    g_lo, g_hi = gutter_col, gutter_col
    while g_lo > 0 and smooth[g_lo - 1] < half_ref:
        g_lo -= 1
    while g_hi < w - 1 and smooth[g_hi + 1] < half_ref:
        g_hi += 1
    valley_width_pct = (g_hi - g_lo + 1) / w

    return valley_width_pct >= 0.02, gutter_col


# ---------------------------------------------------------------------------
# Pre-scan: accurate output-page count
# ---------------------------------------------------------------------------

def count_output_pages(doc, rotation: int) -> int:
    """Quick low-DPI scan to count expected output pages (1 or 2 per PDF page)."""
    mat = fitz.Matrix(30 / 72, 30 / 72)
    total = 0
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        if rotation:
            img = np.rot90(img, k=rotation)
        is_spread, _ = detect_spread(img)
        total += 2 if is_spread else 1
    return total


# ---------------------------------------------------------------------------
# Figure detection — SAM3 (primary) with CV heuristic fallback
# ---------------------------------------------------------------------------

def _build_text_mask(h: int, w: int, ocr_results: list) -> np.ndarray:
    """Boolean mask of pixels covered by OCR text bounding boxes."""
    mask = np.zeros((h, w), dtype=bool)
    for r in ocr_results:
        pts = np.array([[int(p[0]), int(p[1])] for p in r[0]], dtype=np.int32)
        x, y, rw, rh = cv2.boundingRect(pts)
        mask[y : y + rh, x : x + rw] = True
    return mask


def _text_coverage(h: int, w: int, text_mask: np.ndarray) -> float:
    return float(text_mask.sum()) / max(h * w, 1)


def _sam3_find_figures(img: np.ndarray, text_mask: np.ndarray) -> list[tuple[float, str]]:
    """Use SAM3 text prompts to find and label non-text regions.

    SAM3 returns state["masks"] as a (N, 1, H, W) bool tensor and
    state["scores"] as a (N,) float tensor.  The image is encoded once;
    each text prompt call re-uses the cached backbone features.
    """
    h, w = img.shape[:2]
    page_area = h * w

    pil = Image.fromarray(img).convert("RGB")  # SAM3 expects RGB
    state = _sam3_processor.set_image(pil)

    figure_items: list[tuple[float, str]] = []
    seen_centers: list[float] = []

    for prompt, label in _FIGURE_PROMPTS:
        try:
            result = _sam3_processor.set_text_prompt(prompt, state)
        except Exception:
            continue

        masks  = result.get("masks")   # (N, 1, H, W) bool tensor
        scores = result.get("scores")  # (N,) float tensor

        if masks is None or len(masks) == 0:
            continue

        for idx in range(len(masks)):
            # scores may be 0-D or multi-dim; use .max() for safety
            score = float(scores[idx].max())
            if score < _FIG_SCORE_THRESH:
                continue

            # masks[idx] is (1, H, W); squeeze to (H, W)
            mask_t = masks[idx]
            mask_bool = (mask_t[0] if mask_t.dim() == 3 else mask_t).cpu().numpy().astype(bool)

            if mask_bool.shape != (h, w):
                continue

            area_frac = mask_bool.sum() / page_area
            if area_frac < _FIG_MIN_AREA_FRAC or area_frac > _FIG_MAX_AREA_FRAC:
                continue

            overlap = float((mask_bool & text_mask).sum()) / max(int(mask_bool.sum()), 1)
            if overlap > _FIG_TEXT_OVERLAP:
                continue

            ys = np.where(mask_bool.any(axis=1))[0]
            if len(ys) == 0:
                continue
            cy = float(ys.mean())

            if any(abs(cy - c) < h * 0.05 for c in seen_centers):
                continue
            seen_centers.append(cy)
            figure_items.append((cy, f"[{label}]"))

    return figure_items


def _cv_find_figures(img: np.ndarray, text_mask: np.ndarray) -> list[tuple[float, str]]:
    """CV fallback: edge density + aspect ratio for non-text contours."""
    h, w = img.shape[:2]

    kernel = np.ones((40, 40), np.uint8)
    dilated_text = cv2.dilate(text_mask.astype(np.uint8) * 255, kernel)
    non_text = cv2.bitwise_not(dilated_text)
    contours, _ = cv2.findContours(non_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    items: list[tuple[float, str]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < h * w * 0.04 or area > h * w * 0.80:
            continue
        x, y, rw, rh = cv2.boundingRect(contour)
        region = img[y : y + rh, x : x + rw]

        if region.size == 0 or region.mean() > 245 or region.std() < 8:
            continue
        edges = cv2.Canny(region, 50, 150)
        if edges.mean() < 3:
            continue

        aspect = rw / max(rh, 1)
        if aspect > 1.4:
            label = "a map or wide illustration"
        elif aspect < 0.7:
            label = "a tall figure or chart"
        else:
            label = "an image or illustration"

        items.append((y + rh / 2.0, f"[{label}]"))

    return items


# ---------------------------------------------------------------------------
# Layout-aware text extraction
# ---------------------------------------------------------------------------

def _bbox_y_center(bbox) -> float:
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2.0


def extract_ordered_text(img: np.ndarray, ocr_results: list) -> str:
    """Sort OCR results by vertical position, interleaving figure placeholders.

    Uses SAM3 when available and the page has non-trivial non-text regions;
    falls back to CV heuristics otherwise.
    """
    h, w = img.shape[:2]
    text_mask = _build_text_mask(h, w, ocr_results)
    text_items = [(_bbox_y_center(r[0]), r[1]) for r in ocr_results]

    coverage = _text_coverage(h, w, text_mask)
    if coverage >= _SAM3_SKIP_THRESH:
        # Page is almost entirely text — skip figure detection entirely
        figure_items: list[tuple[float, str]] = []
    elif SAM3_AVAILABLE:
        try:
            figure_items = _sam3_find_figures(img, text_mask)
        except Exception:
            figure_items = _cv_find_figures(img, text_mask)
    else:
        figure_items = _cv_find_figures(img, text_mask)

    all_items = sorted(text_items + figure_items, key=lambda t: t[0])
    return "\n".join(text for _, text in all_items)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/cancel", methods=["POST"])
def cancel():
    cancel_event.set()
    return "", 204


@app.route("/convert", methods=["POST"])
def convert():
    def error_stream(msg):
        return Response(
            f'data: {json.dumps({"type": "error", "message": msg})}\n\n',
            mimetype="text/event-stream",
        )

    if "pdf" not in request.files:
        return error_stream("No file uploaded.")
    file = request.files["pdf"]
    if not file.filename.lower().endswith(".pdf"):
        return error_stream("Please upload a .pdf file.")

    pdf_bytes = file.read()
    original_name = file.filename

    def generate():
        cancel_event.clear()
        try:
            yield f'data: {json.dumps({"type": "status", "message": "Opening PDF…"})}\n\n'
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            yield f'data: {json.dumps({"type": "status", "message": "Detecting page orientation…"})}\n\n'
            rotation = detect_doc_rotation(doc)

            total = count_output_pages(doc, rotation)
            yield f'data: {json.dumps({"type": "total", "total": total})}\n\n'

            mat = fitz.Matrix(200 / 72, 200 / 72)
            output_page = 0

            for i, page in enumerate(doc):
                if cancel_event.is_set():
                    yield f'data: {json.dumps({"type": "cancelled"})}\n\n'
                    break

                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

                if rotation:
                    img = np.rot90(img, k=rotation)

                is_spread, split = detect_spread(img)
                halves = (
                    [(img[:, :split], f"{i + 1}a"), (img[:, split:], f"{i + 1}b")]
                    if is_spread
                    else [(img, str(i + 1))]
                )

                for half_img, label in halves:
                    output_page += 1
                    yield f'data: {json.dumps({"type": "progress", "page": output_page, "total": total})}\n\n'

                    yield f'data: {json.dumps({"type": "status", "message": f"OCR — page {label}…"})}\n\n'
                    results = reader.readtext(half_img, detail=1, paragraph=True)

                    if SAM3_AVAILABLE:
                        yield f'data: {json.dumps({"type": "status", "message": f"SAM3 figure detection — page {label}…"})}\n\n'

                    text = extract_ordered_text(half_img, results)
                    block = f"--- Page {label} ---\n{text.strip()}"
                    yield f'data: {json.dumps({"type": "page", "page": output_page, "text": block})}\n\n'

            doc.close()
            output_name = original_name.rsplit(".", 1)[0] + ".txt"
            yield f'data: {json.dumps({"type": "done", "filename": output_name})}\n\n'

        except Exception as exc:
            yield f'data: {json.dumps({"type": "error", "message": str(exc) + chr(10) + traceback.format_exc()})}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
