import json
import traceback
import cv2
import fitz  # pymupdf
import easyocr
import numpy as np
from flask import Flask, render_template, request, Response, stream_with_context

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

print("Loading EasyOCR model (first run may download weights)…")
reader = easyocr.Reader(["en"], gpu=True)  # set gpu=False if you hit issues
print("EasyOCR ready.")


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
    """Return k for np.rot90(img, k) that makes text lines horizontal and right-side-up.

    Strategy:
      1. Fast projection-profile score for all 4 rotations.
      2. If one rotation clearly dominates (>30% more variance), trust it.
      3. Otherwise run OCR on a centre crop at the top-2 candidates and pick
         whichever orientation produces more confident characters — this
         distinguishes k=1 (correct 90° CCW) from k=3 (upside-down 90° CW).
    """
    scores = [_projection_score(np.rot90(img, k=k)) for k in range(4)]
    ranked = sorted(range(4), key=lambda k: scores[k], reverse=True)
    best = scores[ranked[0]]

    # Clear winner — no OCR needed
    if best > scores[ranked[1]] * 1.3:
        return ranked[0]

    # Ambiguous: validate the top-2 candidates with OCR confidence
    h, w = img.shape[:2]
    best_k, best_ocr = ranked[0], -1.0
    for k in ranked[:2]:
        rotated = np.rot90(img, k=k)
        rh, rw = rotated.shape[:2]
        # Centre 33% height × 50% width — avoids blank margins
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
        if img.std() < 5:   # nearly blank page — skip
            continue
        return detect_rotation(img)
    return 0


# ---------------------------------------------------------------------------
# Two-page spread splitting
# ---------------------------------------------------------------------------

def find_page_split(img: np.ndarray) -> int:
    """Find the spine/gutter column (minimum dark-pixel density in the middle third)."""
    binary = (img < 128).astype(np.float32)
    col_sums = binary.sum(axis=0)
    w = img.shape[1]
    lo, hi = w // 3, 2 * w // 3
    return lo + int(col_sums[lo:hi].argmin())


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
        h, w = img.shape[:2]
        total += 2 if w > h * 1.2 else 1
    return total


# ---------------------------------------------------------------------------
# Image region detection — abstract layer (swap body for an LLM vision call later)
# ---------------------------------------------------------------------------

def describe_image_region(img_region: np.ndarray) -> str | None:
    """Describe a non-text region. Replace this body with an LLM vision call for richer labels."""
    return _cv_describe_image(img_region)


def _cv_describe_image(img_region: np.ndarray) -> str | None:
    """Simple CV heuristic: edge density + aspect ratio → rough label."""
    if img_region.size == 0 or img_region.mean() > 245 or img_region.std() < 8:
        return None
    edges = cv2.Canny(img_region, 50, 150)
    if edges.mean() < 3:
        return None
    h, w = img_region.shape[:2]
    aspect = w / max(h, 1)
    if aspect > 1.4:
        return "a map or wide illustration"
    if aspect < 0.7:
        return "a tall figure or chart"
    return "an image or illustration"


# ---------------------------------------------------------------------------
# Layout-aware text extraction with image-placeholder insertion
# ---------------------------------------------------------------------------

def _bbox_y_center(bbox) -> float:
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2.0


def extract_ordered_text(img: np.ndarray, results: list) -> str:
    """Sort OCR results by vertical position and interleave image placeholders."""
    h, w = img.shape[:2]

    text_mask = np.zeros((h, w), dtype=np.uint8)
    for r in results:
        pts = np.array([[int(p[0]), int(p[1])] for p in r[0]], dtype=np.int32)
        cv2.fillPoly(text_mask, [pts], 255)

    kernel = np.ones((40, 40), np.uint8)
    non_text = cv2.bitwise_not(cv2.dilate(text_mask, kernel))
    contours, _ = cv2.findContours(non_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_items: list[tuple[float, str]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < h * w * 0.04 or area > h * w * 0.80:
            continue
        x, y, rw, rh = cv2.boundingRect(contour)
        desc = describe_image_region(img[y : y + rh, x : x + rw])
        if desc:
            image_items.append((y + rh / 2.0, f"[{desc}]"))

    text_items = [(_bbox_y_center(r[0]), r[1]) for r in results]
    all_items = sorted(text_items + image_items, key=lambda t: t[0])
    return "\n".join(text for _, text in all_items)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


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
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

                if rotation:
                    img = np.rot90(img, k=rotation)

                h_img, w_img = img.shape[:2]
                if w_img > h_img * 1.2:
                    split = find_page_split(img)
                    halves = [(img[:, :split], f"{i + 1}a"), (img[:, split:], f"{i + 1}b")]
                else:
                    halves = [(img, str(i + 1))]

                for half_img, label in halves:
                    output_page += 1
                    yield (
                        f'data: {json.dumps({"type": "progress", "page": output_page, "total": total})}\n\n'
                    )
                    results = reader.readtext(half_img, detail=1, paragraph=True)
                    text = extract_ordered_text(half_img, results)
                    block = f"--- Page {label} ---\n{text.strip()}"
                    yield (
                        f'data: {json.dumps({"type": "page", "page": output_page, "text": block})}\n\n'
                    )

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
