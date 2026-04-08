import json
import fitz  # pymupdf
import easyocr
import numpy as np
from flask import Flask, render_template, request, Response, stream_with_context

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

# Load once at startup — downloads ~100 MB of model weights on first run
print("Loading EasyOCR model (first run may download weights)…")
reader = easyocr.Reader(["en"], gpu=True)  # set gpu=False if you hit issues
print("EasyOCR ready.")


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
            total = len(doc)

            yield f'data: {json.dumps({"type": "total", "total": total})}\n\n'

            for i, page in enumerate(doc):
                page_num = i + 1
                yield f'data: {json.dumps({"type": "progress", "page": page_num, "total": total})}\n\n'

                # Rasterise at 200 DPI (72 pt/in baseline)
                mat = fitz.Matrix(200 / 72, 200 / 72)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

                lines = reader.readtext(img, detail=0, paragraph=True)
                text = "\n".join(lines)
                block = f"--- Page {page_num} ---\n{text.strip()}"

                yield f'data: {json.dumps({"type": "page", "page": page_num, "text": block})}\n\n'

            doc.close()
            output_name = original_name.rsplit(".", 1)[0] + ".txt"
            yield f'data: {json.dumps({"type": "done", "filename": output_name})}\n\n'

        except Exception as exc:
            yield f'data: {json.dumps({"type": "error", "message": str(exc)})}\n\n'

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
