# scanned-pdf-to-text

A lightweight local web app that converts scanned PDF books into plain text using OCR — no cloud, no API keys, no internet required after setup.

> **This project was written almost entirely by [Claude Code](https://claude.ai/code) (Anthropic's AI coding assistant) with very little human input. The human provided the idea and a few directional corrections; Claude Code wrote all of the code, made architectural decisions, and set up the project structure.**

---

## What it does

- You upload a scanned PDF (the kind where pages are images, like a photographed or photocopied book)
- Each page is rasterised locally using **PyMuPDF**
- OCR is run on each page using **EasyOCR** (backed by PyTorch — runs on your GPU)
- Text streams back to the browser page-by-page as it's processed
- You can copy the result or download it as a `.txt` file

Everything runs on your own machine. No data leaves your computer.

---

## Requirements

### Python environment

You need a Python 3.10+ environment with **PyTorch already installed**. A conda environment with an existing PyTorch/CUDA setup works perfectly (EasyOCR will use the GPU automatically).

### Python packages

```bash
pip install flask pymupdf easyocr
```

> On first run, EasyOCR will download its model weights (~100 MB). This is a one-time download.

No other external binaries are required (no Tesseract, no Poppler).

---

## Running

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

1. Drag and drop a scanned PDF onto the upload area (or click to browse)
2. Click **Convert**
3. Watch the text fill in live as each page is processed
4. Use **Copy** or **Download .txt** when done

---

## Configuration

In [app.py](app.py):

- `gpu=True` on the `easyocr.Reader(...)` line — set to `False` if you don't have a GPU or run into CUDA issues
- DPI is set to `200` in the rasterisation step — increase to `300` for better accuracy on low-quality scans (uses more memory)
- Max upload size is `300 MB` — adjust `MAX_CONTENT_LENGTH` if needed

---

## Stack

| Component | Library |
|-----------|---------|
| Web server | [Flask](https://flask.palletsprojects.com/) |
| PDF rasterisation | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Streaming | Server-Sent Events (SSE) via Flask |

---

## Credits

Built with [Claude Code](https://claude.ai/code) by Anthropic.
