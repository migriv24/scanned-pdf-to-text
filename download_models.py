"""
download_models.py — pre-download the SAM3 checkpoint from HuggingFace.

Run once before launching the app so the first startup is fast:

    conda activate PY312_SAM3
    huggingface-cli login          # only needed once; stores token in ~/.cache
    python download_models.py

If you skip this, the app will download the model (~2 GB) on its first run.
The checkpoint is cached in HuggingFace's local cache automatically.
"""
import sys

print("Verifying HuggingFace login…")
try:
    from huggingface_hub import HfApi
    HfApi().whoami()
    print("HF login OK.")
except Exception:
    print(
        "\nYou are not logged in to HuggingFace.\n"
        "Run:  huggingface-cli login\n"
        "Then re-run this script.\n"
    )
    sys.exit(1)

print("Downloading SAM3 model via sam3 package (this may take a few minutes)…")
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(load_from_HF=True)
    print("SAM3 model loaded and cached successfully.")
    print("\nYou can now launch the app — startup will be fast.")
except Exception as exc:
    print(f"\nDownload failed: {exc}")
    print(
        "If you see a 401/403 error, you may need to request access at:\n"
        "  https://huggingface.co/facebook/sam3"
    )
    sys.exit(1)
