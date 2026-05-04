"""
launch.pyw — Desktop launcher for PDF to Text.
Runs via pythonw.exe (no console window).

- If server is already running  → just opens the browser tab.
- If not                        → starts Flask, then opens the browser.
- On any startup error          → writes to launch_error.log and shows a popup.
"""
import os
import sys
import socket
import webbrowser
import traceback
from pathlib import Path
from threading import Timer

BASE = Path(__file__).resolve().parent
LOG  = BASE / "launch_error.log"
PORT = 5000
URL  = f"http://localhost:{PORT}"

# Redirect stdout/stderr — pythonw has no console
try:
    _log_fh = open(LOG, "w", encoding="utf-8", buffering=1)
    sys.stdout = _log_fh
    sys.stderr = _log_fh
except Exception:
    pass


def _fatal(msg: str):
    try:
        LOG.write_text(msg, encoding="utf-8")
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            f"PDF to Text failed to start.\n\n{msg}\n\nSee launch_error.log for details.",
            "Launch Error",
            0x10,
        )
    except Exception:
        pass
    sys.exit(1)


try:
    os.chdir(BASE)
    sys.path.insert(0, str(BASE))
    # Suppress duplicate-OpenMP warning that occurs when NumPy and PyTorch share a process
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    def _server_running() -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", PORT))
            s.close()
            return True
        except OSError:
            return False

    if _server_running():
        webbrowser.open(URL)
        sys.exit(0)

    from app import app  # noqa: E402

    Timer(2.0, lambda: webbrowser.open(URL)).start()
    app.run(debug=False, port=PORT, use_reloader=False)

except Exception:
    _fatal(traceback.format_exc())
