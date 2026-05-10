# create_shortcut.ps1
# Run once to create (or re-create) the Desktop shortcut for PDF to Text.
# Usage: right-click, Run with PowerShell
#        (or: powershell -ExecutionPolicy Bypass -File create_shortcut.ps1)

$ProjectDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$IconFile     = Join-Path $ProjectDir "assets\vista_sidebar_2.ico"
$LaunchScript = Join-Path $ProjectDir "launch.pyw"
$ShortcutPath = [System.IO.Path]::Combine(
    [Environment]::GetFolderPath("Desktop"),
    "PDF to Text.lnk"
)

# Find pythonw.exe — try each candidate in order, pick the first one that
# exists AND has all required packages installed.
# PY312_SAM3 is the preferred env: Python 3.12, PyTorch 2.11+CUDA 12.6, SAM3 + all deps.
# PY310_Media_GPU is the fallback if the new env isn't set up yet.
$Required = @("flask", "fitz", "easyocr", "cv2", "numpy")

$candidates = @(
    "C:\Users\migri\.conda\envs\PY312_SAM3\pythonw.exe",
    "C:\Users\migri\.conda\envs\PY310_Media_GPU\pythonw.exe",
    "C:\Python314\pythonw.exe",
    (Join-Path (Split-Path (Get-Command python -ErrorAction SilentlyContinue).Source) "pythonw.exe")
)

$PythonW = $null
foreach ($c in $candidates) {
    if (-not ($c -and (Test-Path $c))) { continue }

    # Derive python.exe from pythonw.exe to run the check (pythonw has no stdout)
    $pythonExe = Join-Path (Split-Path $c) "python.exe"
    if (-not (Test-Path $pythonExe)) { $pythonExe = $c }

    $checkScript = ($Required | ForEach-Object { "import $_" }) -join "; "
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"   # suppress duplicate-OpenMP warning on Windows
    $result = & $pythonExe -c $checkScript 2>&1
    if ($LASTEXITCODE -eq 0) {
        $PythonW = $c
        Write-Host "Using: $PythonW" -ForegroundColor Cyan
        break
    } else {
        Write-Host "Skipping $c (missing packages: $result)" -ForegroundColor Yellow
    }
}

if (-not $PythonW) {
    Write-Host ""
    Write-Host "ERROR: No suitable Python found. Install missing packages with:" -ForegroundColor Red
    Write-Host "  pip install flask pymupdf easyocr opencv-python-headless" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut  = $WshShell.CreateShortcut($ShortcutPath)

$Shortcut.TargetPath       = $PythonW
$Shortcut.Arguments        = "`"$LaunchScript`""
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.Description      = "PDF to Text - scanned PDF OCR converter"
$Shortcut.WindowStyle      = 7

if (Test-Path $IconFile) {
    $Shortcut.IconLocation = "$IconFile,0"
} else {
    Write-Host "WARNING: Icon not found at $IconFile" -ForegroundColor Yellow
}

$Shortcut.Save()

Write-Host ""
Write-Host "Shortcut created: $ShortcutPath" -ForegroundColor Green
Write-Host ""
Write-Host "To use: double-click PDF to Text on your Desktop."
Write-Host "  - First launch starts the Flask server and opens your browser."
Write-Host "  - Double-clicking again while running just reopens the tab."
Write-Host "  - EasyOCR and SAM3 model weights load on first run (may take a moment)."
Write-Host "  - If SAM3 checkpoint is missing, run: conda activate PY312_SAM3 && python download_models.py"
