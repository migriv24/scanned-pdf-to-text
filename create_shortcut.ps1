# create_shortcut.ps1
# Run once (after setup_env.ps1 + download_models.py) to create the Desktop shortcut.
# Usage: right-click -> Run with PowerShell
#        (or: powershell -ExecutionPolicy Bypass -File create_shortcut.ps1)

$ProjectDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$IconFile     = Join-Path $ProjectDir "assets\vista_sidebar_2.ico"
$LaunchScript = Join-Path $ProjectDir "launch.pyw"
$ShortcutPath = [System.IO.Path]::Combine(
    [Environment]::GetFolderPath("Desktop"),
    "PDF to Text.lnk"
)

# ── Step 1: Find a suitable Python ───────────────────────────────────────────
# Try each candidate in order; pick the first one that exists and has all
# required packages installed.
# PY312_SAM3 is preferred (Python 3.12, SAM3, CUDA 12.6).
# PY310_Media_GPU is the fallback if the new env isn't ready yet.

$Required = @("flask", "fitz", "easyocr", "cv2", "numpy")

$candidates = @(
    "$env:USERPROFILE\.conda\envs\PY312_SAM3\pythonw.exe",
    "$env:USERPROFILE\.conda\envs\PY310_Media_GPU\pythonw.exe",
    "C:\Python314\pythonw.exe",
    (Join-Path (Split-Path (Get-Command python -ErrorAction SilentlyContinue).Source) "pythonw.exe")
)

$PythonW  = $null
$EnvLabel = $null

foreach ($c in $candidates) {
    if (-not ($c -and (Test-Path $c))) { continue }

    $pythonExe = Join-Path (Split-Path $c) "python.exe"
    if (-not (Test-Path $pythonExe)) { $pythonExe = $c }

    $checkScript = ($Required | ForEach-Object { "import $_" }) -join "; "
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    $result = & $pythonExe -c $checkScript 2>&1
    if ($LASTEXITCODE -eq 0) {
        $PythonW  = $c
        $EnvLabel = Split-Path (Split-Path $c -Parent) -Leaf
        Write-Host "Using Python: $PythonW" -ForegroundColor Cyan
        break
    } else {
        Write-Host "Skipping $c (missing packages)" -ForegroundColor Yellow
    }
}

if (-not $PythonW) {
    Write-Host ""
    Write-Host "ERROR: No suitable Python found." -ForegroundColor Red
    Write-Host "Run setup_env.ps1 first, then try again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Derive python.exe path for checks below
$pythonExe = Join-Path (Split-Path $PythonW) "python.exe"

# ── Step 2: Check SAM3 model (only when using the SAM3 env) ──────────────────
$hasSam3 = & $pythonExe -c "import sam3" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Checking SAM3 model in HuggingFace cache..." -ForegroundColor DarkGray

    & $pythonExe -c `
        "from huggingface_hub import try_to_load_from_cache; r=try_to_load_from_cache('facebook/sam3','sam3.pt'); exit(0 if r else 1)" `
        2>&1 | Out-Null

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: SAM3 model not found in HuggingFace cache." -ForegroundColor Red
        Write-Host ""
        Write-Host "Download it first by running (in order):" -ForegroundColor Yellow
        Write-Host "  conda activate PY312_SAM3" -ForegroundColor White
        Write-Host "  huggingface-cli login" -ForegroundColor DarkGray
        Write-Host "    (paste your token from https://huggingface.co/settings/tokens)" -ForegroundColor DarkGray
        Write-Host "  python `"$ProjectDir\download_models.py`"" -ForegroundColor White
        Write-Host ""
        Write-Host "Then run create_shortcut.ps1 again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host "SAM3 model: found in cache." -ForegroundColor Green
} else {
    Write-Host "Note: sam3 not installed in this env ($EnvLabel) — using CV fallback for figures." -ForegroundColor Yellow
}

# ── Step 3: Create the shortcut ───────────────────────────────────────────────
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
Write-Host "To use: double-click 'PDF to Text' on your Desktop."
Write-Host "  - First launch loads EasyOCR + SAM3 (30-60 s), then opens the browser."
Write-Host "  - Double-clicking again while running just reopens the tab."
Write-Host "  - Click Stop in the UI to cancel a long conversion without killing the process."
