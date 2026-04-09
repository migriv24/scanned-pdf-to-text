# create_shortcut.ps1
# Run once to create (or re-create) the Desktop shortcut for PDF to Text.
# Usage: right-click → Run with PowerShell
#        (or: powershell -ExecutionPolicy Bypass -File create_shortcut.ps1)

$ProjectDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$IconFile     = Join-Path $ProjectDir "assets\vista_sidebar_2.ico"
$LaunchScript = Join-Path $ProjectDir "launch.pyw"
$ShortcutPath = [System.IO.Path]::Combine(
    [Environment]::GetFolderPath("Desktop"),
    "PDF to Text.lnk"
)

# Find pythonw.exe — prefer the one next to the active python.exe
$PythonW = $null
$candidates = @(
    "C:\Python314\pythonw.exe",
    (Join-Path (Split-Path (Get-Command python -ErrorAction SilentlyContinue).Source) "pythonw.exe")
)
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) { $PythonW = $c; break }
}

if (-not $PythonW) {
    Write-Host "ERROR: pythonw.exe not found. Edit the candidates list in this script." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut  = $WshShell.CreateShortcut($ShortcutPath)

$Shortcut.TargetPath       = $PythonW
$Shortcut.Arguments        = "`"$LaunchScript`""
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.Description      = "PDF to Text — scanned PDF OCR converter"
$Shortcut.WindowStyle      = 7   # 7 = minimised (hides the brief pythonw flash)

if (Test-Path $IconFile) {
    $Shortcut.IconLocation = "$IconFile,0"
} else {
    Write-Host "WARNING: Icon not found at $IconFile — shortcut will use default Python icon." -ForegroundColor Yellow
}

$Shortcut.Save()

Write-Host "Shortcut created: $ShortcutPath" -ForegroundColor Green
Write-Host ""
Write-Host "To use: double-click 'PDF to Text' on your Desktop."
Write-Host "  - First launch starts the Flask server and opens your browser."
Write-Host "  - Double-clicking again while running just reopens the tab."
Write-Host "  - EasyOCR model weights load on first run (may take a moment)."
Write-Host ""
Read-Host "Press Enter to close"
