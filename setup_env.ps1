# setup_env.ps1
# Sets up the PY312_SAM3 conda environment for PDF to Text (SAM3 edition).
# Run from a regular PowerShell window — no administrator rights needed.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1 -SkipDownload
#
# -SkipDownload  Skip the HuggingFace model download step (download manually later).

param([switch]$SkipDownload)

$ErrorActionPreference = "Stop"

$ProjectDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName      = "PY312_SAM3"
$EnvPath      = "$env:USERPROFILE\.conda\envs\$EnvName"
$PythonExe    = "$EnvPath\python.exe"
$PipExe       = "$EnvPath\Scripts\pip.exe"
$TritonDir    = "$EnvPath\Lib\site-packages\triton"
$TritonLangDir = "$TritonDir\language"

$TotalSteps = 7

function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host "  [$n/$TotalSteps] $msg" -ForegroundColor Cyan
    Write-Host "  $("-" * 60)" -ForegroundColor DarkGray
}
function Write-OK($msg)   { Write-Host "  [OK]   $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red }

# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔═══════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║   PDF to Text — Environment Setup     ║" -ForegroundColor Cyan
Write-Host "  ╚═══════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host "  Env:     $EnvName"
Write-Host "  Project: $ProjectDir"
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
Write-Step 1 "Checking prerequisites"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Fail "conda not found in PATH. Install Miniconda: https://docs.conda.io"
    exit 1
}
Write-OK "conda: $((Get-Command conda).Source)"

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpuLine = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null) -join ", "
    Write-OK "GPU: $gpuLine"
} else {
    Write-Warn "nvidia-smi not found — SAM3 will run on CPU (significantly slower)."
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Step 2 "Creating conda environment ($EnvName, Python 3.12)"

if (Test-Path $PythonExe) {
    $v = & $PythonExe --version 2>&1
    Write-OK "Already exists — $v"
} else {
    Write-Host "  Downloading Python 3.12 base (~100 MB)..." -ForegroundColor DarkGray
    conda create -n $EnvName python=3.12 -y
    if ($LASTEXITCODE -ne 0) { Write-Fail "conda create failed."; exit 1 }
    Write-OK "Environment created."
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Step 3 "Installing PyTorch 2.x + CUDA 12.6 (large download — up to 2.5 GB)"

$torchCheck = & $PythonExe -c "import torch; print(torch.__version__)" 2>&1
if ($torchCheck -match "cu12") {
    Write-OK "Already installed: torch $torchCheck"
} else {
    Write-Host "  Source: https://download.pytorch.org/whl/cu126" -ForegroundColor DarkGray
    Write-Host "  This can take 10-30 minutes on a slow connection." -ForegroundColor DarkGray
    & $PipExe install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { Write-Fail "PyTorch install failed."; exit 1 }
    $torchVer = & $PythonExe -c "import torch; print(torch.__version__)" 2>&1
    Write-OK "Installed: torch $torchVer"
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Step 4 "Installing Python packages"

$packages = @(
    "sam3",
    "easyocr",
    "flask",
    "pymupdf",
    "opencv-python-headless",
    "huggingface_hub",
    "psutil"
)

foreach ($pkg in $packages) {
    Write-Host ""
    Write-Host "  > pip install $pkg" -ForegroundColor DarkGray
    & $PipExe install $pkg
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "$pkg install returned non-zero — check output above."
    }
}

Write-OK "Package installs complete."

# ─────────────────────────────────────────────────────────────────────────────
Write-Step 5 "Installing Windows triton stub"
Write-Host "  (triton is Linux-only; this stub lets sam3's image model import on Windows)" -ForegroundColor DarkGray

New-Item -ItemType Directory -Force -Path $TritonDir     | Out-Null
New-Item -ItemType Directory -Force -Path $TritonLangDir | Out-Null

# ── triton/__init__.py ────────────────────────────────────────────────────────
$TritonInit = @'
"""
Windows stub for the triton GPU kernel compilation library.
sam3's video tracker uses triton; the image predictor does not.
This stub + a sys.meta_path finder lets sam3 import cleanly on Windows.
Any triton kernel that is actually *called* will raise a clear RuntimeError.
"""
import sys as _sys
import types as _types


class _MockModule(_types.ModuleType):
    """A ModuleType that silently creates child stubs for any attribute."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        child_name = f"{self.__name__}.{name}"
        child = _MockModule(child_name)
        _sys.modules[child_name] = child
        object.__setattr__(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        return _MockModule(f"{self.__name__}()")

    def __repr__(self):
        return f"<triton stub: {self.__name__}>"


class _TritonSubFinder:
    @staticmethod
    def find_module(fullname, path=None):
        if fullname.startswith("triton.") and fullname not in _sys.modules:
            return _TritonSubLoader()
        return None

    @staticmethod
    def find_spec(fullname, path, target=None):
        if fullname.startswith("triton.") and fullname not in _sys.modules:
            import importlib.machinery
            return importlib.machinery.ModuleSpec(
                fullname, _TritonSubLoader(), is_package=True
            )
        return None


class _TritonSubLoader:
    @staticmethod
    def load_module(fullname):
        if fullname in _sys.modules:
            return _sys.modules[fullname]
        mod = _MockModule(fullname)
        _sys.modules[fullname] = mod
        return mod

    @staticmethod
    def create_module(spec):
        return _MockModule(spec.name)

    @staticmethod
    def exec_module(module):
        pass


if not any(isinstance(f, _TritonSubFinder) for f in _sys.meta_path):
    _sys.meta_path.append(_TritonSubFinder())


def jit(fn=None, **kwargs):
    if fn is None:
        return lambda f: f
    return fn


class Config:
    def __init__(self, kwargs=None, **extra):
        self.kwargs = kwargs or {}


def autotune(configs=None, key=None, **kwargs):
    return lambda f: f


def heuristics(values=None, **kwargs):
    return lambda f: f


def cdiv(a, b):
    return (a + b - 1) // b


def next_power_of_2(n):
    import math
    return 1 if n <= 0 else 2 ** math.ceil(math.log2(max(n, 1)))


from . import language  # noqa: E402
'@

# ── triton/language/__init__.py ───────────────────────────────────────────────
$TritonLang = @'
"""
Windows stub for triton.language — satisfies all import-time symbol access.
Actual kernels are never called in the image-only inference path.
"""

class dtype:
    def __init__(self, name, bitwidth=0):
        self._name = name
        self.bitwidth = bitwidth
    def __repr__(self):
        return f"tl.{self._name}"
    def __eq__(self, other):
        return isinstance(other, dtype) and self._name == other._name
    def __hash__(self):
        return hash(self._name)
    def is_floating_point(self):
        return "float" in self._name or "bfloat" in self._name


float64  = dtype("float64",  64)
float32  = dtype("float32",  32)
float16  = dtype("float16",  16)
bfloat16 = dtype("bfloat16", 16)
int64    = dtype("int64",    64)
int32    = dtype("int32",    32)
int16    = dtype("int16",    16)
int8     = dtype("int8",      8)
int1     = dtype("int1",      1)
uint64   = dtype("uint64",   64)
uint32   = dtype("uint32",   32)
uint16   = dtype("uint16",   16)
uint8    = dtype("uint8",     8)
bool_    = dtype("bool",      1)


class constexpr:
    """Marker for compile-time constants in triton kernel signatures."""
    def __init__(self, value=None):
        self.value = value
    def __repr__(self):
        return f"constexpr({self.value!r})"
    def __add__(self, other): return constexpr(self.value + _val(other))
    def __radd__(self, other): return constexpr(_val(other) + self.value)
    def __mul__(self, other): return constexpr(self.value * _val(other))
    def __rmul__(self, other): return constexpr(_val(other) * self.value)
    def __floordiv__(self, other): return constexpr(self.value // _val(other))
    def __eq__(self, other): return self.value == _val(other)
    def __lt__(self, other): return self.value < _val(other)
    def __le__(self, other): return self.value <= _val(other)
    def __gt__(self, other): return self.value > _val(other)
    def __ge__(self, other): return self.value >= _val(other)
    def __bool__(self): return bool(self.value)
    def __int__(self): return int(self.value)

def _val(x):
    return x.value if isinstance(x, constexpr) else x


class pointer_type:
    def __init__(self, element_ty, const=False):
        self.element_ty = element_ty
        self.const = const


def _noop(*args, **kwargs):
    raise RuntimeError(
        "triton kernel operation called on Windows -- "
        "this should not happen in image-only inference mode"
    )

load = store = atomic_add = atomic_min = atomic_max = atomic_cas = atomic_xchg = _noop
program_id = num_programs = arange = zeros_like = zeros = full = tensor = _noop
abs = exp = log = sqrt = sigmoid = cast = cdiv = _noop
sum = max = min = reduce = _noop
where = minimum = maximum = clamp = _noop
broadcast_to = reshape = ravel = view = trans = cat = flip = dot = _noop
debug_barrier = _noop
'@

Set-Content -Path "$TritonDir\__init__.py"     -Value $TritonInit -Encoding UTF8
Set-Content -Path "$TritonLangDir\__init__.py" -Value $TritonLang -Encoding UTF8

$stubCheck = & $PythonExe -c "import triton, triton.language; print('OK')" 2>&1
if ($stubCheck -match "OK") {
    Write-OK "Triton stub verified."
} else {
    Write-Warn "Triton stub check returned: $stubCheck"
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Step 6 "Verifying sam3 imports cleanly"

$sam3Check = & $PythonExe -c "import sam3; print(sam3.__version__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-OK "sam3 $($sam3Check.Trim()) imports OK."
} else {
    Write-Fail "sam3 import failed:`n$sam3Check"
    Write-Host "  The triton stub may be incomplete. Check the error above." -ForegroundColor Yellow
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Step 7 "SAM3 model download (facebook/sam3, ~2 GB)"

# Check if model is already in HF cache
$modelCached = & $PythonExe -c `
    "from huggingface_hub import try_to_load_from_cache; r=try_to_load_from_cache('facebook/sam3','sam3.pt'); exit(0 if r else 1)" `
    2>&1
$modelReady = ($LASTEXITCODE -eq 0)

if ($modelReady) {
    Write-OK "SAM3 model already in HuggingFace cache — nothing to download."
} elseif ($SkipDownload) {
    Write-Warn "Skipped (--SkipDownload). Run download_models.py when ready."
} else {
    # Check HF login
    $hfUser = & $PythonExe -c `
        "from huggingface_hub import HfApi; info=HfApi().whoami(); print(info.get('name','?'))" `
        2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-OK "HuggingFace logged in as: $($hfUser.Trim())"
        Write-Host ""
        Write-Host "  Downloading SAM3 model (~2 GB) — this will take a few minutes..." -ForegroundColor DarkGray
        & $PythonExe "$ProjectDir\download_models.py"
        if ($LASTEXITCODE -eq 0) {
            Write-OK "Model downloaded and cached."
        } else {
            Write-Warn "Download failed or was interrupted."
            Write-Host "  Retry manually:" -ForegroundColor Yellow
            Write-Host "    conda activate $EnvName" -ForegroundColor White
            Write-Host "    python download_models.py" -ForegroundColor White
        }
    } else {
        Write-Warn "Not logged in to HuggingFace."
        Write-Host ""
        Write-Host "  To download the model, run these commands in order:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "    conda activate $EnvName" -ForegroundColor White
        Write-Host "    huggingface-cli login" -ForegroundColor White
        Write-Host "       (paste your token from https://huggingface.co/settings/tokens)" -ForegroundColor DarkGray
        Write-Host "    python download_models.py" -ForegroundColor White
        Write-Host ""
        Write-Host "  Then run create_shortcut.ps1 to create the Desktop icon." -ForegroundColor Yellow
    }
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "  ════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
if (-not $modelReady -and -not $SkipDownload) {
    Write-Host "  NEXT: download the model (see step 7 above), then:" -ForegroundColor Yellow
} else {
    Write-Host "  NEXT:" -ForegroundColor Green
}
Write-Host "    powershell -ExecutionPolicy Bypass -File create_shortcut.ps1" -ForegroundColor White
Write-Host ""
