import os
import sys
import warnings
from pathlib import Path

AVAILABLE = False
batch_extract = None
build_csr = None

_src_dir = Path(__file__).parent

# torch.utils.cpp_extension.load() invokes ninja via subprocess and looks for it
# in the shell PATH. When the script is run as `.venv/bin/python` without
# activating the venv, the venv's bin directory is not in PATH and ninja is
# not found. Add it explicitly here.
_venv_bin = str(Path(sys.executable).parent)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

try:
    from torch.utils.cpp_extension import load as _cpp_load

    _seal_cpp = _cpp_load(
        name="seal_cpp",
        sources=[str(_src_dir / "seal_ops.cpp")],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )

    batch_extract = _seal_cpp.batch_extract
    build_csr = _seal_cpp.build_csr
    AVAILABLE = True

except Exception as _err:
    warnings.warn(
        f"[seal_cpp] C++ extension unavailable — falling back to Python/scipy. "
        f"Reason: {_err}",
        RuntimeWarning,
        stacklevel=2,
    )
