import sys
from pathlib import Path

import torch

# Add submodule directories to sys.path so their internal packages
# (e.g. efficient_kan, fastkan, fasterkan) are directly importable
# without needing __init__.py inside the git submodules.
_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
for _subdir in [
    _MODULES_DIR / "efficientkan" / "src",  # contains efficient_kan/
    _MODULES_DIR / "fastkan",  # contains fastkan/
    _MODULES_DIR / "fasterkan",  # contains fasterkan/
]:
    _s = str(_subdir)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
