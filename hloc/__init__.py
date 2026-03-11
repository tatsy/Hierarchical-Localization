import os
import logging
import platform

import coloredlogs
from packaging import version

if platform.system() == "Darwin":
    # Avoid potential deadlock when using PyTorch with OpenMP on macOS
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__version__ = "1.5"

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("hloc")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
coloredlogs.install(level="INFO", logger=logger)

try:
    import pycolmap
except ImportError:
    logger.warning("pycolmap is not installed, some features may not work.")
else:
    min_version = version.parse("3.13.0")
    found_version = pycolmap.__version__
    if found_version != "dev":
        ver_number = version.parse(found_version)
        if ver_number < min_version:
            s = f"pycolmap>={min_version}"
            logger.warning(
                f"hloc requires {s} but found pycolmap=={found_version},"
                f'please upgrade with `pip install --upgrade "{s}"`'
            )
