#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
"""
  pyCGNS - Python package for CFD General Notation System

  This packages provides some libs and tools dedicated to the CGNS standard.

  Modules are:

    * CGNS.MAP
    * CGNS.APP
    * CGNS.NAV
    * CGNS.DAT
    * CGNS.VAL
    * CGNS.PAT
    
"""
#
from . import version
from .version import __version__

#
backend_h5py = False
#
try:
    from CGNS.__config__ import show as show_config
except ImportError as e:
    msg = """Error importing CGNS: you cannot import CGNS while
    being in CGNS source directory; please exit the CGNS source
    tree first and relaunch your Python interpreter."""
    raise ImportError(msg) from e


submodules = [
    "MAP",
    "PAT",
]

__all__ = submodules + [
    "show_config",
    "__version__",
]


def __dir__():
    return __all__


#
# --- last line
