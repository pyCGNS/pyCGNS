#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# 
# Default values, would be overwritten during install
#
import sys

# --------------------------------------------------------------------
def pathfromexec(execname):
    import os
    for pth in os.environ['PATH']:
        apth = os.path.normpath(os.path.expanduser(pth))
        if os.path.exists(apth + '/' + execname): return pth
    return ''


mylocal = sys.prefix

# --------------------------------------------------------------------
INCLUDE_DIRS = ["%s/include" % mylocal]
LIBRARY_DIRS = ["%s/lib" % mylocal]

# Use integers instead of booleans, values are used in C files
HAS_H5PY = 0
HAS_HDF5 = 1

try:
    import vtk
    from CGNS.NAV.wvtk import Q7VTK
    HAS_VTK = 1
except ImportError:
    HAS_VTK = 0

HAS_MSW = 0
USE_COMPACT_STORAGE = 1

hdf5path = pathfromexec("h5dump")
HDF5_VERSION = "1.8"
HDF5_PATH_INCLUDES = [hdf5path + "../include"]
HDF5_PATH_LIBRARIES = [hdf5path + "../lib"]
HDF5_LINK_LIBRARIES = ["hdf5"]
HDF5_EXTRA_ARGS = []

NUMPY_VERSION = ""
NUMPY_VERSION_API = ""
NUMPY_PATH_INCLUDES = []
NUMPY_PATH_LIBRARIES = []
NUMPY_LINK_LIBRARIES = []
NUMPY_EXTRA_ARGS = []

PYQT_VERSION = ""
QT_VERSION = ""
CYTHON_VERSION = ""
VTK_VERSION = ""

# cannot manage include orders here...
INCLUDE_DIRS = INCLUDE_DIRS + HDF5_PATH_INCLUDES \
                + NUMPY_PATH_INCLUDES
LIBRARY_DIRS += LIBRARY_DIRS + HDF5_PATH_LIBRARIES \
                + NUMPY_PATH_LIBRARIES
#
# -------------------------------------------------------------------------
# You should not change values beyond this point
#
PFX = '### pyCGNS:'
#
file_pattern = """#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# This file has been generated on [%(DATE)s]
# Using platform [%(PLATFORM)s]

DATE="%(DATE)s"
PLATFORM="%(PLATFORM)s"
PFX="%(PFX)s"

HAS_HDF5=%(HAS_HDF5)s
HAS_H5PY=%(HAS_H5PY)s
HAS_VTK=%(HAS_VTK)s
HAS_MSW=%(HAS_MSW)s
USE_COMPACT_STORAGE=%(USE_COMPACT_STORAGE)s

INCLUDE_DIRS=%(INCLUDE_DIRS)s
LIBRARY_DIRS=%(LIBRARY_DIRS)s

HDF5_VERSION="%(HDF5_VERSION)s"
HDF5_PATH_INCLUDES=%(HDF5_PATH_INCLUDES)s
HDF5_PATH_LIBRARIES=%(HDF5_PATH_LIBRARIES)s
HDF5_LINK_LIBRARIES=%(HDF5_LINK_LIBRARIES)s
HDF5_EXTRA_ARGS=%(HDF5_EXTRA_ARGS)s

NUMPY_VERSION="%(NUMPY_VERSION)s"
NUMPY_VERSION_API="%(NUMPY_VERSION_API)s"
NUMPY_PATH_INCLUDES=%(NUMPY_PATH_INCLUDES)s
NUMPY_PATH_LIBRARIES=%(NUMPY_PATH_LIBRARIES)s
NUMPY_LINK_LIBRARIES=%(NUMPY_LINK_LIBRARIES)s
NUMPY_EXTRA_ARGS=%(NUMPY_EXTRA_ARGS)s

PYQT_VERSION="%(PYQT_VERSION)s"
QT_VERSION="%(QT_VERSION)s"
CYTHON_VERSION="%(CYTHON_VERSION)s"
VTK_VERSION="%(VTK_VERSION)s"

__version__=6 # @@UPDATEVERSION@@
__release__=0 # @@UPDATERELEASE@@
__revision__=0 # @@UPDATEREVISION@@
__vid__="%%s.%%s.%%s"%%(__version__,__release__,__revision__)
__doc__="pyCGNS - %%s - Python package for CGNS (CFD General Notation System)"\
        %%(__vid__)
version=__vid__

NAME="%(NAME)s"
VERSION=__vid__
DESCRIPTION=__doc__
AUTHOR="%(AUTHOR)s"
EMAIL="%(EMAIL)s"
LICENSE="%(LICENSE)s"

"""
#
__version__=6 # @@UPDATEVERSION@@
__release__=0 # @@UPDATERELEASE@@
__revision__=0 # @@UPDATEREVISION@@
__vid__="%s.%s.%s"%(__version__,__release__,__revision__)
__doc__="pyCGNS - %s - Python package for CGNS (CFD General Notation System)"\
        %(__vid__)
version=__vid__
REVISION = __revision__




# common meta data
NAME = 'pyCGNS'
VERSION = __vid__
DESCRIPTION = __doc__
AUTHOR = 'Marc Poinot'
EMAIL = "marc.poinot@safrangroup.com"
LICENSE = "LGPL 2"
URL = "https://github.com/pyCGNS/pyCGNS"

# --- last line
