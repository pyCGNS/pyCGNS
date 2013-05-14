#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
# Change these values to fit your installation
# See notes at end of file about config values
#
import sys

# --------------------------------------------------------------------
def pathfromexec(execname):
  import os
  for pth in os.environ['PATH']:
    apth=os.path.normpath(os.path.expanduser(pth))
    if os.path.exists(apth+'/'+execname): return pth
  return ''

mylocal=sys.prefix

# --------------------------------------------------------------------
INCLUDE_DIRS =['%s/include'%mylocal]
LIBRARY_DIRS =['%s/lib'%mylocal]

HAS_MLL=False
HAS_NUMPY=False
HAS_CHLONE=False
HAS_HDF5=False

# useless with distutils but maybe required for other external libs
PYTHON_VERSION          = "%d.%d"%(sys.version_info[0],sys.version_info[1])
PYTHON_PATH_INCLUDES    = [sys.prefix+'/include']
PYTHON_PATH_LIBRARIES   = [sys.prefix+'/lib']
PYTHON_LINK_LIBRARIES   = [] 
PYTHON_EXTRA_ARGS       = []

hdf5path=pathfromexec('h5dump')
HDF5_VERSION          = '1.8'
HDF5_PATH_INCLUDES    = [hdf5path+'../include']
HDF5_PATH_LIBRARIES   = [hdf5path+'..:lib']
HDF5_LINK_LIBRARIES   = ['hdf5'] 
HDF5_EXTRA_ARGS       = []

MLL_PATH_LIBRARIES    = []
MLL_LINK_LIBRARIES    = []
MLL_PATH_INCLUDES     = []
MLL_VERSION           = ''
MLL_EXTRA_ARGS        = []

NUMPY_VERSION         = ''
NUMPY_PATH_INCLUDES   = []
NUMPY_PATH_LIBRARIES  = []
NUMPY_LINK_LIBRARIES  = []
NUMPY_EXTRA_ARGS      = []

CHLONE_VERSION         = ''
CHLONE_PATH_INCLUDES   = []
CHLONE_PATH_LIBRARIES  = []
CHLONE_LINK_LIBRARIES  = []
CHLONE_EXTRA_ARGS      = []

# cannot manage include orders here...
INCLUDE_DIRS=INCLUDE_DIRS+PYTHON_PATH_INCLUDES+HDF5_PATH_INCLUDES\
             +MLL_PATH_INCLUDES+NUMPY_PATH_INCLUDES+CHLONE_PATH_INCLUDES
LIBRARY_DIRS+=LIBRARY_DIRS+PYTHON_PATH_LIBRARIES+HDF5_PATH_LIBRARIES\
             +MLL_PATH_LIBRARIES+NUMPY_PATH_LIBRARIES+CHLONE_PATH_LIBRARIES
#
# -------------------------------------------------------------------------
# You should not change values beyond this point
#
PFX='### pyCGNS:'
#
__version__=4 # @@UPDATEVERSION@@
__release__=3 # @@UPDATERELEASE@@
__revision__=417 # @@UPDATEREVISION@@
__vid__="%s.%s.%s"%(__version__,__release__,__revision__)
__doc__="""pyCGNS - %s - Python package for CFD General Notation System"""\
        %(__vid__)
version=__vid__
#
WRA_VERSION=__vid__+'.0'
VAL_VERSION=__vid__+'.0'
MAP_VERSION=__vid__+'.0'
NAV_VERSION=__vid__+'.0'
PAT_VERSION=__vid__+'.0'
DAT_VERSION=__vid__+'.0'
APP_VERSION=__vid__+'.0'
#
file_pattern="""#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# This file has been generated on [%(DATE)s]
# Using platform [%(PLATFORM)s]

DATE='%(DATE)s'
PLATFORM='%(PLATFORM)s'
PFX='%(PFX)s'

WRA_VERSION='%(WRA_VERSION)s'
VAL_VERSION='%(VAL_VERSION)s'
MAP_VERSION='%(MAP_VERSION)s'
NAV_VERSION='%(NAV_VERSION)s'
PAT_VERSION='%(PAT_VERSION)s'
DAT_VERSION='%(DAT_VERSION)s'
APP_VERSION='%(APP_VERSION)s'

HAS_HDF5=%(HAS_HDF5)s
HAS_CHLONE=%(HAS_CHLONE)s
HAS_NUMPY=%(HAS_NUMPY)s
HAS_MLL=%(HAS_MLL)s

INCLUDE_DIRS=%(INCLUDE_DIRS)s
LIBRARY_DIRS=%(LIBRARY_DIRS)s

PYTHON_VERSION='%(PYTHON_VERSION)s'
PYTHON_PATH_LIBRARIES=%(PYTHON_PATH_LIBRARIES)s
PYTHON_LINK_LIBRARIES=%(PYTHON_LINK_LIBRARIES)s
PYTHON_PATH_INCLUDES=%(PYTHON_PATH_INCLUDES)s
PYTHON_EXTRA_ARGS=%(PYTHON_EXTRA_ARGS)s

MLL_VERSION='%(MLL_VERSION)s'
MLL_PATH_LIBRARIES=%(MLL_PATH_LIBRARIES)s
MLL_LINK_LIBRARIES=%(MLL_LINK_LIBRARIES)s
MLL_PATH_INCLUDES=%(MLL_PATH_INCLUDES)s
MLL_EXTRA_ARGS=%(MLL_EXTRA_ARGS)s

HDF5_VERSION='%(HDF5_VERSION)s'
HDF5_PATH_INCLUDES=%(HDF5_PATH_INCLUDES)s
HDF5_PATH_LIBRARIES=%(HDF5_PATH_LIBRARIES)s
HDF5_LINK_LIBRARIES=%(HDF5_LINK_LIBRARIES)s
HDF5_EXTRA_ARGS=%(HDF5_EXTRA_ARGS)s

CHLONE_VERSION='%(CHLONE_VERSION)s'
CHLONE_PATH_INCLUDES=%(CHLONE_PATH_INCLUDES)s
CHLONE_PATH_LIBRARIES=%(CHLONE_PATH_LIBRARIES)s
CHLONE_LINK_LIBRARIES=%(CHLONE_LINK_LIBRARIES)s
CHLONE_EXTRA_ARGS=%(CHLONE_EXTRA_ARGS)s

NUMPY_VERSION='%(NUMPY_VERSION)s'
NUMPY_PATH_INCLUDES=%(NUMPY_PATH_INCLUDES)s
NUMPY_PATH_LIBRARIES=%(NUMPY_PATH_LIBRARIES)s
NUMPY_LINK_LIBRARIES=%(NUMPY_LINK_LIBRARIES)s
NUMPY_EXTRA_ARGS=%(NUMPY_EXTRA_ARGS)s

__version__=4 # @@UPDATEVERSION@@
__release__=3 # @@UPDATERELEASE@@
__revision__=417 # @@UPDATEREVISION@@
__vid__="%%s.%%s.%%s"%%(__version__,__release__,__revision__)
__doc__='pyCGNS - %%s - Python package for CFD General Notation System'\
        %%(__vid__)
version=__vid__
"""
#

