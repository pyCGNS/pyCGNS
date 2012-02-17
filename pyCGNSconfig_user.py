#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# Change these values to fit your installation
#

# --- overall directories and libs
#
import subprocess

try:
 h5p=subprocess.check_output(["which","h5dump"])
except:
  try:
    h5p=subprocess.check_output(["whence","h5dump"])
  except:
    h5p=None
if (h5p is not None):
  h5root='/'.join(h5p.split('/')[:-2])
else:
  h5root='/usr/local'

try:
 mllp=subprocess.check_output(["which","cgnsview"])
except:
  try:
    mllp=subprocess.check_output(["whence","cgnsview"])
  except:
    mllp=None
if (mllp is not None):
  mllroot='/'.join(mllp.split('/')[:-2])
else:
  mllroot='/usr/local'

# --- If you leave empty the variables below the build process would
#     try to guess for you, using the PATH you gave above.
#     Change these values only if you need to add/change something.
#     Remove the comment (the first #) and change the value

# --- stuff to add for HDF5 

HDF5_INSTALL=h5root
HDF5_INSTALL2=h5root+'/../'
#HDF5_VERSION          = ''
HDF5_PATH_INCLUDES    = [HDF5_INSTALL+'/include',HDF5_INSTALL2+'/include']
HDF5_PATH_LIBRARIES   = [HDF5_INSTALL+'/lib',HDF5_INSTALL2+'/lib']
#HDF5_LINK_LIBRARIES   = []
#HDF5_EXTRA_ARGS       = []

# --- stuff to add for numpy

#NUMPY_VERSION         = ''
#NUMPY_PATH_INCLUDES   = []
NUMPY_PATH_LIBRARIES  = ['/opt/intel/Compiler/11.1/069/lib/intel64']
#NUMPY_LINK_LIBRARIES  = ['irc']
#NUMPY_EXTRA_ARGS      = [] 

# --- stuff to add for CGNS/MLL (set the HAS_MLL to True)

MLL_INSTALL=mllroot
HAS_MLL               = True 
MLL_PATH_LIBRARIES    = [MLL_INSTALL+'/lib']
#MLL_LINK_LIBRARIES    = []
MLL_PATH_INCLUDES     = [MLL_INSTALL+'/include'] 
#MLL_VERSION           = '' 
#MLL_EXTRA_ARGS        = []

# --- stuff to add for Python

#PYTHON_VERSION          = ''
#PYTHON_PATH_INCLUDES    = []
#PYTHON_PATH_LIBRARIES   = []
#PYTHON_LINK_LIBRARIES   = [] 
#PYTHON_EXTRA_ARGS       = []

# --- stuff to add for CHLone

#CHLONE_VERSION          = ''
CHLONE_PATH_INCLUDES    = ['/home/tools/local/eosz/include']
CHLONE_PATH_LIBRARIES   = ['/home/tools/local/eosz/lib']
#CHLONE_LINK_LIBRARIES   = [] 
#CHLONE_EXTRA_ARGS       = []

#
# -------------------------------------------------------------------------


