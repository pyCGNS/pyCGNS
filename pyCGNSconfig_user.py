#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# Change these values to fit your installation
#

# --- overall directories and libs

mylocal='/home/tools/local/x86_64p' 

INCLUDE_DIRS =['/home/tools/local/x86_64t/include'] 
INCLUDE_DIRS+=['%s/include'%mylocal]
INCLUDE_DIRS+=['%s/lib/python2.5/site-packages/numpy/core/include'%mylocal]

LIBRARY_DIRS =['/home/tools/local/x86_64t/lib']
LIBRARY_DIRS+=['%s/lib'%mylocal] 

# --- If you leave empty the variables below the build process would
#     try to guess for you, using the PATH you gave above.
#     Change these values only if you need to add/change something.
#     Remove the comment (the first #) and change the value

# --- stuff to add for HDF5 

#HDF5_VERSION          = ''
#HDF5_PATH_INCLUDES    = []
#HDF5_PATH_LIBRARIES   = []
#HDF5_LINK_LIBRARIES   = []
#HDF5_EXTRA_ARGS       = []

# --- stuff to add for numpy

#NUMPY_VERSION         = ''
#NUMPY_PATH_INCLUDES   = []
#NUMPY_PATH_LIBRARIES  = []
#NUMPY_LINK_LIBRARIES  = []
#NUMPY_EXTRA_ARGS      = [] 

# --- stuff to add for CGNS/MLL (set the HAS_MLL to True)

#HAS_MLL               = True 
#MLL_PATH_LIBRARIES    = []
#MLL_LINK_LIBRARIES    = []
#MLL_PATH_INCLUDES     = [] 
#MLL_VERSION           = '' 
#MLL_EXTRA_ARGS        = []

#
# -------------------------------------------------------------------------


