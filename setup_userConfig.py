#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# Change these values to fit your installation
#
# --- If you leave empty the variables below the build process would
#     try to guess for you, using your current environment variables.
#     Change these values only if you need to add/change something.
#
# --- stuff to add for numpy (used by NAV)
#     you should not change this, the numpy stuff is expected to be detected
#     using the numpy import itself
#
if False:
  NUMPY_VERSION         = ''
  NUMPY_PATH_INCLUDES   = []
  NUMPY_PATH_LIBRARIES  = ['']
  NUMPY_LINK_LIBRARIES  = ['']
  NUMPY_EXTRA_ARGS      = [] 
#
# --- stuff to add for HDF5 (used by MAP and WRA)
#     the hdf5 install is detected using a 'which h5dump' and then we parse the
#     installation. Thus non-standard installs would require to set these vars
#
if True:
  HDF5_VERSION          = ''
  HDF5_PATH_INCLUDES    = ['C:\Appl\Anaconda2\Library\include']
  HDF5_PATH_LIBRARIES   = ['C:\Appl\Anaconda2\Library\lib']
  HDF5_LINK_LIBRARIES   = ['hdf5']
  HDF5_EXTRA_ARGS       = ['-Wno-return-type']
#
# --- stuff to add for CGNS/MLL (used by WRA)
#     the cgns/mll install is detected using a 'which cgnscheck' then we parse 
#     the installation. Non-standard installs would require to set these vars
#
if False:
  HAS_MLL               = True 
  MLL_VERSION           = '' 
  MLL_PATH_INCLUDES     = [''] 
  MLL_PATH_LIBRARIES    = ['']
  MLL_LINK_LIBRARIES    = ['']
  MLL_EXTRA_ARGS        = []
#
# -------------------------------------------------------------------------


