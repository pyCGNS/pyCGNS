# -------------------------------------------------------------------------
# pyCGNS.MAP - Python package for CFD General Notation System - MAPper
# See license.txt file in the root directory of this Python module source  
# -------------------------------------------------------------------------
import os

# --- pyCGNSconfig search 
import sys
sys.path=['../lib']+sys.path
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('MAP',
                                                ['HDF5','CHLone','numpy'])
# --- 

from  distutils.core import setup, Extension
from  distutils.util import get_platform
sys.prefix=sys.exec_prefix
pfx=pyCGNSconfig.PFX

cf_include_dirs=[]
cf_library_dirs=[]
cf_optional_libs=[]
cf_depends=['readme.txt','CGNS/MAP/SIDStoPython.c']

if (installprocess):
  try:
    if (not pyCGNSconfig.HAS_HDF5):
      print pfx,'ERROR: MAP requires HDF5, check pyCGNSconfig.py file!'
      sys.exit(1)
    if (not pyCGNSconfig.HAS_CHLONE):
      print pfx,'ERROR: MAP requires CHLone, check pyCGNSconfig.py file!'
      sys.exit(1)
    if (not pyCGNSconfig.HAS_NUMPY):
      print pfx,'ERROR: MAP requires Numpy, check pyCGNSconfig.py file!'
      sys.exit(1)
    cf_include_dirs=pyCGNSconfig.HDF5_PATH_INCLUDES+pyCGNSconfig.INCLUDE_DIRS
    cf_library_dirs=pyCGNSconfig.HDF5_PATH_LIBRARIES+pyCGNSconfig.LIBRARY_DIRS
    cf_optional_libs=pyCGNSconfig.HDF5_LINK_LIBRARIES
    cf_include_dirs+=pyCGNSconfig.CHLONE_PATH_INCLUDES
    cf_library_dirs+=pyCGNSconfig.CHLONE_PATH_LIBRARIES
    cf_optional_libs+=pyCGNSconfig.CHLONE_LINK_LIBRARIES
    cf_include_dirs+=pyCGNSconfig.NUMPY_PATH_INCLUDES
    cf_library_dirs+=pyCGNSconfig.NUMPY_PATH_LIBRARIES
    cf_optional_libs+=pyCGNSconfig.NUMPY_LINK_LIBRARIES
  except ValueError:
    print pfx,'ERROR: Bad pyCGNSconfig.py file for MAP!'
    sys.exit(1)

cf_optional_libs+=['CHLone']
if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

# -------------------------------------------------------------------------
setup (
name         = "CGNS.MAP",
version      = pyCGNSconfig.MAP_VERSION,
description  = "pyCGNS MAPper - CGNS/Python mapping with HDF5 load/save",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages=['CGNS.MAP'],
ext_modules  = [Extension('CGNS.MAP',
                sources = ['CGNS/MAP/MAPmodule.c'],
                include_dirs = cf_include_dirs,
                library_dirs = cf_library_dirs,
                libraries    = cf_optional_libs,
                depends      = ['CGNS/MAP/SIDStoPython.h',
                                'CGNS/MAP/SIDStoPython.c'])],
cmdclass={'clean':setuputils.clean}
)

# --- last line
