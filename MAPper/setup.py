#  -------------------------------------------------------------------------
#  pyCGNS.MAP - Python package for CFD General Notation System - MAPper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import os

# --- pyCGNSconfig search 
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('MAP')
# --- 

from  distutils.core import setup, Extension
from  distutils.util import get_platform
sys.prefix=sys.exec_prefix

cf_include_dirs=[]
cf_library_dirs=[]
cf_optional_libs=[]
cf_depends=['readme.txt','CGNS/MAP/SIDStoPython.c']

if (installprocess):
  try:
    if (not pyCGNSconfig.HAS_HDF5):
      print 'pyGCNS[ERROR]: MAP requires HDF5, check pyCGNSconfig.py file!'
      sys.exit(1)
    cf_include_dirs=pyCGNSconfig.INCLUDE_DIRS
    cf_library_dirs=pyCGNSconfig.LIBRARY_DIRS
    cf_optional_libs=pyCGNSconfig.OPTIONAL_LIBS
  except:
    print 'pyGCNS[ERROR]: bad pyCGNSconfig.py file for MAP!'
    sys.exit(1)

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

setup (
name         = "CGNS.MAP",
version      = pyCGNSconfig.MAP_VERSION,
description  = "pyCGNS MAPper - SIDS/Python mapping with HDF5 load/save",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages=['CGNS.MAP'],
ext_modules  = [Extension('CGNS.MAP',
                sources = ['CGNS/MAP/MAPmodule.c',
                           'CGNS/MAP/CHLone_l3.c',
                           'CGNS/MAP/CHLone_error.c'],
                depends      = cf_depends,
                include_dirs = cf_include_dirs,
                library_dirs = cf_library_dirs,
                libraries    = cf_optional_libs)],
cmdclass={'clean':setuputils.clean}
)

# --- last line
