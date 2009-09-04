# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS-to-Python MAPping            
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------

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

setup (
  name         = "CGNS.MAP",
  version      = "0.0.1",
  description  = "pyCGNS MAPping SIDS-to-Python",
  author       = "marc Poinot",
  author_email = "marc.poinot@onera.fr",
  packages=['CGNS'],
  ext_modules  = [Extension('CGNS.MAP',sources = ['MAPmodule.c'],
                  include_dirs = cf_include_dirs,
                  library_dirs = cf_library_dirs,
                  libraries    = cf_optional_libs)],
  cmdclass={'clean':setuputils.clean}
)

# --- last line
