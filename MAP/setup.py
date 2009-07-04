# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS-to-Python MAPping            
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
from  distutils.core import setup, Extension

# --- pyConfig search
import os
import sys
spath=sys.path[:]
sys.path=[os.getcwd(),'%s/..'%(os.getcwd())]
try:
  import pyCGNSconfig
except ImportError:
  print 'pyGCNS[ERROR]: MAP cannot find pyCGNSconfig.py file!'
  sys.exit(1)
sys.path=spath
# ---

try:
  if (not pyCGNSconfig.HAS_HDF5):
    print 'pyGCNS[ERROR]: MAP requires HDF5, please set pyCGNSconfig.py file!'
    sys.exit(1)
  cf_include_dirs=pyCGNSconfig.include_dirs
  cf_library_dirs=pyCGNSconfig.library_dirs
  cf_optional_libs=pyCGNSconfig.optional_libs
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
)

# --- last line
