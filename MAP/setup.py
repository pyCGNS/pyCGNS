# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS-to-Python MAPping            
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
from  distutils.core import setup, Extension
import pyCGNSconfig

include_dirs=[\
    '/home/tools/local/x86_64/include',\
    '/home/tools/local/x86_64/lib/python2.5/site-packages/numpy/core/include'\
]

library_dirs=['/home/tools/local/x86_64/lib']
optional_libs=['cgns','hdf5']

setup (
  name         = "CGNS.MAP",
  version      = "0.1",
  description  = "pyCGNS MAPping SIDS-to-Python",
  author       = "marc Poinot",
  author_email = "marc.poinot@onera.fr",
  ext_modules  = [Extension('CGNS.MAP', 
                           sources=['MAPmodule.c'],
                           include_dirs = include_dirs,
                           library_dirs = library_dirs,
                           libraries    = optional_libs)],
)
