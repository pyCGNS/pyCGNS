#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import os
from  distutils.core import setup, Extension
from  distutils.util import get_platform
from Cython.Distutils import build_ext

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('APP')
# ---
if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

incdirs=['%s/lib/python%s/site-packages/numpy/core/include'\
         %(os.path.normpath(sys.exec_prefix),sys.version[:3]),
         '.',
         'CGNS/APP/probe']

x_mods=[Extension("CGNS.APP.probe.arrayutils",
                  ["CGNS/APP/probe/arrayutils.pyx",
                   "CGNS/APP/probe/hashutils.c"],
                  include_dirs = incdirs,
                  extra_compile_args=[])]

# -------------------------------------------------------------------------
setup (
name         = "CGNS.APP",
version      = pyCGNSconfig.PAT_VERSION,
description  = "pyCGNS - CGNS/Python Application tools and utilities",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages=['CGNS.APP',
          'CGNS.APP.embedded',
          'CGNS.APP.parse',
          'CGNS.APP.demos',
          'CGNS.APP.sids',
          'CGNS.APP.probe',          
          'CGNS.APP.test'],
 ext_modules = x_mods,
cmdclass={'clean':setuputils.clean,'build_ext': build_ext}
)
# --- last line
