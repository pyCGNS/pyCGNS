#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#  
import os
from distutils.core import setup, Extension
from distutils import sysconfig

try:
  from Cython.Distutils import build_ext
  HAS_CYTHON=True
except:
  HAS_CYTHON=False

# skip it now
HAS_CYTHON=False

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('PAT')
# ---

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

if HAS_CYTHON:
  cmdclassdict={'clean':setuputils.clean,'build_ext':build_ext}
  extmods=[ Extension('CGNS.PAT.cgnsutils',
                      ['CGNS/PAT/cgnsutils.pyx'],
                      include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES) ]
else:
  cmdclassdict={'clean':setuputils.clean}
  extmods=[]

setup (
name         = pyCGNSconfig.NAME,
version      = pyCGNSconfig.VERSION,
description  = pyCGNSconfig.DESCRIPTION,
author       = pyCGNSconfig.AUTHOR,
author_email = pyCGNSconfig.EMAIL,
license      = pyCGNSconfig.LICENSE,
packages     = ['CGNS.PAT','CGNS.PAT.SIDS','CGNS.PAT.test'],
ext_modules  = extmods,
cmdclass     = cmdclassdict
)
# --- last line
