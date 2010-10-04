#  -------------------------------------------------------------------------
#  pyCGNS.APP - Python package for CFD General Notation System - APPlicater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import os
from  distutils.core import setup, Extension
from  distutils.util import get_platform

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('APP')
# ---
if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

setup (
name         = "CGNS.APP",
version      = pyCGNSconfig.PAT_VERSION,
description  = "pyCGNS APPlicater - CGNS/Python Application tools and utilities",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages=['CGNS.APP',
          'CGNS.APP.embedded',
          'CGNS.APP.parse',
          'CGNS.APP.demos',
          'CGNS.APP.tests'],
cmdclass={'clean':setuputils.clean}
)
# --- last line
