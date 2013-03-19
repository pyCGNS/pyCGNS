#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# 
from  distutils.core import setup, Extension
from  distutils.util import get_platform
import glob
import os

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('VAL')

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

# ---

setup (
name         = "CGNS.VAL",
version      = pyCGNSconfig.VAL_VERSION,
description  = "pyCGNS - SIDS and sub-grammars verification tool",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages     = ['CGNS.VAL',
                'CGNS.VAL.grammars',
                'CGNS.VAL.suite',
                'CGNS.VAL.suite.treebasics',
                'CGNS.VAL.parse'],
scripts      = ['CGNS/VAL/CGNS.VAL'],
cmdclass={'clean':setuputils.clean}
) # close setup

