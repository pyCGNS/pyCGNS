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

cmdclassdict={'clean':setuputils.clean}

# ---

setup (
name         = pyCGNSconfig.NAME,
version      = pyCGNSconfig.VERSION,
description  = pyCGNSconfig.DESCRIPTION,
author       = pyCGNSconfig.AUTHOR,
author_email = pyCGNSconfig.EMAIL,
license      = pyCGNSconfig.LICENSE,
packages     = ['CGNS.VAL',
                'CGNS.VAL.grammars',
                'CGNS.VAL.suite',
                'CGNS.VAL.suite.elsA',
                'CGNS.VAL.suite.SIDS',
                'CGNS.VAL.parse'],
scripts      = ['CGNS/VAL/CGNS.VAL'],
cmdclass     = cmdclassdict
) # close setup

