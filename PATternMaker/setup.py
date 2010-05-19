#  ---------------------------------------------------------------------------
#  pyCGNS.PAT - Python package for CFD General Notation System - PATternMaker
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#  $Release$
#  ---------------------------------------------------------------------------

import os
from distutils.core import setup
from distutils import sysconfig

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('PAT')
# ---

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

setup (
name         = "CGNS.PAT",
version      = pyCGNSconfig.PAT_VERSION,
description  = "pyCGNS PATternMaker - CGNS/Python patterns for SIDS and other",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages=['CGNS.PAT','CGNS.PAT.SIDS'],
cmdclass={'clean':setuputils.clean}
)
# --- last line
