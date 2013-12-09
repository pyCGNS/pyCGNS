# -------------------------------------------------------------------------
# pyCGNS.MAP - Python package for CFD General Notation System - MAPper
# See license.txt file in the root directory of this Python module source  
# -------------------------------------------------------------------------
import os
from distutils.core import setup, Extension
from distutils import sysconfig

try:
    import CHLone
except:
    print "### pyCGNSERROR: CHLone not found, abort MAP"

import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('MAP')

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

# -------------------------------------------------------------------------
setup (
name         = pyCGNSconfig.NAME,
version      = pyCGNSconfig.VERSION,
description  = pyCGNSconfig.DESCRIPTION,
author       = pyCGNSconfig.AUTHOR,
author_email = pyCGNSconfig.EMAIL,
license      = pyCGNSconfig.LICENSE,
packages     = ['CGNS.MAP','CGNS.MAP.test'],
)

# --- last line
