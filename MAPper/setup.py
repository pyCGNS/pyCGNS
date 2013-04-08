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
name         = "CGNS.MAP",
description  = "pyCGNS MAPper - CGNS/Python mapping with HDF5 load/save",
version      = pyCGNSconfig.MAP_VERSION,
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages=['CGNS.MAP'],
)

# --- last line
