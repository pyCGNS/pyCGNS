# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS PATterns
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
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
