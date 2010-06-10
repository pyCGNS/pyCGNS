#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
# 

# --- FORCE QUIT THIS MODULE IS NOT READY
import sys
print "### pyCGNS: WARNING you cannot use DAT now - wait next release"
print "### pyCGNS: WARNING skip DAT install"
sys.exit(1)
# ---

import os
from distutils.core import setup, Extension

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('DAT')
# ---

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

setup(
name         = "CGNS.DAT",
version      = pyCGNSconfig.DAT_VERSION,
description  = "pyCGNS DATaTracer - DBMS archival and CGNS files tracability",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages     = ['CGNS.DAT',
                'CGNS.DAT.db',
                'CGNS.DAT.db.dbdrivers',                
                'CGNS.DAT.demo'],
scripts      = ['CGNS/DAT/tools/daxDB',
                'CGNS/DAT/tools/daxQT',
                'CGNS/DAT/tools/daxET'],
cmdclass={'clean':setuputils.clean}
) # close setup

