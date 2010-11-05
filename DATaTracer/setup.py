#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
# 
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
scripts      = ['CGNS/DAT/tools/CGNS.DAT',
                'CGNS/DAT/tools/daxQT',
                'CGNS/DAT/tools/CGNS.DAT.create'],
cmdclass={'clean':setuputils.clean}
) # close setup

