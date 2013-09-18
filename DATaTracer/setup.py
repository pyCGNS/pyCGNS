#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
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
cmdclassdict={'clean':setuputils.clean}

setup(
name         = pyCGNSconfig.NAME,
version      = pyCGNSconfig.VERSION,
description  = pyCGNSconfig.DESCRIPTION,
author       = pyCGNSconfig.AUTHOR,
author_email = pyCGNSconfig.EMAIL,
license      = pyCGNSconfig.LICENSE,
packages     = ['CGNS.DAT',
                'CGNS.DAT.db',
                'CGNS.DAT.db.dbdrivers',                
                'CGNS.DAT.demo'],
scripts      = ['CGNS/DAT/tools/CGNS.DAT',
                'CGNS/DAT/tools/daxQT',
                'CGNS/DAT/tools/CGNS.DAT.create'],
cmdclass     = cmdclassdict
)
# --- last line
