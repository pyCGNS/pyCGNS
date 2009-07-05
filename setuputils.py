# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS PATterns
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
import os
import sys
import shutil

from  distutils.core import setup
from  distutils.util import get_platform
from  distutils.command.clean import clean as _clean
#from  distutils.command.install import install as _install

def installConfigFiles(searchpath):
  bptarget='./build/lib/CGNS'
  bxtarget='./build/lib.%s-%s/CGNS'%(get_platform(),sys.version[0:3])
  for d in searchpath:
    if (os.path.exists("%s/pyCGNSconfig.py"%d)):
      try:
        os.makedirs(bptarget)
        os.makedirs(bxtarget)
      except os.error: pass
      shutil.copy("%s/pyCGNSconfig.py"%d,"%s/pyCGNSconfig.py"%bptarget)
      shutil.copy("%s/pyCGNSconfig.py"%d,"%s/pyCGNSconfig.py"%bxtarget)

