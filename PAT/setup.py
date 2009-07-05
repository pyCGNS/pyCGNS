# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS PATterns
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
from  distutils.core import setup
from  distutils.util import get_platform

# --- pyCGNSconfig search
import os
import sys
import shutil

spath=sys.path[:]
sys.path=[os.getcwd(),'%s/..'%(os.getcwd())]
try:
  import pyCGNSconfig
except ImportError:
  print 'pyGCNS[ERROR]: PAT cannot find pyCGNSconfig.py file!'
  sys.exit(1)

bptarget='./build/lib/CGNS'
bxtarget='./build/lib.%s-%s/CGNS'%(get_platform(),sys.version[0:3])
for d in sys.path:
  if (os.path.exists("%s/pyCGNSconfig.py"%d)):
    try:
      os.makedirs(bptarget)
      os.makedirs(bxtarget)
    except os.error: pass
    shutil.copy("%s/pyCGNSconfig.py"%d,"%s/pyCGNSconfig.py"%bptarget)
    shutil.copy("%s/pyCGNSconfig.py"%d,"%s/pyCGNSconfig.py"%bxtarget)

sys.path=spath
# ---

setup (
  name         = "CGNS.PAT",
  version      = "0.1.1",
  description  = "pyCGNS SIDS PATterns",
  author       = "marc Poinot",
  author_email = "marc.poinot@onera.fr",
  packages=['CGNS','CGNS.PAT']
)

# --- last line
