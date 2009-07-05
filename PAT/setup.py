# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS PATterns
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
from distutils.core import setup
from distutils import sysconfig

# --- pyCGNSconfig search
import os
import sys
import shutil

spath=sys.path[:]
sys.path=[os.getcwd(),'%s/..'%(os.getcwd())]
import setuputils
try:
  import pyCGNSconfig
except ImportError:
  print 'pyGCNS[ERROR]: PAT cannot find pyCGNSconfig.py file!'
  sys.exit(1)
sys.path=spath
setuputils.installConfigFiles([os.getcwd(),'%s/..'%(os.getcwd())])
# ---

sys.prefix=sys.exec_prefix

setup (
  name         = "CGNS.PAT",
  version      = "0.1.1",
  description  = "pyCGNS SIDS PATterns",
  author       = "marc Poinot",
  author_email = "marc.poinot@onera.fr",
  packages=['CGNS','CGNS.PAT']
)

# --- last line
