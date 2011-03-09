#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $KeyFile$
#  $Release$
#  -------------------------------------------------------------------------
from  distutils.core import setup, Extension
from  distutils.util import get_platform
import glob
import os

try:
  from Cython.Distutils import build_ext
  HAS_CYTHON=True
except:
  HAS_CYTHON=False

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('NAV')
# ---

if installprocess:
  from optparse import OptionParser
  from distutils.core import setup, Extension
  import re

  sys.path.append('.')
  from pyCGNSconfig import version as __vid__

  from optparse import OptionParser

  parser = OptionParser()
  parser.add_option("--prefix",dest="prefix")
  try:
    (options, args) = parser.parse_args(sys.argv)
  except optparse.OptionError: pass

  icondirprefix=sys.prefix
  try:
    if (options.prefix != None): icondirprefix=options.prefix
    fg=open("./CGNS/NAV/gui/s7globals_.py",'r')
    llg=fg.readlines()
    fg.close()
    gg=open("./CGNS/NAV/gui/s7globals.py",'w+')
    for lg in llg:
      if (lg[:31]=='    self.s7icondirectoryprefix='):
        gg.write('    self.s7icondirectoryprefix="%s"\n'%icondirprefix)
      else:
        gg.write(lg)
    gg.close()
  except KeyError: pass

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

if HAS_CYTHON:
  cmdclassdict={'clean':setuputils.clean,'build_ext':build_ext}
  extmods=[ Extension('CGNS.NAV.gui.s7vtkView',
                           ['CGNS/NAV/gui/s7vtkView.pyx'],
                           include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES) ]
else:
  cmdclassdict={'clean':setuputils.clean}
  extmods=[]

setup (
name         = "CGNS.NAV",
version      = pyCGNSconfig.NAV_VERSION,
description  = "pyCGNS NAVigator - CGNS/Python trees navigator and editor",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages     = ['CGNS.NAV','CGNS.NAV.gui','CGNS.NAV.supervisor'],
scripts      = ['CGNS/NAV/CGNS.NAV'],
data_files   = [('share/CGNS/NAV/icons',glob.glob('CGNS/NAV/gui/icons/*'))],
ext_modules  = extmods,
cmdclass     = cmdclassdict
)
 
# --- last line
