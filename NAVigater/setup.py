# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - NAVigator
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
from  distutils.core import setup, Extension
from  distutils.util import get_platform
import glob
import os

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

setup (
name         = "CGNS.NAV",
version      = pyCGNSconfig.NAV_VERSION,
description  = "pyCGNS NAVigator - CGNS/Python trees navigator and editor",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
packages     = ['CGNS.NAV','CGNS.NAV.gui','CGNS.NAV.supervisor'],
scripts      = ['CGNS/NAV/CGNS.NAV'],
data_files   = [('share/CGNS/NAV/icons',glob.glob('CGNS/NAV/gui/icons/*'))],

cmdclass={'clean': setuputils.clean}
)
 
# --- last line
