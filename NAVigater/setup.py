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
import sys

cui='/home/tools/local/tiamat/bin/pyside-uic'
crc='/home/tools/local/tiamat/bin/pyside-rcc'
ccy='cython'

try:
  from Cython.Distutils import build_ext
  HAS_CYTHON=True
except:
  HAS_CYTHON=False
  raise 'Cannot build CGNS.NAV without cython'
  sys.exit()

# --- pyCGNSconfig search
sys.path=['../lib']+sys.path
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('NAV',['numpy'])
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


  if (not os.path.exists("build")): os.system("ln -sf ../build build")
  setuputils.installConfigFiles()

  modnamelist=[
      'Q7TreeWindow',
      'Q7MainWindow',
      'Q7OptionsWindow',
      'Q7FormWindow',
      'Q7FileWindow',
      'Q7VTKWindow'
      ]
  modgenlist=[]
  modextlist=[Extension("CGNS.NAV.mtree", ["CGNS/NAV/mtree.pyx"],
                        include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES,
                        library_dirs = pyCGNSconfig.NUMPY_PATH_LIBRARIES,
                        libraries    = pyCGNSconfig.NUMPY_LINK_LIBRARIES,
                        )]
  for m in modnamelist:
     modextlist+=[Extension("CGNS.NAV.%s"%m, ["CGNS/NAV/G/%s.pyx"%m],
                            include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES,
                            library_dirs = pyCGNSconfig.NUMPY_PATH_LIBRARIES,
                            libraries    = pyCGNSconfig.NUMPY_LINK_LIBRARIES,
                            )]
     g=("CGNS/NAV/T/%s.ui"%m,"CGNS/NAV/G/%s.pyx"%m)
     if (os.path.getmtime(g[0])>os.path.getmtime(g[1])): modgenlist+=[m]
                  
  for m in modgenlist:
      print 'Generate from updated GUI templates: ',m
      com="(%s -o CGNS/NAV/G/%s.pyx CGNS/NAV/T/%s.ui;(cd CGNS/NAV/G;%s -a %s.pyx))"%(cui,m,m,ccy,m)
      os.system(com)
         
  if (os.path.getmtime('CGNS/NAV/R/Res.qrc')>os.path.getmtime('CGNS/NAV/Res_rc.py')):
      print 'Generate from updated GUI Ressources'
      com="(%s -o CGNS/NAV/Res_rc.py CGNS/NAV/R/Res.qrc)"%(crc)
      os.system(com)
  cmdclassdict={'clean':setuputils.clean,'build_ext':build_ext}
else:
  cmdclassdict={'clean':setuputils.clean}
  modextlist=[]

print pyCGNSconfig.NUMPY_PATH_LIBRARIES
setup (
name         = "CGNS.NAV",
version      = pyCGNSconfig.NAV_VERSION,
description  = "pyCGNS NAVigator - CGNS/Python trees navigator and editor",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
packages     = ['CGNS.NAV'],
scripts      = ['CGNS/NAV/CGNS.NAV'],
ext_modules  = modextlist,
cmdclass     = cmdclassdict
)
 
# --- last line
