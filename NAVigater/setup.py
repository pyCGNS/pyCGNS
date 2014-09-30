#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# 
from  distutils.core import setup, Extension
from  distutils.util import get_platform
import glob
import os
import sys

cui='pyside-uic'
crc='pyrcc4' #'pyside-rcc'
ccy='cython'

try:
  from Cython.Distutils import build_ext
except:
  raise Exception('Cannot build CGNS.NAV without cython')
  sys.exit()

try:
  import PySide.QtCore
except:
  raise Exception('Cannot build CGNS.NAV without PySide (Qt)')
  sys.exit()

try:
  import vtk
except:
  raise Exception('Cannot build CGNS.NAV without vtk')
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
  from optparse import OptionParser,OptionError

  fakefile="./CGNS/NAV/fake.pxi"

  parser = OptionParser()
  parser.add_option("--force",dest="forcerebuild",action="store_true")
  parser.add_option("--prefix",dest="prefix")
  parser.add_option("--dist",dest="dist")
  parser.add_option("--compiler",dest="compiler")
  parser.add_option("--build-base",dest="build-base")
  parser.add_option("--format",dest="format")
  try:
    (options, args) = parser.parse_args(sys.argv)
    if (options.forcerebuild):
      setuputils.touch(fakefile)
  except OptionError: pass

  setuputils.installConfigFiles()
  modnamelist=[
      'Q7TreeWindow',
      'Q7DiffWindow',
      'Q7MergeWindow',
      'Q7ControlWindow',
      'Q7OptionsWindow',
      'Q7FormWindow',
      'Q7FileWindow',
      'Q7QueryWindow',
      'Q7SelectionWindow',
      'Q7InfoWindow',
      'Q7DiagWindow',
      'Q7LinkWindow',
      'Q7VTKPlotWindow',
      'Q7HelpWindow',
      'Q7ToolsWindow',
      'Q7PatternWindow',
      'Q7AnimationWindow',
      'Q7MessageWindow',
      'Q7LogWindow',
      'Q7VTKWindow'
      ]
  modgenlist=[]
  modextlist=[]
  for mfile in ['mtree','mparser','mquery','mcontrol','mtable','mpattern',
                'diff','mdifftreeview','merge','mmergetreeview']:
     modextlist+=[Extension("CGNS.NAV.%s"%mfile,["CGNS/NAV/%s.pyx"%mfile,
                                                 fakefile],
                           include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES,
                           library_dirs = pyCGNSconfig.NUMPY_PATH_LIBRARIES,
                           libraries    = pyCGNSconfig.NUMPY_LINK_LIBRARIES,
                           )]
  for m in modnamelist:
     modextlist+=[Extension("CGNS.NAV.%s"%m, ["CGNS/NAV/G/%s.pyx"%m,
                                              fakefile],
                            include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES,
                            library_dirs = pyCGNSconfig.NUMPY_PATH_LIBRARIES,
                            libraries    = pyCGNSconfig.NUMPY_LINK_LIBRARIES,
                            )]
     g=("CGNS/NAV/T/%s.ui"%m,"CGNS/NAV/G/%s.pyx"%m)
     if (not os.path.exists(g[1])
         or os.path.getmtime(g[0])>os.path.getmtime(g[1])): modgenlist+=[m]
                  
  modextlist+=[Extension("CGNS.NAV.temputils",["CGNS/NAV/temputils.pyx",
                                               fakefile],
                         include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES,
                         library_dirs = pyCGNSconfig.NUMPY_PATH_LIBRARIES,
                         libraries    = pyCGNSconfig.NUMPY_LINK_LIBRARIES,
                       )]

  for m in modgenlist:
      print '### pyCGNS: Generate from updated GUI templates: ',m
      com="(%s -o CGNS/NAV/G/%s.pyx CGNS/NAV/T/%s.ui;(cd CGNS/NAV/G;%s -a %s.pyx))2>/dev/null"%(cui,m,m,ccy,m)
      print com
      os.system(com)
         
  if (os.path.getmtime('CGNS/NAV/R/Res.qrc')>os.path.getmtime('CGNS/NAV/Res_rc.py')):
      print '### pyCGNS: Generate from updated GUI Ressources'
      com="(%s -o CGNS/NAV/Res_rc.py CGNS/NAV/R/Res.qrc)2>/dev/null"%(crc)
      print com
      os.system(com)
  cmdclassdict={'clean':setuputils.clean,'build_ext':build_ext}
else:
  cmdclassdict={'clean':setuputils.clean}
  modextlist=[]

setup (
name         = pyCGNSconfig.NAME,
version      = pyCGNSconfig.VERSION,
description  = pyCGNSconfig.DESCRIPTION,
author       = pyCGNSconfig.AUTHOR,
author_email = pyCGNSconfig.EMAIL,
license      = pyCGNSconfig.LICENSE,
packages     = ['CGNS.NAV'],
scripts      = ['CGNS/NAV/CGNS.NAV'],
ext_modules  = modextlist,
cmdclass     = cmdclassdict
)
 
# --- last line
