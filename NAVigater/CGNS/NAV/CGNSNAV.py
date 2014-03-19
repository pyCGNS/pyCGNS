#!c:\Python27\python.exe
#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import getopt
import sys
import os

try:
  import numpy
except:
  print """CGNS.NAV: FATAL error, cannot import numpy..."""
  sys.exit(-1)
  
prog=sys.argv[0]
dpref='./'
if (prog[0]=='/'):
  dpref=os.path.dirname(prog)
else:
  for pp in os.environ['PATH'].split(':'):
    if (os.path.exists("%s/%s"%(pp,prog))):
      dpref=os.path.dirname("%s/%s"%(pp,prog))
      break
ppref=dpref+'/../lib/python%s/site-packages'%sys.version[:3]
ppref=os.path.normpath(os.path.abspath(ppref))
if (ppref not in sys.path):
  sys.path.append(ppref)

ppref=dpref+'/../Lib/site-packages'
ppref=os.path.normpath(os.path.abspath(ppref))
if (ppref not in sys.path):
  sys.path.append(ppref)

ppref=dpref+'/build/exe/win32-2.7'
ppref=os.path.normpath(os.path.abspath(ppref))
if (ppref not in sys.path):
  sys.path.append(ppref)

try:
  import CGNS.MAP
except:
  print """CGNS.NAV: FATAL error, cannot import CGNS.MAP..."""
  sys.exit(-1)
  
try:
  import PySide.QtCore
except:
  print """CGNS.NAV: FATAL error, cannot import PySide.QtCore..."""
  sys.exit(-1)
  
try:
  import vtk
except:
  print """CGNS.NAV: FATAL error, cannot import vtk..."""
  sys.exit(-1)
  
try:
  import PySide 
  import PySide.QtCore 
  import PySide.QtGui
except:
  print """CGNS.NAV: FATAL error, cannot import PySide..."""
  sys.exit(-1)

import CGNS.NAV.Res_rc
import CGNS.NAV.temputils
from CGNS.NAV.moption import Q7OptionContext as OCTXT

def usage():
  print """%s v%s
  -R : recurse on all tree
  -l : open last file used
  -g : open VTK view

  -P <profiles-paths> : override CGNSPROFILEPATH variable for profile search
                        path variable. It is a ':' separated list of paths.
  -h : help
  -v : verbose (trace)
  """%(OCTXT._ToolName,OCTXT._ToolVersion)
  sys.exit(-1)
 
try:
  os.environ['DISPLAY']
except KeyError:
  pass
  #print """%s: you should define a DISPLAY variable..."""%(OCTXT._ToolName)
  #sys.exit(-1)
  
try:
  ppath=os.environ['CGNSPROFILEPATH']
except KeyError:
  ppath=""
  
try:
 opts, args = getopt.getopt(sys.argv[1:], "lgRhvP:")
except getopt.GetoptError:
 usage()

output=None
verbose=False
last=False
recurse=False
wvtk=False
olist=[]

for o, a in opts:
  if (o == "-R"): recurse=True
  if (o == "-l"): last=True
  if (o == "-P"): ppath=a
  if (o == "-v"): verbose=True
  if (o == "-g"): wvtk=True
  if (o in ("-h", "--help")): usage()

flags=(recurse,last,verbose,wvtk)

import CGNS.NAV.script
CGNS.NAV.script.run(sys.argv,args,flags,ppath)

# --------------------------------------------------------------------
