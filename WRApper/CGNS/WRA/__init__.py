#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.pyCGNSconfig

# config="# pyCGNS v%s - The Python CGNS API\n"%CGNS.pyCGNSconfig.version
# config+="# produced %s on %s %s %s"%(CGNS.pyCGNSconfig.DATE,
#                                      CGNS.pyCGNSconfig.PLATFORM[0],
#                                      CGNS.pyCGNSconfig.PLATFORM[2],
#                                      CGNS.pyCGNSconfig.PLATFORM[4])

# if (CGNS.pyCGNSconfig.HAS_MLL == True):
#   config+="""#
# # using CGNS Librarie v%s found in %s
# """%(CGNS.pyCGNSconfig.MLL_VERSION,CGNS.pyCGNSconfig.PATH_MLL)
# else:
#   config+="""# Built without CGNS library"""

# if (CGNS.pyCGNSconfig.HAS_CHLONE == True):
#   config+="""# using %s found in %s\n"""%(CGNS.pyCGNSconfig.CHLONE_VERSION,
#                                           CGNS.pyCGNSconfig.PATH_CHLONE)

# vnum=""
# if (CGNS.pyCGNSconfig.NUM_LIBRARY == "numpy"):
#   import numpy
#   vnum="v"+numpy.version.version

#config+="# using module %s %s\n"%(CGNS.pyCGNSconfig.NUM_LIBRARY,vnum)

def test():
  import sys
  syspathold=sys.path
  sys.path.append("%s/%s"%(sys.prefix,'share/CGNS/WRA/test'))
  import CGNSWRAtest
#  CGNSWRAtest.showConfig()
  CGNSWRAtest.run(sys.path)
  sys.path=syspathold
#

# compat for enums
from CGNS.PAT.cgnskeywords import *

