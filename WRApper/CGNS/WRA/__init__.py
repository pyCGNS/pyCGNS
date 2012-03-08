#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.pyCGNSconfig

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

