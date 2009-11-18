#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS.MAP - Python package for CFD General Notation System - MAPper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $File$
#  $Node$
#  $Last$
#  ------------------------------------------------------------------------- 
import CGNS.MAP
import time
import os

import T0
tree=T0.tree
links=T0.links

try:
  os.unlink("T0.py.cgns")
except os.error: pass
print '# CGNS.MAP.save'
flags=CGNS.MAP.S2P_TRACE
start=time.clock()
CGNS.MAP.save("T0.py.cgns",tree,links,flags)
end=time.clock()
print '# time =',end-start

