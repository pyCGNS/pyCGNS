#!/usr/bin/env python
# -------------------------------------------------------------------------
# pyCGNS.MAP - CFD General Notation System - SIDS-to-Python MAPping            
# See license.txt file in the root directory of this Python module source  
# -------------------------------------------------------------------------
#
import CGNS.MAP
import time
import os

import T0
tree=T0.data

try:
  os.unlink("T0.py.cgns")
except os.error: pass
print '# CGNS.MAP.save'
flags=CGNS.MAP.S2P_TRACE
start=time.clock()
CGNS.MAP.save("T0.py.cgns",tree,[],flags)
end=time.clock()
print '# time =',end-start

