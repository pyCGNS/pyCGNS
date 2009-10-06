#!/usr/bin/env python
# -------------------------------------------------------------------------
# pyCGNS.MAP - CFD General Notation System - SIDS-to-Python MAPping            
# See license.txt file in the root directory of this Python module source  
# -------------------------------------------------------------------------
#
import CGNS.MAP
import time

print 'CGNS.MAP.load(links)'
flags=CGNS.MAP.S2P_FOLLOWLINKS|CGNS.MAP.S2P_TRACE
start=time.clock()
(tree,links)=CGNS.MAP.load("T1.cgns",flags)
print links
end=time.clock()
print 'time =',end-start

