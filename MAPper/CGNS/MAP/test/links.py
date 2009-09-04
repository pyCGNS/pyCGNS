#!/usr/bin/env python
# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS-to-Python MAPping            
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
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

