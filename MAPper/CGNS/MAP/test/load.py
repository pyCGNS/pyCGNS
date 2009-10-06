#!/usr/bin/env python
# -------------------------------------------------------------------------
# pyCGNS.MAP - CFD General Notation System - SIDS-to-Python MAPping            
# See license.txt file in the root directory of this Python module source  
# -------------------------------------------------------------------------
#
import CGNS.MAP
import time
import numpy
import sys

numpy.set_printoptions(threshold=sys.maxint)

print '# CGNS.MAP.load '
flags=CGNS.MAP.S2P_FOLLOWLINKS|CGNS.MAP.S2P_TRACE
start=time.clock()
(tree,links)=CGNS.MAP.load("./T0.cgns",flags)
f=open('T0.py','w+')
f.write('from numpy import *\n')
f.write('data=')
f.write(str(tree))
f.close()
end=time.clock()
print '# time =',end-start


