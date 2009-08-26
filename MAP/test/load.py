#!/usr/bin/env python
# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS-to-Python MAPping            
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
import CGNS.MAP
import time
import numpy
import sys

numpy.set_printoptions(threshold=sys.maxint)

print '# CGNS.MAP.load '
flags=CGNS.MAP.S2P_FOLLOWLINKS|CGNS.MAP.S2P_TRACE
start=time.clock()
(tree,links)=CGNS.MAP.load("./5blocks.hdf",flags)
print 'from numpy import *'
print 'data=',
print tree
end=time.clock()
print '# time =',end-start


