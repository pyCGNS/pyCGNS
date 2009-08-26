#!/usr/bin/env python
# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS-to-Python MAPping            
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
#
import CGNS.MAP
import time
import os

import Z
tree=Z.data

try:
  os.unlink("Z.cgns")
except os.error: pass
print '# CGNS.MAP.save'
flags=CGNS.MAP.S2P_TRACE
start=time.clock()
CGNS.MAP.save("Z.cgns",tree,[],flags)
end=time.clock()
print '# time =',end-start

