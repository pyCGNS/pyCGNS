#!/usr/bin/env python
# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS-to-Python MAPping            
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
import CGNS.MAP

print CGNS.MAP.flags

for k in CGNS.MAP.flags:
  print k, CGNS.MAP.__dict__[k]
  
