#!/usr/bin/env python
# -------------------------------------------------------------------------
# pyCGNS.MAP - CFD General Notation System - SIDS-to-Python MAPping
# See license.txt file in the root directory of this Python module source
# -------------------------------------------------------------------------
#
import CGNS.MAP

print(CGNS.MAP.flags)

for k in CGNS.MAP.flags:
    print(k, CGNS.MAP.__dict__[k])
