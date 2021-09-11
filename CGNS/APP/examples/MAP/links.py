#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS.APP - Python package for CFD General Notation System - APPlicater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.MAP
import time

print("CGNS.MAP.load(links)")
flags = CGNS.MAP.S2P_DEFAULTS | CGNS.MAP.S2P_FOLLOWLINKS | CGNS.MAP.S2P_TRACE
start = time.clock()
(tree, links, paths) = CGNS.MAP.load("T1.cgns", flags=flags)
print(links)
end = time.clock()
print("time =", end - start)
