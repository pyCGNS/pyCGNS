#!/usr/bin/env python
# -------------------------------------------------------------------------
# pyCGNS.PAT - CFD General Notation System -
# See license.txt file in the root directory of this Python module source
# -------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU

T = CGL.newCGNSTree()
B = CGL.newBase(T, "Base", 3, 3)
Z = CGL.newZone(B, "Zone1")
Z = CGL.newZone(B, "Zone2")
Z = CGL.newZone(B, "Zone3")
Z = CGL.newZone(B, "Zone4")

l1 = CGU.getAllNodesByTypeList(T, ["CGNSTree_t", "CGNSBase_t", "Zone_t"])
l2 = CGU.getAllNodesByTypeSet(T, ["Zone_t", "IndexArray_t"])

print(l1)
print(l2)

print(CGU.getNodeByPath(l1[0], T))

l3 = CGU.getAllNodesByTypeList(B, ["CGNSBase_t", "Zone_t"])
l4 = CGU.getAllNodesByTypeSet(B, ["Zone_t"])

print(l3)
print(l4)

print(CGU.getNodeByPath(B, l3[0]))
