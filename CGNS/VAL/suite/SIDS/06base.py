#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy as NPY

TESTS = []

#  -------------------------------------------------------------------------
tag = "multi bases"
diag = True
T = CGL.newCGNSTree()
for i in range(10):
    CGL.newBase(T, "{Base#%.2d}" % i, 3, 3)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad base dims #1"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
b[1] = None
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad base dims #2"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base#1}", 3, 3)
b[1] = NPY.array([0, 0], dtype="int32")
b = CGL.newBase(T, "{Base#2}", 3, 3)
b[1] = NPY.array([-2, 0], dtype="int32")
b = CGL.newBase(T, "{Base#3}", 3, 3)
b[1] = NPY.array([2, 2], dtype="float32")
b = CGL.newBase(T, "{Base#4}", 3, 3)
b[1] = NPY.array([1, 2], dtype="int32")
b = CGL.newBase(T, "{Base#5}", 3, 3)
b[1] = NPY.array([[1, 2]], dtype="int32")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
