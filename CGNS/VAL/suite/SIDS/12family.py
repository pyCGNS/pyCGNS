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
tag = "base level family #1"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base#001}", 3, 3)
CGL.newFamily(b, "{Family#001}")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "family elsewhere than base"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base#001}", 3, 3)
u = CGL.newUserDefinedData(b, "{UD#001}")
CGL.newFamily(u, "{Family#001}")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
