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
tag = "empty tree"
diag = True
T = CGL.newCGNSTree()
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad tree #1"
diag = False
T = CGL.newCGNSTree()
T[2] = []
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad version #1"
diag = False
T = CGL.newCGNSTree()
v = CGU.hasChildName(T, CGK.CGNSLibraryVersion_s)
v[1] = NPY.array([6.8], dtype="float32")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad version #2"
diag = False
T = CGL.newCGNSTree()
v = CGU.hasChildName(T, CGK.CGNSLibraryVersion_s)
v[1] = NPY.array([3])
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad struct #1"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base#1}", 3, 3)
b.append(None)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad struct #2"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base#1}", 3, 3)
b[2] = {}
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad name #1"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family#1}")
d[0] = None
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad name #2"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family#1}")
d = CGL.newFamily(b, "{Family#2}")
d[0] = "{Family#1}"
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad name #3"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family#2}")
d[0] = "/A/B/C"
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad name #4"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family#2}")
d[0] = "A" * 33
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad name #5"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family#1}")
d[0] = ""
d = CGL.newFamily(b, "{Family#2}")
d[0] = " "
d = CGL.newFamily(b, "{Family#3}")
d[0] = " " * 10
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad value #1"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
b[1] = "Oups"
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad value #2"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
b[1] = NPY.array([True, False], dtype="b")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad SIDS type"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
b[3] = None
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
