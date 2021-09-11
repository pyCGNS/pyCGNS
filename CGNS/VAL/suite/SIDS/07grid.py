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
tag = "grid 1D"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 1, 1)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
d = CGL.newDataArray(g, CGK.CoordinateX_s, NPY.ones((5,), dtype="float64", order="F"))
g = CGL.newGridCoordinates(z, "{Grid#002}")
d = CGL.newDataArray(g, CGK.CoordinateR_s, NPY.ones((5,), dtype="float64", order="F"))
g = CGL.newGridCoordinates(z, "{Grid#003}")
d = CGL.newDataArray(g, CGK.CoordinateXi_s, NPY.ones((5,), dtype="float64", order="F"))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "grid 2D"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 2, 2)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
d = CGL.newDataArray(g, CGK.CoordinateX_s, NPY.ones((5, 7), dtype="float64", order="F"))
d = CGL.newDataArray(g, CGK.CoordinateY_s, NPY.ones((5, 7), dtype="float64", order="F"))
g = CGL.newGridCoordinates(z, "{Grid#002}")
d = CGL.newDataArray(
    g, CGK.CoordinateXi_s, NPY.ones((5, 7), dtype="float64", order="F")
)
d = CGL.newDataArray(
    g, CGK.CoordinateEta_s, NPY.ones((5, 7), dtype="float64", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "grid 3D #1"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
d = CGL.newDataArray(
    g, CGK.CoordinateX_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
d = CGL.newDataArray(
    g, CGK.CoordinateY_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
d = CGL.newDataArray(
    g, CGK.CoordinateZ_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "grid 3D #2"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 2, 3)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
d = CGL.newDataArray(g, CGK.CoordinateX_s, NPY.ones((5, 7), dtype="float64", order="F"))
d = CGL.newDataArray(g, CGK.CoordinateY_s, NPY.ones((5, 7), dtype="float64", order="F"))
d = CGL.newDataArray(g, CGK.CoordinateZ_s, NPY.ones((5, 7), dtype="float64", order="F"))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "grid empty #1"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
g = CGL.newGridCoordinates(z, "{Grid#002}")
g = CGL.newGridCoordinates(z, "{Grid#003}")
g = CGL.newGridCoordinates(z, "{Grid#004}")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "grid empty #2"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
r = CGL.newUserDefinedData(g, "{UserDefinedData}")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "grid bad dims #1"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
d = CGL.newDataArray(
    g, CGK.CoordinateX_s, NPY.ones((4, 7, 9), dtype="float64", order="F")
)
d = CGL.newDataArray(
    g, CGK.CoordinateY_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
d = CGL.newDataArray(
    g, CGK.CoordinateZ_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
