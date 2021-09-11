#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy as NPY

TESTS = []

#  -------------------------------------------------------------------------
tag = "gridconnectivity1to1"
diag = True
vertexsize = 20
cellsize = 7
ntris = 12


def makeCorrectTree():
    T = CGL.newCGNSTree()
    b = CGL.newBase(T, "Base", 3, 3)
    z1 = CGL.newZone(
        b, "Zone1", NPY.array([[5, 4, 0], [7, 6, 0], [5, 4, 0]], order="F")
    )
    g = CGL.newGridCoordinates(z1, "GridCoordinates")
    d = CGL.newDataArray(
        g, CGK.CoordinateX_s, NPY.ones((5, 7, 5), dtype="float64", order="F")
    )
    d = CGL.newDataArray(
        g, CGK.CoordinateY_s, NPY.ones((5, 7, 5), dtype="float64", order="F")
    )
    d = CGL.newDataArray(
        g, CGK.CoordinateZ_s, NPY.ones((5, 7, 5), dtype="float64", order="F")
    )
    s = NPY.array([[vertexsize, cellsize, 0]], dtype="int32", order="F")
    z2 = CGU.copyNode(z1, "Zone2")
    b[2].append(z2)
    z = [z1, z2]
    return (T, b, z)


(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join1_2",
    "Zone2",
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([+1, +2, +3]),
)
zgc = CGL.newZoneGridConnectivity(z[1])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join2_1",
    "Zone1",
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([+1, +2, +3]),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity1to1 bad datatype"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join1_2",
    "Zone2",
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([[1, 1], [1, 7], [5, 5]]),
    NPY.array([+1, +2, +3]),
)
gc[1] = NPY.array([1], order="F")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity1to1 absent opposite GridConnectivity"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join1_2",
    "Zone2",
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([+1, +2, +3]),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity1to1 unvalid opposite GridConnectivity"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join1_2",
    "Zone2",
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([+1, +2, +3]),
)
zgc = CGL.newZoneGridConnectivity(z[1])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join2_1",
    "Zone1",
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([+1, +2, +3]),
)
CGU.removeChildByName(gc, CGK.PointRangeDonor_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity1to1 bad shape on Transform"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join1_2",
    "Zone2",
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([[+1, +2, +3]]),
)
zgc = CGL.newZoneGridConnectivity(z[1])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join2_1",
    "Zone1",
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([+1, +2, +3]),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity1to1 bad Transform"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join1_2",
    "Zone2",
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([+1, +2, -4]),
)
zgc = CGL.newZoneGridConnectivity(z[1])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join2_1",
    "Zone1",
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([+1, +2, +4]),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity1to1 several 0 in Transform"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join1_2",
    "Zone2",
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([+1, 0, 0]),
)
zgc = CGL.newZoneGridConnectivity(z[1])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join2_1",
    "Zone1",
    NPY.array([[1, 5], [1, 7], [5, 5]]),
    NPY.array([[1, 1], [1, 7], [1, 5]]),
    NPY.array([+1, +2, +3]),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity1to1 not a face"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join1_2",
    "Zone2",
    NPY.array([[1, 5], [1, 7], [1, 5]]),
    NPY.array([[1, 5], [1, 7], [1, 5]]),
    NPY.array([+1, +2, +3]),
)
zgc = CGL.newZoneGridConnectivity(z[1])
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join2_1",
    "Zone1",
    NPY.array([[1, 5], [1, 7], [1, 5]]),
    NPY.array([[1, 5], [1, 7], [1, 5]]),
    NPY.array([+1, +2, +3]),
)
TESTS.append((tag, T, diag))
