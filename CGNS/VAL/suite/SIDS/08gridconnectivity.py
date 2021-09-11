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
tag = "gridconnectivity"
diag = True
vertexsize = 20
cellsize = 7
ntris = 12


def makeCorrectTree():
    T = CGL.newCGNSTree()
    b = CGL.newBase(T, "Base", 3, 3)
    z1 = CGL.newZone(
        b, "Zone1", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F")
    )
    g = CGL.newGridCoordinates(z1, "GridCoordinates")
    d = CGL.newDataArray(
        g, CGK.CoordinateX_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
    )
    d = CGL.newDataArray(
        g, CGK.CoordinateY_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
    )
    d = CGL.newDataArray(
        g, CGK.CoordinateZ_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
    )
    s = NPY.array([[vertexsize, cellsize, 0]], dtype="int32", order="F")
    z2 = CGL.newZone(b, "Zone2", s, CGK.Unstructured_s)
    g = CGL.newGridCoordinates(z2, "GridCoordinates")
    d = CGL.newDataArray(
        g, CGK.CoordinateX_s, NPY.ones((vertexsize), dtype="float64", order="F")
    )
    d = CGL.newDataArray(
        g, CGK.CoordinateY_s, NPY.ones((vertexsize), dtype="float64", order="F")
    )
    d = CGL.newDataArray(
        g, CGK.CoordinateZ_s, NPY.ones((vertexsize), dtype="float64", order="F")
    )
    tetras = CGL.newElements(
        z2,
        "TETRAS",
        CGK.TETRA_4_s,
        NPY.array([[1, cellsize]], "i", order="F"),
        NPY.ones((cellsize * 4), dtype="int32"),
    )
    tris = CGL.newElements(
        z2,
        "TRIS",
        CGK.TRI_3_s,
        NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
        NPY.ones((ntris * 3), dtype="int32"),
    )
    z3 = CGU.copyNode(z1, "Zone3")
    b[2].append(z3)
    z4 = CGU.copyNode(z2, "Zone4")
    b[2].append(z4)
    z = [z1, z2, z3, z4]
    return (T, b, z)


(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity(zgc, "join1_3", "Zone3", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(gc, value=NPY.array([[1, 1], [1, 7], [1, 9]], order="F"))
CGL.newPointRange(
    gc, name=CGK.PointRangeDonor_s, value=NPY.array([[1, 1], [1, 7], [1, 9]], order="F")
)
gc = CGL.newGridConnectivity(zgc, "join1_2", "Zone2", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(gc, value=NPY.array([[1, 5], [1, 1], [1, 9]], order="F"))
CGL.newIndexArray(
    gc,
    CGK.PointListDonor_s,
    value=NPY.array([range(cellsize + 1, cellsize + ntris + 1)], order="F"),
)
CGL.newGridLocation(gc, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity bad datatype"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity(zgc, "join1_2", "Zone2", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(gc, value=NPY.array([[1, 5], [1, 1], [1, 9]], order="F"))
CGL.newIndexArray(
    gc,
    CGK.PointListDonor_s,
    value=NPY.array([range(cellsize + 1, cellsize + ntris + 1)], order="F"),
)
CGL.newGridLocation(gc, value=CGK.FaceCenter_s)
gc[1] = NPY.array([1], order="F")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity unfound zonedonor"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity(zgc, "join1_2", "Zone6", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(gc, value=NPY.array([[1, 5], [1, 1], [1, 9]], order="F"))
CGL.newIndexArray(
    gc,
    CGK.PointListDonor_s,
    value=NPY.array([range(cellsize + 1, cellsize + ntris + 1)], order="F"),
)
CGL.newGridLocation(gc, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity bad pointrange/list"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity(zgc, "join1_2", "Zone2", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(gc, value=NPY.array([[1, 5], [1, 1], [1, 9]], order="F"))
CGL.newIndexArray(
    gc, CGK.PointList_s, value=NPY.array([[1, 5], [1, 1], [1, 9]], order="F")
)
CGL.newIndexArray(
    gc,
    CGK.PointListDonor_s,
    value=NPY.array([range(cellsize + 1, cellsize + ntris + 1)], order="F"),
)
CGL.newGridLocation(gc, value=CGK.FaceCenter_s)
gc = CGL.newGridConnectivity(zgc, "join1_3", "Zone3", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(
    gc, name=CGK.PointRangeDonor_s, value=NPY.array([[1, 1], [1, 7], [1, 9]], order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity bad gridlocation"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity(zgc, "join1_2a", "Zone2", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(gc, value=NPY.array([[1, 4], [1, 1], [1, 8]], order="F"))
CGL.newIndexArray(
    gc, CGK.PointListDonor_s, value=NPY.array([range(1, cellsize)], order="F")
)
CGL.newGridLocation(gc, value=CGK.CellCenter_s)
gc = CGL.newGridConnectivity(zgc, "join1_2b", "Zone2", ctype=CGK.Overset_s)
CGL.newPointRange(gc, value=NPY.array([[1, 5], [1, 7], [1, 1]], order="F"))
CGL.newIndexArray(
    gc,
    CGK.PointListDonor_s,
    value=NPY.array([range(cellsize + 1, cellsize + ntris + 1)], order="F"),
)
CGL.newGridLocation(gc, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity bad range on zonedonor #1"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity(zgc, "join1_2", "Zone2", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(gc, value=NPY.array([[1, 5], [1, 1], [1, 9]], order="F"))
CGL.newIndexArray(
    gc,
    CGK.PointListDonor_s,
    value=NPY.array([range(cellsize - 1, cellsize + ntris + 1)], order="F"),
)
CGL.newGridLocation(gc, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity bad range on zonedonor #2"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity(zgc, "join1_2", "Zone2", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(gc, value=NPY.array([[1, 5], [1, 1], [1, 9]], order="F"))
CGL.newIndexArray(
    gc,
    CGK.PointListDonor_s,
    value=NPY.array([range(cellsize + 2, cellsize + ntris + 2)], order="F"),
)
CGL.newGridLocation(gc, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity bad range on zonedonor #3"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity(zgc, "join1_3", "Zone3", ctype=CGK.Overset_s)
CGL.newPointRange(gc, value=NPY.array([[1, 1], [1, 7], [1, 9]], order="F"))
CGL.newPointRange(
    gc, name=CGK.PointRangeDonor_s, value=NPY.array([[1, 1], [1, 7], [1, 9]], order="F")
)
CGL.newGridLocation(gc, value=CGK.CellCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "gridconnectivity abutting not a face"
diag = False
(T, b, z) = makeCorrectTree()
zgc = CGL.newZoneGridConnectivity(z[0])
gc = CGL.newGridConnectivity(zgc, "join1_3", "Zone3", ctype=CGK.Abutting1to1_s)
CGL.newPointRange(gc, value=NPY.array([[1, 5], [1, 7], [1, 9]], order="F"))
CGL.newPointRange(
    gc, name=CGK.PointRangeDonor_s, value=NPY.array([[1, 5], [1, 7], [1, 9]], order="F")
)
TESTS.append((tag, T, diag))
