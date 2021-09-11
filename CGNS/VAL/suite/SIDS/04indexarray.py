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
tag = "indexarray"
diag = True


def makeCorrectTree(vertexsize, cellsize):
    T = CGL.newCGNSTree()
    b = CGL.newBase(T, "Base", 3, 3)
    s = NPY.array([[vertexsize, cellsize, 0]], dtype="int32", order="F")
    z = CGL.newZone(b, "Zone", s, CGK.Unstructured_s)
    g = CGL.newGridCoordinates(z, "GridCoordinates")
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
        z,
        "TETRAS",
        CGK.TETRA_4_s,
        NPY.array([[1, cellsize]], "i", order="F"),
        NPY.ones((cellsize * 4), dtype="int32"),
    )
    zbc = CGL.newZoneBC(z)
    return (T, b, z, zbc)


vertexsize = 20
cellsize = 7
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray bad parent"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
i = n[2][0]
z[2].append(CGU.copyNode(i, "PointList"))  # unauthorized parent node
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray bad name"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
i = n[2][0]
i[0] = "ElementList"  # unauthorized name
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray bad data type"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
i = n[2][0]
i[1] = NPY.ones((1, cellsize), dtype="float64", order="F")  # unauthorized data type
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray bad child"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
i = n[2][0]
CGL.newDataArray(i, "{DataArray}")  # unauthorized child
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray element index out of range"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(2, cellsize + 2)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)  # element index out of range
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray element appear several times (Warning)"
diag = True
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [[1, 1, 1, 1, 1, 1, 1]],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)  # values appear several times in list
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray gridlocation vertex"
diag = True
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [[1, 2, 3, 4, vertexsize]],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)  # element index out of range
g = CGL.newGridLocation(n, value=CGK.Vertex_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray gridlocation vertex index out of range"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [[1, 2, 0, 4, vertexsize + 1]],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)  # element index out of range
g = CGL.newGridLocation(n, value=CGK.Vertex_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray gridlocation face"
diag = True
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
ntris = 12
tris = CGL.newElements(
    z,
    "TRIS",
    CGK.TRI_3_s,
    NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
    NPY.ones((ntris * 3), dtype="int32"),
)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(cellsize + 1, cellsize + ntris + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray gridlocation face index out of range #1"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
ntris = 12
tris = CGL.newElements(
    z,
    "TRIS",
    CGK.TRI_3_s,
    NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
    NPY.ones((ntris * 3), dtype="int32"),
)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(cellsize + 1, cellsize + ntris + 2)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray gridlocation face index out of range #2"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
ntris = 12
tris = CGL.newElements(
    z,
    "TRIS",
    CGK.TRI_3_s,
    NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
    NPY.ones((ntris * 3), dtype="int32"),
)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(cellsize, cellsize + ntris + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray InwardNormalList"
diag = True
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
inl = CGL.newIndexArray(n, CGK.InwardNormalList_s, value=NPY.ones([3, cellsize]))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray InwardNormalList bad shape #1"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
inl = CGL.newIndexArray(n, CGK.InwardNormalList_s, value=NPY.ones([2, cellsize]))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray InwardNormalList bad shape #2"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
inl = CGL.newIndexArray(n, CGK.InwardNormalList_s, value=NPY.ones([3, cellsize + 1]))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray InwardNormalList bad shape #3"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
inl = CGL.newIndexArray(n, CGK.InwardNormalList_s, value=NPY.ones([cellsize]))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexarray InwardNormalList bad shape #4"
diag = False
(T, b, z, zbc) = makeCorrectTree(vertexsize, cellsize)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
CGU.nodeDelete(T, CGU.getNodeByPath(n, "BC/" + CGK.PointList_s))
inl = CGL.newIndexArray(n, CGK.InwardNormalList_s, value=NPY.ones([2, cellsize]))
TESTS.append((tag, T, diag))
