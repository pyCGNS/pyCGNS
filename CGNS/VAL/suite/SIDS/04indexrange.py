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
tag = "indexrange"
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
    return (T, b, z)


vertexsize = 20
cellsize = 7
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexrange index bad ordered"  # this raises a warning, not an error
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[cellsize, 1]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
# element range not ordered
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexrange bad node shape"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([1, cellsize], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
# ElementRange bad node shape
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexrange on BC_t"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
ntris = 11
tris = CGL.newElements(
    z,
    "TRIS",
    CGK.TRI_3_s,
    NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
    NPY.ones((ntris * 3), dtype="int32"),
)
n = CGL.newBoundary(
    z,
    "BC",
    [range(cellsize + 1, cellsize + ntris + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
CGU.removeChildByName(n, CGK.PointList_s)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[cellsize + 1, cellsize + ntris]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=n,
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexrange on BC_t PointRange index out of range #1"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
ntris = 11
tris = CGL.newElements(
    z,
    "TRIS",
    CGK.TRI_3_s,
    NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
    NPY.ones((ntris * 3), dtype="int32"),
)
n = CGL.newBoundary(
    z,
    "BC",
    [range(cellsize + 1, cellsize + ntris + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
CGU.removeChildByName(n, CGK.PointList_s)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[cellsize + 1, cellsize + ntris + 1]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=n,
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexrange on BC_t PointRange index out of range #2"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
ntris = 11
tris = CGL.newElements(
    z,
    "TRIS",
    CGK.TRI_3_s,
    NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
    NPY.ones((ntris * 3), dtype="int32"),
)
n = CGL.newBoundary(
    z,
    "BC",
    [range(cellsize + 1, cellsize + ntris + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
CGU.removeChildByName(n, CGK.PointList_s)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[cellsize, cellsize + ntris]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=n,
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "indexrange on BC_t ElementRange index out of range"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
ntris = 11
tris = CGL.newElements(
    z,
    "TRIS",
    CGK.TRI_3_s,
    NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
    NPY.ones((ntris * 3), dtype="int32"),
)
n = CGL.newBoundary(
    z,
    "BC",
    [range(cellsize, cellsize + ntris + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
CGU.removeChildByName(n, CGK.PointList_s)
CGU.newNode(
    CGK.ElementRange_s,
    NPY.array([[cellsize + 1, cellsize + ntris + 1]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=n,
)
TESTS.append((tag, T, diag))
