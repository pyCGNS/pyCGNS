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
tag = "elements"
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
tag = "elements bad elementsizeboundary"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
tetras[1][1] = cellsize
zbc = CGL.newZoneBC(z)
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
tag = "elements elementsizeboundary BC correctly defined"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
tetras[1][1] = cellsize - 1
zbc = CGL.newZoneBC(z)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(1, cellsize)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements elementsizeboundary faces not on BC/GC (Warning)"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
tetras[1][1] = cellsize - 1
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements elementsizeboundary bndfaces multiply defined on BC (Warning)"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
tetras[1][1] = cellsize - 1
zbc = CGL.newZoneBC(z)
n = CGL.newBoundary(
    zbc,
    "BC1",
    [range(1, cellsize)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
n = CGL.newBoundary(
    zbc,
    "BC2",
    [range(2, cellsize - 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.CellCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements elementsizeboundary bndfaces multiply defined on BC/GC (Warning)"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
ntris = 12
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
tris = CGL.newElements(
    z,
    "TRIS",
    CGK.TRI_3_s,
    NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
    NPY.ones((ntris * 3), dtype="int32"),
)
tris[1][1] = ntris
zbc = CGL.newZoneBC(z)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(cellsize + 1, cellsize + ntris + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
z2 = CGU.copyNode(z, "Zone2")
b[2].append(z2)
zgc = CGL.newZoneGridConnectivity(z)
gc = CGL.newGridConnectivity(zgc, "join1_2", "Zone2", ctype=CGK.Abutting1to1_s)
CGL.newIndexArray(
    gc,
    CGK.PointList_s,
    value=NPY.array([range(cellsize + 2, cellsize + ntris)], order="F"),
)
CGL.newIndexArray(
    gc,
    CGK.PointListDonor_s,
    value=NPY.array([range(cellsize + 2, cellsize + ntris)], order="F"),
)
CGL.newGridLocation(gc, value=CGK.FaceCenter_s)
zgc = CGL.newZoneGridConnectivity(z2)
gc = CGL.newGridConnectivity(zgc, "join2_1", "Zone", ctype=CGK.Abutting1to1_s)
CGL.newIndexArray(
    gc,
    CGK.PointList_s,
    value=NPY.array([range(cellsize + 2, cellsize + ntris)], order="F"),
)
CGL.newIndexArray(
    gc,
    CGK.PointListDonor_s,
    value=NPY.array([range(cellsize + 2, cellsize + ntris)], order="F"),
)
CGL.newGridLocation(gc, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements absent children"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
CGU.removeChildByName(tetras, CGK.ElementRange_s)  # ElementRange child absent
CGU.removeChildByName(
    tetras, CGK.ElementConnectivity_s
)  # ElementConnectivity child absent
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements out of range"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
tetras[2][0][1][0] = vertexsize + 1  # ElementConnectity element out of range
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements bad child shape"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones(((cellsize - 1) * 4), dtype="int32"),
)  # bad ElementConnectivity node shape
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements correct NGON shape"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
element = CGL.newElements(
    z,
    "NGON",
    CGK.NGON_n_s,
    NPY.array([[cellsize + 1, cellsize + cellsize]], "i", order="F"),
    NPY.array(
        [
            4,
            9,
            9,
            9,
            9,
            5,
            9,
            9,
            9,
            9,
            9,
            3,
            9,
            9,
            9,
            5,
            9,
            9,
            9,
            9,
            9,
            3,
            9,
            9,
            9,
            5,
            9,
            9,
            9,
            9,
            9,
            4,
            9,
            9,
            9,
            9,
        ],
        dtype="int32",
        order="F",
    ),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements uncorrect NGON shape"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
element = CGL.newElements(
    z,
    "NGON",
    CGK.NGON_n_s,
    NPY.array([[cellsize + 1, cellsize + cellsize]], "i", order="F"),
    NPY.array(
        [4, 9, 9, 9, 9, 5, 9, 9, 9, 9, 9, 3, 9, 9, 9, 5, 9, 9, 9, 9, 9, 4, 9, 9, 9, 9],
        dtype="int32",
        order="F",
    ),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements correct MIXED shape"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
element = CGL.newElements(
    z,
    "MIXED",
    CGK.MIXED_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.array(
        [
            CGK.ElementType[CGK.QUAD_4_s],
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.PYRA_5_s],
            9,
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.TRI_3_s],
            9,
            9,
            9,
            CGK.ElementType[CGK.PYRA_5_s],
            9,
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.TRI_3_s],
            9,
            9,
            9,
            CGK.ElementType[CGK.PYRA_5_s],
            9,
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.QUAD_4_s],
            9,
            9,
            9,
            9,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 5, 11, 15, 21, 25, 31, 36], dtype="int32", order="F"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements uncorrect MIXED shape #1"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
element = CGL.newElements(
    z,
    "MIXED",
    CGK.MIXED_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.array(
        [
            CGK.ElementType[CGK.QUAD_4_s],
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.PYRA_5_s],
            9,
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.TRI_3_s],
            9,
            9,
            9,
            CGK.ElementType[CGK.PYRA_5_s],
            9,
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.QUAD_4_s],
            9,
            9,
            9,
            9,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 5, 11, 15, 21, 26], dtype="int32", order="F"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements uncorrect MIXED shape #2"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
element = CGL.newElements(
    z,
    "MIXED",
    CGK.MIXED_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.array(
        [
            CGK.ElementType[CGK.QUAD_4_s],
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.PYRA_5_s],
            9,
            9,
            9,
            9,
            9,
            999,
            CGK.ElementType[CGK.TRI_3_s],
            9,
            9,
            9,
            CGK.ElementType[CGK.PYRA_5_s],
            9,
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.QUAD_4_s],
            9,
            9,
            9,
            9,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 5, 12, 16, 22, 27], dtype="int32", order="F"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements uncorrect element within MIXED"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
element = CGL.newElements(
    z,
    "MIXED",
    CGK.MIXED_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.array(
        [
            CGK.ElementType[CGK.QUAD_4_s],
            9,
            9,
            9,
            9,
            999,
            9,
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.TRI_3_s],
            9,
            9,
            9,
            CGK.ElementType[CGK.PYRA_5_s],
            9,
            9,
            9,
            9,
            9,
            CGK.ElementType[CGK.QUAD_4_s],
            9,
            9,
            9,
            9,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 11, 15, 21, 26], dtype="int32", order="F"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements FACE"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nface = CGL.newElements(
    z,
    "NFACE",
    CGK.NFACE_n_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.array(
        [
            cellsize + 1,
            cellsize + 2,
            -cellsize - 3,
            cellsize + 4,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            cellsize + 5,
            cellsize + 1,
            -cellsize - 2,
            cellsize + 3,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            -cellsize - 5,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            cellsize + 5,
            cellsize + 1,
            cellsize + 2,
            -cellsize - 3,
            cellsize + 4,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 20, 25, 29], dtype="int32", order="F"),
)
ngon = CGL.newElements(
    z,
    "NGON",
    CGK.NGON_n_s,
    NPY.array([[cellsize + 1, cellsize + 5]], "i", order="F"),
    NPY.array(
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 21], dtype="int32", order="F"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements FACE inconsistent dataarray"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nface = CGL.newElements(
    z,
    "NFACE",
    CGK.NFACE_n_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.array(
        [
            cellsize + 1,
            cellsize + 2,
            -cellsize - 3,
            cellsize + 4,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            cellsize + 5,
            cellsize + 1,
            -cellsize - 2,
            cellsize + 3,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            -cellsize - 5,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            cellsize + 5,
            cellsize + 1,
            cellsize + 2,
            -cellsize - 3,
            cellsize + 4,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array(
        [0, 4, 9999997, 10000000, 10000005, 10000008, 10000013, 10000017],
        dtype="int32",
        order="F",
    ),
)
ngon = CGL.newElements(
    z,
    "NGON",
    CGK.NGON_n_s,
    NPY.array([[cellsize + 1, cellsize + 5]], "i", order="F"),
    NPY.array(
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 21], dtype="int32", order="F"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements FACE bad node shape"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nface = CGL.newElements(
    z,
    "NFACE",
    CGK.NFACE_n_s,
    NPY.array([[1, cellsize - 1]], "i", order="F"),
    NPY.array(
        [
            cellsize + 1,
            cellsize + 2,
            -cellsize - 3,
            cellsize + 4,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            cellsize + 5,
            cellsize + 1,
            -cellsize - 2,
            cellsize + 3,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            -cellsize - 5,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            cellsize + 5,
            cellsize + 1,
            cellsize + 2,
            -cellsize - 3,
            cellsize + 4,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 20, 25, 29], dtype="int32", order="F"),
)
ngon = CGL.newElements(
    z,
    "NGON",
    CGK.NGON_n_s,
    NPY.array([[cellsize + 1, cellsize + 5]], "i", order="F"),
    NPY.array(
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 21], dtype="int32", order="F"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements FACE face index out of range"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nface = CGL.newElements(
    z,
    "NFACE",
    CGK.NFACE_n_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.array(
        [
            cellsize + 1,
            cellsize + 2,
            -cellsize - 3,
            cellsize + 4,
            cellsize + 1,
            cellsize + 2,
            999999,
            cellsize + 4,
            cellsize + 5,
            cellsize + 1,
            -cellsize - 2,
            cellsize + 3,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            -cellsize - 5,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 1,
            cellsize + 2,
            cellsize + 3,
            cellsize + 4,
            cellsize + 5,
            cellsize + 1,
            cellsize + 2,
            -cellsize - 3,
            cellsize + 4,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 20, 25, 29], dtype="int32", order="F"),
)
ngon = CGL.newElements(
    z,
    "NGON",
    CGK.NGON_n_s,
    NPY.array([[cellsize + 1, cellsize + 5]], "i", order="F"),
    NPY.array(
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 21], dtype="int32", order="F"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelements and parentelementsposition"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 8), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
pe = CGL.newParentElements(quads, NPY.ones((nquads, 2), dtype="int32", order="F"))
pp = CGL.newParentElementsPosition(
    quads, NPY.ones((nquads, 2), dtype="int32", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelementsposition without parentelements"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 8), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
pp = CGL.newParentElementsPosition(
    quads, NPY.ones((nquads, 2), dtype="int32", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements bad parentelements shape and parentelementsposition datatype"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 8), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
pe = CGL.newParentElements(quads, NPY.ones((nquads - 1, 2), dtype="int32", order="F"))
pp = CGL.newParentElementsPosition(
    quads, NPY.ones((nquads, 2), dtype="float64", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelements and parentelementsposition on bad element type (Warning)"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 8), dtype="int32"),
)
pe = CGL.newParentElements(tetras, NPY.ones((cellsize, 2), dtype="int32", order="F"))
pp = CGL.newParentElementsPosition(
    tetras, NPY.ones((cellsize, 2), dtype="int32", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelements bad values #1"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 8), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
pe = CGL.newParentElements(
    quads, NPY.array([[1, 2], [1, cellsize + 1], [1, 2]], dtype="int32", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelements bad values #2"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nhexa = 2
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, nhexa]], "i", order="F"),
    NPY.ones((nhexa * 8), dtype="int32"),
)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[nhexa + 1, cellsize]], "i", order="F"),
    NPY.ones(((cellsize - nhexa) * 4), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
pe = CGL.newParentElements(quads, NPY.ones((nquads, 2), dtype="int32", order="F"))
pe[1][0][1] = nhexa + 1
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelements with boundary faces"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nhexa = 2
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, nhexa]], "i", order="F"),
    NPY.ones((nhexa * 8), dtype="int32"),
)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[nhexa + 1, cellsize]], "i", order="F"),
    NPY.ones(((cellsize - nhexa) * 4), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
pe = CGL.newParentElements(
    quads, NPY.array([[1, 0], [2, 0], [2, 0]], dtype="int32", order="F")
)
zbc = CGL.newZoneBC(z)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(cellsize + 1, cellsize + nquads + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelements with boundary faces but no BC or GC (warning)"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nhexa = 2
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, nhexa]], "i", order="F"),
    NPY.ones((nhexa * 8), dtype="int32"),
)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[nhexa + 1, cellsize]], "i", order="F"),
    NPY.ones(((cellsize - nhexa) * 4), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
pe = CGL.newParentElements(
    quads, NPY.array([[1, 0], [2, 0], [2, 0]], dtype="int32", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelements with boundary faces but badly defined"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nhexa = 2
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, nhexa]], "i", order="F"),
    NPY.ones((nhexa * 8), dtype="int32"),
)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[nhexa + 1, cellsize]], "i", order="F"),
    NPY.ones(((cellsize - nhexa) * 4), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
pe = CGL.newParentElements(quads, NPY.zeros((nquads, 2), dtype="int32", order="F"))
zbc = CGL.newZoneBC(z)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(cellsize + 1, cellsize + nquads + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelements and elementsizeboundary compatible"
diag = True
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nhexa = 2
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, nhexa]], "i", order="F"),
    NPY.ones((nhexa * 8), dtype="int32"),
)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[nhexa + 1, cellsize]], "i", order="F"),
    NPY.ones(((cellsize - nhexa) * 4), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
quads[1][1] = nquads
pe = CGL.newParentElements(
    quads, NPY.array([[1, 0], [2, 0], [2, 0]], dtype="int32", order="F")
)
zbc = CGL.newZoneBC(z)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(cellsize + 1, cellsize + nquads + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelements and elementsizeboundary not compatible"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
nhexa = 2
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, nhexa]], "i", order="F"),
    NPY.ones((nhexa * 8), dtype="int32"),
)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[nhexa + 1, cellsize]], "i", order="F"),
    NPY.ones(((cellsize - nhexa) * 4), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
quads[1][1] = nquads
pe = CGL.newParentElements(quads, NPY.ones((nquads, 2), dtype="int32", order="F"))
zbc = CGL.newZoneBC(z)
n = CGL.newBoundary(
    zbc,
    "BC",
    [range(cellsize + 1, cellsize + nquads + 1)],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointList_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "elements parentelementsposition bad face position"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize)
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 8), dtype="int32"),
)
nquads = 3
quads = CGL.newElements(
    z,
    "QUADS",
    CGK.QUAD_4_s,
    NPY.array([[cellsize + 1, cellsize + nquads]], "i", order="F"),
    NPY.ones((nquads * 4), dtype="int32"),
)
pe = CGL.newParentElements(quads, NPY.ones((nquads, 2), dtype="int32", order="F"))
pp = CGL.newParentElementsPosition(
    quads, NPY.ones((nquads, 2), dtype="int32", order="F") * 7
)  # 7 > 6 faces for hexas
TESTS.append((tag, T, diag))
