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
tag = "zone structured"
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
tag = "zone bad zonetype"
diag = False
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
zt = CGU.hasChildName(z, CGK.ZoneType_s)
zt[1] = CGU.setStringAsArray("Untruscutred")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone bad zone size"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
d = CGL.newDataArray(g, CGK.CoordinateX_s, NPY.ones(5, dtype="float64", order="F"))
d = CGL.newDataArray(g, CGK.CoordinateY_s, NPY.ones(5, dtype="float64", order="F"))
d = CGL.newDataArray(g, CGK.CoordinateZ_s, NPY.ones(5, dtype="float64", order="F"))
zt = CGU.hasChildName(z, CGK.ZoneType_s)
zt[1] = CGU.setStringAsArray(CGK.Unstructured_s)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, 4]], "i", order="F"),
    NPY.ones((4 * 4), dtype="int32"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone unstructured bad dims"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
z = CGL.newZone(b, "{Zone}", NPY.array([[4, 5, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
d = CGL.newDataArray(g, CGK.CoordinateX_s, NPY.ones(4, dtype="float64", order="F"))
d = CGL.newDataArray(g, CGK.CoordinateY_s, NPY.ones(4, dtype="float64", order="F"))
d = CGL.newDataArray(g, CGK.CoordinateZ_s, NPY.ones(4, dtype="float64", order="F"))
zt = CGU.hasChildName(z, CGK.ZoneType_s)
zt[1] = CGU.setStringAsArray(CGK.Unstructured_s)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, 5]], "i", order="F"),
    NPY.ones((5 * 4), dtype="int32"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone structured bad dims"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 7, 0], [9, 8, 0]], order="F"))
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
tag = "zone structured bad numpy array order"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 7, 9], [4, 6, 8], [0, 0, 0]], order="F"))
g = CGL.newGridCoordinates(z, "GridCoordinates")
d = CGL.newDataArray(
    g, CGK.CoordinateX_s, NPY.ones((5, 4, 0), dtype="float64", order="F")
)
d = CGL.newDataArray(
    g, CGK.CoordinateY_s, NPY.ones((5, 4, 0), dtype="float64", order="F")
)
d = CGL.newDataArray(
    g, CGK.CoordinateZ_s, NPY.ones((5, 4, 0), dtype="float64", order="F")
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone unstructured no elements"
diag = False


def makeUnstTree(vertexsize, cellsize):
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
(T, b, z) = makeUnstTree(vertexsize, cellsize)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone unstructured correct ElementRange combination"
diag = True
(T, b, z) = makeUnstTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, 3]], "i", order="F"),
    NPY.ones((3 * 4), dtype="int32"),
)
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[4, cellsize]], "i", order="F"),
    NPY.ones((4 * 8), dtype="int32"),
)
element = CGL.newElements(
    z,
    "NGON",
    CGK.NGON_n_s,
    NPY.array([[cellsize + 1, cellsize + 6]], "i", order="F"),
    NPY.array(
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 20, 24], dtype="int32", order="F"),
)
hexas2 = CGL.newElements(
    z,
    "HEXAS2",
    CGK.HEXA_8_s,
    NPY.array([[cellsize + 7, cellsize + 8]], "i", order="F"),
    NPY.ones((2 * 8), dtype="int32"),
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone unstructured bad ElementRange combination #1"
diag = False
(T, b, z) = makeUnstTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, 3]], "i", order="F"),
    NPY.ones((3 * 4), dtype="int32"),
)
hexas = CGL.newElements(
    z,
    "HEXAS",
    CGK.HEXA_8_s,
    NPY.array([[3, 7]], "i", order="F"),
    NPY.ones((5 * 8), dtype="int32"),
)  # Bad combination of ElementRange
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone unstructured bad ElementRange combination #2"
diag = False
(T, b, z) = makeUnstTree(vertexsize, cellsize)
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
    NPY.array([[cellsize + 1 + 1, cellsize + 7 + 1]], "i", order="F"),
    NPY.array(
        [
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 20, 25, 29], dtype="int32", order="F"),
)  # should be cellsize+1,cellsize+7
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone unstructured bad ElementRange combination #3"
diag = False
(T, b, z) = makeUnstTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.ones((cellsize * 4), dtype="int32"),
    NPY.array([[1, cellsize]], "i", order="F"),
)
element = CGL.newElements(
    z,
    "NGON",
    CGK.NGON_n_s,
    NPY.array([[cellsize, cellsize + 7 - 1]], "i", order="F"),
    NPY.array(
        [
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
        dtype="int32",
        order="F",
    ),
    NPY.array([0, 4, 9, 12, 17, 20, 25, 29], dtype="int32", order="F"),
)  # should be cellsize+1,cellsize+7
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone unstructured NFACE without NGON"
diag = False
(T, b, z) = makeUnstTree(vertexsize, cellsize)
tetras = CGL.newElements(
    z,
    "TETRAS",
    CGK.TETRA_4_s,
    NPY.array([[1, cellsize]], "i", order="F"),
    NPY.ones((cellsize * 4), dtype="int32"),
)
element = CGL.newElements(
    z,
    "NFACE",
    CGK.NFACE_n_s,
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
tag = "zone structured full BC and GridConnect"
diag = True


def makeStTree():
    T = CGL.newCGNSTree()
    b = CGL.newBase(T, "{Base}", 3, 3)
    z1 = CGL.newZone(
        b, "{Zone1}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F")
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
    z2 = CGU.copyNode(z1, "{Zone2}")
    b[2].append(z2)
    zgc = CGL.newZoneGridConnectivity(z1)
    gc = CGL.newGridConnectivity1to1(
        zgc,
        "join1_2",
        "{Zone2}",
        NPY.array([[1, 1], [1, 4], [1, 9]]),
        NPY.array([[5, 5], [3, 7], [1, 9]]),
        NPY.array([-1, +2, +3]),
    )
    zgc = CGL.newZoneGridConnectivity(z2)
    gc = CGL.newGridConnectivity1to1(
        zgc,
        "join2_1",
        "{Zone1}",
        NPY.array([[5, 5], [3, 7], [1, 9]]),
        NPY.array([[1, 1], [1, 4], [1, 9]]),
        NPY.array([-1, +2, +3]),
    )
    zbc = CGL.newZoneBC(z1)
    n = CGL.newBoundary(
        zbc,
        "{BC1_1}",
        [[5, 5], [1, 7], [1, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC1_2}",
        [[1, 5], [1, 1], [1, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC1_3}",
        [[1, 5], [7, 7], [1, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC1_4}",
        [[1, 5], [1, 7], [1, 1]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC1_5}",
        [[1, 5], [1, 7], [9, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC1_6}",
        [[1, 1], [4, 7], [1, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    zbc = CGL.newZoneBC(z2)
    n = CGL.newBoundary(
        zbc,
        "{BC2_1}",
        [[1, 1], [1, 7], [1, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC2_2}",
        [[1, 5], [1, 1], [1, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC2_3}",
        [[1, 5], [7, 7], [1, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC2_4}",
        [[1, 5], [1, 7], [1, 1]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC2_5}",
        [[1, 5], [1, 7], [9, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    n = CGL.newBoundary(
        zbc,
        "{BC2_6}",
        [[5, 5], [1, 3], [1, 9]],
        btype=CGK.Null_s,
        family=None,
        pttype=CGK.PointRange_s,
    )
    g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
    z = [z1, z2]
    return (T, b, z)


(T, b, z) = makeStTree()
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone structured uncomplete BC and GridConnect (warning)"
diag = False
(T, b, z) = makeStTree()
pth = CGU.getAllNodesByTypeOrNameList(z[0], ["Zone_t", "ZoneBC_t", "{BC1_1}"])[0]
CGU.removeChildByName(CGU.getNodeByPath(z[0], CGU.getPathAncestor(pth)), "{BC1_1}")
pth = CGU.getAllNodesByTypeOrNameList(
    z[1], ["Zone_t", "ZoneGridConnectivity_t", "join2_1"]
)[0]
CGU.removeChildByName(CGU.getNodeByPath(z[1], CGU.getPathAncestor(pth)), "join2_1")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zone structured doubly defined BC and GridConnect (warning)"
diag = False
(T, b, z) = makeStTree()
pth = CGU.getAllNodesByTypeOrNameList(z[0], ["Zone_t", "ZoneBC_t"])[0]
zbc = CGU.getNodeByPath(z[0], pth)
n = CGL.newBoundary(
    zbc,
    "{BC1_1b}",
    [[5, 5], [1, 2], [1, 2]],
    btype=CGK.Null_s,
    family=None,
    pttype=CGK.PointRange_s,
)
g = CGL.newGridLocation(n, value=CGK.FaceCenter_s)
pth = CGU.getAllNodesByTypeOrNameList(z[1], ["Zone_t", "ZoneGridConnectivity_t"])[0]
zgc = CGU.getNodeByPath(z[1], pth)
gc = CGL.newGridConnectivity1to1(
    zgc,
    "join2_1b",
    "{Zone1}",
    NPY.array([[1, 1], [1, 2], [1, 2]]),
    NPY.array([[1, 1], [1, 4], [1, 9]]),
    NPY.array([-1, +2, +3]),
)
TESTS.append((tag, T, diag))
