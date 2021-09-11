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
tag = "zonesubregion"
diag = True


def makeCorrectTree(vertexsize, cellsize, ntris):
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
    tris = CGL.newElements(
        z,
        "TRIS",
        CGK.TRI_3_s,
        NPY.array([[cellsize + 1, cellsize + ntris]], "i", order="F"),
        NPY.ones((ntris * 3), dtype="int32"),
    )
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
    return (T, b, z)


vertexsize = 20
cellsize = 7
ntris = 11
(T, b, z) = makeCorrectTree(vertexsize, cellsize, ntris)
zsr1 = CGU.newNode(
    "SubRegion1",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[1, vertexsize]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=zsr1,
)
CGL.newDataArray(zsr1, "var", value=NPY.ones(vertexsize, dtype="float64", order="F"))
zsr2 = CGU.newNode(
    "SubRegion2",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointList_s,
    NPY.array([range(1, vertexsize + 1)], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexArray_ts,
    parent=zsr2,
)
CGL.newDataArray(zsr2, "var", value=NPY.ones(vertexsize, dtype="float64", order="F"))
zsr3 = CGU.newNode(
    "SubRegion3",
    NPY.array([2], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.BCRegionName_s, CGU.setStringAsArray("BC"), [], CGK.Descriptor_ts, parent=zsr3
)
CGL.newDataArray(zsr3, "var", value=NPY.ones(ntris, dtype="float64", order="F"))
zsr5 = CGU.newNode(
    "SubRegion5",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[1, cellsize]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=zsr5,
)
CGL.newGridLocation(zsr5, value=CGK.CellCenter_s)
CGL.newDataArray(zsr5, "var", value=NPY.ones(cellsize, dtype="float64", order="F"))
zsr6 = CGU.newNode(
    "SubRegion6",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointList_s,
    NPY.array([range(1, cellsize + 1)], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexArray_ts,
    parent=zsr6,
)
CGL.newGridLocation(zsr6, value=CGK.CellCenter_s)
CGL.newDataArray(zsr6, "var", value=NPY.ones(cellsize, dtype="float64", order="F"))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zonesubregion bad node value"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize, ntris)
zsr1 = CGU.newNode(
    "SubRegion1",
    NPY.array([4], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[1, vertexsize]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=zsr1,
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zonesubregion bad Grid Location"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize, ntris)
zsr1 = CGU.newNode(
    "SubRegion1",
    NPY.array([2], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[1, cellsize]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=zsr1,
)
CGL.newGridLocation(zsr1, value=CGK.CellCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zonesubregion bad children"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize, ntris)
zsr1 = CGU.newNode(
    "SubRegion1",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[1, vertexsize]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=zsr1,
)
CGU.newNode(
    CGK.PointList_s,
    NPY.array([range(1, vertexsize + 1)], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexArray_ts,
    parent=zsr1,
)
CGU.newNode(
    CGK.BCRegionName_s, CGU.setStringAsArray("bc"), [], CGK.Descriptor_ts, parent=zsr1
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zonesubregion bad regionname"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize, ntris)
zsr1 = CGU.newNode(
    "SubRegion1",
    NPY.array([2], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.BCRegionName_s, CGU.setStringAsArray("bc"), [], CGK.Descriptor_ts, parent=zsr1
)
zsr4 = CGU.newNode(
    "SubRegion4",
    NPY.array([2], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.GridConnectivityRegionName_s,
    CGU.setStringAsArray("GC"),
    [],
    CGK.Descriptor_ts,
    parent=zsr4,
)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zonesubregion bad dataarray shapes"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize, ntris)
zsr1 = CGU.newNode(
    "SubRegion1",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[1, vertexsize]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=zsr1,
)
CGL.newDataArray(
    zsr1, "var", value=NPY.ones(vertexsize + 1, dtype="float64", order="F")
)
zsr2 = CGU.newNode(
    "SubRegion2",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointList_s,
    NPY.array([range(1, vertexsize + 1)], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexArray_ts,
    parent=zsr2,
)
CGL.newDataArray(
    zsr2, "var", value=NPY.ones(vertexsize + 1, dtype="float64", order="F")
)
zsr3 = CGU.newNode(
    "SubRegion3",
    NPY.array([2], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.BCRegionName_s, CGU.setStringAsArray("BC"), [], CGK.Descriptor_ts, parent=zsr3
)
CGL.newDataArray(zsr3, "var", value=NPY.ones(cellsize, dtype="float64", order="F"))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zonesubregion bad point list on cell and face subregions"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize, ntris)
zsr6 = CGU.newNode(
    "SubRegion6",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointList_s,
    NPY.array([range(1, cellsize + 2)], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexArray_ts,
    parent=zsr6,
)
CGL.newGridLocation(zsr6, value=CGK.CellCenter_s)
zsr7 = CGU.newNode(
    "SubRegion7",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointList_s,
    NPY.array([range(cellsize, cellsize + ntris)], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexArray_ts,
    parent=zsr7,
)
CGL.newGridLocation(zsr7, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "zonesubregion bad point range on cell and face subregions"
diag = False
(T, b, z) = makeCorrectTree(vertexsize, cellsize, ntris)
zsr5 = CGU.newNode(
    "SubRegion5",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[1, cellsize + 1]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=zsr5,
)
CGL.newGridLocation(zsr5, value=CGK.CellCenter_s)
zsr8 = CGU.newNode(
    "SubRegion8",
    NPY.array([3], dtype=NPY.int32, order="F"),
    [],
    CGK.ZoneSubRegion_ts,
    parent=z,
)
CGU.newNode(
    CGK.PointRange_s,
    NPY.array([[cellsize, cellsize + ntris]], dtype=NPY.int32, order="F"),
    [],
    CGK.IndexRange_ts,
    parent=zsr8,
)
CGL.newGridLocation(zsr8, value=CGK.FaceCenter_s)
TESTS.append((tag, T, diag))
