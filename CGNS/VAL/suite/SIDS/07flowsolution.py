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
tag = "flowsolution unstructured"
diag = True


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
    tetras = CGL.newElements(
        z,
        "TETRAS",
        CGK.TETRA_4_s,
        NPY.array([[1, cellsize]], "i", order="F"),
        NPY.ones((cellsize * 4), dtype="int32"),
    )
    return (T, b, z)


vertexsize = 20
cellsize = 7
(T, b, z) = makeUnstTree(vertexsize, cellsize)
sol1 = CGL.newFlowSolution(z, name="sol1", gridlocation=CGK.Vertex_s)
CGL.newDataArray(sol1, "var", value=NPY.ones(vertexsize, dtype="float64", order="F"))
sol2 = CGL.newFlowSolution(z, name="sol2", gridlocation=CGK.CellCenter_s)
CGL.newDataArray(sol2, "var", value=NPY.ones(cellsize, dtype="float64", order="F"))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "flowsolution bad dataarray dim"
diag = False
(T, b, z) = makeUnstTree(vertexsize, cellsize)
sol1 = CGL.newFlowSolution(z, name="sol1", gridlocation=CGK.Vertex_s)
CGL.newDataArray(
    sol1, "var", value=NPY.ones(vertexsize + 1, dtype="float64", order="F")
)
sol2 = CGL.newFlowSolution(z, name="sol2", gridlocation=CGK.CellCenter_s)
CGL.newDataArray(sol2, "var", value=NPY.ones(cellsize + 1, dtype="float64", order="F"))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "flowsolution structured"
diag = True


def makeStTree(vertexsize, cellsize):
    T = CGL.newCGNSTree()
    b = CGL.newBase(T, "{Base}", 3, 3)
    z = CGL.newZone(
        b,
        "{Zone}",
        NPY.array(
            [
                [vertexsize[0], cellsize[0], 0],
                [vertexsize[1], cellsize[1], 0],
                [vertexsize[2], cellsize[2], 0],
            ],
            order="F",
        ),
    )
    g = CGL.newGridCoordinates(z, "GridCoordinates")
    d = CGL.newDataArray(
        g, CGK.CoordinateX_s, NPY.ones(tuple(vertexsize), dtype="float64", order="F")
    )
    d = CGL.newDataArray(
        g, CGK.CoordinateY_s, NPY.ones(tuple(vertexsize), dtype="float64", order="F")
    )
    d = CGL.newDataArray(
        g, CGK.CoordinateZ_s, NPY.ones(tuple(vertexsize), dtype="float64", order="F")
    )
    return (T, b, z)


vertexsize = [5, 7, 9]
cellsize = [i - 1 for i in vertexsize]
(T, b, z) = makeStTree(vertexsize, cellsize)
sol1 = CGL.newFlowSolution(z, name="sol1", gridlocation=CGK.Vertex_s)
CGL.newDataArray(
    sol1, "var", value=NPY.ones(tuple(vertexsize), dtype="float64", order="F")
)
sol2 = CGL.newFlowSolution(z, name="sol2", gridlocation=CGK.CellCenter_s)
CGL.newDataArray(
    sol2, "var", value=NPY.ones(tuple(cellsize), dtype="float64", order="F")
)
sol3 = CGL.newFlowSolution(z, name="sol3", gridlocation=CGK.Vertex_s)
CGL.newRind(sol3, NPY.ones(2 * 3, dtype="int32", order="F"))
dsize = [i + 1 + 1 for i in vertexsize]
CGL.newDataArray(sol3, "var", value=NPY.ones(tuple(dsize), dtype="float64", order="F"))
sol4 = CGL.newFlowSolution(z, name="sol4", gridlocation=CGK.CellCenter_s)
CGL.newRind(sol4, NPY.ones(2 * 3, dtype="int32", order="F"))
dsize = [i + 1 + 1 for i in cellsize]
CGL.newDataArray(sol4, "var", value=NPY.ones(tuple(dsize), dtype="float64", order="F"))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "flowsolution bad dataarray dim"
diag = False
(T, b, z) = makeStTree(vertexsize, cellsize)
sol1 = CGL.newFlowSolution(z, name="sol1", gridlocation=CGK.Vertex_s)
dsize = [i + 1 for i in vertexsize]
CGL.newDataArray(sol1, "var", value=NPY.ones(tuple(dsize), dtype="float64", order="F"))
sol2 = CGL.newFlowSolution(z, name="sol2", gridlocation=CGK.CellCenter_s)
dsize = [i + 1 for i in cellsize]
CGL.newDataArray(sol2, "var", value=NPY.ones(tuple(dsize), dtype="float64", order="F"))
sol3 = CGL.newFlowSolution(z, name="sol3", gridlocation=CGK.Vertex_s)
CGL.newRind(sol3, NPY.ones(2 * 3, dtype="int32", order="F"))
dsize = [i for i in vertexsize]
CGL.newDataArray(sol3, "var", value=NPY.ones(tuple(dsize), dtype="float64", order="F"))
sol4 = CGL.newFlowSolution(z, name="sol4", gridlocation=CGK.CellCenter_s)
CGL.newRind(sol4, NPY.ones(2 * 3, dtype="int32", order="F"))
dsize = [i for i in cellsize]
CGL.newDataArray(sol4, "var", value=NPY.ones(tuple(dsize), dtype="float64", order="F"))
TESTS.append((tag, T, diag))
