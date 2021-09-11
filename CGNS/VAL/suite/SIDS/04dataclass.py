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
tag = "base dataclass #1"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newDataClass(b, CGK.NondimensionalParameter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "base dataclass #2"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newDataClass(b, CGK.NormalizedByDimensional_s)
d[0] = "dataclass"
d = CGL.newDataClass(b, CGK.NormalizedByDimensional_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "base dataclass #3"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newDataClass(b, CGK.NormalizedByDimensional_s)
d[1] = CGU.setStringAsArray("NormalizedByDimensionnal")
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "all levels dataclass"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family}")
d = CGL.newDataClass(b, CGK.NondimensionalParameter_s)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
d = CGL.newDataClass(z, CGK.NondimensionalParameter_s)
g = CGL.newGridCoordinates(z, CGK.GridCoordinates_s)
w = CGL.newDataArray(
    g, CGK.CoordinateX_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateY_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateZ_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
d = CGL.newDataClass(g, CGK.NondimensionalParameter_s)
f = CGL.newFlowSolution(z)
d = CGL.newDataClass(f, CGK.NondimensionalParameter_s)
a = CGL.newDataArray(
    f, "{DataArray}", value=NPY.ones((4, 6, 8), dtype="float64", order="F")
)
d = CGL.newDataClass(a, CGK.NondimensionalParameter_s)
n = CGL.newZoneBC(z)
d = CGL.newDataClass(n, CGK.NondimensionalParameter_s)
q = CGL.newBC(n, "{BC}", family="{Family}")
d = CGL.newDataClass(q, CGK.NondimensionalParameter_s)
s = CGL.newBCDataSet(q, "{Set#01}")
d = CGL.newDataClass(s, CGK.NondimensionalParameter_s)
c = CGL.newBCData(s, CGK.Dirichlet_s)
d = CGL.newDataClass(c, CGK.NondimensionalParameter_s)
r = CGL.newReferenceState(b)
d = CGL.newDataClass(r, CGK.NondimensionalParameter_s)
a = CGL.newAxisymmetry(b)
d = CGL.newDataClass(a, CGK.NondimensionalParameter_s)
a = CGL.newRotatingCoordinates(b)
d = CGL.newDataClass(a, CGK.NondimensionalParameter_s)
a = CGL.newDiscreteData(z, "{DiscreteData}")
d = CGL.newDataClass(a, CGK.NondimensionalParameter_s)
i = CGL.newBaseIterativeData(b, "{BaseIterativeData}")
d = CGL.newDataClass(i, CGK.NondimensionalParameter_s)
i = CGL.newZoneIterativeData(z, "{ZoneIterativeData}")
d = CGL.newDataClass(i, CGK.NondimensionalParameter_s)
m = CGL.newRigidGridMotion(
    z, "{RigidGridMotion}", vector=NPY.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newArbitraryGridMotion(z, "{ArbitraryGridMotion}")
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
x = CGL.newZoneGridConnectivity(z)
x = CGL.newGridConnectivity(x, "{GridConnectivity}", z[0])
CGL.newPointRange(
    x, value=NPY.array([[1, 1], [1, 1], [1, 1]], dtype=NPY.int32, order="F")
)
p = CGL.newGridConnectivityProperty(x)
m = CGL.newPeriodic(p)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
w = CGL.newConvergenceHistory(b)
d = CGL.newDataClass(w, CGK.NondimensionalParameter_s)
i = CGL.newIntegralData(b, "{IntegralData}")
d = CGL.newDataClass(i, CGK.NondimensionalParameter_s)
i = CGL.newUserDefinedData(b, "{UserDefinedData}")
d = CGL.newDataClass(i, CGK.NondimensionalParameter_s)
i = CGL.newGravity(b)
d = CGL.newDataClass(i, CGK.NondimensionalParameter_s)
f = CGL.newFlowEquationSet(b)
d = CGL.newDataClass(f, CGK.NondimensionalParameter_s)
m = CGL.newGasModel(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newViscosityModel(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newThermalConductivityModel(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newThermalRelaxationModel(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newChemicalKineticsModel(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newTurbulenceClosure(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newTurbulenceModel(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newEMConductivityModel(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newEMMagneticFieldModel(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
m = CGL.newEMElectricFieldModel(f)
d = CGL.newDataClass(m, CGK.NondimensionalParameter_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "all levels dataclass/dimensionalunits"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family}")
d = CGL.newDataClass(b, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(b)
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
d = CGL.newDataClass(z, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(z)
g = CGL.newGridCoordinates(z, CGK.GridCoordinates_s)
w = CGL.newDataArray(
    g, CGK.CoordinateX_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateY_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateZ_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
d = CGL.newDataClass(g, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(g)
f = CGL.newFlowSolution(z)
d = CGL.newDataClass(f, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(f)
a = CGL.newDataArray(
    f, "{DataArray}", value=NPY.ones((4, 6, 8), dtype="float64", order="F")
)
d = CGL.newDataClass(a, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(a)
n = CGL.newZoneBC(z)
d = CGL.newDataClass(n, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(n)
q = CGL.newBC(n, "{BC}", family="{Family}")
d = CGL.newDataClass(q, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(q)
s = CGL.newBCDataSet(q, "{BCDataSet}")
d = CGL.newDataClass(s, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(s)
c = CGL.newBCData(s, CGK.Neumann_s)
d = CGL.newDataClass(c, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(c)
r = CGL.newReferenceState(b)
d = CGL.newDataClass(r, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(r)
a = CGL.newAxisymmetry(b)
d = CGL.newDataClass(a, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(a)
a = CGL.newRotatingCoordinates(b)
d = CGL.newDataClass(a, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(a)
a = CGL.newDiscreteData(z, "{DiscreteData}")
d = CGL.newDataClass(a, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(a)
i = CGL.newBaseIterativeData(b, "{BaseIterativeData}")
d = CGL.newDataClass(i, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(i)
i = CGL.newZoneIterativeData(z, "{ZoneIterativeData}")
d = CGL.newDataClass(i, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(i)
m = CGL.newRigidGridMotion(
    z, "{RigidGridMotion}", vector=NPY.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(m)
m = CGL.newArbitraryGridMotion(z, "{ArbitraryGridMotion}")
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(m)
x = CGL.newZoneGridConnectivity(z)
x = CGL.newGridConnectivity(x, "{GridConnectivity}", z[0])
CGL.newPointRange(
    x, value=NPY.array([[1, 1], [1, 1], [1, 1]], dtype=NPY.int32, order="F")
)
p = CGL.newGridConnectivityProperty(x)
m = CGL.newPeriodic(p)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(m)
w = CGL.newConvergenceHistory(b)
d = CGL.newDataClass(w, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(w)
i = CGL.newIntegralData(b, "{IntegralData}")
d = CGL.newDataClass(i, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(i)
i = CGL.newUserDefinedData(b, "{UserDefinedData}")
d = CGL.newDataClass(i, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(i)
i = CGL.newGravity(b)
d = CGL.newDataClass(i, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(i)
f = CGL.newFlowEquationSet(b)
d = CGL.newDataClass(f, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(f)
m = CGL.newGasModel(f)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
u = CGL.newDimensionalUnits(m)
m = CGL.newViscosityModel(f)
u = CGL.newDimensionalUnits(m)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
m = CGL.newThermalConductivityModel(f)
u = CGL.newDimensionalUnits(m)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
m = CGL.newThermalRelaxationModel(f)
u = CGL.newDimensionalUnits(m)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
m = CGL.newChemicalKineticsModel(f)
u = CGL.newDimensionalUnits(m)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
m = CGL.newTurbulenceClosure(f)
u = CGL.newDimensionalUnits(m)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
m = CGL.newTurbulenceModel(f)
u = CGL.newDimensionalUnits(m)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
m = CGL.newEMConductivityModel(f)
u = CGL.newDimensionalUnits(m)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
m = CGL.newEMMagneticFieldModel(f)
u = CGL.newDimensionalUnits(m)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
m = CGL.newEMElectricFieldModel(f)
d = CGL.newDataClass(m, CGK.NormalizedByDimensional_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "dataclass dimensional without dimensionalunits"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family}")
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
g = CGL.newGridCoordinates(z, CGK.GridCoordinates_s)
w = CGL.newDataArray(
    g, CGK.CoordinateX_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateY_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateZ_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
f = CGL.newFlowSolution(z)
a = CGL.newDataArray(
    f, "{DataArray}", value=NPY.ones((4, 6, 8), dtype="float64", order="F")
)
d = CGL.newDataClass(a, CGK.NormalizedByDimensional_s)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "exponents without dimensionalunits"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family}")
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
g = CGL.newGridCoordinates(z, CGK.GridCoordinates_s)
w = CGL.newDataArray(
    g, CGK.CoordinateX_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateY_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateZ_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
f = CGL.newFlowSolution(z)
a = CGL.newDataArray(
    f, "{DataArray}", value=NPY.ones((4, 6, 8), dtype="float64", order="F")
)
d = CGL.newDimensionalExponents(a)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "dimensionalunits without dataclass"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family}")
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
g = CGL.newGridCoordinates(z, CGK.GridCoordinates_s)
w = CGL.newDataArray(
    g, CGK.CoordinateX_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateY_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateZ_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
f = CGL.newFlowSolution(z)
a = CGL.newDataArray(
    f, "{DataArray}", value=NPY.ones((4, 6, 8), dtype="float64", order="F")
)
d = CGL.newDimensionalUnits(a)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "bad dimensionalunits #1"
diag = False
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base}", 3, 3)
d = CGL.newFamily(b, "{Family}")
z = CGL.newZone(b, "{Zone}", NPY.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
g = CGL.newGridCoordinates(z, CGK.GridCoordinates_s)
w = CGL.newDataArray(
    g, CGK.CoordinateX_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateY_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
w = CGL.newDataArray(
    g, CGK.CoordinateZ_s, NPY.ones((5, 7, 9), dtype="float64", order="F")
)
f = CGL.newFlowSolution(z)
a = CGL.newDataArray(
    f, "{DataArray}", value=NPY.ones((4, 6, 8), dtype="float64", order="F")
)
d = CGL.newDimensionalUnits(a)
d[1] = "Meter"
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
