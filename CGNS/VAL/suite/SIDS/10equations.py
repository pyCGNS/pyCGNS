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
tag = "flow equation set"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base#001}", 3, 3)
f = CGL.newFlowEquationSet(b)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "governing equations"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base#001}", 3, 3)
f = CGL.newFlowEquationSet(b)
g = CGL.newGoverningEquations(f)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "diffusion model"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base#001}", 3, 3)
f = CGL.newFlowEquationSet(b)
g = CGL.newGoverningEquations(f)
d = CGL.newDiffusionModel(g, NPY.ones(6, dtype="int32"))
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
tag = "flow equation set all models #1"
diag = True
T = CGL.newCGNSTree()
b = CGL.newBase(T, "{Base#001}", 3, 3)
f = CGL.newFlowEquationSet(b)
g = CGL.newGoverningEquations(f)
d = CGL.newDiffusionModel(g, NPY.zeros(6, dtype="int32"))
m = CGL.newGasModel(f)
m = CGL.newThermalConductivityModel(f)
m = CGL.newViscosityModel(f)
m = CGL.newTurbulenceModel(f)
m = CGL.newTurbulenceClosure(f)
m = CGL.newThermalRelaxationModel(f)
m = CGL.newChemicalKineticsModel(f)
m = CGL.newEMElectricFieldModel(f)
m = CGL.newEMMagneticFieldModel(f)
m = CGL.newEMConductivityModel(f)
TESTS.append((tag, T, diag))

#  -------------------------------------------------------------------------
