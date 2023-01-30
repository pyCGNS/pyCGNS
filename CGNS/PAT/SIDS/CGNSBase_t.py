#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
from .. import cgnslib as C
from .. import cgnserrors as E
from .. import cgnskeywords as K
import numpy as N

data = C.newBase(None, "{Base}", 3, 3)
C.newZone(data, "{Zone}", N.array([[5, 4, 0], [7, 6, 0], [9, 8, 0]], order="F"))
C.newSimulationType(data)
C.newIntegralData(data, "{IntegralData}")
C.newBaseIterativeData(data, "{BaseIterativeData}")
C.newConvergenceHistory(data)
C.newFamily(data, "{Family}")
C.newFlowEquationSet(data)
C.newReferenceState(data)
C.newAxisymmetry(data)
C.newRotatingCoordinates(data)
C.newGravity(data)
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")

status = "6.2"
comment = "Full SIDS with all optionals children"
pattern = [data, status, comment]
