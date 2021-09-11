#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as C
import CGNS.PAT.cgnserrors as E
import CGNS.PAT.cgnskeywords as K
import numpy as N

#
data = C.newFlowSolution(None)
C.newDataArray(data, "{DataArray}")
C.newRind(data, N.array([0, 0, 0, 0, 1, 1]))
C.newGridLocation(data)
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")
#
status = "7.7"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
