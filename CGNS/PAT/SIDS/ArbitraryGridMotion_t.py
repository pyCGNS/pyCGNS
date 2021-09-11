#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#

#
import CGNS.PAT.cgnslib as C
import CGNS.PAT.cgnskeywords as K
import CGNS.PAT.cgnserrors as E
import numpy as N

#
#
data = C.newArbitraryGridMotion(None, "{ArbitraryGridMotion}")
C.newRind(data, N.array([0, 0, 0, 0, 1, 1]))
C.newGridLocation(data)
C.newDataArray(data, K.GridVelocityX_s)
C.newDataArray(data, K.GridVelocityY_s)
C.newDataArray(data, K.GridVelocityZ_s)
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")
#
status = "11.3"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
