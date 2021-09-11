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
data = C.newAxisymmetry(None)
C.newDataArray(data, K.AxisymmetryAngle_s)
C.newDataArray(data, K.CoordinateNames_s)
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")
#
status = "7.5"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
