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
data = C.newDataArray(None, "{DataArray}")
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newDimensionalExponents(data)
C.newDataConversion(data)
C.newDescriptor(data, "{Descriptor}")
#
status = "7.1"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
