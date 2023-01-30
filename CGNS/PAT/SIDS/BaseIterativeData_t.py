#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
from .. import cgnslib as C
from .. import cgnserrors as E
from .. import cgnskeywords as K
import numpy as N

#
data = C.newBaseIterativeData(None, K.BaseIterativeData_s)
C.newDataArray(data, K.NumberOfZones_s)
C.newDataArray(data, K.NumberOfFamilies_s)
C.newDataArray(data, K.ZonePointers_s)
C.newDataArray(data, K.FamilyPointers_s)
C.newDataArray(data, "{DataArray}")
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")
#
status = "11.1.1"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
