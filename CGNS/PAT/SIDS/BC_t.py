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
data = C.newBC(None, "{BC}")
C.newIndexArray(data, K.InwardNormalList_s)
C.newReferenceState(data)
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")
C.newOrdinal(data)
#
status = "9.3"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
