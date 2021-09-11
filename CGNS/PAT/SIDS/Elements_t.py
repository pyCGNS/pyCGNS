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
data = C.newElements(None, "{Elements_t}")
C.newDataArray(data, K.ParentData_s)
C.newRind(data, N.array([0, 0, 0, 0, 1, 1]))
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")
#
status = "7.3"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
