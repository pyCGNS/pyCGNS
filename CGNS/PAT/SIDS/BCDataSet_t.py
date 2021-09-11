#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as C
import CGNS.PAT.cgnserrors as E
import CGNS.PAT.cgnskeywords as K
import numpy as N
import copy

#
from . import BCData_t

#
data = C.newBCDataSet(None, "{BCDataSet}")
C.newGridLocation(data)
C.newPointRange(data)
C.newPointList(data)
C.newDescriptor(data, "{Descriptor}")
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newReferenceState(data)
C.newUserDefinedData(data, "{UserDefinedData}")
#
d1 = copy.deepcopy(BCData_t.pattern[0])
d1[0] = K.NeumannData_s
data[2].append(d1)
#
d2 = copy.deepcopy(BCData_t.pattern[0])
d2[0] = K.DirichletData_s
data[2].append(d2)
#
status = "9.4"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
