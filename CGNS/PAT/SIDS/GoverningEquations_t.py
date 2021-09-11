#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as C
import CGNS.PAT.cgnserrors as E
import CGNS.PAT.cgnskeywords as K
import numpy as N

data = C.newGoverningEquations(None)
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")
C.newDiffusionModel(data)
status = "-"
comment = "SIDS GoverningEquations_t"
pattern = [data, status, comment]
