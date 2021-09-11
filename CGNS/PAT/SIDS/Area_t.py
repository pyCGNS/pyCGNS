#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as C
import CGNS.PAT.cgnskeywords as K
import CGNS.PAT.cgnserrors as E
import numpy as N

#
#
data = C.newArea(None)
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")
#
status = "11.3"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
