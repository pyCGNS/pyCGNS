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
from . import BCDataSet_t

#
data = C.newFamilyBC(None)
data[2].append(BCDataSet_t.pattern[0])
#
status = "12.8"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
