#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
from __future__ import unicode_literals
import CGNS.PAT.cgnslib as C
import CGNS.PAT.cgnserrors as E
import CGNS.PAT.cgnskeywords as K
import numpy as N

data = ["ZoneIterativeData", None, [], "ZoneIterativeData_t"]
status = ""
comment = ""
pattern = [data, status, comment]
