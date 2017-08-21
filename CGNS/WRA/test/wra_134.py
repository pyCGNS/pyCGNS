#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#134 - nholes/hole_info/hole_read/hole_id')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll35.hdf', Mll.MODE_READ)
t = a.nholes(1, 1)
u = a.hole_info(1, 1, 1)
o = a.hole_read(1, 1, 1)
p = a.hole_id(1, 1, 1)
a.close()
