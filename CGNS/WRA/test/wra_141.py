#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as NPY
import CGNS.PAT.cgnskeywords as CK

print('CGNS.WRA.mll', '#141 - axisym_read')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll42.hdf', Mll.MODE_READ)
(base, refpoint, axis) = a.axisym_read(1)
a.close()

# ---
