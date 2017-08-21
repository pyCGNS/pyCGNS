#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#120 - ndescriptors')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll23.hdf', Mll.MODE_READ)
a.gopath("/Base")
p = a.ndescriptors()
a.close()
