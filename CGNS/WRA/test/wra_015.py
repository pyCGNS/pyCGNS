#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sy
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function
import CGNS.WRA.mll as Mll
import numpy as NPY
import CGNS.PAT.cgnskeywords as CK

print('CGNS.WRA.mll', '#015 - array_write')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll14.hdf', Mll.MODE_WRITE)

a.gopath("/Base/Zone 01/discrete")
a.array_write('coordinates', NPY.ones((2, 4, 6), dtype=NPY.int32))

a.close()

# ---
