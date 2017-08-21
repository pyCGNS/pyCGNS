#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#125 - conversion_info/conversion_read')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll28.hdf', Mll.MODE_READ)
a.gopath('/Base/Zone 01/GridCoordinates/CoordinateX')
p = a.conversion_info()
t = a.conversion_read()
a.close()
