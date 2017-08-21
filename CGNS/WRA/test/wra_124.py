#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#124 - expfull_read')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll27.hdf', Mll.MODE_READ)
a.gopath('/Base/Zone 01/GridCoordinates/CoordinateX')
p = a.nexponents()
t = a.exponents_info()
o = a.expfull_read()
a.gopath('/Base/Zone 01/GridCoordinates/CoordinateY')
p = a.nexponents()
t = a.exponents_info()
o = a.expfull_read()
a.gopath('/Base/Zone 01/GridCoordinates/CoordinateZ')
p = a.nexponents()
a.close()
