#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#129 - ptset_info')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll30.hdf', Mll.MODE_READ)
a.gopath('/Base/Zone 01/ZoneBC/BC')
p = a.ptset_info()
a.close()
