#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#137 - simulation_type_read')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll38.hdf', Mll.MODE_READ)
n = a.simulation_type_read(1)
a.close()
