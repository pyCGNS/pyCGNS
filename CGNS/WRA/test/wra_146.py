#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#146 - conn_periodic_read')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll46.hdf', Mll.MODE_READ)
n = a.conn_periodic_read(1, 1, 1)
a.close()
