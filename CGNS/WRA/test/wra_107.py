#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function
import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#107 - ndiscrete/discrete_read/discrete_size/discrete_ptset_info')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll14.hdf', Mll.MODE_READ)
t = a.ndiscrete(1, 1)
for i in range(t):
    p = a.discrete_read(1, 1, i + 1)
    o = a.discrete_size(1, 1, i + 1)
    u = a.discrete_ptset_info(1, 1, i + 1)
a.close()
