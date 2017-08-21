#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function
import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#110 - n1to1/conn_1to1_read/conn_1to1_id')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/001Disk.cgns', Mll.MODE_READ)
t = a.n1to1(1, 1)
for i in range(t):
    p = a.conn_1to1_read(1, 1, i + 1)
    r = a.conn_1to1_id(1, 1, i + 1)
a.close()
