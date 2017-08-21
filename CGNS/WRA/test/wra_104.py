#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function
import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#104 - array_read/array_read_as')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/001Disk.cgns', Mll.MODE_READ)
a.gopath("/Disk/zone1/GridCoordinates")
i = a.narrays()
for j in range(i):
    t = a.array_info(j + 1)
a.gopath("/Disk/.Solver#Compute")
i = a.narrays()
for j in range(i):
    t1 = a.array_info(j + 1)
    t = a.array_read(j + 1)
    t = a.array_read_as(j + 1, 5)
a.close()
