#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import CGNS.PAT.cgnskeywords as CK
import numpy as N

print('CGNS.WRA.mll', '#131 - nfields/field_info/field_read/field_id')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll32.hdf', Mll.MODE_READ)
b = a.nfields(1, 1, 1)
t = a.field_info(1, 1, 1, 1)
u = a.field_read(1, 1, 1, 'data array #1', CK.RealDouble)
u = a.field_read(1, 1, 1, 'data array #2', CK.RealDouble)
u = a.field_read(1, 1, 1, 'data array #3', CK.RealDouble)
for i in range(b):
    t = a.field_id(1, 1, 1, i + 1)
a.close()
