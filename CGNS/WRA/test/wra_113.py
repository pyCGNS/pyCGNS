#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function
import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#113 - convergence_read/state_read')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll20.hdf', Mll.MODE_READ)
a.gopath('/Base')
t = a.convergence_read()
p = a.state_read()
a.close()
