#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as N

print('CGNS.WRA.mll', '#151 - delete_node')

# ----------------------------------------------------------------------
a = Mll.pyCGNS('tmp/testmll51.hdf', Mll.MODE_MODIFY)
a.gopath('/Base/Zone 01')
a.delete_node('Initialize')
a.close()
