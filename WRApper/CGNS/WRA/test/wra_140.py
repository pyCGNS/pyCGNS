#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#140 - gravity_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll41.hdf',Mll.MODE_READ)
n=a.gravity_read(1)
a.close()
