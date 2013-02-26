#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#132 - sol_ptset_info/sol_ptset_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll33.hdf',Mll.MODE_READ)
b=a.sol_ptset_info(1,1,3)
#t=a.sol_ptset_read(1,1,3)
a.close()
