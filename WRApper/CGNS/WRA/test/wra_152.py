#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#152 - cell_dim/index_dim'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll51.hdf',Mll.MODE_READ)
d=a.cell_dim(1)
t=a.index_dim(1,1)
a.close()

