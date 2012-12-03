#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#147 - conn_1to1_periodic_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll47.hdf',Mll.MODE_READ)
n=a.conn_1to1_periodic_read(1,1,1)
a.close()
