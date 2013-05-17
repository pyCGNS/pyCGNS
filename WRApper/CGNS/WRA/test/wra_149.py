#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#149 - conn_1to1_average_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll49.hdf',Mll.MODE_READ)
n=a.conn_1to1_average_read(1,1,1)
a.close()
