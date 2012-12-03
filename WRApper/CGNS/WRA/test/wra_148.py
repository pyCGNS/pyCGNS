#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#148 - conn_average_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll48.hdf',Mll.MODE_READ)
n=a.conn_average_read(1,1,1)
a.close()
