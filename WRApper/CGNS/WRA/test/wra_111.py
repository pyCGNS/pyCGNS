#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#111 - n1to1_global/conn_1to1_read_global'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/cgns/001Disk.cgns',Mll.MODE_READ)
t=a.n1to1_global(1)
#p=a.conn_1to1_read_global(1)
a.close()
