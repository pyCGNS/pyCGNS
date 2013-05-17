#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#114 - nconns/conn_info/conn_read/conn_id'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/114Disk.cgns',Mll.MODE_READ)
t=a.nconns(1,1)
for i in range(t):
    p=a.conn_info(1,1,i+1)
    o=a.conn_read(1,1,i+1)
    u=a.conn_id(1,1,i+1)
a.close()    
