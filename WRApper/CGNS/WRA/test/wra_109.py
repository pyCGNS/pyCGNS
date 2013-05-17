#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#109 - nzconns/zconn_read/zconn_get/nconns'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/001Disk.cgns',Mll.MODE_READ)
t=a.nzconns(1,1)
for i in range(t):
    p=a.zconn_read(1,1,i+1)
    o=a.zconn_get(1,1)
    z=a.nconns(1,1)
a.close()
