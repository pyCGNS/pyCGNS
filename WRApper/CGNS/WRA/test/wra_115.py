import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/114Disk.cgns',Mll.MODE_READ)

t=a.nconns(1,1)
print t

for i in range(t):
    p=a.conn_info(1,1,i+1)
    print p
    o=a.conn_read_short(1,1,i+1)
    print o
    
## tmp/cgns/101Disk.cgns
## tmp/cgns/102Disk.cgns
## tmp/cgns/104Disk.cgns
## tmp/cgns/114Disk.cgns
## tmp/cgns/123Disk.cgns
## tmp/cgns/131Disk.cgns
