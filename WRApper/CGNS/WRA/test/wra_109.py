import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/001Disk.cgns',Mll.MODE_READ)

t=a.nzconns(1,1)
print t
for i in range(t):
    p=a.zconn_read(1,1,i+1)
    print p
    o=a.zconn_get(1,1)
    print o
    z=a.nconns(1,1)
    print z
 
a.close()
