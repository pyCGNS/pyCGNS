import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/001Disk.cgns',Mll.MODE_READ)

t=a.n1to1(1,1)
print t
for i in range(t):
    p=a._1to1_read(1,1,i+1)
    print p
    r=a._1to1_id(1,1,i+1)
    print r
a.close()
