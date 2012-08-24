import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll34.hdf',Mll.MODE_READ)
b=a.nsubregs(1,1)
print b

for i in range(b):
    t=a.subreg_info(1,1,i+1)
    print t
    o=a.subreg_ptset_read(1,1,i+1)
    print o
    u=a.subreg_bcname_read(1,1,i+1)
    print u
    z=a.subreg_gcname_read(1,1,i+1)
    print z

a.close()
