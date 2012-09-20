import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll37.hdf',Mll.MODE_READ)

n=a.n_arbitrary_motions(1,1)
print n

for i in range(n):
    t=a.arbitrary_motion_read(1,1,i+1)
    print t

a.close()
