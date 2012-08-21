import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll14.hdf',Mll.MODE_READ)

t=a.ndiscrete(1,1)
for i in range(t):
    p=a.discrete_read(1,1,i+1)
    print p
    o=a.discrete_size(1,1,i+1)
    print o
    u=a.discrete_ptset_info(1,1,i+1)
    print u
    z=a.discrete_ptset_read(1,1,i+1)
    print z  
a.close()
