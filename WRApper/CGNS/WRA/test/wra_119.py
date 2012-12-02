import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll22.hdf',Mll.MODE_READ)

a.gopath("/Base")
p=a.nintegrals()
print p
for i in range(p):
    t=a.integral_read(i+1)
    print t
a.close()
