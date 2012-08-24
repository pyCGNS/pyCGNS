import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll35.hdf',Mll.MODE_READ)

t=a.nholes(1,1)
print t

u=a.hole_info(1,1,1)
print u

o=a.hole_read(1,1,1)
print o, o.dtype

p=a.hole_id(1,1,1)
print p

a.close()
