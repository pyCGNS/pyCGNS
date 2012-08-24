import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll32.hdf',Mll.MODE_READ)
b=a.nfields(1,1,1)
print b

t=a.field_info(1,1,1,1)
print t

u=a.field_read(1,1,1,'data array1',3,N.array([1,1,1]),N.array([2,4,6]))
print u
u=a.field_read(1,1,1,'data array2',4,N.array([1,1,1]),N.array([2,2,2]))
print u
u=a.field_read(1,1,1,'data array3',2,N.array([1,1,1]),N.array([2,4,6]))
print u

for i in range(b):
    t=a.field_id(1,1,1,i+1)
    print t

a.close()
