import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll33.hdf',Mll.MODE_READ)
b=a.sol_ptset_info(1,1,3)
print b
t=a.sol_ptset_read(1,1,3)
print t

a.close()
