import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll42.hdf',Mll.MODE_READ)

n=a.axisym_read(1)
print n

a.close()
