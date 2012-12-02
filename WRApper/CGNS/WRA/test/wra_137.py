import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll38.hdf',Mll.MODE_READ)

n=a.simulation_type_read(1)
print n

a.close()
