import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll48.hdf',Mll.MODE_READ)

n=a.conn_average_read(1,1,1)
print n

a.close()
