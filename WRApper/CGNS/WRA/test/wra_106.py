import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll12.hdf',Mll.MODE_READ)

a.gopath('/Base/Zone 01/GridCoordinates')
i=a.nuser_data()
print i
t=a.user_data_read(1)
print t

a.close()
