import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll14.hdf',Mll.MODE_READ)

a.gopath("/Base/Zone 01/discrete")
a.array_write('coordinates',2,3,N.array([2,4,6],dtype=N.int32),N.ones((2,4,6),dtype=N.int32))

a.close()
