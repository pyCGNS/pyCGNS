import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------

a=Mll.pyCGNS('tmp/testmll02.hdf',Mll.MODE_WRITE)
a.base_write('Base',3,3)
a.zone_write(1,'Zone 01',N.array([[3,5,7],[2,4,6],[0,0,0]]),Mll.Structured)
a.zone_write(1,'Zone 02',N.array([[3,5,7],[2,4,6],[0,0,0]]),Mll.Structured)
a.zone_write(1,'Zone 03',N.array([[3,5,7],[2,4,6],[0,0,0]]),Mll.Structured)
a.close()

