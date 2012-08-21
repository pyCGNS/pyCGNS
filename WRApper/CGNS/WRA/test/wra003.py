import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/naca12.hdf',Mll.MODE_READ)
## a.boco_read(1,1,2)
## a.boco_info(1,1,2)
## a.boco_info(1,1,1)
## a.boco_gridlocation_read(1,1,1)
## i=a.dataset_read(1,1,1,1)
a.gopath("/base1/dom1/GridCoordinates")
i=a.narrays()
print i
a.gopath("/base1/dom1/ZoneBC/bcu_1_10")
i=a.narrays()
print i
a.gopath("/base1/dom1/sol_1000")
i=a.narrays()
print i
for j in range(i):
    t=a.array_info(j+1)
    print t
a.close()

