import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/001Disk.cgns',Mll.MODE_READ)
a.gopath("/Disk/zone1/GridCoordinates")
i=a.narrays()
print i
for j in range(i):
    t=a.array_info(j+1)
    print t
    
a.gopath("/Disk/.Solver#Compute")
i=a.narrays()
print i
for j in range(i):
    t1=a.array_info(j+1)
    print t1
    t=a.array_read(j+1)
    print t
    t=a.array_read_as(j+1,5)
    print t

a.close()
