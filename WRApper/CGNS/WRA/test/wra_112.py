import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/cgns/001Disk.cgns',Mll.MODE_READ)

t=a.nconns(1,1)
print t

a.close()
