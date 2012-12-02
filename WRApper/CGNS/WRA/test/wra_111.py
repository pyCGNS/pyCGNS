import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/cgns/001Disk.cgns',Mll.MODE_READ)

t=a.n1to1_global(1)

p=a._1to1_read_global(1)


a.close()
