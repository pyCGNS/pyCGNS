#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#105 - nuser_data/'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/cgns/001Disk.cgns',Mll.MODE_READ)
a.gopath("/Disk/zone1/GridCoordinates")
i=a.nuser_data()
for j in range(i):
    t=a.array_info(j+1)
a.close()
