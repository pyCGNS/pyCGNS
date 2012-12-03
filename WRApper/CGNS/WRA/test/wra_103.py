#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#103 - gopath/narrays/array_info'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/naca12.hdf',Mll.MODE_READ)
a.gopath("/base1/dom1/GridCoordinates")
i=a.narrays()
a.gopath("/base1/dom1/ZoneBC/bcu_1_10")
i=a.narrays()
a.gopath("/base1/dom1/sol_1000")
i=a.narrays()
for j in range(i):
    t=a.array_info(j+1)
a.close()

