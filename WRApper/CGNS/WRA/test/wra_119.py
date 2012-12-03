#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#119 - nintegrals/integral_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll22.hdf',Mll.MODE_READ)
a.gopath("/Base")
p=a.nintegrals()
for i in range(p):
    t=a.integral_read(i+1)
a.close()
