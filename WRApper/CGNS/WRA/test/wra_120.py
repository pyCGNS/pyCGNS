#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#120 - ndescriptors'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll23.hdf',Mll.MODE_READ)
a.gopath("/Base")
p=a.ndescriptors()
a.close()
