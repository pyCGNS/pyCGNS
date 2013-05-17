#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#154 - rind_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll50.hdf',Mll.MODE_READ)
a.gopath('/Base/Zone 01/GridCoordinates')
r=a.rind_read()
a.close()
