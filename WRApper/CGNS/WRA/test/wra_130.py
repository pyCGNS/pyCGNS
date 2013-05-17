#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#130 - famname_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll31.hdf',Mll.MODE_READ)
a.gopath('/Base/Zone 01')
b=a.famname_read()
a.close()

