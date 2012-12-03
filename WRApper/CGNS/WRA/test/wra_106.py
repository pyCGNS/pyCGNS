#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#106 - user_data_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll12.hdf',Mll.MODE_READ)
a.gopath('/Base/Zone 01/GridCoordinates')
i=a.nuser_data()
t=a.user_data_read(1)
a.close()
