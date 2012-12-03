#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#137 - simulation_type_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll38.hdf',Mll.MODE_READ)
n=a.simulation_type_read(1)
a.close()
