#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#144 - bc_wallfunction_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll44.hdf',Mll.MODE_READ)
n=a.bc_wallfunction_read(1,1,1)
a.close()
