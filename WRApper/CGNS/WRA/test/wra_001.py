#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll

print 'CGNS.WRA.mll','#001 - pyCGNS/base_write/close'

a=Mll.pyCGNS('tmp/testmll.hdf',Mll.MODE_WRITE)
a.base_write('Base',3,3)
a.close()

# ---

