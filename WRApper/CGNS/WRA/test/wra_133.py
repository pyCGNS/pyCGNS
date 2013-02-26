#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#133 - nsubregs/subreg_info/subreg_ptset/subreg_bcname/subreg_gcname_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll34.hdf',Mll.MODE_READ)
b=a.nsubregs(1,1)
for i in range(b):
    t=a.subreg_info(1,1,i+1)
    o=a.subreg_ptset_read(1,1,i+1)
    u=a.subreg_bcname_read(1,1,i+1)
    z=a.subreg_gcname_read(1,1,i+1)
a.close()
