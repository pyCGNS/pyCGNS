#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#136 - n_arbitrary_motions/arbitrary_motion_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll37.hdf',Mll.MODE_READ)
n=a.n_arbitrary_motions(1,1)
for i in range(n):
    t=a.arbitrary_motion_read(1,1,i+1)
a.close()
