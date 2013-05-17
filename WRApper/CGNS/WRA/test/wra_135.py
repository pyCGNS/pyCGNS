#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll 
import numpy as NPY
import CGNS.PAT.cgnskeywords as CK

print 'CGNS.WRA.mll','#135 - n_rigid_motions/rigid_motion_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll36.hdf',Mll.MODE_READ)

t=a.n_rigid_motions(1,1)
p=a.rigid_motion_read(1,1,1)
a.close()

# ---
