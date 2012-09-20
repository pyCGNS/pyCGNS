import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll36.hdf',Mll.MODE_READ)

t=a.n_rigid_motions(1,1)
print t

p=a.rigid_motion_read(1,1,1)
print p

a.close()
