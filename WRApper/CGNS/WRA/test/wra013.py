import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll20.hdf',Mll.MODE_READ)
a.gopath('/Base')
t=a.convergence_read()
print t
p=a.state_read()
print p

a.close()
