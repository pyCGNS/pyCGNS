import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll20.hdf',Mll.MODE_READ)

a.gopath("/Base/FlowEquationSet/GoverningEquations")
t=a.diffusion_read()
print t

a.close()
