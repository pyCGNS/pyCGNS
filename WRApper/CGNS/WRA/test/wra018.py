import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll21.hdf',Mll.MODE_READ)

a.gopath("/Base/FlowEquationSet")
t=a.model_read('GasModel_t')
print t
a.close()
