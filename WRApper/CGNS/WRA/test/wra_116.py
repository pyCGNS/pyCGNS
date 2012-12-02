import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll20.hdf',Mll.MODE_READ)

a.gopath("/Base")
i=a.equationset_read()
print i
o=a.equationset_chemistry_read()
print o
p=a.equationset_elecmagn_read()
print p
a.gopath("/Base/FlowEquationSet")
t=a.governing_read()
print t

a.close()
