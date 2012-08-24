import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll24.hdf',Mll.MODE_READ)

a.gopath("/Base")
p=a.nunits()
print p
t=a.units_read()
print t
a.close()
