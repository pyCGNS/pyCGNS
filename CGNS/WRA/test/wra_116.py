#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#116 - equationset/equationset_*/governing_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll20.hdf',Mll.MODE_READ)
a.gopath("/Base")
i=a.equationset_read()
o=a.equationset_chemistry_read()
p=a.equationset_elecmagn_read()
a.gopath("/Base/FlowEquationSet")
t=a.governing_read()
a.close()
