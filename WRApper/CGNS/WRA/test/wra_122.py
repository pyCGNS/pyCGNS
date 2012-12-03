#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#122 - unitsfull_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll25.hdf',Mll.MODE_READ)
a.gopath("/Base")
p=a.nunits()
t=a.unitsfull_read()
a.close()

