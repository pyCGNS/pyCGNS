#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#117 - diffusion_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll20.hdf',Mll.MODE_READ)
a.gopath("/Base/FlowEquationSet/GoverningEquations")
t=a.diffusion_read()
a.close()
