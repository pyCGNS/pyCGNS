#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#112 - nconns'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/cgns/001Disk.cgns',Mll.MODE_READ)
t=a.nconns(1,1)
a.close()
