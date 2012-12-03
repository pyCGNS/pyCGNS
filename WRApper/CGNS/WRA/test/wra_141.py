#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll 
import numpy as NPY
import CGNS.PAT.cgnskeywords as CK

print 'CGNS.WRA.mll','#141 - axisym_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll42.hdf',Mll.MODE_READ)
n=a.axisym_read(1)
a.close()

# ---
