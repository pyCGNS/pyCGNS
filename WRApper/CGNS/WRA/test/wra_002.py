#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import CGNS.PAT.cgnskeywords as CK
import numpy as NPY

print 'CGNS.WRA.mll','#002 - zone_write'

a=Mll.pyCGNS('tmp/testmll02.hdf',Mll.MODE_WRITE)
a.base_write('Base',3,3)
a.zone_write(1,'Zone 01',NPY.array([[3,5,7],[2,4,6],[0,0,0]]),CK.Structured)
a.zone_write(1,'Zone 02',NPY.array([[3,5,7],[2,4,6],[0,0,0]]),CK.Structured)
a.zone_write(1,'Zone 03',NPY.array([[3,5,7],[2,4,6],[0,0,0]]),CK.Structured)
a.close()

# ---

