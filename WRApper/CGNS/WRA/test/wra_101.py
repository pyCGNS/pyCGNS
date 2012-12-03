#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#101 - boco_read/boco_info/boco_gridlocation_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll06.hdf',Mll.MODE_READ)
i=a.boco_read(1,1,2)
i=a.boco_info(1,1,2)
i=a.boco_info(1,1,1)
i=a.boco_gridlocation_read(1,1,1)
a.close()

