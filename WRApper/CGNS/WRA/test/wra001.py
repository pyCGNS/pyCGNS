import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll06.hdf',Mll.MODE_READ)
i=a.boco_read(1,1,2)
print i
i=a.boco_info(1,1,2)
i=a.boco_info(1,1,1)
i=a.boco_gridlocation_read(1,1,1)
print i
a.close()

