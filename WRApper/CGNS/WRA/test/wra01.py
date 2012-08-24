import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------

a=Mll.pyCGNS('tmp/testmll.hdf',Mll.MODE_WRITE)
a.base_write('Base',3,3)
a.close()

