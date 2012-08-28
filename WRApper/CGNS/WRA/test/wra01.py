import CGNS.WRA.mll as Mll
import numpy as N
import CGNS.PAT.cgnskeywords as CK

# ----------------------------------------------------------------------

a=Mll.pyCGNS('tmp/testmll.hdf',Mll.MODE_WRITE)
a.base_write('Base',3,3)
a.close()

