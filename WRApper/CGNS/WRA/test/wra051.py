import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll51.hdf',Mll.MODE_MODIFY)

a.gopath('/Base/Zone 01')
a.delete_node('Initialize')

a.close()

