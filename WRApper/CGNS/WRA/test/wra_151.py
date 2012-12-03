#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#151 - delete_node'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll51.hdf',Mll.MODE_MODIFY)
a.gopath('/Base/Zone 01')
a.delete_node('Initialize')
a.close()

