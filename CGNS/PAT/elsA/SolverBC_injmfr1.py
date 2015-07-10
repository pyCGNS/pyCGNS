#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnsutils    as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy                 as NPY

data=CGL.newUserDefinedData(None,'Solver#BC')
CGL.newDataArray(data,'type',CGU.setStringAsArray("injmfr1"))
CGL.newDataArray(data,'tvx',CGU.setDoubleAsArray(1.0))
CGL.newDataArray(data,'tvy',CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'tvz',CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'surf_massflow',CGU.setDoubleAsArray(50.0))
CGL.newDataArray(data,'stagnation_enthalpy',CGU.setDoubleAsArray(301140.0))
CGL.newDataArray(data,'injtur1',CGU.setDoubleAsArray(1e-08))
CGL.newDataArray(data,'injtur2',CGU.setDoubleAsArray(1e-06))
    
status='0.1'
comment='ONERA/elsA CFD pattern'
pattern=[data, status, comment]
