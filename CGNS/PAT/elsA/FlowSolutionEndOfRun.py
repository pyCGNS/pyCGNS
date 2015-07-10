#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnsutils    as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy                 as NPY

data=CGL.newUserDefinedData(None,'Solver#Compute')
CGL.newDataArray(data,'artviscosity',CGU.setStringAsArray("dissca"))
CGL.newDataArray(data,'avcoef_k2',CGU.setDoubleAsArray(1.0))
CGL.newDataArray(data,'avcoef_k4',CGU.setDoubleAsArray(0.032))
CGL.newDataArray(data,'avcoef_sigma',CGU.setDoubleAsArray(0.0))
    
status='0.1'
comment='ONERA/elsA CFD pattern - Generic Solver#Compute'
pattern=[data, status, comment]
