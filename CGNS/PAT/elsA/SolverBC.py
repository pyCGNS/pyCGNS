#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnserrors   as CGE
import CGNS.PAT.cgnskeywords as CGK
import numpy                 as NPY

data=CGL.newUserDefinedData(None,'Solver#BC')
CGL.newDataArray('type',CGU.setStringAsArray("wall"))
    
status='0.1'
comment='ONERA/elsA CFD pattern'
pattern=[data, status, comment]
