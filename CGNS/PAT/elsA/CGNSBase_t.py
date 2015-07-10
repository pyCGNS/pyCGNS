#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnserrors   as CGE
import CGNS.PAT.cgnskeywords as CGK
import numpy                 as NPY

data=CGL.newBase(None,'{Base}',3,3)
CGL.newZone(data,'{Zone}',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
CGL.newSimulationType(data)
CGL.newConvergenceHistory(data)
CGL.newFlowEquationSet(data)
CGL.newReferenceState(data)
CGL.newDataClass(data)
CGL.newDimensionalUnits(data)
CGL.newUserDefinedData(data,'FingerPrint')

status='0.1'
comment='Standard mono-Zone base'
pattern=[data, status, comment]
