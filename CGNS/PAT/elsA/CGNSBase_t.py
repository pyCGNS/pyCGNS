#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnserrors   as CGE
import CGNS.PAT.cgnskeywords as CGK
import numpy                 as NPY

data=C.newBase(None,'{Base}',3,3)
C.newZone(data,'{Zone}',N.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
C.newSimulationType(data)
C.newConvergenceHistory(data)
C.newFamily(data,'SURFACES')
C.newFlowEquationSet(data)
C.newReferenceState(data)
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data,'FingerPrint')

status='0.1'
comment='ONERA/elsA CFD pattern'
pattern=[data, status, comment]
