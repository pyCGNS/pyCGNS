#  ---------------------------------------------------------------------------
#  pyCGNS.PAT - Python package for CFD General Notation System - PATternMaker
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#  $Release$
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as C
import CGNS.PAT.cgnserrors   as E
import CGNS.PAT.cgnskeywords as K
import numpy             as N

data=C.newBase(None,'{Base}',3,3)
C.newZone(data,'{Zone}')
C.newSimulationType(data)
C.newIntegralData(data,'{IntegralData}')
C.newBaseIterativeData(data)
C.newConvergenceHistory(data)
C.newFamily(data,'{Family}')
C.newFlowEquationSet(data)
C.newReferenceState(data)
C.newAxisymmetry(data)
C.newRotatingCoordinates(data)
C.newGravity(data)
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data,'{UserDefinedData}')
C.newDescriptor(data,'{Descriptor}')

status='6.2'
comment='Full SIDS with all optionals'
pattern=[data, status, comment]
