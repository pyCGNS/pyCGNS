# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 47 $ $Date: 2008-01-24 12:00:02 +0100 (Thu, 24 Jan 2008) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
import CGNS.cgnslib      as C
import CGNS.cgnserrors   as E
import CGNS.cgnskeywords as K
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
