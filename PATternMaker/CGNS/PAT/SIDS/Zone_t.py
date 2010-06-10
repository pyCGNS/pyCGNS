# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 47 $ $Date: 2008-01-24 12:00:02 +0100 (Thu, 24 Jan 2008) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
import CGNS.PAT.cgnslib      as C
import CGNS.PAT.cgnserrors   as E
import CGNS.PAT.cgnskeywords as K
import numpy             as N

data=C.newZone(None,'{Zone}')

g1=C.newGridCoordinates(data,"GridCoordinates")
C.newRigidGridMotion(data,"{RigidGridMotion}")
C.newArbitraryGridMotion(data,"{ArbitraryGridMotion}")
C.newFlowSolution(data,"{FlowSolution}")
C.newDiscreteData(data,"{DiscreteData}")
C.newIntegralData(data,"{IntegralData}")
C.newZoneGridConnectivity(data,"{GridConnectivity}")
C.newBoundary(data,"{BC}")
C.newZoneIterativeData(data,"{ZoneIterativeData}")
C.newReferenceState(data)
C.newRotatingCoordinates(data)
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newFlowEquationSet(data)
C.newConvergenceHistory(data,K.ZoneConvergenceHistory_s)
C.newUserDefinedData(data,'{UserDefinedData}')
C.newDescriptor(data,'{Descriptor}')
C.newOrdinal(data)

status='6.3'
comment='Full SIDS with all optionals'
pattern=[data, status, comment]
