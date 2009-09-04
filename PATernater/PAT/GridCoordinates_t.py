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

data=C.newCoordinates(None)
C.newRind(data,N.array([0,0,0,0,1,1]))
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data,'{UserDefinedData}')
C.newDescriptor(data,'{Descriptor}')

C.newDataArray(data,K.CoordinateX_s)
C.newDataArray(data,K.CoordinateY_s)
C.newDataArray(data,K.CoordinateZ_s)

status='7.1'
comment='Full SIDS with all optionals'
pattern=[data, status, comment]
