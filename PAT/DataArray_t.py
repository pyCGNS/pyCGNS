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
#
data=C.newDataArray(None,'{DataArray}')
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newDimensionalExponents(data)
C.newDataConversion(data)
C.newDescriptor(data,'{Descriptor}')
#
status='7.1'
comment='Full SIDS with all optionals'
pattern=[data, status, comment]
#
