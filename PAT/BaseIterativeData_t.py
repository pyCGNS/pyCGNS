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
data=C.newBaseIterativeData(None)
C.newDataArray(data,K.NumberOfZones_s)
C.newDataArray(data,K.NumberOfFamilies_s)
C.newDataArray(data,K.ZonePointers_s)
C.newDataArray(data,K.FamilyPointers_s)
C.newDataArray(data,'{DataArray}')
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data,'{UserDefinedData}')
C.newDescriptor(data,'{Descriptor}')
#
status='11.1.1'
comment='Full SIDS with all optionals'
pattern=[data, status, comment]
#
