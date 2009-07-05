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
import copy
#
import BCData_t
#
data=C.newBCDataSet(None,'{BCDataSet}')
C.newGridLocation(data)
C.newPointRange(data)
C.newPointList(data)
C.newDescriptor(data,'{Descriptor}')
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newReferenceState(data)
C.newUserDefinedData(data,'{UserDefinedData}')
#
d1=copy.deepcopy(BCData_t.pattern[0])
d1[0]=K.NeumannData_s
data[2].append(d1)
#
d2=copy.deepcopy(BCData_t.pattern[0])
d2[0]=K.DirichletData_s
data[2].append(d2)
#
status='9.4'
comment='Full SIDS with all optionals'
pattern=[data, status, comment]
#
