# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 58 $ $Date: 2008-08-20 15:55:47 +0200 (Wed, 20 Aug 2008) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
import CGNS.cgnslib      as C
import CGNS.cgnserrors   as E
import CGNS.cgnskeywords as K
import numpy             as N

data=C.newGravity(None)
status='-'
comment=''
childrentypes=[
    [K.DataArray_ts]+C.typeListA
    ]
datatypes=[C.MT]
cardinality=[C.one_N]
names=[]
pattern=[data,status,comment,childrentypes,datatypes,names]
