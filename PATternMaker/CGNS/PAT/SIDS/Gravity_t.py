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
