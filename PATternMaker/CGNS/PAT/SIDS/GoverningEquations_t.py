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

data=C.newGoverningEquations(None)
C.newUserDefinedData(data,'{UserDefinedData}')
C.newDescriptor(data,'{Descriptor}')
C.newDiffusionModel(data)
status='-'
comment='SIDS GoverningEquations_t'
childrentypes=[
    K.DiffusionModel_ts, # '"int[1+...+IndexDimension]"',
    K.Descriptor_ts,
    K.UserDefinedData_ts,
    ]
datatypes=[C.C1]
cardinality=[C.zero_N]
names=[]
pattern=[data,status,comment,childrentypes,datatypes,names]
