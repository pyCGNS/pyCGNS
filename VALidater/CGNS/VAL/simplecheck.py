#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import CGNS.PAT.cgnsutils as CGU

CHECK_NONE=0
CHECK_GOOD=1
CHECK_WARN=2
CHECK_FAIL=3
CHECK_USER=4

def checkTree(T):
    diag={}
    paths=CGU.getPathFullTree(T)
    for p in paths:
        diag[p]=(CHECK_GOOD,"ok")
    return diag
    
    
