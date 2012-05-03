#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

CHECK_NONE=0
CHECK_GOOD=1
CHECK_WARN=2
CHECK_FAIL=3
CHECK_USER=4

def checkSingle(T,node):
    stt=CHECK_GOOD
    msg=[]
    if (not CGU.checkNodeName(node[0])):
        stt=CHECK_FAIL
        msg+=[(CHECK_FAIL,'Name [%s] is not valid'%node[0])]
    parent=CGU.getParentFromNode(T,node)
    lchildren=CGU.childNames(parent)
    if (lchildren):
      lchildren.remove(node[0])
      if (node[0] in lchildren):
          stt=CHECK_FAIL
          msg+=[(CHECK_FAIL,'Name [%s] is a duplicated child name'%node[0])]
    tlist=CGU.getNodeAllowedChildrenTypes(parent,node)
    if (node[3] not in tlist):
        stt=CHECK_FAIL
        msg+=[(CHECK_FAIL,'Type [%s] not allowed for this node'%node[3])]
    dlist=CGU.getNodeAllowedDataTypes(node)
    dt=CGU.getValueDataType(node)
    if (dt not in dlist):
        stt=CHECK_FAIL
        msg+=[(CHECK_FAIL,'Datatype [%s] not allowed for this node'%dt)]
    return [stt,msg]
        
def checkTree(T):
    diag={}
    paths=CGU.getPathFullTree(T)
    for p in paths:
        node=CGU.getNodeByPath(T,p)
        diag[p]=checkSingle(T,node)
#    showDiag(diag)
    return diag

def showDiag(diag):
    for p in diag:
        if (diag[p][0]!=CHECK_GOOD):
            print p
            for d in diag[p][1]:
                print '    ',d[1]
    
