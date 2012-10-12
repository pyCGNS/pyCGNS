#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils      as CGU
import CGNS.PAT.cgnstypes      as CGT
import CGNS.PAT.cgnskeywords   as CGK
import CGNS.PAT.cgnserrors     as CGE
import CGNS.VAL.parse.messages as CGM

import inspect

OLD_VERSION          = 'G001'
BAD_VERSION          = 'G002'
INVALID_NAME         = 'G003'
DUPLICATED_NAME      = 'G004'
INVALID_PATH         = 'G005'
NODE_BADDATA         = 'G006'
NODE_CHILDRENNOTLIST = 'G007'
NODE_NAMENOTSTRING   = 'G008'
NODE_NOTALIST        = 'G009'
NODE_EMPTYLIST       = 'G010'

UNKNOWN_SIDSTYPE     = 'S001'
INVALID_SIDSTYPE_P   = 'S002'
INVALID_SIDSTYPE     = 'S003'
INVALID_DATATYPE     = 'S004'
FORBIDDEN_CHILD      = 'S005'
SINGLE_CHILD         = 'S006'
MANDATORY_CHILD      = 'S007'


genericmessages={
BAD_VERSION:'CGNSLibraryVersion is incorrect',
OLD_VERSION:'CGNSLibraryVersion [%s] is too old for current check level',
INVALID_NAME:'Name [%s] is not valid',
INVALID_PATH:'PANIC: Cannot find node with path [%s]',
DUPLICATED_NAME:'Name [%s] is a duplicated child name',
UNKNOWN_SIDSTYPE:'Unknown SIDS type [%s]',
INVALID_SIDSTYPE_P:'SIDS type [%s] not allowed as child of [%s]',
INVALID_SIDSTYPE:'SIDS type [%s] not allowed for this node',
INVALID_DATATYPE:'DataType [%s] not allowed for this node',
FORBIDDEN_CHILD:'Node [%s] of type [%s] is not allowed as child',
SINGLE_CHILD:'Node [%s] of type [%s] is allowed only once as child',
MANDATORY_CHILD:'Node [%s] of type [%s] is mandatory',
NODE_EMPTYLIST:'PANIC: Node is empty list or None (child of [%s])',
NODE_NOTALIST:'PANIC: Node is not a list of 4 objects (child of [%s])',
NODE_NAMENOTSTRING:'PANIC: Node name is not a string (child of [%s])',
NODE_CHILDRENNOTLIST:'PANIC: Node children is not a list (child of [%s])',
NODE_BADDATA:'PANIC: Node data is not numpy.ndarray or None (child of [%s])',
}

class GenericContext(dict):
  def __getitem__(self,i):
    if (i not in self): self[i]=None
    return dict.__getitem__(self,i)

class GenericParser(object):
  # --------------------------------------------------------------------
  def __init__(self,log=None):
    self.keywordlist=CGK.cgnsnames
    if (log is None):
      self.log=CGM.DiagnosticLog()
    self.log.addMessages(genericmessages)
    self.context=GenericContext()
  # --------------------------------------------------------------------
  def listDiagnostics(self):
    return self.log.listMessages()
  # --------------------------------------------------------------------
  def checkLeafStructure(self,T,path,node):
    stt=CGM.CHECK_GOOD
    try:
      CGU.checkNode(node,dienow=True)
    except CGE.CE.cgnsNameError(1):
      stt=CGM.CHECK_FAIL
      self.log.push(path,CGM.CHECK_FAIL,NODE_EMPTYLIST)
    except CGE.CE.cgnsNameError(2):
      stt=CGM.CHECK_FAIL
      self.log.push(path,CGM.CHECK_FAIL,NODE_NOTALIST)
    except CGE.CE.cgnsNameError(3):
      stt=CGM.CHECK_FAIL
      self.log.push(path,CGM.CHECK_FAIL,NODE_NAMENOTASTRING)
    except CGE.CE.cgnsNameError(4):
      stt=CGM.CHECK_FAIL
      self.log.push(path,CGM.CHECK_FAIL,NODE_CHILDRENNOTLIST)
    except CGE.CE.cgnsNameError(5):
      stt=CGM.CHECK_FAIL
      self.log.push(path,CGM.CHECK_FAIL,NODE_BADDATA)
    return stt
  # --------------------------------------------------------------------
  def checkLeaf(self,T,path,node):
    if (not hasattr(self,'methods')):
      self.methods=[]
      for m in inspect.getmembers(self):
        if ((m[0][-2:]=='_t') or (m[0][-7:]=='_n')): self.methods+=[m[0]]
    parent=CGU.getParentFromNode(T,node)
    status1=self.checkSingleNode(T,path,node,parent)
    status2=status1
    ntype=CGU.getTypeAsGrammarToken(node[3])
    if ((len(node)==4) and (ntype in self.methods)):
      status2=apply(getattr(self,ntype),[path,node,parent,T,self.log])
    status1=CGM.getWorst(status1,status2)
    return status1
  # --------------------------------------------------------------------
  def checkSingleNode(self,T,path,node,parent):
    stt=CGM.CHECK_GOOD
    if (not CGU.checkNodeName(node)):
      stt=self.log.push(path,CGM.CHECK_FAIL,INVALID_NAME,node[0])
    lchildren=CGU.childNames(parent)
    if (lchildren):
      lchildren.remove(node[0])
      if (node[0] in lchildren):
        stt=self.log.push(path,CGM.CHECK_FAIL,DUPLICATED_NAME,node[0])
    tlist=CGU.getNodeAllowedChildrenTypes(parent,node)
    if (CGU.getTypeAsGrammarToken(node[3]) not in tlist):
      if (parent is not None):
        stt=self.log.push(path,CGM.CHECK_FAIL,INVALID_SIDSTYPE_P,
                          node[3],parent[3])
      else:
        stt=self.log.push(path,CGM.CHECK_FAIL,INVALID_SIDSTYPE,node[3])
    dlist=CGU.getNodeAllowedDataTypes(node)
    dt=CGU.getValueDataType(node)
    if (dt not in dlist):
      stt=self.log.push(path,CGM.CHECK_FAIL,INVALID_DATATYPE,dt)
    if (node[3] not in CGT.types.keys()):
      stt=self.log.push(path,CGM.CHECK_FAIL,UNKNOWN_SIDSTYPE,node[3])
    else:
      stt=self.checkCardinalityOfChildren(T,path,node,parent)
    return stt
  # --------------------------------------------------------------------
  def checkTree(self,T,trace=False):
    status1=CGM.CHECK_GOOD
    if (trace): print '### Parsing node paths...'
    paths=CGU.getPathFullTree(T)
    sz=len(paths)
    ct=1
    for path in paths:
      if (trace): print '### Check node [%.6d/%.6d]\r'%(ct,sz),
      node=CGU.getNodeByPath(T,path)
      status2=CGM.CHECK_GOOD
      if (node is None):
        status2=self.log.push(path,CGM.CHECK_FAIL,INVALID_PATH,path)
      if (status2==CGM.CHECK_GOOD):
        status2=self.checkLeafStructure(T,path,node)
      if (status2==CGM.CHECK_GOOD):
        status2=self.checkLeaf(T,path,node)
      status1=status2
      ct+=1
    return status1
  # --------------------------------------------------
  def checkCardinalityOfChildren(self,T,path,node,parent):
      stt=CGM.CHECK_GOOD
      for child in node[2]:
        card=CGT.types[node[3]].cardinality(child[3])
        if (card==CGT.C_00):
          stt=CGM.CHECK_FAIL
          cpath='%s/%s'%(path,child[0])
          self.log.push(path,stt,FORBIDDEN_CHILD,cpath,child[3])
        if (card in [CGT.C_11,CGT.C_01]):
          if ([c[3] for c in node[2]].count(child[3])>1):
            stt=CGM.CHECK_FAIL
            self.log.push(path,stt,SINGLE_CHILD,child[0],child[3])
      for tchild in CGT.types[node[3]].children:
        card=CGT.types[node[3]].cardinality(tchild[0])
        if (card in [CGT.C_11,CGT.C_1N]):
          if ([c[3] for c in node[2]].count(tchild[0])<1):
            stt=CGM.CHECK_FAIL
            self.log.push(path,stt,MANDATORY_CHILD,tchild[1],tchild[0])
      return stt
  # --------------------------------------------------------------------
  def CGNSLibraryVersion_t(self,pth,node,parent,tree,log):
    stt=CGM.CHECK_OK
    try:
      version=int(node[1][0]*1000)
      if (version < 2400):
        stt=CGM.CHECK_FAIL
        self.log.push(pth,stt,OLD_VERSION,version)
    except Exception:
      stt=CGM.CHECK_FAIL
      self.log.push(pth,stt,BAD_VERSION)
    return stt
      
# --- last line
