#  ---------------------------------------------------------------------------
#  pyCGNS.PAT - Python package for CFD General Notation System - PATternMaker
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#  $Release$
#  ---------------------------------------------------------------------------

import CGNS.PAT.cgnskeywords as CK
import CGNS.PAT.cgnstypes    as CT
import CGNS.PAT.cgnserrors   as CE
import CGNS

import numpy as NPY
import os.path
import string

# -----------------------------------------------------------------------------
# support functions
# -----------------------------------------------------------------------------
def checkName(name):
  if (type(name) != type("s")): raise CE.cgnsException(22)
  if (len(name) == 0):          raise CE.cgnsException(23)
  if ('/' in name):             raise CE.cgnsException(24)
  if (len(name) > 32):          raise CE.cgnsException(25)

# -----------------------------------------------------------------------------
def checkDuplicatedName(parent,name):
  if (not parent): return
  if (parent[2] == None): return
  checkName(name)
  for nc in parent[2]:
    if (nc[0] == name): raise CE.cgnsException(102,(name,parent[0]))

# -----------------------------------------------------------------------------
def concatenateForArrayChar(nlist):
  nl=[]
  for n in nlist:
    if (type(n)==type('')): nl+=[setStringAsArray(("%-32s"%n)[:32])]
    else:
      checkArrayChar(n)
      nl+=[setStringAsArray(("%-32s"%n.tostring())[:32])]
  r=NPY.array(NPY.array(nl,order='Fortran').T,order='Fortran')
  return r

# -----------------------------------------------------------------------------
def getValueType(v):
  if (v == None): return None
  if (type(v) == type(NPY.array((1,)))):
    if (v.dtype.kind in ['S','a']): return CK.Character_s
    if (v.dtype.char in ['f']):     return CK.RealSingle_s
    if (v.dtype.char in ['d']):     return CK.RealDouble_s
    if (v.dtype.char in ['i']):     return CK.Integer_s
    if (v.dtype.char in ['l']):     return CK.Integer_s
  return None
   
# -----------------------------------------------------------------------------
def setValue(node,value):
  t=getValueType(value)
  if (t == None): node[1]=None
  if (t in [CK.Integer_s,CK.RealDouble_s,
            CK.RealSingle_s,CK.Character_s]): node[1]=value
  return node
  
# -----------------------------------------------------------------------------
def setStringAsArray(a):
  if ((type(a)==type(NPY.array((1))))
      and (a.shape != ()) and (a.dtype.char=='S')):
    return a
  if ((type(a)==type("")) or (type(a)==type(NPY.array((1))))):
    return NPY.array(tuple(a),dtype='|S',order='Fortran')
  return None

# -----------------------------------------------------------------------------
# useless
def getValue(node):
  v=node[1]
  t=getValueType(v)
  if (t == None):           return None
  if (t == CK.Integer_s):    return v
  if (t == CK.RealDouble_s): return v
  if (t == CK.Character_s):  return v
  return v
  
# -----------------------------------------------------------------------------
def checkArray(a):
  return
  if (type(a) != type(NPY.array((1)))): raise CE.cgnsException(109)
  if ((len(a.shape)>1) and not NPY.isfortran(a)):
    raise CE.cgnsException(710)  

# -----------------------------------------------------------------------------
def checkArrayChar(a):
  checkArray(a)
  if (a.dtype.char not in ['S','a']):  raise CE.cgnsException(105)
  return a

# -----------------------------------------------------------------------------
def checkArrayReal(a):
  checkArray(a)
  if (a.dtype.char not in ['d','f']):  raise CE.cgnsException(106)
  return a

# -----------------------------------------------------------------------------
def checkArrayInteger(a):
  checkArray(a)
  if (a.dtype.char not in ['i','u']):  raise CE.cgnsException(107)
  return a

# -----------------------------------------------------------------------------
def checkType(parent,stype,name):
  if (parent == None): return None
  if (parent[3] != stype): 
    raise CE.cgnsException(103,(name,stype))
  return None

# -----------------------------------------------------------------------------
def checkParentType(parent,stype):
  if (parent == None):     return False
  if (parent[3] != stype): return False
  return True

# -----------------------------------------------------------------------------
def checkTypeList(parent,ltype,name):
  if (parent == None): return None
  if (parent[3] not in ltype): 
    raise CE.cgnsException(104,(name,ltype))
  return None

# -----------------------------------------------------------------------------
def checkParent(node,dienow=0):
  if (node == None): return 1
  return checkNode(node,dienow)

# -----------------------------------------------------------------------------
def checkNode(node,dienow=0):
  if (node in [ [], None ]):
    if (dienow): raise CE.cgnsException(1)
    return 0
  if (type(node) != type([3,])):
    if (dienow): raise CE.cgnsException(2)
    return 0
  if (len(node) != 4):
    if (dienow): raise CE.cgnsException(2)
    return 0
  if (type(node[0]) != type("")):
    if (dienow): raise CE.cgnsException(3)
    return 0
  if (type(node[2]) != type([3,])):
    if (dienow): raise CE.cgnsException(4,node[0])
    return 0
  if ((node[1] != None) and (type(node[1])) != type(NPY.array([3,]))):
    if (dienow): raise CE.cgnsException(5,node[0])
    return 0
  return 1
    
# -----------------------------------------------------------------------------
def isRootNode(node,dienow=0):
  if (node in [ [], None ]):         return 0
  versionfound=0
  if (not checkNode(node)):          return 0
  for n in node[2]:
     if (not checkNode(n,dienow)):   return 0 
     if (     (n[0] == CK.CGNSLibraryVersion_s)
          and (n[3] == CK.CGNSLibraryVersion_ts) ):
         if versionfound: raise CE.cgnsException(99)
         versionfound=1
     elif ( n[3] != CK.CGNSBase_ts ): return 0
  if (versionfound):                 return 1
  else:                              return 0
       
# -----------------------------------------------------------------------------
# Arbitrary and incomplete node comparison (lazy evaluation)
def sameNode(nodeA,nodeB):
  if (not (checkNode(nodeA) and checkNode(nodeB))): return 0
  if (nodeA[0] != nodeB[0]):                        return 0
  if (nodeA[3] != nodeB[3]):                        return 0
  if (type(nodeA[1]) != type(nodeB[1])):            return 0
  if (len(nodeA[1])  != len(nodeB[1])):             return 0
  if (len(nodeA[2])  != len(nodeB[2])):             return 0
  # no recursion on children yet 
  return 1      

# -----------------------------------------------------------------------------
def newNode(name,value,children,type,parent=None):
  node=[name, value, children, type]
  if (parent): parent[2].append(node)
  return node

# --------------------------------------------------
def hasFortranFlag(node):
  if (node[1]==None): return 1
  if (node[1]==[]):   return 1
  if (type(node[1])==type('')): return 1 # link
  if (not node[1].shape): return 1
  if (len(node[1].shape)==1): return 1  
  return NPY.isfortran(node[1])

# --------------------------------------------------
def getNodeShape(node):
  r="-"
  if   (node[1]==None): r="-"
  elif (node[1]==[]):   r="-"
  elif (node[3]==''):   r="-"
  elif (node[1].shape in ['',(0,),()]): r="[0]"
  else: r=str(list(node[1].shape))
  return r

# --------------------------------------------------
def getNodeType(node):
  data=node[1]
  if (node[0] == 'CGNSLibraryVersion_t'):
    return CK.R4 # ONLY ONE R4 IN ALL SIDS !
  if ( data in [None, []] ):
    return CK.MT
  if ( (type(data) == type(NPY.ones((1,)))) ):
    if (data.dtype.char in ['S','c','s']):    return CK.C1
    if (data.dtype.char in ['f','F']):        return CK.R4
    if (data.dtype.char in ['D','d']):        return CK.R8
    if (data.dtype.char in ['l','i','I']):    return CK.I4
  if ((type(data) == type([])) and (len(data))): # oups !
    if (type(data[0]) == type("")):           return CK.C1 
    if (type(data[0]) == type(0)):            return CK.I4 
    if (type(data[0]) == type(0.0)):          return CK.R8
  return '??'

# --------------------------------------------------
def checkPath(path):
  return True

# --------------------------------------------------
def removeFirstPathItem(path):
  p=path.split('/')
  if ((p[0]=='') and (p>2)):
    return string.join(['']+p[2:],'/')
  elif (p>1):
    return string.join(p[1:],'/')
  else:
    return '/'

# --------------------------------------------------
def getValueByPath(node, path):
  """node : Root node of the tree parse
     path : Path string of the target node

     Gets the value of the CGNS/Python node with name 'path'.
     Returns None if the node is not found.
  """
  n=getNodeFromPath(path.split('/'),node)
  if (n==-1): return None
  return n[1]

# --------------------------------------------------
def getNodeByPath(path,tree):
  if (not checkPath(path)): return None
  if (path[0]=='/'): path=path[1:]
  if (tree[3]==CK.CGNSTree_ts):
    path=string.join(path.split('/')[1:],'/')
  n=getNodeFromPath(path.split('/'),tree)
  if (n==-1): return None
  return n

# --------------------------------------------------
def nodeByPath(path,tree):
  if (not checkPath(path)): return None
  if (path[0]=='/'): path=path[1:]
  if (tree[3]==CK.CGNSTree_ts):
    path=string.join(path.split('/')[1:],'/')
    n=getNodeFromPath(path.split('/'),tree)
  else:
    n=getNodeFromPath(path.split('/'),[None,None,[tree],None])
  if (n==-1): return None
  return n

# --------------------------------------------------
def removeChildByName(parent,name):
  for n in range(len(parent[2])):
    if (parent[2][n][0] == name):
        del parent[2][n]
        return None
  return None

# --------------------------------------------------
def removeNodeFromPath(path,node):
  target=getNodeFromPath(path,node)
  if (len(path)>1):
    father=getNodeFromPath(path[:-1],node)
    father[2].remove(target)
  else:
    # Root node child
    for c in node[2]:
      if (c[0] == path[0]): node[2].remove(target)
    
# --------------------------------------------------
def getNodeFromPath(path,node):
  """path : path to look for as a list of strings
     node : root node for the tree parse
  """
  for c in node[2]:
    if (c[0] == path[0]):
      if (len(path) == 1): return c
      return getNodeFromPath(path[1:],c)
  return -1

# --------------------------------------------------
def getPathFromNode(node,rootnode,path=''):
  if (node == rootnode):
    return path
  for c in rootnode[2]:
    p=getPathFromNode(node,c,path+'/'+c[0])
    if (p): return p
  return None

# --------------------------------------------------
def getAllNodesByTypeList(typelist,tree):
  if (tree[3] != typelist[0]): return None
  n=getAllNodesFromTypeList(typelist[1:],tree[2],"%s"%tree[0],[])
  if (n==-1): return None
  return n

# --------------------------------------------------
def getAllNodesFromTypeList(typelist,node,path,result):
  for c in node:
    if (c[3] == typelist[0]):
      if (len(typelist) == 1):
        result.append("%s/%s"%(path,c[0]))
      else:
        getAllNodesFromTypeList(typelist[1:],c[2],"%s/%s"%(path,c[0]),result)
  return result

# --------------------------------------------------
def getPaths(tree,path,plist):
  for c in tree[2]:
    plist.append(path+'/'+c[0])
    getPaths(c,path+'/'+c[0],plist)
 
# --------------------------------------------------   
def getAllPaths(tree):
  plist=[]
  path=''
  getPaths(tree,path,plist)
  return plist

# --------------------------------------------------
def childNames(node):
  r=[]
  if (node == None): return r
  for c in node[2]:
    r.append(c[0])
  return r

# --------------------------------------------------
def getAllNodesByTypeList(typelist,tree):
  if (tree[3]==CK.CGNSTree_ts): start="/%s"%tree[0]
  else:                         start="%s"%tree[0]
  n=getAllNodesFromTypeList(typelist,tree[2],start,[])
  return n

# --------------------------------------------------
def getAllNodesByTypeList2(typelist,tree):
  if (tree[3]==CK.CGNSTree_ts): start=""
  else:                         start="%s"%tree[0]
  n=getAllNodesFromTypeList(typelist,tree[2],start,[])
  return n

# --------------------------------------------------
def getAllNodesFromTypeList(typelist,node,path,result):
  for c in node:
    if (c[3] in typelist):
      result.append("%s/%s"%(path,c[0]))
    getAllNodesFromTypeList(typelist,c[2],"%s/%s"%(path,c[0]),result)
  return result

# --------------------------------------------------
def getNodeAllowedChildrenTypes(pnode,node):
  tlist=[]
  if (node[3] == CK.CGNSTree_ts): return tlist
  if (node[3] == None): return [CK.CGNSBase_ts,CK.CGNSLibraryVersion_ts]
  try:
    for cn in CT.types[pnode[3]].children:
      if (cn[0] not in tlist): tlist+=[cn[0]]
  except:
    pass
  return tlist

# --------------------------------------------------
def getNodeAllowedDataTypes(node):
  tlist=[]
  try:
    tlist=CT.types[node[3]].datatype
  except:
    pass
  return tlist

# -----------------------------------------------------------------------------
def hasChildrenType(parent,ntype):
  if (not parent): return None
  r=[]
  for n in parent[2]:
    if (n[3] == ntype): r.append(n)
  if r: return r
  return None

# -----------------------------------------------------------------------------
def hasChildName(parent,name):
  if (not parent): return None
  for n in parent[2]:
    if (n[0] == name): return n
  return None

# --------------------------------------------------
def hasChildNodeOfType(node,ntype):
  if (node == None): return 0
  for cn in node[2]:
    if (cn[3]==ntype): return 1
  return 0

# --------------------------------------------------
def stringValueMatches(node,reval):
  if (node == None):             return 0
  if (node[1] == None):          return 0  
  if (getNodeType(node)!=CK.C1): return 0
  tn=type(node[1])
  if   (tn==type('')): vn=node[1]
  elif (tn == type(NPY.ones((1,))) and (node[1].dtype.char in ['S','c','s'])):
    vn=node[1].tostring()
  else: return 0
  return re.match(reval,vn)

# --------------------------------------------------
def checkLinkFile(lkfile,lksearch=['']):
  found=(None,None)
  if (lksearch==[]): lksearch=['']
  for spath in lksearch:
    sfile=os.path.normpath(spath+'/'+lkfile)
    if (os.path.exists(sfile)):
      found=(os.path.normpath(spath),os.path.normpath(lkfile))
      break
  return found
    
# --------------------------------------------------
def copyArray(a):
  if (a==None): return None
  if (a==[]):   return None
  if (NPY.isfortran(a)):
    b=NPY.array(a,order='Fortran')
  else:
    b=NPY.array(a)
  return b

# --------------------------------------------------
def copyNode(n):
  newn=[n[0],copyArray(n[1]),deepcopyNodeList(n[2]),n[3]]
  return newn

# --------------------------------------------------
def deepcopyNodeList(la):
  if (not la): return la
  ra=[]
  for a in la:
    ra.append(copyNode(a))
  return ra

# ----
