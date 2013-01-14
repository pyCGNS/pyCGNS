#  ---------------------------------------------------------------------------
#  pyCGNS.PAT - Python package for CFD General Notation System - PATternMaker
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------

import CGNS
import CGNS.PAT.cgnskeywords as CK
import CGNS.PAT.cgnstypes    as CT
import CGNS.PAT.cgnserrors   as CE

import numpy as NPY

import os.path
import string
import re

# -----------------------------------------------------------------------------
def nodeCreate(name,value,children,type,parent=None,dienow=False):
  """
  Create a new node with and bind it to its parent::

    import CGNS.PAT.cgnskeywords as CK

    n=createNode('Base',numpy([3,3]),[],CK.CGNSBase_ts)
    z=createNode('ReferenceState',None,[],CK.ReferenceState_ts,parent=n)

  - Args:
   * `name`: node name as a string
   * `value`: node value as a numpy array
   * `children`: list of node children 
   * `type`: CGNS type as a string
   * `parent`: parent node where to insert the new node (default: None)
   * `dienow`: If True raises an exception in case of problem (default: False)

  - Return:
   * The new node

  - Remarks:
   * If parent is None (default) node is orphan
   * Full-checks the node with `checkNodeCompliant` only if `dienow` is True.

  """
  return newNode(name,value,children,type,parent,dienow)
    
# --------------------------------------------------
def nodeCopy(node,newname=None):
  """
  Creates a new node sub-tree as a copy of the argument node sub-tree.
  A deep copy is performed on the node, including the values, which can
  lead to a very large memory use::

    n1=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity')
    n2=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity/Connect1')
    n3=nodeCopy(n2,'Connect2')
    nodeChild(n1,n3)

  - Args:
   * `node`: node to copy
   * `name`: new node (copy) name

  - Return:
   * The new node

  - Remarks:
   * Full-checks the node with `checkNodeCompliant` only if `dienow` is True.
   * The new node name is the same by default, thus user would have to check
     for potential duplicated name.
   
  """
  if (newname is None): newname=node[0]
  return copyNode(node,newname)

def copyNode(n,newname):
  newn=[newname,copyArray(n[1]),deepcopyNodeList(n[2]),n[3]]
  return newn

# --------------------------------------------------
def nodeDelete(tree,node,legacy=False):
  """
  Deletes a node from a tree::

    import CGNS.PAT.cgnslib as CL
        
    T =CL.newCGNSTree()
    b1=CL.newBase(T,'Base',3,3)
    z1=CL.newZone(b1,'Zone1',numpy.array([1,2,3]))
    z2=CL.newZone(b1,'Zone2',numpy.array([1,2,3]))
    print getPathFullTree(T)
    # ['/CGNSLibraryVersion', '/Base', '/Base/Zone1', '/Base/Zone1/ZoneType', '/Base/Zone2', '/Base/Zone2/ZoneType']
    nodeDelete(T,z1)
    print getPathFullTree(T)
    # ['/CGNSLibraryVersion', '/Base', '/Base/Zone2', '/Base/Zone2/ZoneType']

  - Args:
   * `tree`: target tree where to find the node to remove
   * `node`: node to remove (actual CGNS/Python node or node name as absolute path)

  - Return:
   * The tree argument (without the deleted node)

  - Remarks:
   * Uses :py:func:`checkSameNode`.
   * The actual memory of the node only if no other reference to this node is found by Python.
   
  """
  if (type(node)==str): path=node
  else: path=getPathFromNode(tree,node)
  if (path is not None):
    pp=getPathAncestor(path)
    pc=getPathLeaf(path)
    if (pp!=path):
      np=getNodeByPath(tree,pp)
      if (np is not None):
        removeChildByName(np,pc)
  return tree

# --------------------------------------------------
def deepcopyNodeList(la):
  if (not la): return la
  ra=[]
  for a in la:
    ra.append(copyNode(a,a[0]))
  return ra

# -----------------------------------------------------------------------------
# support functions
# -----------------------------------------------------------------------------
def checkNodeName(node,dienow=False):
  """
  Checks if the name is CGNS/Python compliant node name:

  .. highlight:: python

     if (not checkNodeName(node)):
        print 'Such name ',name',' not allowed'

  - Args:
   * `node`: the CGNS/Python node to check

  - Return:
   * True if the name has a correct syntax

  - Remarks:
   * The function is the same as :py:func:`checkName` but with a node as
     arg instead of string
   * see also :py:func:`checkNodeCompliant`
  
  """
  if (len(node)!=4): 
    if (dienow): raise CE.cgnsNameError(2)
    return False
  return checkName(node[0],dienow)

def checkName(name,dienow=False):
  """
  Checks if the name is CGNS/Python compliant node name.

  .. code-block:: python
         
     if (not checkName(name)):
        print 'Such name ',name',' not allowed'

  - Args:
   * `name`: the string to check

  - Return:
   * True if the name has a correct syntax

  - Remarks:
   * Type of name should be a Python string
   * Name cannot be empty
   * No '/' in the name
   * A single '.' or '..' are not allowed
   * A name with ' ' as first char is not allowed
   * A name with only ' ' is not allowed
   * Raises :ref:`cgnsnameerror` codes 22,23,24,25,29,31,32 if `dienow` is True
  
  """
  if (type(name) not in [str, unicode]):
    if (dienow): raise CE.cgnsNameError(22)
    return False
  if (len(name) == 0):
    if (dienow): raise CE.cgnsNameError(23)
    return False
  if ('/' in name):
    if (dienow): raise CE.cgnsNameError(24,name)
    return False
  if (len(name) > 32):
    if (dienow): raise CE.cgnsNameError(25,name)
    return False
  if (name in ['.','..']):
    if (dienow): raise CE.cgnsNameError(29)
    return False
  if (name.count(' ')==len(name)):
    if (dienow): raise CE.cgnsNameError(31)
    return False
  if (name[0]==' '):
    if (dienow): raise CE.cgnsNameError(32)
    return False
  return True

# --------------------------------------------------
def setChild(parent,node):
  """
  Adds a child node to the parent node children list::

    n1=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity')
    n2=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity/Connect1')
    n3=nodeCopy(n2)
    setChild(n1,n3)
    
  - Args:
   * `parent`: the parent node
   * `node`: the child node to add to parent

  - Return:
   * The parent node

  - Remarks:
   * No check is performed on duplicated child name or any other validity.

  """
  parent[2].append(node)
  return parent

# -----------------------------------------------------------------------------
def checkDuplicatedName(parent,name,dienow=False):
  """
  Checks if the name is not already in the children list of the parent::

    count=1
    while (not checkDuplicatedName(node,'solution#%.3d'%count)): count+=1

  - Args:
   * `parent`: the parent node
   * `name`:   the child name to look for

  - Return:
   * True if the child *IS NOT* duplicated
   * False if the child *IS* duplicated

  - Remarks:
   * Sorry about the legacy interface, True means not ok...
     (see :py:func:`checkHasChildName`)
   * Raises :ref:`cgnsnameerror` code 102 if `dienow` is True
  
  """
  if (not parent): return True
  if (parent[2] == None): return True
  for nc in parent[2]:
    if (nc[0] == name):
      if (dienow): raise CE.cgnsNameError(102,(name,parent[0]))
      return False
  return True

# -----------------------------------------------------------------------------
def checkHasChildName(parent,name,dienow=False):
  """
  Checks if the name is in the children list of the parent::

    count=1
    while (checkHasChildName(node,'solution#%.3d'%count)): count+=1

  - Args:
   * `parent`: the parent node
   * `name`:   the child name to look for

  - Return:
   * True if the child exists

  - Remarks:
   * Raises :ref:`cgnsnameerror` code 102 if `dienow` is True
  
  """
  return not checkDuplicatedName(parent,name,dienow)

# -----------------------------------------------------------------------------
def checkNodeType(node,cgnstype=[],dienow=False):
  """
  Check the CGNS type of a node. The type can be a single value or
  a list of values. Each type to check is a string such as
  `CGNS.PAT.cgnskeywords.CGNSBase_ts` constant for example.
  If the list is empty, the check uses the list of all existing CGNS types::
     
    import CGNS.PAT.cgnskeywords as CK

    n=createNode('Base',numpy([3,3]),[],CK.CGNSBase_ts)
    checkNodeType(n)
    checkNodeType(n,['Zone_t',CK.CGNSBase_t])
    
  - Args:
   * `node`: the CGNS/Python node to check
   * `cgnstype`: a list of strings with the types to check

  - Return:
   * True if the type is a CGNSType or a type in the argument list.
   * None if the parent is None (`may change to have consistent return`)

  - Remarks:
   * raises :ref:`cgnstypeerror` codes 103,104,40 if `dienow` is True
  
  """
  return checkType(node,cgnstype,'',dienow)

def checkType(parent,ltype,name,dienow=False):
  if (parent == None): return None
  if ((ltype==[]) and (parent[3] in CK.cgnstypes)):
    return True
  if ((type(ltype)==list) and (parent[3] in ltype)):
    return True
  if (parent[3] == ltype): 
    return True
  if ((ltype==[]) and (parent[3] not in CK.cgnstypes)):
    if (dienow): raise CE.cgnsTypeError(40,(parent,parent[3]))
    return False
  if (parent[3] != ltype): 
    if (dienow): raise CE.cgnsTypeError(103,(parent,ltype))
    return False
  if ((type(ltype)==list) and (parent[3] not in ltype)): 
    if (dienow): raise CE.cgnsTypeError(104,(parent,ltype))
    return False
  return True

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
def checkNode(node,dienow=False):
  """
  Checks if a node is a compliant CGNS/Python node structure of list.
  The syntax for a compliant node structure is:
  
  .. code-block:: c

    [<name:string>, <value:numpy>, <children:list-of-nodes>, <cgnstype:string>]

  With the following exception: a `value` can be None.

  The function checks the syntax of the node and the types of its contents,
  it doesn't perform sub checks such as `checkNodeName`, `checkNodeType`...

  You should always check first the node structure, a function such as
  `checkNodeName` blindly access to the first item of the list and would raise
  an exception if the structure is not correct.

  - Args:
   * `node`: the CGNS/Python node to check

  - Return:
   * True if the node is ok

  - Remarks:
   * see also :py:func:`checkNodeCompliant`
   * Raises :ref:`cgnsnodeerror` codes 1,2,3,4,5 if `dienow` is True
  
  """
  if (node in [ [], None ]):
    if (dienow): raise CE.cgnsException(1)
    return False
  if (type(node) != type([3,])):
    if (dienow): raise CE.cgnsException(2)
    return False
  if (len(node) != 4):
    if (dienow): raise CE.cgnsException(2)
    return False
  if (type(node[0]) not in [str, unicode]):
    if (dienow): raise CE.cgnsException(3)
    return False
  if (type(node[2]) != type([3,])):
    if (dienow): raise CE.cgnsException(4,node[0])
    return False
  if ((node[1] != None) and (type(node[1])) != type(NPY.array([3,]))):
    if (dienow): raise CE.cgnsException(5,node[0])
    return False
  return True
    
# -----------------------------------------------------------------------------
def checkRootNode(node,legacy=False,dienow=False):
  """
  Checks if a node is the CGNS/Python tree root node.
  If `legacy` is True, then `[None, None, [children], None]` is
  accepted as Root. If it is not True (default) then a CGNS/Python node
  of type `CGNSTree_t` is expected as root node.
  Children contains then the `CGNSLibraryVersion`
  and `CGNSBase` nodes as flat list.
  The `flat` pattern with a list containing `CGNSLibraryVersion`
  and zero or more `CGNSBase` nodes is not accepted. You should use a trick as
  the one above by giving this pattern as child of a fake node.

  .. code-block:: python
         
     # flatpattern is a CGNS/Python list with a `CGNSLibraryVersion` node
     # and `CGNSBase` nodes at the same level.
     # we build a temporary children list
   
     tmp=[None, None, [flatpattern], None]
     if (checkRootNode(tmp,True)):
       # do smomething

  - Args:
   * `node`: the CGNS/Python node to check
   * `legacy`: boolean, True means you accept non-CGNSTree_t node

  - Return:
   * True if the node is a root node

  - Remarks:
   * Raises :ref:`cgnsnodeerror` codes 90,91,99 if `dienow` is True
  
  """
  return isRootNode(node,legacy,dienow)

def isRootNode(node,legacy=False,version=False,dienow=False):
  if (node in [ [], None ]):
    if (dienow): raise CE.cgnsNodeError(90)
    return False
  versionfound=0
  if (not checkNode(node,dienow)): return False
  start=[None,None,[node],None]
  if ((not legacy) and (node[3] != CK.CGNSTree_ts)): return False
  if ((not legacy) and (node[3] == CK.CGNSTree_ts)): start=node
  if (legacy): start=node
  for n in start[2]:
     if (not checkNode(n,dienow)): return False 
     if (     (n[0] == CK.CGNSLibraryVersion_s)
          and (n[3] == CK.CGNSLibraryVersion_ts) ):
         if versionfound and dienow:
           raise CE.cgnsNodeError(99)
           return False
         versionfound=1
         if (version and n[1] is not None): return n[1][0]
     elif ( n[3] != CK.CGNSBase_ts ):
       if (dienow): raise CE.cgnsNodeError(91)
       return False
  if (versionfound): return True
  else:              return True
       
def getVersion(tree):
  v=isRootNode(tree,version=True)
  if (v): return v
  return 0.0

# -----------------------------------------------------------------------------
# Arbitrary and incomplete node comparison (lazy evaluation)
def checkSameNode(nodeA,nodeB,dienow=False):
  """
  Checks if two nodes have the same contents: same name, same CGNS/SIDS type,
  same number of children, same value type (None of numpy).
  The children are not parsed, the value itself is not checked
  (see :py:func:`checkSameValue`)::

    if (checkSameNode(nodeA,nodeB)):
      nodeB=copyNode(nodeB)

  - Args:
   * `node A`: CGNS/Python node
   * `node B`: CGNS/Python node

  - Return:
   * True if the nodes are same

  - Remarks:
   * Raises :ref:`cgnsnodeerror` code 30 if `dienow` is True

  """
  return sameNode(nodeA,nodeB,dienow)
  
def sameNode(nodeA,nodeB,dienow=False):
  same=True
  if (not (checkNode(nodeA) and checkNode(nodeB))): same=False
  elif (nodeA[0] != nodeB[0]):                      same=False
  elif (nodeA[3] != nodeB[3]):                      same=False
  elif (type(nodeA[1]) != type(nodeB[1])):          same=False
  elif (not checkSameValue(nodeA,nodeB)):           same=False
  elif (len(nodeA[2]) != len(nodeB[2])):            same=False
  if (not same and dienow):
    raise cgnsNodeError(30,(nodeA[0],nodeB[0]))
  return same

# -----------------------------------------------------------------------------
def checkSameValue(nodeA,nodeB,dienow=False):
  """
  Checks if two nodes have the same value. There is no tolerance on actual
  array values when these are compared one to one. This could lead to time
  consuming operations for large arrays.

  .. code-block:: python
         
     if (checkSameValue(nodeA,nodeB)):
       # do something

  - Args:
   * `node A`: CGNS/Python node
   * `node B`: CGNS/Python node

  - Return:
   * True if the nodes have same value

  - Remarks:
   * Raises :ref:`cgnsnodeerror` code 30 if `dienow` is True

  """
  vA=nodeA[1]
  vB=nodeB[1]
  if ((vA is None) and (vB is None)):     return True
  if ((vA is None) and (vB is not None)): return False
  if ((vA is not None) and (vB is None)): return False
  if ((type(vA)==NPY.ndarray) and (type(vB)!=NPY.ndarray)): return False
  if ((type(vA)!=NPY.ndarray) and (type(vB)==NPY.ndarray)): return False
  if ((type(vA)==NPY.ndarray) and (type(vB)==NPY.ndarray)):
    if (vA.dtype != vB.dtype): return False
    if (vA.shape != vB.shape): return False
    return NPY.all(NPY.equal(vA,vB))
  return (vA == vB)

# -----------------------------------------------------------------------------
def checkArray(a,dienow=False):
  """
  Check if the array value of a node is a numpy array.

  .. code-block:: python
         
     if (checkArray(node[1])):
       # do something
 
  - Args:
   * `a`: value to check

  - Return:
   * True if the array is suitable as CGNS/Python value

  - Remarks:
   * Raises error codes 109,170 if `dienow` is True

  """
  if (type(a) != type(NPY.array((1)))):
    if (dienow): raise CE.cgnsException(109)
    return False
  if ((len(a.shape)>1) and not NPY.isfortran(a)):
    if (dienow): raise CE.cgnsException(710)
    return False
  return True

# -----------------------------------------------------------------------------
def checkArrayChar(a,dienow=False):
  checkArray(a)
  if (a.dtype.char not in ['S','a']):  raise CE.cgnsException(105)
  return a

# -----------------------------------------------------------------------------
def checkArrayReal(a,dienow=False):
  checkArray(a)
  if (a.dtype.char not in ['d','f']):  raise CE.cgnsException(106)
  return a

# -----------------------------------------------------------------------------
def checkArrayInteger(a,dienow=False):
  checkArray(a)
  if (a.dtype.char not in ['i','u']):  raise CE.cgnsException(107)
  return a

# -----------------------------------------------------------------------------
def checkNodeCompliant(node,parent,dienow=False):
  """
  Performs all possible checks on a node. Can raise any of the exceptions
  related to node checks (:py:func:`checkNodeName`, :py:func:`checkNodeType`,
  :py:func:`checkArray`...).
  
  .. code-block:: python
         
     if (checkNodeCompliant(node)):
       # do something

  - Args:
   * `node`: CGNS/Python node to check
   * `parent`: CGNS/Python parent node to check

  - Return:
   * True if all controls are ok

  - Remarks:
   * Calls :py:func:`checkNode`,:py:func:`checkNodeName`,
     :py:func:`checkDuplicatedName`, :py:func:`checkArray`,
     :py:func:`checkNodeType`

  """
  r=checkNode(node,dienow=dienow)\
    and checkNodeName(node,dienow=dienow)\
    and checkDuplicatedName(parent,node[0],dienow=dienow)\
    and checkArray(node[2],dienow=dienow)\
    and checkNodeType(node,dienow=dienow)
  return r

# -----------------------------------------------------------------------------
def newNode(name,value,children,type,parent=None,dienow=False):
  node=[name, value, children, type]
  if (dienow): checkNodeCompliant(node,parent,dienow)
  if (parent): parent[2].append(node)
  return node

# -----------------------------------------------------------------------------
def concatenateForArrayChar(nlist):
  nl=[]
  for n in nlist:
    if (type(n)==type('')): nl+=[setStringAsArray(("%-32s"%n)[:32])]
    else:
      checkArrayChar(n)
      nl+=[setStringAsArray(("%-32s"%n.tostring())[:32])]
  r=NPY.array(NPY.array(nl,order='F').T,order='F')
  return r

# -----------------------------------------------------------------------------
def concatenateForArrayChar2D(nlist):
  return concatenateForArrayChar(nlist)

# -----------------------------------------------------------------------------
def concatenateForArrayChar3D(nlist):
  rr=[]
  for p in nlist:
    nl=[]
    for n in p:
      if (type(n)==type('')): nl+=[setStringAsArray(("%-32s"%n)[:32])]
      else:
        checkArrayChar(n)
        nl+=[setStringAsArray(("%-32s"%n.tostring())[:32])]
    rr.append(NPY.array(NPY.array(nl,order='F'),order='F'))
  r=NPY.array(rr,order='F').T
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
  if ((type(a) in [str, unicode]) or (type(a)==type(NPY.array((1))))):
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
  
# --------------------------------------------------
def hasFortranFlag(node):
  if (node[1]==None):           return True
  if (node[1]==[]):             return True
  if (type(node[1])==type('')): return True # link
  if (not node[1].shape):       return True
  if (len(node[1].shape)==1):   return True  
  return NPY.isfortran(node[1])

# --------------------------------------------------
def getValueShape(node):
  """
  Returns the value data shape for a CGNS/Python node for **display purpose**.
  If the shape cannot be determined a `-` is returned.

  .. code-block:: python
         
     print 'node data shape is %s'%getValueShape(node)

  - Args:
   * `node`: CGNS/Python node

  - Return:
   * A string with the shape

  """
  return getNodeShape(node)

def getNodeShape(node):
  r="-"
  if   (node[1]==None): r="-"
  elif (node[1]==[]):   r="-"
  elif (node[3]==''):   r="-"
  elif (node[1].shape in ['',(0,),()]): r="[0]"
  else: r=str(list(node[1].shape))
  return r

def getShape(node):
  r=0
  if   (node[1]==None): r=(0,)
  elif (node[1]==[]):   r=(0,)
  elif (node[3]==''):   r=(0,)
  elif (node[1].shape in ['',(0,),()]): r=(0,)
  else: r=node[1].shape
  return r

# --------------------------------------------------
def getValueDataType(node):
  """
  Returns the value data type for a CGNS/Python node for **display purpose**.

  .. code-block:: python
         
     print 'node data type is %s'%getValueDataType(node)

  - Args:
   * `node`: CGNS/Python node

  - Return:
   * A string in [`C1`,`I4`,`I8`,`R4`,`R8`,`??`]

  - Remarks:
   * `??` is returned if datatype is not one of [`C1`,`I4`,`I8`,`R4`,`R8`]

  """
  return getNodeType(node)

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
    if (data.dtype.char in ['i','I']):        return CK.I4
    if (data.dtype.char in ['l']):            return CK.I8
  if ((type(data) == type([])) and (len(data))): # oups !
    if (type(data[0]) == type("")):           return CK.C1 
    if (type(data[0]) == type(0)):            return CK.I4 
    if (type(data[0]) == type(0.0)):          return CK.R8
  return '??'

# --------------------------------------------------
def hasFirstPathItem(path,sidstype=CK.CGNSTree_s):
  if ((len(path)>0) and (path[0]=='/')): path=path[1:]
  p=path.split('/')
  if ((len(p)>1) and (sidstype==p[0])): return True
  return False
    
# --------------------------------------------------
def removeFirstPathItem(path):
  p=path.split('/')
  if ((p[0]=='') and (p>2)):
    return string.join(['']+p[2:],'/')
  elif (len(p)>1):
    return string.join(p[1:],'/')
  else:
    return '/'

# --------------------------------------------------
def getNodeByPath(tree,path):
  """
  Returns the CGNS/Python node with the argument path::

   zbc=getNodeByPath(T,'/Base/Zone001/ZoneBC')
   bc1=getNodeByPath(zbc,'wall01')

  The path is compared as a string, you should provide the exact path
  if you have a sub-tree or a tree with its `CGNSTree` fake node. The
  following lines are not equivalent (sic!)::

   zbc=getNodeByPath(T,'/Base/Zone001/ZoneBC')
   zbc=getNodeByPath(T,'/CGNSTree/Base/Zone001/ZoneBC')
   

  - Args:
   * `tree`: the target tree to parse
   * `path`: a string representing an absolute or relative path

  - Return:
   * The CGNS/Python node matching the path

  - Remark:
   * Returns None if the path is not found
   * No wildcards allowed (see :py:func:`getPathByNameFilter` and :py:func:`getPathByNameFilter` )
     
  """
  if (path=='/'): return tree
  if (not checkPath(path)): return None
  if (path[0]=='/'): path=path[1:]
  if (tree[3]==CK.CGNSTree_ts): T=tree
  else: T=[CK.CGNSTree_s,None,[tree],CK.CGNSTree_ts]
  n=getNodeFromPath(path.split('/'),T)
  if (n==-1): return None
  return n

# --------------------------------------------------
def getValueByPath(tree,path):
  """
  Returns the value of a CGNS/Python node with the argument path::

   import CGNS.PAT.cgnskeywords as CK
  
   v=getNodeByPath(T,'/Base/Zone001/ZoneType')

   if (v == CK.Structured_s): print 'Structured Zone Found'
  
  - Args:
   * `tree`: the target tree to parse
   * `path`: a string representing an absolute or relative path

  - Return:
   * The CGNS/Python node value matching the path

  - Remark:
   * Returns None if the path is not found
   * No wildcards allowed (see :py:func:`getPathByNameFilter` and :py:func:`getPathByNameFilter` )
   
  """
  n=getNodeByPath(tree,path)
  if (n==-1): return None
  return n[1]

# --------------------------------------------------
def getChildrenByPath(tree,path):
  """
  Returns the children list of a CGNS/Python node with the argument path.

  .. code-block:: python

     import CGNS.PAT.cgnskeywords as CK

     for bc in getChildrenByPath(T,'/Base/Zone01/ZoneBC'):
       if (bc[3] == CK.BC_ts): 
         print 'BC found:', bc[0]

  - Args:
   * `tree`: the target tree to parse
   * `path`: a string representing an absolute or relative path

  - Return:
   * The CGNS/Python node children list of node matching the path

  - Remark:
   * Returns None if the path is not found
   * No wildcards allowed (see :py:func:`getPathByNameFilter` and :py:func:`getPathByNameFilter` )
   
  """
  n=getNodeByPath(tree,path)
  if (n==-1): return None
  return n[2]

# --------------------------------------------------
def getNextChildSortByType(node,parent=None,criteria=CK.cgnstypes):
  """
  Iterator, returns the children list of the argument CGNS/Python
  sorted using the CGNS type then the name. The `sortlist` gives
  an alternate sort list/dictionnary.

  .. code-block:: python

     for child in getNextChildSortByType(node):
         print 'Next child:', child[0]

     zonesort=[CGK.Elements_ts, CGK.Family_ts, CGK.ZoneType_ts]
     for child in getNextChildSortByType(node,criteria=mysort):
         print 'Next child:', child[0]

     mysort={CGK.Zone_t: zonesort}
     for child in getNextChildSortByType(node,parent,mysort):
         print 'Next child:', child[0]

  - Args:
   * `node`: the target node
   * `parent`: the parent node
   * `criteria`: a list or a dictionnary used as the sort criteria

  - Return:
   * This is an iterator, it returns a CGNS/Python node 

  - Remark:
   * The function is an iterator
   * If criteria is a list of type, the sort order for the type is the
     list order. If it is a dictionnary, its keys are the parent types
     and the values are list of types.
   
  """
  def sortbytypesasincriteria(a,b):
    if ((a[0] in a[2]) and (b[0] in b[2])):
      if (a[2].index(a[0])>b[2].index(b[0])): return  1
      if (a[2].index(a[0])<b[2].index(b[0])): return -1
    if (a[1]>b[1]): return  1
    if (a[1]<b[1]): return -1
    return 0
  
  __criteria=[]
  if (type(criteria)==list):
    __criteria=criteria
  if (    (type(criteria)==dict)
      and (parent is not None)
      and (criteria.has_key(parent[3]))):
    __criteria=criteria[parent[3]]
  r=[]
  for i in range(len(node[2])):
    c=node[2][i]
    r+=[(c[3],c[0],__criteria,i)]
  r.sort(sortbytypesasincriteria)
  for i in r:
    yield node[2][i[3]]

# --------------------------------------------------
def getTypeByPath(tree,path):
  """
  Returns the CGNS type of a CGNS/Python node with the argument path.

  .. code-block:: python

     import CGNS.PAT.cgnskeywords as CK

     if (getTypeByPath(T,'/Base/Zone01/ZoneBC/'):
       if (bc[3] == CK.BC_ts): 
         print 'BC found:', bc[0]

  - Args:
   * `tree`: the target tree to parse
   * `path`: a string representing an absolute or relative path

  - Return:
   * The CGNS/Python node CGNS/SIDS type (string)

  - Remark:
   * Returns None if the path is not found
   * No wildcards allowed (see :py:func:`getPathByTypeFilter` and :py:func:`getPathByNameFilter` )
   
  """
  n=getNodeByPath(tree,path)
  if (n is not None): return n[3]
  return None

# --------------------------------------------------
def getPathByNameFilter(tree,filter=None):
  """
  Returns a list of paths from T matching the filter. The filter is a
  `regular expression <http://docs.python.org/library/re.html>`_
  used to match the path of **node names**::

   import CGNS.PAT.cgnskeywords as CK

   for path in filterPathByName(T,'/Base[0-1]/domain\..*/.*/.*/FamilyName'):
      print 'FamilyName ',path,' is ',path[2]

  - Args:
   * `tree`: the target tree to parse
   * `filter`: a regular expresion for the complete path to math to

  - Return:
   * A list of paths (strings) matching the path pattern

  - Remark:
   * Returns empty list if no match
   
  """
  return []

# --------------------------------------------------
def getPathByTypeFilter(tree,filter=None):
  """
  Returns a list of paths from T matching the filter. The filter is a
  `regular expression <http://docs.python.org/library/re.html>`_
  used to match the path of **node types**::

   import CGNS.PAT.cgnskeywords as CK

   for path in filterPathByType(T,'/.*/.*/.*/BC_t'):
     for child in getChildrenByPath(T,path):
      if (child[3]==CK.FamilyName_t):
        print 'BC ',path,' belongs to ',child[2]

  - Args:
   * `tree`: the target tree to parse
   * `filter`: a regular expression for the complete path to math to

  - Return:
   * A list of paths (strings) matching the types-path pattern

  - Remark:
   * Returns empty list if no match
   
  """
  return []

# --------------------------------------------------
def nodeByPath(path,tree):
  if (not checkPath(path)): return None
  if (path[0]=='/'): path=path[1:]
  if (tree[3]==CK.CGNSTree_ts):
#    path=string.join([CK.CGNSTree_s]+path.split('/')[1:],'/')
#    path=string.join(path.split('/')[1:],'/')
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
  """
  Beware: this parse starts with children, not current node...
  """
  for c in node[2]:
    if (c[0] == path[0]):
      if (len(path) == 1): return c
      return getNodeFromPath(path[1:],c)
  return -1

# --------------------------------------------------
def getParentFromNode(tree,node):
  """
  Returns the parent node of a node. If the node is root node, itself is
  returned::

   parent=getParentFromNode(T,node)

  - Args:
   * `tree`: the CGNS/Python target tree to parse
   * `node`: the child node 

  - Return:
   * A list of paths (strings) matching the types-path pattern

  - Remark:
   * Returns itself if node is root
   
  """
  pn=getPathFromNode(tree,node)
  pp=getPathAncestor(pn)
  np=getNodeByPath(tree,pp)
  return np
  
# --------------------------------------------------
def getPathFromNode(tree,node,path=''):
  """
  Returns the path from a node in a tree. The argument tree is parsed and
  a path is built-up until the node is found. The node object is compared
  to the tree nodes, if you have multiple references to the same node, the
  first found is used for the path::

   # T is a compliant CGNS/Python tree
   
   path=getPathFromNode(T,node)
   getNodeByPath(T,getPathAncestor(path))

  Args:
   * `tree`: the target tree to parse
   * `node`: the target node to find

  Remark:
   * Returns None if not found
   
  """
  if checkSameNode(node,tree): return path
  for c in tree[2]:
    p=getPathFromNode(c,node,path+'/'+c[0])
    if (p): return p
  return None

# --------------------------------------------------
def getAllNodesByTypeOrNameList(tree,typeornamelist):
  """
  Returns a list of paths from the argument tree with nodes matching
  the list of types or names. The list you give is the list you would
  have if you pick the node type or the node name during the parse.

  .. code-block:: python

     tnlist=['CGNSTree_t','Base#001','Zone_t']

     for path in getAllNodesByTypeOrNameList(T,tnlist):
        node=getNodeByPath(T,path)
        # do something with node

  Would return all the zones of the named base.
  See also :py:func:`getAllNodesByTypeSet`
  See also :py:func:`getAllNodesByTypeList`

  - Args:
   * `tree`: the start node of the CGNS tree to parse
   * `typeornamelist`: the (ordered) list of types

  - Return:
   * a list of strings, each string is the path to a matching node

  - Remark:
   * the first comparison is performed on name, then on type. If you have
     a node name that matches a type, the node is included in the result.
 
  """
  if ((tree[0]!=typeornamelist[0]) and (tree[3]!=typeornamelist[0])):
    return None
  if (tree[3]==CK.CGNSTree_ts): start=""
  else:                         start="%s"%tree[0]
  n=getAllNodesFromTypeOrNameList(typeornamelist[1:],tree[2],start,[])
  if (n==-1): return None
  return n
  
# --------------------------------------------------
def getAllNodesFromTypeOrNameList(tnlist,node,path,result):
  for c in node:
    if ((c[0]==tnlist[0]) or (c[3]==tnlist[0])):
      if (len(tnlist) == 1):
        result.append("%s/%s"%(path,c[0]))
      else:
        getAllNodesFromTypeOrNameList(tnlist[1:],
                                      c[2],"%s/%s"%(path,c[0]),result)
  return result
  
# --------------------------------------------------
def getAllNodesByTypeList(tree,typelist):
  """
  Returns a list of paths from the argument tree with nodes matching
  the list of types. The list you give is the list you would have if you
  pick the node type during the parse.

  .. code-block:: python

     tlist=['CGNSTree_t','CGNSBase_t','Zone_t']

     for path in getAllNodesByTypeList(T,tlist):
        node=getNodeByPath(T,path)
        # do something with node

  Would return all the zones of your tree.
  See also :py:func:`getAllNodesByTypeSet`

  - Args:
   * `tree`: the start node of the CGNS tree to parse
   * `typelist`: the (ordered) list of types

  - Return:
   * a list of strings, each string is the path to a matching node
 
  """
  if (tree[3]!=typelist[0]): return None
  if (tree[3]==CK.CGNSTree_ts): start=""
  else:                         start="%s"%tree[0]
  n=getAllNodesFromTypeList(typelist[1:],tree[2],start,[])
  if (n==-1): return None
  return n

# --------------------------------------------------
def getAllNodesFromTypeList(typelist,node,path,result):
  for c in node:
    if (c[3]==typelist[0]):
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
  plist.sort()
  return plist

def wsort(a,b):
    if (a[0] < b[0]): return -1
    if (a[0] > b[0]): return  1
    if (a[1] < b[1]): return -1
    if (a[1] > b[1]): return  1
    return 0

# --------------------------------------------------   
def getPathFullTree(tree,width=False):
  """
  Returns the list of all possible node paths of a CGNS/Python tree.

  .. code-block:: python

     for paths in getPathFullTree(T):
        # do something
        
  - Args:
   * `tree`: the CGNS/Python target tree to parse
   * `width`: a boolean set to True (default is False) for width sorting

  - Return:
   * A list of strings, each is a path
   
  - Remark:
   * Returns [] is tree empty or invalid
   * When sorting with width the paths are listed as width-first parse
  
  """
  r=getAllPaths(tree)
  if (width):
    s=[]
    for p in r:
      s.append((p.count('/'),p))
    s.sort(cmp=wsort)
    r=[]
    for p in s:
      r.append(p[1])
  return r

# --------------------------------------------------   
def checkPath(path,dienow=False):
  """
  Checks the compliance of a path, which is basically a UNIX-like
  path with constraints on each node name.

    checkPath('/Base/Zone/ZoneBC')

  - Args:
   * `path`: path to check (string)

  - Return:
   * True if the path is ok, False if a problem is found
  
  """
  if ((type(path) not in [str,unicode]) or not path): return False
  if (path[0]=='/'): path=path[1:]
  for p in path.split('/'):
    if (not checkName(p,dienow)): return False
  return True

# --------------------------------------------------   
def hasSameRootPath(pathroot,pathtocompare):
  """
  Compares two paths::

    >>>hasSameRootPath('/Base/Zone/ZoneBC','/Base/ZoneBC/BC#2/Data')
    True
    >>>hasSameRootPath('/Base/Zone/ZoneBC','/Base/ZoneBC#2')
    False

  - Args:
   * `pathroot`: root path to compare
   * `pathtocompare`: path which is supposed to have rootpath as substring

  - Return:
   * Ture if 'rootpath' is a prefix of 'pathtocompare'

  - Remarks:
   * Each node name is a token, see example below: the second example
     doesn't match as a path while it matches as a string.
   
  """
  l1=getPathToList(pathroot)
  l2=getPathToList(pathtocompare)
  if (len(l1) > len(l2)): return False
  for i in range(len(l1)):
    if (l1[i]!=l2[i]): return False
  return True

# --------------------------------------------------   
def getPathListCommonAncestor(pathlist):
  """
  Finds the common ancestor for all paths in list::

    >>>p=['/Base/Zone/Data-A','/Base/Zone/Data-D','/Base/Zone/ZoneBC/BC1']
    >>>print getPathListCommonAncestor(p)
    '/Base/Zone'

  - Args:
   * `pathlist`: list of path strings
   
  - Return:
   * The common root path (at least '/')
   
  """
  if (len(pathlist)==0): return '/'
  if (len(pathlist)==1): return pathlist[0]
  lp=[]
  for p in pathlist:
    if (p=='/'): return '/'
    lp.append(getPathToList(p,True))
  t=lp[0]
  for p in lp:
    n=0
    r=t
    m=min(len(p),len(r))
    for n in range(m):
      if (p[n]!=r[n]):
        t=r[:n]
        break
    else:
      t=r[:n+1]
  if (t):
    t=['']+t
    c='/'.join(t)
  else:
    c='/'
  return c

# --------------------------------------------------   
def getPathToList(path,nofirst=False,noroot=True):
  """
  Return the path as a list of node names::

    >>>print getPathToList('/Base/Zone/ZoneBC')
    ['','Base','Zone','ZoneBC']
    >>>print getPathToList('/Base/Zone/ZoneBC',True)
    ['Base','Zone','ZoneBC']
    >>>print getPathToList('/')
    []

  - Args:
   * `path`: path string to split
   * `nofirst`: Removes the first empty string that appears for absoulte paths (default: False)
   * `noroot`: If true then removes the CGNS/HDF5 root if found (default: True)

  - Return:
   * The list of path elements as strings
   * With '/' as argument, the function returns an empty list

  - Remarks:
   * The path is first processed by :py:func:`getPathNormalize` before its split
   
  """
  lp=[]
  if (path is None): return []
  if (len(path)>0):
    path=getPathNormalize(path)
    if (noroot): path=getPathNoRoot(path)
    if (nofirst and (path[0]=='/')): path=path[1:]
    if (path not in ['/','']): lp=path.split('/')
  return lp

# --------------------------------------------------   
def getPathAncestor(path,level=1):
  """
  Return the path of the node parent of the argument node path::

    >>>print getPathAncestor('/Base/Zone/ZoneBC')
    '/Base/Zone'

  - Args:
   * `path`: path string of the child node 
   * `level`: number of levels back from the child (default: 1 means the father of the node)

  - Return:
   * The ancestor path
   * If the path is '/' its ancestor is None.
   
  """
  lp=getPathToList(path)
  if ((len(lp)==2) and (lp[0]=='')):  ancestor='/'
  elif (len(lp)>1):                   ancestor='/'.join(lp[:-1])
  elif (len(lp)==1):                  ancestor='/'
  else:                               ancestor=None
  if (level >1):
    ancestor=getPathAncestor(ancestor,level-1)
  return ancestor

# --------------------------------------------------   
def getPathLeaf(path):
  """
  Return the leaf node name of the path::

    >>>print getPathLeaf('/Base/Zone/ZoneBC')
    'ZoneBC'

  - Args:
   * `path`: path string of the child node 

  - Return:
   * The leaf node name
   * If the path is '/' the function returns ''

  """
  leaf=''
  lp=getPathToList(path)
  if (len(lp)>0): leaf=lp[-1]
  return leaf

# --------------------------------------------------   
def getPathNoRoot(path):
  """
  Return the path without the implementation node 'HDF5 Mother node'
  if detected as first element::

    >>>print getPathNoRoot('/HDF5 Mother Node/Base/Zone/ZoneBC')
    ['Base','Zone','ZoneBC']

  - Args:
   * `path`: path string to check

  - Return:
   * The new path without `HDF5 Mother node` if found

  - Remarks:
   * The path is processed by :py:func:`getPathNormalize`
   
  """
  if (path is None): return '/'
  if (path == ''):   return '/'
  path=getPathNormalize(path)
  if (path=='/'): return path
  lp=path.split('/')
  if (lp[0] in [CK.CGNSHDF5ROOT_s, CK.CGNSTree_s]): lp=lp[1:]
  if ((lp[0]=='')
      and (len(lp)>1)
      and (lp[1] in [CK.CGNSHDF5ROOT_s, CK.CGNSTree_s])): lp=[lp[0]]+lp[2:]
  path='/'.join(lp)
  return path

# --------------------------------------------------   
def getPathAsTypes(tree,path):
  """
  Return the list of types corresponding to the argument path in the tree::

    >>>getPathAsTypes(T,'/Base/Zone/ZoneBC')
    ['CGNSBase_t','Zone_t','ZoneBC_t']

  - Args:
   * `tree`: target tree
   * `path`: path to parse in the tree

  - Return:
   * The list of CGNS types found
   * `None` if the path is not found
   
  """
  ltypes=[]
  leg=True
  if (checkRootNode(tree,legacy=False)):
    p=getPathToList(path)
    if (p and (p[0] != CK.CGNSTree_ts)):
      path='/'+CK.CGNSTree_s+'/'+path
      leg=False
  path=getPathNormalize(path)
  while path not in ['/', None]:
    t=getTypeByPath(tree,path)
    ltypes+=[t]
    path=getPathAncestor(path)
  if (leg and (len(ltypes)>0)): ltypes=ltypes[:-1]
  ltypes.reverse()
  return ltypes

# --------------------------------------------------   
def getPathNormalize(path):
  """
  Return the same path as minimal string, removes `////` and `/./` and
  other simplifiable UNIX-like path elements.

  .. code-block:: python

     # a checkPath here would fail, because single or double dots are not
     # allowed as a node name. But actually this is allowed as a
     # path description
     p=getPathNormalize('///Base/././//Zone/../Zone/./ZoneBC//.')
    
     # would return '/Base/Zone/ZoneBC'
     if (checkPath(p)):
        # do something

  - Args:
   * `path`: path string to simplify

  - Return:
   * The simplified path

  - Remarks:
   * Uses *os.path.normpath*
   * Before its normalization a path can be **non-compliant**
   
  """
  path=os.path.normpath(path)
  return path

# --------------------------------------------------
def childNames(node):
  """
  Gets the children names

  .. code-block:: python

     for c in childNames(node):
       # do something
       
  - Args:
   * `node`: CGNS/Python node

  - Return:
   * List of children names
  
  """
  r=[]
  if (node == None): return r
  for c in node[2]:
    r.append(c[0])
  return r

# --------------------------------------------------
def getAllNodesByTypeSet2(typelist,tree):
  if (tree[3]==CK.CGNSTree_ts): start="/%s"%tree[0]
  else:                         start="%s"%tree[0]
  n=getAllNodesFromTypeSet(typelist,tree[2],start,[])
  return n

# --------------------------------------------------
def getAllNodesByTypeSet(tree,typeset):
  """
  Returns a list of paths from the argument tree with nodes matching
  one of the types in the list.

  .. code-block:: python

     #  Would return all the zones and BCs of your tree.
     tset=['BC_t','Zone_t']

     for path in getAllNodesByTypeSet(T,tset):
        node=getNodeByPath(T,path)
        # do something

  - Args:
   * `tree`: the start node of the CGNS tree to parse
   * `typeset`: the list of types

  - Return:
   * a list of strings, each string is the path to a matching node

  - Remarks:
   * See also :py:func:`getAllNodesByTypeList`

  """
  if (tree[3]==CK.CGNSTree_ts): start=""
  else:                         start="%s"%tree[0]
  n=getAllNodesFromTypeSet(typeset,tree[2],start,[])
  return n

# --------------------------------------------------
def getAllNodesFromTypeSet(typelist,node,path,result):
  for c in node:
    if (c[3] in typelist):
      result.append("%s/%s"%(path,c[0]))
    getAllNodesFromTypeSet(typelist,c[2],"%s/%s"%(path,c[0]),result)
  return result

# --------------------------------------------------
def getNodeAllowedChildrenTypes(pnode,node):
  """
  Returns all allowed CGNS-types for the node. The parent is mandatory.

  .. code-block:: python

     if (node[2] not in getNodeAllowedChildrenTypes(parent,node)):
        # do something

  - Args:
   * `pnode`: CGNS/Python parent node of second argument
   * `node`:  CGNS/Python node

  - Return:
   * A list of CGNS/SIDS types (strings)
  
  """
  tlist=[]
  if (node[3] == CK.CGNSTree_ts): return tlist
  try:
    if ((node[3] == None) or (pnode == None)):
      ctl=CT.types[CK.CGNSTree_ts]
    else:
      ctl=CT.types[pnode[3]]
    for cn in ctl.children:
      if (cn[0] not in tlist): tlist+=[cn[0]]
  except:
    pass
  return tlist

# --------------------------------------------------
def getNodeAllowedDataTypes(node):
  """
  Returns a list of string with all allowed CGNS data types for the node.

  .. code-block:: python

     if (getValueDataType(node) not in getNodeAllowedDataTypes(node)):
        # do something

  - Args:
   * `node`:  CGNS/Python node

  - Return:
   * A list of CGNS/SIDS value data types (strings)
   * see also :py:func:`getValueDataType`
  
  """
  tlist=[]
  try:
    tlist=CT.types[node[3]].datatype
  except:
    pass
  return tlist

# -----------------------------------------------------------------------------
def hasChildType(parent,ntype):
  """
  checks if the parent node has a child with given type.

  .. code-block:: python

     node=getNodeByPath(T,'/Base/Zone/BC'):
     if (hasChildType(node, 'AdditionalFamily_t')):
        # do something

  - Args:
   * `parent`: CGNS/Python parent node
   * `ntype`: CGNS/SIDS node type (string)

  - Return:
   * True if at least one child with such type is found

  """
  if (not parent): return None
  r=[]
  for n in parent[2]:
    if (n[3] == ntype): r.append(n)
  if r: return r
  return None

# -----------------------------------------------------------------------------
def hasChildName(parent,name,dienow=False):
  """
  Checks if the name is in the children list of the parent::

    node=hasChildName(parent,CGK.ZoneGridConnectivity_s)
    if (node is None):
       node=CGL.newZoneGridConnectivity(parent)

  - Args:
   * `parent`: the parent node
   * `name`:   the child name to look for

  - Return:
   * the actual child node if the child exists
   * None if the child is not found

  - Remarks:
   * Raises :ref:`cgnsnameerror` code 102 if `dienow` is True
  
  """
  if (not parent): return None
  if (parent[2] == None): return None
  for nc in parent[2]:
    if (nc[0] == name):
      if (dienow): raise CE.cgnsNameError(102,(name,parent[0]))
      return nc
  return None

# --------------------------------------------------
def getTypeAsGrammarToken(ntype):
  if (ntype in CK.weirdSIDStypes): return CK.weirdSIDStypes[ntype]
  return ntype
    
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
def stringValueInList(node,listval):
  if (node == None):             return 0
  if (node[1] == None):          return 0  
  if (getNodeType(node)!=CK.C1): return 0
  tn=type(node[1])
  if   (tn==type('')): vn=node[1]
  elif (tn == type(NPY.ones((1,))) and (node[1].dtype.char in ['S','c','s'])):
    vn=node[1].tostring()
  else: return 0
  return vn in listval

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
    b=NPY.array(a,order='Fortran',copy=True)
  else:
    b=NPY.array(a,copy=True)
  return b

# ----
