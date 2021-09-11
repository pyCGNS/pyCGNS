#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnslib        as CGL
cimport numpy as NCY
import numpy as NPY
from cpython cimport bool as py_bool 

def checkReserved(node,childtype,childname,pth,log):
    childnode=CGU.hasChildName(node,childname)
    if (childnode is None): return True
    if (childnode[3]!=childtype):
        log.push(pth,'S120',childname,childtype)
        return False
    return True

def checkChildValue(tnode,tshape,tdtype,pth,log,rs,tmin=None,tmax=None):
   if (tnode is not None):
       if (CGU.getShape(tnode)!=tshape):
           rs=log.push(pth,'S192',CGU.getShape(tnode),tshape,tnode[0])
       elif (CGU.getValueDataType(tnode) not in tdtype):
           rs=log.push(pth,'S196',tnode[0])
       elif ((tmin is not None) and (tnode[1][0] < tmin)):
           rs=log.push(pth,'S197',tnode[0])
       elif ((tmax is not None) and (tnode[1][0] > tmax)):
           rs=log.push(pth,'S197',tnode[0])
   return rs

def getBase(pth,node,parent,tree,log):
    lpth=CGU.getPathToList(pth,True)
    if (lpth==[]): return None
    if (lpth==[CGK.CGNSTree_ts]): return None
    if (lpth[0]!=[CGK.CGNSTree_ts]): return lpth[0]
    return lpth[1]
  
cpdef py_bool indexInRangeList(int idx,rl):
  """return True if index in range index list rl 
    - Args:
     * `var`: integer
     * `rl`: list of range index [[i1min,i1max],[i2min,i2max],..]
  """
  for r in rl:
    if (idx>=r[0] and idx<=r[1]):
      return True
  return False

cpdef int countIdxInRangeList(int idx,rl):
  """return the number of times index appears in range index list rl 
    - Args:
     * `var`: integer
     * `rl`: list of range index [[i1min,i1max],[i2min,i2max],..]
  """
  cdef int n
  n = 0
  for r in rl:
    if (idx>=r[0] and idx<=r[1]): n += 1
  return n

def allIndexInElementRangeList(NCY.ndarray t,erl,pth,log,rs):
  cdef int idx
  cdef py_bool test

  for idx in t:
    test = False
    for et in erl:
      if (indexInRangeList(idx,erl[et])): 
        test=True
        break
    if (not test): 
      rs=log.push(pth,'S206')
      return rs
  return rs

# -----------------------------------------------------------------------------
def getElementTypes(level,physicaldimension):
  """Element types of certain level for certain physical dimension::

    # The level may be Cell, Face, Edge or Vertex

    # The physical dimension is within 0-3
  
  - Args:
   * `level`: may be Cell, Face, Edge or Vertex (`string`)
   * `physicaldimension`: physical dimension of CFD problem (`int`)

  - Return:
   * The list of Element Types of considered level for considered physical dimension
  """
  if   (level == CGK.Cell_s and physicaldimension == 3):
    return CGK.ElementType3D+[CGK.NFACE_n]
  elif (level == CGK.Cell_s and physicaldimension == 2):
    return CGK.ElementType2D+[CGK.NFACE_n]
  elif (level == CGK.Cell_s and physicaldimension == 1):
    return CGK.ElementType1D+[CGK.NFACE_n]
    
  elif (level == CGK.Face_s and physicaldimension == 3):
    return CGK.ElementType2D+[CGK.NGON_n]    
  elif (level == CGK.Face_s and physicaldimension == 2):
    return CGK.ElementType1D+[CGK.NGON_n] 
  elif (level == CGK.Face_s and physicaldimension == 1):
    return CGK.ElementType0D+[CGK.NGON_n] 

  elif (level == CGK.Edge_s and physicaldimension == 3):
    return CGK.ElementType1D
  elif (level == CGK.Edge_s and physicaldimension == 2):
    return CGK.ElementType0D
  
  elif (level == CGK.Vertex_s and physicaldimension == 3):
    return CGK.ElementType0D 
        
  else :
    return []  

# -----------------------------------------------------------------------------
def getAuthElementTypes(facetype):
  """Authorized Element types for a Face type::
  
  - Args:
   * `facetype`: Face Type (`int`)

  - Return:
   * The list of Element Types authorized for the considered face type
  """  
  if (facetype in CGK.ElementType2D):
    if (facetype in CGK.ElementType_tri):
      return CGK.ElementType_trionly+CGK.ElementType_triquad+[CGK.MIXED,CGK.NFACE_n]
    elif (facetype in CGK.ElementType_quad):
      return CGK.ElementType_quadonly+CGK.ElementType_triquad+[CGK.MIXED,CGK.NFACE_n]
    else:
      return CGK.ElementType3D+[CGK.NFACE_n]
  elif (facetype==CGK.NGON_n):
    return CGK.ElementType3D+[CGK.NFACE_n]
  else:
    return []

# -----------------------------------------------------------------------------
def getAuthGridLocation(celldim,regioncelldim):
  """Authorized GridLocation wrt celldim and regioncelldim::
  
  - Args:
   * `celldim`: Cell dimension (`int`)
   * `regioncelldim`: Region cell dimension (`int`)

  - Return:
   * The list of GridLocation authorized for celldim/regioncelldim
  """  
  if (celldim == 3):
    if (regioncelldim == 3):
      return [CGK.Vertex_s,
              CGK.EdgeCenter_s,
              CGK.FaceCenter_s,CGK.IFaceCenter_s,CGK.JFaceCenter_s,CGK.KFaceCenter_s,
              CGK.CellCenter_s]
    elif (regioncelldim == 2):
      return [CGK.Vertex_s,
              CGK.EdgeCenter_s,
              CGK.FaceCenter_s,CGK.IFaceCenter_s,CGK.JFaceCenter_s,CGK.KFaceCenter_s]
    elif (regioncelldim == 1):
      return [CGK.Vertex_s,
              CGK.EdgeCenter_s]
    elif (regioncelldim == 0):
      return [CGK.Vertex_s]    
  elif (celldim == 2):
    if (regioncelldim == 2):
      return [CGK.Vertex_s,
              CGK.EdgeCenter_s,
              CGK.CellCenter_s]
    elif (regioncelldim == 1):
      return [CGK.Vertex_s,
              CGK.EdgeCenter_s]    
    elif (regioncelldim == 0):
      return [CGK.Vertex_s]    
  elif (celldim == 1):
    if (regioncelldim == 1):
      return [CGK.Vertex_s,
              CGK.CellCenter_s]
    elif (regioncelldim == 0):
      return [CGK.Vertex_s]          
  return []

def getGridLocation(node,pth=None,log=None,rs=None):
  return getChildNodeValueWithDefault(node,CGK.GridLocation_ts,CGK.Vertex_s,
                                      pth=pth,log=log,rs=rs)

def getGridConnectivityType(node,pth=None,log=None,rs=None):
  return getChildNodeValueWithDefault(node,CGK.GridConnectivityType_ts,CGK.Overset_s,
                                      pth=pth,log=log,rs=rs)
  
def getChildNodeValueWithDefault(node,ntype,defaultv,pth=None,log=None,rs=None):
  p = CGU.hasChildType(node,ntype) # get child by type
  if (not p): # child is absent
    v=defaultv # affect default value
    if rs: rs=log.push(pth,'s0000.0123',ntype,defaultv)
  else: v=p[0][1].tostring() # child is present
  if rs: return v,rs
  else: return v  

def listLength(node):
  """the structure function ListLength is used to specify the number of entities (e.g. vertices) 
     corresponding to a given PointRange or PointList. 
     If PointRange is specified, then ListLength is obtained from the number of points (inclusive)
     between the beginning and ending indices of PointRange. 
     If PointList is specified, then ListLength is the number of indices in the list of points. 
     In this situation, ListLength becomes a user input along with the indices of the list PointList. 
     By user we mean the application code that is generating the CGNS database::
     
  - Args:
   * `node`: cgns node, should be a node of type PointList or PointRange

  - Return:
   * ListLength, number of elements as defined in PointList/PointRange
  """
  CGU.checkNode(node)
  listlength = -1
  if (node[0] == CGK.PointList_s):
    listlength = node[1].size
  elif (node[0] == CGK.PointRange_s):
    shp=CGU.getShape(node)
    if (len(shp)==2):
      if (    shp[0] in range(1,3+1) 
          and shp[1]==2
          and CGU.getValueDataType(node) in [CGK.I4,CGK.I8]):
        pr = node[1]
        listlength = 0        
        for d in range(shp[0]):
          listlength += max(pr[d])-min(pr[d])+1
  return listlength

# -----------------------------------------------------------------------------
def getListLength(node):
  """return ListLength, number of elements as defined in PointList/PointRange node or child::
  
  - Args:
   * `node`: cgns node, should have a child or be itself a node of type PointList or PointRange

  - Return:
   * ListLength, number of elements as defined in PointList/PointRange
  """
  listlength = -1
  if (node[0] in [CGK.PointList_s,CGK.PointRange_s]):
    listlength = listLength(node)
  else:
    if (CGU.hasChildName(node,CGK.PointList_s)):
      listlength = listLength(CGU.hasChildName(node,CGK.PointList_s))
    if (CGU.hasChildName(node,CGK.PointRange_s)):
      listlength = listLength(CGU.hasChildName(node,CGK.PointRange_s))
  return listlength

def dataSize(node,indexdimension,vertexsize,cellsize):
  """The function DataSize[] is the size of flow solution data arrays. 
     If Rind is absent then DataSize represents only the core points; 
     it will be the same as VertexSize or CellSize depending on GridLocation.::
     
  - Args:
   * `node`: cgns node

  - Return:
   * DataSize, one-dimensional int array of length IndexDimension 
  """
  gridloc=getGridLocation(node)
  datasize=tuple([])
  if (CGU.hasChildName(node,CGK.PointList_s) or CGU.hasChildName(node,CGK.PointRange_s)):
    datasize = tuple([getListLength(node)])
  elif (not CGU.hasChildType(node,CGK.Rind_ts)):
    if (gridloc == CGK.Vertex_s):
      datasize = vertexsize
    elif (gridloc == CGK.CellCenter_s):
      datasize = cellsize
  elif (CGU.hasChildType(node,CGK.Rind_ts)):
    rind=CGU.hasChildType(node,CGK.Rind_ts)[0]
    if (    CGU.getShape(rind)==(2*indexdimension,)
        and CGU.getValueDataType(rind) in [CGK.I4,CGK.I8]):
      datasize=[]
      if (gridloc == CGK.Vertex_s):
        for d in range(indexdimension):
          datasize.append( vertexsize[d]
                          +rind[1][2*d]+rind[1][2*d+1])
      elif (gridloc == CGK.CellCenter_s):
        for d in range(indexdimension):
          datasize.append( cellsize[d]
                          +rind[1][2*d]+rind[1][2*d+1])
      datasize=tuple(datasize)
  return datasize

def getZoneType(tree,path):
  """Return type Structured|Unstructured of zone in path::
     
  - Args:
   * `tree`: cgns tree
   * `path`: node path in tree

  - Return:
   * Structured|Unstructured
  """
  l=CGU.getPathAsTypes(tree,path)
  for (i,var) in enumerate(l):
    if (var == CGK.Zone_ts): break
  if (i==len(l)): return None
  else:
    zpth=CGU.getPathAncestor(path,level=len(l)-i-1)
    zone=CGU.getNodeByPath(tree,zpth)
    zt=CGU.hasChildName(zone,CGK.ZoneType_s)
    if (zt is not None):
      return zt[1].tostring()
    else:
      return None

def getZoneName(tree,path):
  """Return name of zone ::
     
  - Args:
   * `tree`: cgns tree
   * `path`: node path in tree

  - Return:
   * zone name
  """
  name=None
  l=CGU.getPathAsTypes(tree,path)
  for (i,var) in enumerate(l):
    if (var == CGK.Zone_ts): break
  if (i!=len(l)):
    zpth=CGU.getPathAncestor(path,level=len(l)-i-1)
    zone=CGU.getNodeByPath(tree,zpth)
    name=zone[0]
  return name

def getLevelFromGridLoc(gridloc):
  if (gridloc.endswith(CGK.FaceCenter_s)):
    level = CGK.Face_s
  elif (gridloc == CGK.CellCenter_s):
    level = CGK.Cell_s
  elif (gridloc == CGK.Vertex_s):
    level = CGK.Vertex_s
  elif (gridloc == CGK.EdgeCenter_s):
    level = CGK.Edge_s
  else:
    level = None
  return level

def getElementTypeRangeList(level,physicaldimension,elementrangelist):
  """extract from elementrangelist the range list corresponding to Element 
     types of certain level for certain physical dimension::

    # The level may be Cell, Face, Edge or Vertex
    # The physical dimension is within 0-3
  
  - Args:
   * `level`: may be Cell, Face, Edge or Vertex (`string`)
   * `physicaldimension`: physical dimension of CFD problem (`int`)
   * `elementrangelist`: dictionary element type <-> list of ranges

  - Return:
   * a dictionary element type of the considered level <-> list of ranges
  """
  etypes = getElementTypes(level,physicaldimension)
  etyperl = elementrangelist.copy()
  for k in elementrangelist:
    if k not in etypes: del etyperl[k]
  return etyperl

def getIndices(tree,zonepath,nodetype,indextype,indexdimension):
  """look for all indices of indextype for all nodes of type nodetype in zone
     defined through its path in tree::

    # typical nodetype: BC_t or ZoneGridConnectivity
  
  - Args:
   * `tree`: tree
   * `zonepath`: zone path in tree
   * `nodetype`: node type to look for index in
   * `indextype`: IndexRange or IndexArray
   * `indexdimension`: IndexDimension of zone

  - Return:
   * a list of ranges or arrays, depending on indextype
  """
  pth = []
  if (nodetype in [CGK.ZoneBC_ts,CGK.BC_ts]):
    #pth = CGU.getAllNodesByTypeSet(tree,typeset)
    for child in CGU.getAuthChildren(CGL.newZoneBC(None)):
      searchpath=CGU.getPathToList(zonepath,True)+[CGK.ZoneBC_ts]+[child[0]]
      pth += CGU.getAllNodesByTypeOrNameList(tree,[CGK.CGNSTree_ts]+searchpath+[indextype])
  elif (nodetype in [CGK.ZoneGridConnectivity_ts,CGK.GridConnectivity_ts,CGK.GridConnectivity1to1_ts]):
    for child in CGU.getAuthChildren(CGL.newZoneGridConnectivity(None)):
      searchpath=CGU.getPathToList(zonepath,True)+[CGK.ZoneGridConnectivity_ts]+[child[0]]
      # cannot consider indextype since should not consider Point[Range/List]Donor
      if (indextype == CGK.IndexRange_ts):
        pth += CGU.getAllNodesByTypeOrNameList(tree,[CGK.CGNSTree_ts]+searchpath+[CGK.PointRange_s])
        pth += CGU.getAllNodesByTypeOrNameList(tree,[CGK.CGNSTree_ts]+searchpath+[CGK.ElementRange_s])
      elif (indextype == CGK.IndexArray_ts):
        pth += CGU.getAllNodesByTypeOrNameList(tree,[CGK.CGNSTree_ts]+searchpath+[CGK.PointList_s])
  pth = [path for path in pth if path]
  if (indextype not in [CGK.IndexRange_ts,CGK.IndexArray_ts]): return []
  if (indextype == CGK.IndexRange_ts):
    rl = []
    if (pth):
      for path in pth:
        node = CGU.getNodeByPath(tree,path)
        if (CGU.getShape(node) == (indexdimension,2)):
          rl.append(node[1])
    return rl
  elif (indextype == CGK.IndexArray_ts):
    al = NPY.empty((indexdimension,0),dtype=NPY.int32)
    if (pth):  
      for path in pth:
        node = CGU.getNodeByPath(tree,path)
        if (CGU.getShape(node)[0] == indexdimension):
          al = NPY.concatenate([al,node[1]],axis=1)
    return al  

def getIndicesOnBCandGC(tree,zonepath,indexdim):
  """look for all IndexRange and IndexArray on BC and GridConnectivity of a zone
  
  - Args:
   * `tree`: tree
   * `zonepath`: zone path in tree

  - Return:
   * a list of ranges and a list of arrays
  """
  rl = []
  al = NPY.empty((indexdim,0),dtype=NPY.int32)
  rl += getIndices(tree,zonepath,CGK.ZoneBC_ts,CGK.IndexRange_ts,indexdim)
  rl += getIndices(tree,zonepath,CGK.ZoneGridConnectivity_ts,CGK.IndexRange_ts,indexdim)
  al = NPY.concatenate([al,getIndices(tree,zonepath,CGK.ZoneBC_ts,CGK.IndexArray_ts,indexdim)],axis=1)
  al = NPY.concatenate([al,getIndices(tree,zonepath,CGK.ZoneGridConnectivity_ts,CGK.IndexArray_ts,indexdim)],axis=1)
  return (rl,al) 

def getFaceNumber(facerange,zd):
  """return an integer within [0..celldimension*2-1] corresponding to the face of facerange,
     0 <-> imin
     1 <-> imax
     ...
     5 <-> kmax
     return also the face indexes
     Note: this face indexing is not coherent with CGNS SIDS where, on 3D hexaedra:
     1 <-> kmin
     2 <-> jmin
     3 <-> imax
     4 <-> jmax
     5 <-> imin
     6 <-> kmax
     but the last is not coherent with CGNS SIDs edge indexing, where, on 2D quad:
     1 <-> jmin
     2 <-> imax
     3 <-> jmax
     4 <-> imin 
  - Args:
   * `facerange`: IndexRange of the face
   * `zd`: zone node value

  - Return:
   * five integers corresponding to the face of facerange and its indices
  """
  face = -1
  if (    facerange.ndim == 2
      and facerange.shape[1] == 2
      and zd.ndim == 2
      and facerange.shape[0] == zd.shape[0]):
    indexdimension = facerange.shape[0]
    for idx in range(indexdimension):
      if (    facerange[idx,0]==facerange[idx,1]
          and face == -1):
        if (facerange[idx,0] == 1):
          face = 2*idx
        elif (facerange[idx,0] == zd[idx,0]):
          face = 2*idx+1          
      elif (    facerange[idx,0]==facerange[idx,1]
            and face != -1): # this is not a face
        face = -1
        break
    if (face in [0,1]):
      idx_i = 1
      idx_j = 2
    elif (face in [2,3]):
      idx_i = 0
      idx_j = 2    
    elif (face in [4,5]):
      idx_i = 0
      idx_j = 1
  if (face != -1):
    if (indexdimension == 1):
      return (face,1,2,1,2)
    elif (indexdimension == 2):
      return (face,min(facerange[idx_i]),max(facerange[idx_i]),1,2)
    elif (indexdimension == 3):
      return (face,min(facerange[idx_i]),max(facerange[idx_i]),min(facerange[idx_j]),max(facerange[idx_j]))
    else:
      return (face,0,0,0,0)
  else:
    return (face,0,0,0,0)

def bdnName(bnd):
  if   (bnd==0): bnd_name='imin'
  elif (bnd==1): bnd_name='imax'
  elif (bnd==2): bnd_name='jmin'
  elif (bnd==3): bnd_name='jmax'
  elif (bnd==4): bnd_name='kmin'
  elif (bnd==5): bnd_name='kmax'
  else: bnd_name=''
  return bnd_name

def initBndFaces(zd):
  """return NPY.zeros of sizes the faces of a structured zone, face centered
  
  - Args:
   * `zd`: zone dimension (zone node value)

  - Return:
   * a list of NPY.zeros of sizes the faces of the structured zone, face centered
  """
  celldimension=zd.shape[0]
  if (celldimension == 1):
    bndfaces=NPY.zeros([1,1],dtype=NPY.int32),\
             NPY.zeros([1,1],dtype=NPY.int32)
  elif (celldimension == 2):
    bndfaces=NPY.zeros([zd[1,1],1],dtype=NPY.int32),\
             NPY.zeros([zd[1,1],1],dtype=NPY.int32),\
             NPY.zeros([zd[0,1],1],dtype=NPY.int32),\
             NPY.zeros([zd[0,1],1],dtype=NPY.int32)
  elif (celldimension == 3):
    bndfaces=NPY.zeros([zd[1,1],zd[2,1]],dtype=NPY.int32),\
             NPY.zeros([zd[1,1],zd[2,1]],dtype=NPY.int32),\
             NPY.zeros([zd[0,1],zd[2,1]],dtype=NPY.int32),\
             NPY.zeros([zd[0,1],zd[2,1]],dtype=NPY.int32),\
             NPY.zeros([zd[0,1],zd[1,1]],dtype=NPY.int32),\
             NPY.zeros([zd[0,1],zd[1,1]],dtype=NPY.int32)
  else:
    bndfaces=NPY.empty(0)
  return bndfaces

def isAFace(pr):
  """return True if pointrange is a face
  
  - Args:
   * `pr`: IndexRange node value

  - Return:
   * True if pointrange is a face
  """
  answer = False
  if (pr.ndim == 2 and pr.shape[1] == 2):
    for d in range(pr.shape[0]):
      if (pr[d][0] == pr[d][1]):
        answer = True
        break 
  return answer
                     
def hasOneAndOnlyOneChildAmong(node,namelist):
  """return True if node has one and only one child with name among listed
  
  - Args:
   * `node`: node
   * `namelist`: list of names

  - Return:
   * True if node has one and only one child with name among listed
  """
  count = 0
  for name in namelist:
    if (CGU.hasChildName(node,name)): count += 1
  if (count != 1):
    return False
  else:
    return True
