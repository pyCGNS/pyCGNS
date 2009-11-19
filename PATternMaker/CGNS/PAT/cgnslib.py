#  ---------------------------------------------------------------------------
#  pyCGNS.PAT - Python package for CFD General Notation System - PATternMaker
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#  $Release$
#  ---------------------------------------------------------------------------

import CGNS.PAT.cgnskeywords as K
import CGNS.PAT.cgnserrors   as E
import CGNS

__CGNS_LIBRARY_VERSION__=2.4

import types
import numpy as N

C1='C1'
MT='MT'
I4='I4'
I8='I8'
R4='R4'
R8='R8'
LK='LK'
DT=[C1,MT,I4,I8,R4,R8,LK] # LK declared as data type

zero_N=(0,-1)
one_N=(1,-1)
N_N=(-1,-1)
zero_one=(0,1)

typeListA=[
    K.Descriptor_ts,
    K.UserDefinedData_ts,
    K.DataClass_ts,
    K.DimensionalUnits_ts
    ]

# -----------------------------------------------------------------------------
# support functions

# -----------------------------------------------------------------------------
def hasChildName(parent,name):
  if (not parent): return None
  for n in parent[2]:
    if (n[0] == name): return n
  return None

# -----------------------------------------------------------------------------
def removeChildByName(parent,name):
  for n in range(len(parent[2])):
    if (parent[2][n][0] == name):
        del parent[2][n]
        return None
  return None

# -----------------------------------------------------------------------------
def checkName(name):
  if (type(name) != type("s")): raise E.cgnsException(22)
  if (len(name) == 0): raise E.cgnsException(23)
  if ('/' in name): raise E.cgnsException(24)
  if (len(name) > 32): raise E.cgnsException(25)

# -----------------------------------------------------------------------------
def checkDuplicatedName(parent,name):
  if (not parent): return
  if (parent[2] == None): return
  checkName(name)
  for nc in parent[2]:
    if (nc[0] == name): raise E.cgnsException(102,(name,parent[0]))

# -----------------------------------------------------------------------------
def hasChildrenType(parent,ntype):
  if (not parent): return None
  r=[]
  for n in parent[2]:
    if (n[3] == ntype): r.append(n)
  if r: return r
  return None

# -----------------------------------------------------------------------------
def concatenateForArrayChar(nlist):
  s=""
  for n in nlist:
    s+=("%-32s"%n)[:32]
  r=N.array(s,dtype='c') #.reshape(len(nlist),32)
  return r

# -----------------------------------------------------------------------------
def getValueType(v):
  if (v == None):            return None
  if (type(v) == type(3)):   return K.Integer_s
  if (type(v) == type(3.0)): return K.RealDouble_s
  if (type(v) == type('a')): return K.Character_s
  if ((type(v) == type([])) or (type(v) == type((1,)))):  
    if (len(v) > 1):  return getValueType(v[0])
    else:             return None
  if (type(v) == type(N.array((1,)))):
    if (v.dtype.char in ['S','c']):        return K.Character_s
    if (v.dtype.char in ['f','F']):        return K.RealDouble_s
    if (v.dtype.char in ['D','d']):        return K.RealDouble_s
    if (v.dtype.char in ['l','i','I']):    return K.Integer_s
  return None
   
# -----------------------------------------------------------------------------
def getValue(node):
  v=node[1]
  t=getValueType(v)
  if (t == None):           return None
  if (t == K.Integer_s):    return v
  if (t == K.RealDouble_s): return v
  if (t == K.Character_s):  return v.tostring()
  return v
  
# -----------------------------------------------------------------------------
def checkArray(a):
  if (type(a) != type(N.array((1)))): raise E.cgnsException(109)

# -----------------------------------------------------------------------------
def checkArrayChar(a):
  if ((type(a) != type(N.array((1)))) and (type(a) != type("s"))): 
    raise E.cgnsException(105)
  if (type(a) == type("s")): return N.array(a,dtype='c')
  return a

# -----------------------------------------------------------------------------
def checkArrayReal(a):
  if ((type(a) != type([])) and
      (type(a) != type((1,))) and
      (type(a) != type(N.array(1.0)))):
        raise E.cgnsException(106)
  if (type(a[0]) != type(1.3)):         raise E.cgnsException(106)
  return a

# -----------------------------------------------------------------------------
def checkArrayInteger(a):

  if ((type(a) != type([])) and
      (type(a) != type((1,))) and
      (a.dtype.char!='I') and
      (type(a) != type(N.array(1)))): raise E.cgnsException(107)   
  return a

# -----------------------------------------------------------------------------
def checkType(parent,stype,name):
  if (parent == None): return None
  if (parent[3] != stype): 
    raise E.cgnsException(103,(name,stype))
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
    raise E.cgnsException(104,(name,ltype))
  return None

# -----------------------------------------------------------------------------
def checkParent(node,dienow=0):
  if (node == None): return 1
  return checkNode(node,dienow)

# -----------------------------------------------------------------------------
def checkNode(node,dienow=0):
  if (node in [ [], None ]):
    if (dienow): raise E.cgnsException(1)
    return 0
  if (type(node) != type([3,])):
    if (dienow): raise E.cgnsException(2)
    return 0
  if (len(node) != 4):
    if (dienow): raise E.cgnsException(2)
    return 0
  if (type(node[0]) != type("")):
    if (dienow): raise E.cgnsException(3)
    return 0
  if (type(node[2]) != type([3,])):
    if (dienow): raise E.cgnsException(4,node[0])
    return 0
  return 1
    
# -----------------------------------------------------------------------------
def isRootNode(node,dienow=0):
  """ isRootNode : Check wether a node is a CGNS root node (returns 1)
or not a root node (returns 0).    
A root node is a list of a single CGNSLibraryVersion_t node and zero or more
CGNSBase_t nodes. We do not check first level type, no standard for it, even
if we set it to CGNSTree.
"""
  if (node in [ [], None ]):         return 0
  versionfound=0
  if (not checkNode(node)):          return 0
  for n in node[2]:
     if (not checkNode(n,dienow)):   return 0 
     if (     (n[0] == K.CGNSLibraryVersion_s)
          and (n[3] == K.CGNSLibraryVersion_ts) ):
         if versionfound: raise E.cgnsException(5)
         versionfound=1
     elif ( n[3] != K.CGNSBase_ts ): return 0
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

# =============================================================================
# MLL-like calls
# - every call takes a parent as argument. If the parent is present, the new
#   sub-tree is inserted in the parent child list. In all cases the call
#   returns the created sub-tree
# - some function are not MLL based
# - function patterns
#   newXXX :   creates a new XXX_t type node
#   updateXXX: updates fields in the XXX_t node
#   checkXXX:  check if node is ok for SIDS and for Python/CGNS
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def newDataClass(parent,value=K.UserDefined_s):
  """-DataClass node creation -DataClass
  
  'newNode:N='*newDataClass*'(parent:N,value:A)'
  
  If a parent is given, the new <node> is added to the parent children list.
  The value argument is a DataClass enumerate. No child allowed.
  Returns a new <node> representing a DataClass_t sub-tree."""
  checkDuplicatedName(parent,K.DataClass_s)
  node=newNode(K.DataClass_s,checkArrayChar(value),[],K.DataClass_ts,parent)
  return checkDataClass(node)

def updateDataClass(node,value):
  checkNode(node)  
  if (value!=None): node[1]=value
  return checkDataClass(node)

def checkDataClass(node,parent=None):
  checkNode(node)
  checkName(node[0])
  if (node[0] != K.DataClass_s):  raise E.cgnsException(26,node[0])
  if (node[3] != K.DataClass_ts): raise E.cgnsException(27,node[3])
  if (len(node[2]) != 0):         raise E.cgnsException(28,node[0])
  value=getValue(node)
  if (value not in K.DataClass_l):  raise E.cgnsException(207,value)
  if (parent != None):
     checkTypeList(parent,[K.DataArray_ts,K.CGNSBase_ts,K.Zone_ts,
                           K.GridCoordinates_ts,K.Axisymmetry_ts,
                           K.RotatingCoordinates_ts,K.FlowSolution_ts,
                           K.Periodic_ts,K.ZoneBC_ts,K.BC_ts,K.BCDataSet_ts,
                           K.BCData_ts,K.FlowEquationSet_ts,K.GasModel_ts,
                           K.ViscosityModel_ts,K.ThermalConductivityModel_ts,
                           K.TurbulenceClosure_ts,K.TurbulenceModel_ts,
                           K.ThermalRelaxationModel_ts,
                           K.ChemicalKineticsModel_ts,
                           K.EMElectricFieldModel_ts,K.EMMagneticFieldModel_ts,
                           K.EMConductivityModel_ts,K.BaseIterativeData_ts,
                           K.ZoneIterativeData_ts,K.RigidGridMotion_ts,
                           K.ArbitraryGridMotion_ts,K.ReferenceState_ts,
                           K.ConvergenceHistory_ts,
                           K.DiscreteData_ts,K.IntegralData_ts,
                           K.UserDefinedData_ts,K.Gravity_ts]
                   ,K.DataClass_s)
  return node
  
# -----------------------------------------------------------------------------
def newDescriptor(parent,name,value=''):
  """-Descriptor node creation -Descriptor
  
  'newNode:N='*newDescriptor*'(parent:N,name:S,text:A)'
  
  No child allowed.
  Returns a new <node> representing a Descriptor_t sub-tree."""
  checkDuplicatedName(parent,name)
  node=newNode(name,checkArrayChar(value),[],K.Descriptor_ts,parent)
  return checkDescriptor(node)
  
def checkDescriptor(node,parent=None):
  checkNode(node)
  checkName(node[0])
  if (node[3] != K.Descriptor_ts): raise E.cgnsException(27,node[3])
  if (len(node[2]) != 0):          raise E.cgnsException(28,node[0])
  value=getValue(node)
  if (getValueType(value) != K.Character_s): raise E.cgnsException(110,node[0])
  if (parent != None):
     checkTypeList(parent,[K.DataArray_ts,K.CGNSBase_ts,K.Zone_ts,
                           K.GridCoordinates_ts,K.Elements_ts,K.Axisymmetry_ts,
                           K.RotatingCoordinates_ts,K.FlowSolution_ts,
                           K.ZoneGridConnectivity_ts,K.GridConnectivity1to1_ts,
                           K.GridConnectivity_ts,K.GridConnectivityProperty_ts,
                           K.AverageInterface_ts,K.OversetHoles_ts,
                           K.Periodic_ts,K.ZoneBC_ts,K.BC_ts,K.BCDataSet_ts,
                           K.BCData_ts,K.FlowEquationSet_ts,K.GasModel_ts,
                           K.BCProperty_ts,K.WallFunction_ts,K.Area_ts,
                           K.GoverningEquations_ts,
                           K.ViscosityModel_ts,K.ThermalConductivityModel_ts,
                           K.TurbulenceClosure_ts,K.TurbulenceModel_ts,
                           K.ThermalRelaxationModel_ts,
                           K.ChemicalKineticsModel_ts,
                           K.EMElectricFieldModel_ts,K.EMMagneticFieldModel_ts,
                           K.EMConductivityModel_ts,K.BaseIterativeData_ts,
                           K.ZoneIterativeData_ts,K.RigidGridMotion_ts,
                           K.ArbitraryGridMotion_ts,K.ReferenceState_ts,
                           K.ConvergenceHistory_ts,
                           K.DiscreteData_ts,K.IntegralData_ts,
                           K.Family_ts,K.GeometryReference_ts,
                           K.UserDefinedData_ts,K.Gravity_ts]
                   ,K.DataClass_s)
  return node

# -----------------------------------------------------------------------------
def newDimensionalUnits(parent,value=[K.Meter_s,K.Kelvin_s,
                                      K.Second_s,K.Radian_s,K.Kilogram_s]):
  """-DimensionalUnits node creation -DimensionalUnits
  
  'newNode:N='*newDimensionalUnits*'(parent:N,value=[K.MassUnits,K.LengthUnits,
                                      K.TimeUnits,K.TemperatureUnits, K.AngleUnits])'
                                      
  If a parent is given, the new <node> is added to the parent children list.
  new <node> is composed of a set of enumeration types : MassUnits,LengthUnits,TimeUnits,TemperatureUnits,AngleUnits are required
  Returns a new <node> representing a DimensionalUnits_t sub-tree.
  chapter 4.3 
  """
  if (len(value) != 5): raise E.cgnsException(202)
  checkDuplicatedName(parent,K.DimensionalUnits_s)
  # --- loop over values to find all required units
  vunit=[K.Null_s,K.Null_s,K.Null_s,K.Null_s,K.Null_s]
  for v in value:
    if (v not in K.AllUnits_l): raise E.cgnsException(203,v)
    if ((v in K.MassUnits_l) and (v not in [K.Null_s,K.UserDefined_s])):
      if (v in vunit): raise E.cgnsException(204,v)
      else:            vunit[0]=v
    if ((v in K.LengthUnits_l) and (v not in [K.Null_s,K.UserDefined_s])):
      if (v in vunit): raise E.cgnsException(204,v)
      else:            vunit[1]=v
    if ((v in K.TimeUnits_l) and (v not in [K.Null_s,K.UserDefined_s])):
      if (v in vunit): raise E.cgnsException(204,v)
      else:            vunit[2]=v
    if ((v in K.TemperatureUnits_l) and (v not in [K.Null_s,K.UserDefined_s])):
      if (v in vunit): raise E.cgnsException(204,v)
      else:            vunit[3]=v
    if ((v in K.AngleUnits_l) and (v not in [K.Null_s,K.UserDefined_s])):
      if (v in vunit): raise E.cgnsException(204,v)
      else:            vunit[4]=v
  node=newNode(K.DimensionalUnits_s,concatenateForArrayChar(vunit),[],
               K.DimensionalUnits_ts,parent)
  snode=newNode(K.AdditionalUnits_s,
                concatenateForArrayChar([K.Null_s,K.Null_s,K.Null_s]),[],
                K.AdditionalUnits_ts,node)
  return node

# -----------------------------------------------------------------------------
def newDimensionalExponents(parent,
                            MassExponent=0,LengthExponent=0,TimeExponent=0,
                            TemperatureExponent=0,AngleExponent=0):
  """-DimensionalExponents node creation -DimensionalExponents
  
  'newNode:N='*newDimensionalExponents*'(parent:N,MassExponent:r,LengthExponent:r,TimeExponent:r,TemperatureExponent:r,AngleExponent:r)'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a DimensionalExponents_t sub-tree.
  chapter 4.4
  """
  checkDuplicatedName(parent,K.DimensionalExponents_s)
  node=newNode(K.DimensionalExponents_s,N.array([MassExponent,LengthExponent,TimeExponent,TemperatureExponent,AngleExponent],'d'),[],K.DimensionalExponents_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newGridLocation(parent,value=K.CellCenter_s):
  """-GridLocation node creation -GridLocation
  
  'newNode:N='*newGridLocation*'(parent:N,value:K.GridLocation)'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a GridLocation_t sub-tree.
  chapter 4.5
  """
  checkDuplicatedName(parent,K.GridLocation_s)
  if (value not in K.GridLocation_l): raise E.cgnsException(200,value)
  node=newNode(K.GridLocation_s,value,[],K.GridLocation_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newIndexArray(parent,name,value=[]):
  checkDuplicatedName(parent,name)
  node=newNode(name,value,[],K.IndexArray_ts,parent)
  return node
  
def newPointList(parent,name=K.PointList_s,value=[]):
  """-PointList node creation -PointList
  
  'newNode:N='*newPointList*'(parent:N,name:S,value:[])'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a IndexArray_t sub-tree.
  chapter 4.6
  """
  checkDuplicatedName(parent,name)
  node=newNode(name,value,[],K.IndexArray_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newPointRange(parent,name=K.PointRange_s,value=[]):
  """-PointRange node creation -PointRange
  
  'newNode:N='*newPointRange*'(parent:N,name:S,value:[])'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a IndexRange_t sub-tree.
  chapter 4.7
  """
  checkDuplicatedName(parent,name)
  node=newNode(name,value,[],K.IndexRange_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newRind(parent,value):                                    
  """-Rind node creation -Rind
  
  'newNode:N='*newRind*'(parent:N,value=A)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a Rind_t sub-tree.
  chapter 4.8
  """
  checkDuplicatedName(parent,K.Rind_s)
  # check value wrt base dims
  node=newNode(K.Rind_s,value,[],K.Rind_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newDataConversion(parent,ConversionScale=1.0,ConversionOffset=1.0):
  """-DataConversion node creation -DataConversion
  
  'newNode:N='*newDataConversion*'(parent:N,ConversionScale:r,ConversionOffset:r)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a DataConversion_t sub-tree.
  chapter  5.1.1
  """
  checkDuplicatedName(parent,K.DataConversion_s)
  node=newNode(K.DataConversion_s,N.array([ConversionScale,ConversionOffset],'d'),[],K.DataConversion_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newCGNS():
  """-Tree node creation -Tree

  'newNode:N='*newCGNS*'()'

  Returns a new <node> representing a CGNS tree root.
  This is not a SIDS type.
  """
  node=[K.CGNSLibraryVersion_s,__CGNS_LIBRARY_VERSION__,[],
        K.CGNSLibraryVersion_ts]
  badnode=[K.CGNSTree_s,None,[node],K.CGNSTree_ts]
  return badnode

# ----------------------------------------------------------------------------
def newSimulationType(parent,stype=K.NonTimeAccurate_s):
  """-SimulationType node creation -SimulationType
  
  'newNode:N='*newSimulationType*'(parent:N,stype=K.SimulationType)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a SimulationType_t sub-tree.
  chapter 6.2
  """
  if (parent): checkNode(parent)
  checkDuplicatedName(parent,K.SimulationType_s)
  checkType(parent,K.CGNSBase_ts,K.SimulationType_s)
  if (stype not in K.SimulationType_l): raise E.cgnsException(205,stype)
  node=newNode(K.SimulationType_s,stype,[],K.SimulationType_ts,parent)
  return node
  
# ----------------------------------------------------------------------------
  
def newBase(tree,name,ncell,nphys):
  """-Base node creation -Base
  
  'newNode:N='*newBase*'(parent:N,name:S,ncell:[1,2,3],nphys:[1,2,3])'
  
  Returns a new <node> representing a CGNSBase_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list,
  that is to the base list of the parent CGNSTree.
  Maps the 'cg_base_write' MLL
  chapter 6.2
  """
  if (ncell not in [1,2,3]): raise E.cgnsException(10,name)
  if (nphys not in [1,2,3]): raise E.cgnsException(11,name)
  if (nphys < ncell):        raise E.cgnsException(12,name)
  if ((tree != None) and (not checkNode(tree))):
     raise E.cgnsException(6,name)
  if ((tree != None) and (tree[0] == K.CGNSTree_s)): parent=tree[2]  
  else:                                              parent=tree
  checkDuplicatedName(["<root node>",None,parent],name)
  node=newNode(name,N.array([ncell,nphys],dtype=N.int32),[],K.CGNSBase_ts)
  if (parent != None): parent.append(node)
  return node

def numberOfBases(tree):
  return len(hasChildrenType(tree,K.CGNSBase_ts))

def readBase(tree,name):
  b=hasChildName(tree,name)
  if (b == None): raise E.cgnsException(21,name)
  if (b[3] != K.CGNSBase_ts): raise E.cgnsException(20,(K.CGNSBase_ts,name))
  return (b[0],b[1])
  
def updateBase(tree,name=None,ncell=None,nphys=None):
  if (ncell not in [1,2,3]): raise E.cgnsException(10,name)
  if (nphys not in [1,2,3]): raise E.cgnsException(11,name)
  if (nphys < ncell):        raise E.cgnsException(12,name)
  
  if (tree): checkNode(tree)

  
  if (tree[3] != K.CGNSBase_ts): raise E.cgnsException(20,(K.CGNSBase_ts,name))
  if(name!=None): tree[0]=name
  if(ncell!=None and nphys!=None and tree): tree[1]=N.array([ncell,nphys] ) 	  	  
  else: raise E.cgnsException(12)  
  
 
# -----------------------------------------------------------------------------
def newOrdinal(parent,value=0):
  """-Ordinal node creation -Ordinal
  
  'newNode:N='*newOrdinal*'(parent:N,value=i)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a Ordinal_t sub-tree.
  chapter 6.3
  """
  checkDuplicatedName(parent,K.Ordinal_s)
  node=newNode(K.Ordinal_s,value,[],K.Ordinal_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newZone(parent,name,size=(2,2,2),ztype=K.Structured_s,family=''):
  """-Zone node creation -Zone
  
  'newNode:N='*newZone*'(parent:N,name:S,size:(I*),ztype:K.ZoneType)'
  
  Returns a new <node> representing a Zone_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list.
  Maps the 'cg_zone_write' MLL
  chapter 6.3
  """
  asize=None
  if (ztype not in K.ZoneType_l): raise E.cgnsException(206,ztype)
  if ((len(size) == 3) and (ztype == K.Structured_s)):
    size=(size[0],size[1],size[2],size[0]-1,size[1]-1,size[2]-1,0,0,0)
    asize=N.array(size,dtype=N.int32).reshape(3,3)
  if ((len(size) == 2) and (ztype == K.Structured_s)):
    size=(size[0],size[1],size[0]-1,size[1]-1,0,0)
    asize=N.array(size,dtype=N.int32).reshape(2,3)
  if ((len(size) == 1) and (ztype == K.Structured_s)):
    size=(size[0][1],size[0]-1,0)
    asize=N.array(size,dtype=N.int32).reshape(1,3)
  if (ztype == K.Unstructured_s):
    asize=N.array(size,dtype=N.int32)
  if (asize == None): raise E.cgnsException(999) 
  checkDuplicatedName(parent,name)
  znode=newNode(name,asize,[],K.Zone_ts,parent)
  newNode(K.ZoneType_s,ztype,[],K.ZoneType_ts,znode)
  if (family): newNode(K.FamilyName_s,family,[],K.FamilyName_ts,znode)
  return znode

def numberOfZones(tree,basename):
  b=hasChildName(tree,basename)
  if (b == None): raise E.cgnsException(21,basename)
  if (b[3] != K.CGNSBase_ts): raise E.cgnsException(20,(K.CGNSBase_ts,name))
  return len(hasChildrenType(b,K.Zone_ts))

def readZone(tree,basename,zonename,gtype=None):
  b=hasChildName(tree,basename)
  if (b == None): raise E.cgnsException(21,basename)
  if (b[3] != K.CGNSBase_ts): raise E.cgnsException(20,(K.CGNSBase_ts,name))
  z=hasChildName(b,zonename)
  if (z == None): raise E.cgnsException(21,zonename)
  if (z[3] != K.Zone_ts): raise E.cgnsException(20,(K.Zone_ts,name))
  if gtype: 
    zt=hasChildName(z,K.ZoneType_s)
    if (zt == None): raise E.cgnsException(21,K.ZoneType_s)
    return (z[0],z[1],zt[1])
  else:
    return (z[0],z[1])

def typeOfZone(tree,basename,zonename):
  return readZone(tree,basename,zonename,1)[2]

# -----------------------------------------------------------------------------
def newGridCoordinates(parent,name):
  """-GridCoordinates node creation -Grid
  
  'newNode:N='*newGridCoordinates*'(parent:N,name:S)'
  
  Returns a new <node> representing a GridCoordinates_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list.
  """
  node=newNode(name,None,[],K.GridCoordinates_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newDataArray(parent,name,value=None):
  """-DataArray node creation -Global
  
  'newNode:N='*newDataArray*'(parent:N,name:S,value:A)'
  
  Returns a new <node> representing a DataArray_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list.
  chapter 5.1
  """
  checkDuplicatedName(parent,name)
  if (type(value) in [type(3), type(3.2), type("s")]): vv=N.array([value])
  else: vv=value
  if (vv != None): checkArray(vv)
  node=newNode(name,vv,[],K.DataArray_ts,parent)
  return node

def numberOfDataArrays(parent):
  return len(hasChildrenType(parent,K.DataArray_ts))

def readDataArray(parent,name):
  n=hasChildName(parent,name)
  if (n == None): raise E.cgnsException(21,name)
  if (n[3] != K.DataArray_ts): raise E.cgnsException(20,(K.DataArray_ts,name))
  return n[1]

# -----------------------------------------------------------------------------
def newDiscreteData(parent,name):
  """-DiscreteData node creation -DiscreteData
  
  'newNode:N='*newDiscreteData*'(parent:N,name:S)'
  
   If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a DiscreteData_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   chapter 6.3
  """
  checkDuplicatedName(parent,name)    
  node=newNode(name,None,[],K.DiscreteData_ts,parent)
  return node 
  
# -----------------------------------------------------------------------------
def newElements(parent,elementstype=K.UserDefined_s,elementsconnectivity=None,
                elementsrange=None):
  """-Elements node creation -Elements
  
  'newNode:N='*newAElements*'(parent:N,elementsType:K.ElementType,value:K.ElementConnectivity)'
  
   Returns a new <node> representing a Element_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   If the parent has already a child name Element then
   only the ElementType,IndexRange_t,ElementConnectivity are created.
   chapter 7.3 Add node :ElementType,IndexRange_t are required
               Add DataArray : ElementConnectivity is required
  """
  enode=hasChildName(parent,K.Element_s)
  if (enode == None):
    enode=newNode(K.Element_s,None,[],K.Element_ts,parent)
  if (elementstype not in K.ElementType_l):
    raise E.cgnsException(250,elementstype)
  checkDuplicatedName(enode,K.ElementType_s)   
  ccnode=newNode(K.ElementType_s,elementstype,[],K.ElementType_ts,enode)
  newDataArray(enode,K.ElementConnectivity_s,elementsconnectivity)
  checkDuplicatedName(enode,K.ElementRange_s) 
  cnode=newNode(K.ElementRange_s,elementsrange,[],K.IndexRange_ts,enode)  
  return enode

# -----------------------------------------------------------------------------
def newZoneBC(parent):
  return newNode(K.ZoneBC_s,None,[],K.ZoneBC_ts,parent)

def newBC(parent,bname,brange=[0,0,0,0,0,0],
          btype=K.Null_s,bcType=K.Null_s,
          family=K.Null_s,pttype=K.PointRange_s):
  return newBoundary(parent,bname,brange,btype,bcType,pttype) 

def newBoundary(parent,bname,brange=[0,0,0,0,0,0],
                btype=K.Null_s,family=None,pttype=K.PointRange_s): 
  """-BC node creation -BC
  
  'newNode:N='*newBoundary*'(parent:N,bname:S,brange:[*i],btype:S)'
  
  Returns a new <node> representing a BC_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list.
  Parent should be Zone_t, returned node is parent.
  If the parent has already a child name ZoneBC then
   only the BC_t,IndexRange_t are created.
  chapter 9.3 Add IndexRange_t required
  """
  checkDuplicatedName(parent,bname)
  zbnode=hasChildName(parent,K.ZoneBC_s)
  if (zbnode == None): zbnode=newNode(K.ZoneBC_s,None,[],K.ZoneBC_ts,parent)
  bnode=newNode(bname,btype,[],K.BC_ts,zbnode)
  if (pttype==K.PointRange_s):
    arange=N.array(list(brange),dtype=N.int32).reshape(3,2)
    newNode(K.PointRange_s,arange,[],K.IndexRange_ts,bnode)
  else:
    arange=N.array(list(brange),dtype=N.int32)
    newNode(K.PointList_s,arange,[],K.IndexArray_ts,bnode)
  if (family): newNode(K.FamilyName_s,family,[],K.FamilyName_ts,bnode)
  return bnode
  
# -----------------------------------------------------------------------------
def newBCDataSet(parent,name,valueType=K.Null_s):
  """-BCDataSet node creation -BCDataSet
  
  'newNode:N='*newBCDataSet*'(parent:N,name:S,valueType:K.BCTypeSimple)'
  
   If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a BCDataSet_t sub-tree.  
   chapter 9.4 Add node BCTypeSimple is required
  """
  node=hasChildName(parent,name)
  if (node == None):    
    node=newNode(name,None,[],K.BCDataSet_ts,parent)
  if (valueType not in K.BCTypeSimple_l):
    raise E.cgnsException(252,valueType)
  checkDuplicatedName(node,K.BCTypeSimple_s)    
  nodeType=newNode(K.BCTypeSimple_s,valueType,[],K.BCTypeSimple_ts,node)
  return node

# ---------------------------------------------------------------------------  
def newBCData(parent,name):
  """-BCData node creation -BCData
  
  'newNode:N='*newBCData*'(parent:N,name:S)'
  
   Returns a new <node> representing a BCData_t sub-tree. 
   chapter 9.5 
  """
  checkDuplicatedName(parent,name)    
  node=newNode(name,None,[],K.BCData_ts,parent)
  return node 
  
# -----------------------------------------------------------------------------
def newBCProperty(parent,wallfunction=K.Null_s,area=K.Null_s):
  """-BCProperty node creation -BCProperty
  
  'newNode:N='*newBCProperty*'(parent:N)'
  
   Returns a new <node> representing a BCProperty_t sub-tree.  
   If a parent is given, the new <node> is added to the parent children list.
   chapter 9.6
  """
  checkDuplicatedName(parent,K.BCProperty_s)    
  node=newNode(K.BCProperty_s,None,[],K.BCProperty_ts,parent)
  wf=newNode(K.WallFunction_s,None,[],K.WallFunction_ts,node)
  newNode(K.WallFunctionType_s,wallfunction,[],K.WallFunctionType_ts,wf)
  ar=newNode(K.Area_s,None,[],K.Area_ts,node)
  newNode(K.AreaType_s,area,[],K.AreaType_ts,ar)
  return node 

# -----------------------------------------------------------------------------
def newCoordinates(parent,name=K.GridCoordinates_s,value=None):
  """-GridCoordinates_t node creation with name GridCoordinates -Grid
  
  'newNode:N='*newCoordinates*'(parent:N,name:S,value:A)'

  Creates a new <node> representing a GridCoordinates_t sub-tree with
  the coordinate DataArray given as argument. This creates both the
  GridCoordinates_t with GridCoordinates name and DataArray_t with the
  argument name. Usually used to create the default grid.
  If the GridCoordinates_t with name GridCoordinates already exists then
  only the DataArray is created.
  If a parent is given, the new GridCoordinates_t <node> is added to the
  parent children list, in all cases the DataArray is child of
  GridCoordinates_t node.
  The returned node always is the DataArray_t node.
  chapter 7.1
  """
  checkDuplicatedName(parent,name)
  gnode=hasChildName(parent,K.GridCoordinates_s)
  if (gnode == None): gnode=newGridCoordinates(parent,K.GridCoordinates_s)
  node=newDataArray(gnode,name,value)
  return node
  
# -----------------------------------------------------------------------------
def newAxisymmetry(parent,refpoint=[0.0,0.0,0.0],axisvector=[0.0,0.0,0.0]):
  """-Axisymmetry node creation -Axisymmetry
  
  'newNode:N='*newAxisymmetry*'(parent:N,refpoint:A,axisvector:A)'
  
  refpoint,axisvector should be a real array.
  Returns a new <node> representing a K.Axisymmetry_t sub-tree.   
  chapter 7.5 Add DataArray AxisymmetryAxisVector,AxisymmetryReferencePoint are required
  """
  if (parent): checkNode(parent)
  checkType(parent,K.CGNSBase_ts,K.Axisymmetry_s)
  checkDuplicatedName(parent,K.Axisymmetry_s)
  checkArrayReal(refpoint)
  checkArrayReal(axisvector)
  node=newNode(K.Axisymmetry_s,None,[],K.Axisymmetry_ts,parent)
  n=hasChildName(parent,K.AxisymmetryReferencePoint_s)
  if (n == None): 
    n=newDataArray(node,K.AxisymmetryReferencePoint_s,N.array(refpoint))
  n=hasChildName(parent,K.AxisymmetryAxisVector_s)
  if (n == None): 
    n=newDataArray(node,K.AxisymmetryAxisVector_s,N.array(axisvector))
  return node

# -----------------------------------------------------------------------------
def newRotatingCoordinates(parent,rotcenter=[0.0,0.0,0.0],ratev=[0.0,0.0,0.0]):
  """-RotatingCoordinates node creation -RotatingCoordinates
  
  'newNode:N='*newRotatingCoordinates*'(parent:N,rotcenter=A,ratev=A)'
  
   Returns a new <node> representing a RotatingCoordinates_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   rotcenter,ratev should be a real array.
   chapter  7.6 Add DataArray RotationRateVector,RotationCenter are required   
  """ 
  if (parent): checkNode(parent)
  checkTypeList(parent,[K.CGNSBase_ts,K.Zone_ts,K.Family_ts],
                K.RotatingCoordinates_s)
  checkDuplicatedName(parent,K.RotatingCoordinates_s)
  checkArrayReal(rotcenter)
  checkArrayReal(ratev)
  node=newNode(K.RotatingCoordinates_s,None,[],K.RotatingCoordinates_ts,parent)
  n=hasChildName(node,K.RotationCenter_s)
  if (n == None): 
    n=newDataArray(node,K.RotationCenter_s,N.array(rotcenter))
  n=hasChildName(node,K.RotationRateVector_s)
  if (n == None): 
    n=newDataArray(node,K.RotationRateVector_s,N.array(ratev))
  return node

# -----------------------------------------------------------------------------
def newFlowSolution(parent,name='{FlowSolution}',gridlocation=None):
  """-Solution node creation -Solution
  
  'newNode:N='*newSolution*'(parent:N,name:S,gridlocation:None)'
  
  Returns a new <node> representing a FlowSolution_t sub-tree. 
  chapter 7.7
  """
  checkDuplicatedName(parent,name)
  node=newNode(name,None,[],K.FlowSolution_ts,parent)
  return node  
  
# -----------------------------------------------------------------------------
def newZoneGridConnectivity(parent,name,ctype=K.Null_s,donor=''):
  """-GridConnectivity node creation -Grid
  
  'newNode:N='*newZoneGridConnectivity*'(parent:N,name:S,ctype:S)'

  Creates a ZoneGridConnectivity_t sub-tree with
  a sub-node depending on the type of connectivity.
  This sub-node is returned.
  If a parent is given, the new <node> is added to the parent children list,
  the parent should be a Zone_t.
  chapter 8.1
  """
  checkDuplicatedName(parent,name)
  cnode=hasChildName(parent,K.ZoneGridConnectivity_s)  
  if (cnode == None):   
    cnode=newNode(K.ZoneGridConnectivity_s,
                  None,[],K.ZoneGridConnectivity_ts,parent)
  node=newNode(name,donor,[],ctype,cnode)
  return node
  
# -----------------------------------------------------------------------------
def newGridConnectivity1to1(parent,name,dname,window,dwindow,trans):
  """-GridConnectivity1to1 node creation -Grid
  
  'newNode:N='*newGridConnectivity1to1*'(parent:N,name:S,dname:S,window:[i*],dwindow:[i*],trans:[i*])'
  
  Creates a ZoneGridConnectivity1to1_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list,
  the parent should be a Zone_t.
  The returned node is the GridConnectivity1to1_t
  chapter 8.2
  """
  cnode=hasChildName(parent,K.ZoneGridConnectivity_s)
  if (cnode == None):
    cnode=newNode(K.ZoneGridConnectivity_s,
                  None,[],K.ZoneGridConnectivity_ts,parent)
  zcnode=newNode(name,dname,[],K.GridConnectivity1to1_ts,cnode)
  newNode("Transform",N.array(list(trans),'i'),[],
          "int[IndexDimension]",zcnode)   
  newNode(K.PointRange_s,N.array(list(window),'i'),[],
          K.IndexRange_ts,zcnode)   
  newNode(K.PointRangeDonor_s,N.array(list(dwindow),'i'),[],
          K.IndexRange_ts,zcnode)   
  return zcnode

# -----------------------------------------------------------------------------
def newGridConnectivityProperty(parent): 
  """-GridConnectivityProperty node creation -GridConnectivityProperty
  
  'newNode:N='*newGridConnectivityProperty*'(parent:N)'
  
   Returns a new <node> representing a GridConnectivityProperty_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   chapter 8.5 
  """
  checkDuplicatedName(parent,K.GridConnectivityProperty_s)   
  nodeType=newNode(K.GridConnectivityProperty_s,None,[],
                   K.GridConnectivityProperty_ts,parent)
  return nodeType

def  newPeriodic(parent,rotcenter=[0.0,0.0,0.0],ratev=[0.0,0.0,0.0],trans=[0.0,0.0,0.0]):
  """-Periodic node creation -Periodic
  
  'newNode:N='*newPeriodic*'(parent:N,rotcenter=A,ratev=A,trans=A)'
  
   Returns a new <node> representing a Periodic_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name Periodic then
   only the RotationCenter,RotationAngle,Translation are created.
   rotcenter,ratev,trans should be a real array.
   chapter 8.5.1 Add DataArray RotationCenter,RotationAngle,Translation are required
  """
  if (parent): checkNode(parent)  
  checkArrayReal(rotcenter)
  checkArrayReal(ratev)
  checkArrayReal(trans)
  cnode=hasChildName(parent,K.Periodic_s)
  if (cnode == None):
    cnode=newNode(K.Periodic_s,None,[],K.Periodic_ts,parent)
  n=hasChildName(cnode,K.RotationCenter_s)
  if (n == None): 
    newDataArray(cnode,K.RotationCenter_s,N.array(rotcenter))
  n=hasChildName(cnode,K.RotationAngle_s)
  if (n == None): 
    newDataArray(cnode,K.RotationAngle_s,N.array(ratev))
  n=hasChildName(cnode,K.Translation_s)
  if (n == None): 
    newDataArray(cnode,K.Translation_s,N.array(trans)) 
  return cnode
  
# -----------------------------------------------------------------------------
def newAverageInterface(parent,valueType=K.Null_s):
  """-AverageInterface node creation -AverageInterface
  
  'newNode:N='*newAverageInterface*'(parent:N,valueType:K.AverageInterfaceType)'
  
   Returns a new <node> representing a AverageInterface_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   If the parent has already a child name AverageInterface then
   only the AverageInterfaceType is created.
   chapter 8.5.2
  """
  node=hasChildName(parent,K.AverageInterface_s)
  if (node == None):       
    node=newNode(K.AverageInterface_s,None,[],
                 K.AverageInterface_ts,parent)
  if (valueType not in K.AverageInterfaceType_l):
    raise E.cgnsException(253,valueType)
  checkDuplicatedName(node,K.AverageInterfaceType_s) 
  nodeType=newNode(K.AverageInterfaceType_s,valueType,[],
                   K.AverageInterfaceType_ts,node)
  return node
  
# -----------------------------------------------------------------------------
def newOversetHoles(parent,name,hrange):
  """-OversetHoles node creation -OversetHoles
  
  'node:N='*newOversetHoles*'(parent:N,name:S,hrange:list)'

  Creates a OversetHoles_t sub-tree.  
  the parent should be a Zone_t.
  If a parent is given, the new <node> is added to the parent children list.
  chapter 8.6 Add PointList or List( PointRange ) are required
  """
  cnode=hasChildName(parent,K.ZoneGridConnectivity_s)
  if (cnode == None):
    cnode=newNode(K.ZoneGridConnectivity_s,None,[],K.ZoneGridConnectivity_ts,parent)
  checkDuplicatedName(cnode,name)   
  node=newNode(name,None,[],K.OversetHoles_ts,cnode)
  #if(pname!=None and value!=None):
    #newPointList(node,pname,value)
  if hrange!=None:  
   newPointRange(node,K.PointRange_s,N.array(list(hrange),'i'))
   #newNode(K.PointRange_s,N.array(list(hrange),'i'),[],K.IndexRange_ts,node)
  return node

# -----------------------------------------------------------------------------
def newFlowEquationSet(parent):
  """-FlowEquationSet node creation -FlowEquationSet
  
  'newNode:N='*newFlowEquationSet*'(parent:N)'
  
  If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a K.FlowEquationSet_t sub-tree.  
   chapter 10.1
  """
  if (parent): checkNode(parent)
  checkDuplicatedName(parent,K.FlowEquationSet_s)
  checkTypeList(parent,[K.CGNSBase_ts,K.Zone_ts],K.FlowEquationSet_s)     
  node=newNode(K.FlowEquationSet_s,None,[],K.FlowEquationSet_ts,parent)  
  return node   
    
def newGoverningEquations(parent,valueType=K.Euler_s):
  """-GoverningEquations node creation -GoverningEquations
  
  'newNode:N='*newGoverningEquations*'(parent:N,valueType:K.GoverningEquationsType)'
  
   Returns a new <node> representing a K.GoverningEquations_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name GoverningEquations then
   only the GoverningEquationsType is created.
   chapter  10.2 Add node GoverningEquationsType is required   
  """
  node=hasChildName(parent,K.GoverningEquations_s)
  if (node == None):    
    node=newNode(K.GoverningEquations_s,None,[],K.GoverningEquations_ts,parent)
  if (valueType not in K.GoverningEquationsType_l):
      raise E.cgnsException(221,valueType)
  checkDuplicatedName(parent,K.GoverningEquationsType_s,)
  nodeType=newNode(K.GoverningEquationsType_s,valueType,[],
                     K.GoverningEquationsType_ts,node)
  return node
  
# -----------------------------------------------------------------------------
def newGasModel(parent,valueType=K.Ideal_s):
  """-GasModel node creation -GasModel
  
  'newNode:N='*newGasModel*'(parent:N,valueType:K.GasModelType)'
  
   Returns a new <node> representing a K.GasModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name GasModel then
   only the GasModelType is created. 
   chapter 10.3 Add node GasModelType is required  
  """
  node=hasChildName(parent,K.GasModel_s)
  if (node == None):       
    node=newNode(K.GasModel_s,None,[],K.GasModel_ts,parent)
  if (valueType not in K.GasModelType_l): raise E.cgnsException(224,valueType)
  checkDuplicatedName(node,K.GasModelType_s)  
  nodeType=newNode(K.GasModelType_s,valueType,[],K.GasModelType_ts,node)
  return node
  
def newThermalConductivityModel(parent,valueType=K.SutherlandLaw_s):   
  """-ThermalConductivityModel node creation -ThermalConductivityModel
  
  'newNode:N='*newThermalConductivityModel*'(parent:N,valueType:K.ThermalConductivityModelType)'
  
   Returns a new <node> representing a K.ThermalConductivityModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ThermalConductivityModel then
   only the ThermalConductivityModelType is created. 
   chapter 10.5 Add node ThermalConductivityModelType is required     
  """
  node=hasChildName(parent,K.ThermalConductivityModel_s)
  if (node == None):    
    node=newNode(K.ThermalConductivityModel_s,None,[],
                 K.ThermalConductivityModel_ts,parent)
  if (valueType not in K.ThermalConductivityModelType_l):
    raise E.cgnsException(227,valueType)
  checkDuplicatedName(node,K.ThermalConductivityModelType_s)
  nodeType=newNode(K.ThermalConductivityModelType_s,valueType,[],
                     K.ThermalConductivityModelType_ts,node)  
  return node

def newViscosityModel(parent,valueType=K.SutherlandLaw_s): 
  """-ViscosityModel node creation -ViscosityModel
  
  'newNode:N='*newViscosityModel*'(parent:N,valueType:K.ViscosityModelType)'
  
   Returns a new <node> representing a K.ViscosityModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ViscosityModel then
   only the ViscosityModelType is created. 
   chapter 10.4 Add node ViscosityModelType is (r)       
  """  
  node=hasChildName(parent,K.ViscosityModel_s)
  if (node == None):    
    node=newNode(K.ViscosityModel_s,None,[],K.ViscosityModel_ts,parent)    
  if (valueType not in K.ViscosityModelType_l):
    raise E.cgnsException(230,valueType) 
  checkDuplicatedName(node,K.ViscosityModelType_s)  
  nodeType=newNode(K.ViscosityModelType_s,valueType,[],
                     K.ViscosityModelType_ts,node)  
  return node

def newTurbulenceClosure(parent,valueType=K.EddyViscosity_s):   
  """-TurbulenceClosure node creation -TurbulenceClosure
  
  'newNode:N='*newTurbulenceClosure*'(parent:N,valueType:K.TurbulenceClosureType)'  
   Returns a new <node> representing a K.TurbulenceClosure_t sub-tree.  
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name TurbulenceClosure then
   only the ViscosityModelType is created.
   chapter 10.5 Add node TurbulenceClosureType is (r)       
  """
  node=hasChildName(parent,K.TurbulenceClosure_s)
  if (node == None):    
    node=newNode(K.TurbulenceClosure_s,None,[],K.TurbulenceClosure_ts,parent)
  if (valueType not in K.TurbulenceClosureType_l):
    raise E.cgnsException(233,valueType)
  checkDuplicatedName(node,K.TurbulenceClosureType_s)
  nodeType=newNode(K.TurbulenceClosureType_s,valueType,[],
                     K.TurbulenceClosure_ts,node)  
  return node

def newTurbulenceModel(parent,valueType=K.OneEquation_SpalartAllmaras_s): 
  """-TurbulenceModel node creation -TurbulenceModel
  
  'newNode:N='*newTurbulenceModel*'(parent:N,valueType:K.TurbulenceModelType)'
  
   Returns a new <node> representing a K.TurbulenceModel_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name TurbulenceModel then
   only the TurbulenceModelType is created.
   chapter 10.6.2 Add node TurbulenceModelType is (r)  
  """ 
  node=hasChildName(parent,K.TurbulenceModel_s)
  if (node == None):
    node=newNode(K.TurbulenceModel_s,None,[],K.TurbulenceModel_ts,parent)
  if (valueType not in K.TurbulenceModelType_l):
    raise E.cgnsException(236,valueType)  
  checkDuplicatedName(node,K.TurbulenceModelType_s)
  nodeType=newNode(K.TurbulenceModelType_s,valueType,[],
                     K.TurbulenceModelType_ts,node)
  return node

def newThermalRelaxationModel(parent,valueType):
  """-ThermalRelaxationModel node creation -ThermalRelaxationModel
  
  'newNode:N='*newThermalRelaxationModel*'(parent:N,valueType:K.ThermalRelaxationModelType)'
  
   Returns a new <node> representing a K.ThermalRelaxationModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ThermalRelaxationModel then
   only the ThermalRelaxationModelType is created.  
   chapter 10.7 Add node ThermalRelaxationModelType is (r)
  """
  node=hasChildName(parent,K.ThermalRelaxationModel_s) 
  if (node == None):          
    node=newNode(K.ThermalRelaxationModel_s,None,[],
                 K.ThermalRelaxationModel_ts,parent)
  if (valueType not in K.ThermalRelaxationModelType_l):
    raise E.cgnsException(239,valueType) 
  checkDuplicatedName(node,K.ThermalRelaxationModelType_s)   
  nodeType=newNode(K.ThermalRelaxationModelType_s,valueType,[],
                   K.ThermalRelaxationModelType_ts,node)
  return node

def newChemicalKineticsModel(parent,valueType=K.Null_s):
  """-ChemicalKineticsModel node creation -ChemicalKineticsModel
  
  'newNode:N='*newChemicalKineticsModel*'(parent:N,valueType:K.ChemicalKineticsModelType)'
  
   Returns a new <node> representing a K.ChemicalKineticsModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ChemicalKineticsModel then
   only the ChemicalKineticsModelType is created. 
   chapter 10.8 Add node ChemicalKineticsModelType is (r)  
  """
  node=hasChildName(parent,K.ChemicalKineticsModel_s) 
  if (node == None):             
    node=newNode(K.ChemicalKineticsModel_s,None,[],
                 K.ChemicalKineticsModel_ts,parent)
  if (valueType not in K.ChemicalKineticsModelType_l):
    raise E.cgnsException(242,valueType)
  checkDuplicatedName(node,K.ChemicalKineticsModelType_s)     
  nodeType=newNode(K.ChemicalKineticsModelType_s,valueType,[],
                     K.ChemicalKineticsModelType_ts,node)
  return node

def newEMElectricFieldModel(parent,valueType=K.UserDefined_s):
  """-EMElectricFieldModel node creation -EMElectricFieldModel
  
  'newNode:N='*newEMElectricFieldModel*'(parent:N,valueType:K.EMElectricFieldModelType)'
  
   Returns a new <node> representing a K.EMElectricFieldModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
    If the parent has already a child name EMElectricFieldModel then
   only the EMElectricFieldModelType is created. 
   chapter 10.9 Add node EMElectricFieldModelType is (r)  
  """
  node=hasChildName(parent,K.EMElectricFieldModel_s)   
  if (node == None):           
    node=newNode(K.EMElectricFieldModel_s,None,[],
                 K.EMElectricFieldModel_ts,parent)
  if (valueType not in K.EMElectricFieldModelType_l):
    raise E.cgnsException(245,valueType)
  checkDuplicatedName(node,K.EMElectricFieldModelType_s)  
  nodeType=newNode(K.EMElectricFieldModelType_s,valueType,[],
                   K.EMElectricFieldModelType_ts,node)
  return node

def newEMMagneticFieldModel(parent,valueType=K.UserDefined_s):
  """-EMMagneticFieldModel node creation -EMMagneticFieldModel
  
  'newNode:N='*newEMMagneticFieldModel*'(parent:N,valueType:K.EMMagneticFieldModelType)'
  
   Returns a new <node> representing a K.EMMagneticFieldModel_t sub-tree.  
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name EMMagneticFieldModel_s then
   only the EMMagneticFieldModelType is created. 
   chapter 10.9.2 Add node EMMagneticFieldModelType is (r)  
  """
  node=hasChildName(parent,K.EMMagneticFieldModel_s)   
  if (node == None):            
    node=newNode(K.EMMagneticFieldModel_s,None,[],
                 K.EMMagneticFieldModel_ts,parent)
  if (valueType not in K.EMMagneticFieldModelType_l):
    raise E.cgnsException(248,valueType)  
  checkDuplicatedName(node,K.EMMagneticFieldModelType_s) 
  nodeType=newNode(K.EMMagneticFieldModelType_s,valueType,[],
                   K.EMMagneticFieldModelType_ts,node)
  return node

def newEMConductivityModel(parent,valueType=K.UserDefined_s):
  """-EMConductivityModel node creation -EMConductivityModel
  
  'newNode:N='*newEMConductivityModel*'(parent:N,valueType:K.EMConductivityModelType)'
  
   Returns a new <node> representing a K.EMConductivityModel_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name EMConductivityModel then
   only the EMConductivityModelType is created. 
   chapter 10.9.3 Add node EMConductivityModelType is (r)  
  """
  node=hasChildName(parent,K.EMConductivityModel_s)  
  if (node == None):             
    node=newNode(K.EMConductivityModel_s,None,[],
                 K.EMConductivityModel_ts,parent)
  if (valueType not in K.EMConductivityModelType_l):
    raise E.cgnsException(218,stype)  
  checkDuplicatedName(node,K.EMConductivityModelType_s)  
  nodeType=newNode(K.EMConductivityModelType_s,valueType,[],
                   K.EMConductivityModelType_ts,node)
  return node

# -----------------------------------------------------------------------------
def newBaseIterativeData(parent,nsteps=0,itype=K.IterationValues_s):
  """-BaseIterativeData node creation -BaseIterativeData
  
   'newNode:N='*newBaseIterativeData*'(parent:N,nsteps:I,itype:E)'
  
   Returns a new <node> representing a BaseIterativeData_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter 11.1.1
   NumberOfSteps is required, TimeValues or IterationValues are required
  """ 
  
  if (parent): checkNode(parent)
  checkDuplicatedName(parent,K.BaseIterativeData_s)
  checkType(parent,K.CGNSBase_ts,K.BaseIterativeData_ts)
  if ((type(nsteps) != type(1)) or (nsteps < 0)): raise E.cgnsException(209)
  node=newNode(K.BaseIterativeData_s,nsteps,[],K.BaseIterativeData_ts,parent)
  if (itype not in [K.IterationValues_s, K.TimeValues_s]):
    raise E.cgnsException(210,(K.IterationValues_s, K.TimeValues_s))
  newNode(itype,None,[],K.DataArray_ts,node)  
  return node

# -----------------------------------------------------------------------------
def newZoneIterativeData(parent,name):
  """-ZoneIterativeData node creation -ZoneIterativeData
  
  'newNode:N='*newZoneIterativeData*'(parent:N,name:S)'
  
   Returns a new <node> representing a ZoneIterativeData_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  11.1.2
  """ 
  checkDuplicatedName(parent,name)
  node=newNode(name,None,[],K.ZoneIterativeData_ts,parent)
  return node

# ---------------------------------------------------------------------------  
def newRigidGridMotion(parent,name,valueType=K.Null_s,vector=[0.0,0.0,0.0]):
  """-RigidGridMotion node creation -RigidGridMotion
  
  'newNode:N='*newRigidGridMotion*'(parent:N,name:S,valueType:K.RigidGridMotionType,vector:A)'
  
  If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a K.RigidGridMotion_t sub-tree.  
   If the parent has already a child name RigidGridMotion then
   only the RigidGridMotionType is created and OriginLocation is created
   chapter 11.2 Add Node RigidGridMotionType and add DataArray OriginLocation are the only required
  """
  if (parent): checkNode(parent)  
  checkDuplicatedName(parent,name)
  node=newNode(name,None,[],K.RigidGridMotion_ts,parent)
  
  if (valueType not in K.RigidGridMotionType_l):
      raise E.cgnsException(254,valueType)
  checkDuplicatedName(parent,K.RigidGridMotionType_s,)
  nodeType=newNode(K.RigidGridMotionType_s,valueType,[],
                     K.RigidGridMotionType_ts,node)
  n=hasChildName(parent,K.OriginLocation_s)
  if (n == None): 
    n=newDataArray(node,K.OriginLocation_s,N.array(vector))
  return node
  
#-----------------------------------------------------------------------------
def newReferenceState(parent,name=K.ReferenceState_s):
  """-ReferenceState node creation -ReferenceState
  
  'newNode:N='*newReferenceState*'(parent:N,name:S)'
  
   Returns a new <node> representing a ReferenceState_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.1  """   
  if (parent): checkNode(parent)
  node=hasChildName(parent,name)
  if (node == None):
    checkDuplicatedName(parent,name)
    node=newNode(name,None,[],K.ReferenceState_ts,parent)
  return node

#-----------------------------------------------------------------------------
def newConvergenceHistory(parent,name=K.GlobalConvergenceHistory_s,
			  iterations=0):
  """-ConvergenceHistory node creation -ConvergenceHistory
  
  'newNode:N='*newConvergenceHistory*'(parent:N,name:S,iterations:i)'
  
   Returns a new <node> representing a ConvergenceHistory_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.3  """   
  if (name not in K.ConvergenceHistory_l): raise E.cgnsException(201,name)
  if (parent):
    checkNode(parent)
    checkTypeList(parent,[K.CGNSBase_ts,K.Zone_ts],name)
  if (name == K.GlobalConvergenceHistory_s):
    checkType(parent,K.CGNSBase_ts,name)
  if (name == K.ZoneConvergenceHistory_s):
    checkType(parent,K.Zone_ts,name)
  checkDuplicatedName(parent,name)
  node=newNode(name,iterations,[],K.ConvergenceHistory_ts,parent)
  return node

#-----------------------------------------------------------------------------
def newIntegralData(parent,name):
  """-IntegralData node creation -IntegralData
  
  'newNode:N='*newIntegralData*'(parent:N,name:S)'
  
   Returns a new <node> representing a IntegralData_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.5 
  """ 
  checkDuplicatedName(parent,name)
  node=newNode(name,None,[],K.IntegralData_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newFamily(parent,name):
  """-Family node creation -Family

  'newNode:N='*newFamily*'(parent:N,name:S)'
  
   Returns a new <node> representing a Family_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.6 
  """ 
  if (parent): checkNode(parent)
  checkType(parent,K.CGNSBase_ts,name)
  checkDuplicatedName(parent,name)
  node=newNode(name,None,[],K.Family_ts,parent)
  return node

def newFamilyName(parent,family=None): 
  return newNode(K.FamilyName_s,family,[],K.FamilyName_ts,parent)

# -----------------------------------------------------------------------------
def newGeometryReference(parent,name='{GeometryReference}',
                         valueType=K.UserDefined_s):
  """-GeometryReference node creation -GeometryReference
  
  'newNode:N='*newGeometryReference*'(parent:N,name:S,valueType:K.GeometryFormat)'
  
   Returns a new <node> representing a K.GeometryFormat_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name K.GeometryReference then
   only the .GeometryFormat is created
   chapter  12.7 Add node K.GeometryFormat_t is (r) and GeometryFile_t definition not find but is required (CAD file)
  """
  node=hasChildName(parent,K.GeometryReference_s)
  if (node == None):    
    node=newNode(name,None,[],K.GeometryReference_ts,parent)
  if (valueType not in K.GeometryFormat_l):
      raise E.cgnsException(256,valueType)
  checkDuplicatedName(node,K.GeometryFormat_s)
  nodeType=newNode(K.GeometryFormat_s,valueType,[],
                     K.GeometryFormat_ts,node)
  return node
  
# -----------------------------------------------------------------------------
def newFamilyBC(parent,valueType=K.UserDefined_s): 
  """-FamilyBC node creation -FamilyBC
  
  'newNode:N='*newFamilyBC*'(parent:N,valueType:K.BCTypeSimple/K.BCTypeCompound)'
  
   Returns a new <node> representing a K.FamilyBC_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name FamilyBC then
   only the BCType is created
   chapter  12.8 Add node BCType is required   
  """ 
  node=hasChildName(parent,K.FamilyBC_s)
  if (node == None):    
    node=newNode(K.FamilyBC_s,None,[],K.FamilyBC_ts,parent)
  if (    valueType not in K.BCTypeSimple_l
      and valueType not in K.BCTypeCompound_l):
      raise E.cgnsException(257,valueType)
  checkDuplicatedName(node,K.BCType_s)
  nodeType=newNode(K.BCType_s,valueType,[],
                     K.BCType_ts,node)
  return node

# -----------------------------------------------------------------------------
def newArbitraryGridMotion(parent,name,valuetype=K.Null_s):
  """-ArbitraryGridMotion node creation -ArbitraryGridMotion
  
  'newNode:N='*newArbitraryGridMotion*'(parent:N,name:S,valuetype:E)'
  
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name RigidGridMotion then
   only the RigidGridMotionType is created.
   The valuetype enumerate should be in the K.ArbitraryGridMotionType list.
   Returns a new <node> representing a ArbitraryGridMotionType_t sub-tree.  
   chapter 11.3 Add Node ArbitraryGridMotionType is required
  """
  node=None
  if (parent): node=hasChildName(parent,name)
  if (node == None):
    node=newNode(name,None,[],K.ArbitraryGridMotion_ts,parent)      
  if (valuetype not in K.ArbitraryGridMotionType_l):
    raise E.cgnsException(255,valueType) 
  checkDuplicatedName(node,K.ArbitraryGridMotionType_s)     
  nodeType=newNode(K.ArbitraryGridMotionType_s,valuetype,[],
                   K.ArbitraryGridMotionType_ts,node)
  return node
  
# -----------------------------------------------------------------------------
def newUserDefinedData(parent,name):
  """-UserDefinedData node creation -UserDefinedData
  
  'newNode:N='*newUserDefinedData*'(parent:N,name:S)'
  
   Returns a new <node> representing a UserDefinedData_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   chapter  12.9  
  """ 
  checkDuplicatedName(parent,name)
  node=newNode(name,None,[],K.UserDefinedData_ts,parent)
  return node
 
# -----------------------------------------------------------------------------
def newGravity(parent,gvector=[0.0,0.0,0.0]):
  """-Gravity node creation -Gravity
  
  'newNode:N='*newGravity*'(parent:N,gvector:A)'
  
   Returns a new <node> representing a Gravity_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   gvector should be a real array
   chapter  12.10 Add DataArray GravityVector is required   
  """ 
  if (parent): checkNode(parent)
  checkType(parent,K.CGNSBase_ts,K.Gravity_s)
  checkDuplicatedName(parent,K.Gravity_s)
  checkArrayReal(gvector)
  node=newNode(K.Gravity_s,None,[],K.Gravity_ts,parent)
  n=hasChildName(parent,K.GravityVector_s)
  if (n == None): 
    n=newDataArray(node,K.GravityVector_s,N.array(gvector))
  return node

# -----------------------------------------------------------------------------
def newField(parent,name,value):
  checkDuplicatedName(parent,name)
  node=newDataArray(parent,name,value)
  return node

# -----------------------------------------------------------------------------
def newModel(parent,name,label,value):
  checkDuplicatedName(parent,name)
  node=newNode(name,value,[],label,parent)
  return node
    
# -----------------------------------------------------------------------------
def newDiffusionModel(parent):
  # the diffusion_t doesn't exist. We use the cgnspatch file to keep
  # track of this...
  checkDuplicatedName(parent,K.DiffusionModel_s)
  node=newNode(K.DiffusionModel_s,None,[],K.DiffusionModel_ts,parent)
  return node

# -----------------------------------------------------------------------------
#def newSection():
#  pass

# -----------------------------------------------------------------------------
#def newParentData():
#  pass

# -----------------------------------------------------------------------------
#def newPart():
#  pass

