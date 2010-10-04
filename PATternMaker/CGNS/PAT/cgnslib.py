#  ---------------------------------------------------------------------------
#  pyCGNS.PAT - Python package for CFD General Notation System - PATternMaker
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#  $Release$
#  ---------------------------------------------------------------------------

import CGNS.PAT.cgnskeywords as CG_K
import CGNS.PAT.cgnserrors   as CG_E
import CGNS

__CGNS_LIBRARY_VERSION__=2.4

import types
import numpy as NPY

C1=CG_K.C1
MT=CG_K.MT
I4=CG_K.I4
I8=CG_K.I8
R4=CG_K.R4
R8=CG_K.R8
LK=CG_K.LK
DT=[C1,MT,I4,I8,R4,R8,LK] # LK declared as data type

zero_N=(0,-1)
one_N=(1,-1)
N_N=(-1,-1)
zero_one=(0,1)

typeListA=[
    CG_K.Descriptor_ts,
    CG_K.UserDefinedData_ts,
    CG_K.DataClass_ts,
    CG_K.DimensionalUnits_ts
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
  if (type(name) != type("s")): raise CG_E.cgnsException(22)
  if (len(name) == 0): raise CG_E.cgnsException(23)
  if ('/' in name): raise CG_E.cgnsException(24)
  if (len(name) > 32): raise CG_E.cgnsException(25)

# -----------------------------------------------------------------------------
def checkDuplicatedName(parent,name):
  if (not parent): return
  if (parent[2] == None): return
  checkName(name)
  for nc in parent[2]:
    if (nc[0] == name): raise CG_E.cgnsException(102,(name,parent[0]))

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
    if (v.dtype.kind in ['S','a']): return CG_K.Character_s
    if (v.dtype.char in ['f']):     return CG_K.RealSingle_s
    if (v.dtype.char in ['d']):     return CG_K.RealDouble_s
    if (v.dtype.char in ['i']):     return CG_K.Integer_s
    if (v.dtype.char in ['l']):     return CG_K.Integer_s
  return None
   
# -----------------------------------------------------------------------------
def setValue(node,value):
  t=getValueType(value)
  if (t == None): node[1]=None
  if (t in [CG_K.Integer_s,CG_K.RealDouble_s,
            CG_K.RealSingle_s,CG_K.Character_s]): node[1]=value
  return node
  
# -----------------------------------------------------------------------------
def setStringAsArray(a):
  if (type(a)==type("")):
    return NPY.array(tuple(a),dtype='|S',order='Fortran')
  if (type(a)==type(NPY.array((1)))):
    return a
  return None

# -----------------------------------------------------------------------------
# useless
def getValue(node):
  v=node[1]
  t=getValueType(v)
  if (t == None):           return None
  if (t == CG_K.Integer_s):    return v
  if (t == CG_K.RealDouble_s): return v
  if (t == CG_K.Character_s):  return v
  return v
  
# -----------------------------------------------------------------------------
def checkArray(a):
  if (type(a) != type(NPY.array((1)))): raise CG_E.cgnsException(109)
  if ((len(a.shape)>1) and not NPY.isfortran(a)):
    raise CG_E.cgnsException(710)  

# -----------------------------------------------------------------------------
def checkArrayChar(a):
  checkArray(a)
  if (a.dtype.char not in ['S','a']):  raise CG_E.cgnsException(105)
  return a

# -----------------------------------------------------------------------------
def checkArrayReal(a):
  checkArray(a)
  if (a.dtype.char not in ['d','f']):  raise CG_E.cgnsException(106)
  return a

# -----------------------------------------------------------------------------
def checkArrayInteger(a):
  checkArray(a)
  if (a.dtype.char not in ['i','u']):  raise CG_E.cgnsException(107)
  return a

# -----------------------------------------------------------------------------
def checkType(parent,stype,name):
  if (parent == None): return None
  if (parent[3] != stype): 
    raise CG_E.cgnsException(103,(name,stype))
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
    raise CG_E.cgnsException(104,(name,ltype))
  return None

# -----------------------------------------------------------------------------
def checkParent(node,dienow=0):
  if (node == None): return 1
  return checkNode(node,dienow)

# -----------------------------------------------------------------------------
def checkNode(node,dienow=0):
  if (node in [ [], None ]):
    if (dienow): raise CG_E.cgnsException(1)
    return 0
  if (type(node) != type([3,])):
    if (dienow): raise CG_E.cgnsException(2)
    return 0
  if (len(node) != 4):
    if (dienow): raise CG_E.cgnsException(2)
    return 0
  if (type(node[0]) != type("")):
    if (dienow): raise CG_E.cgnsException(3)
    return 0
  if (type(node[2]) != type([3,])):
    if (dienow): raise CG_E.cgnsException(4,node[0])
    return 0
  if ((node[1] != None) and (type(node[1])) != type(NPY.array([3,]))):
    if (dienow): raise CG_E.cgnsException(5,node[0])
    return 0
  return 1
    
# -----------------------------------------------------------------------------
def isRootNode(node,dienow=0):
#   """isRootNode :
#   Check wether a node is a CGNS root node (returns 1)
#   or not a root node (returns 0).    
#   A root node is a list of a single CGNSLibraryVersion_t node and zero or more
#   CGNSBase_t nodes. We do not check first level type, no standard for it, even
#   if we set it to CGNSTree."""
  if (node in [ [], None ]):         return 0
  versionfound=0
  if (not checkNode(node)):          return 0
  for n in node[2]:
     if (not checkNode(n,dienow)):   return 0 
     if (     (n[0] == CG_K.CGNSLibraryVersion_s)
          and (n[3] == CG_K.CGNSLibraryVersion_ts) ):
         if versionfound: raise CG_E.cgnsException(99)
         versionfound=1
     elif ( n[3] != CG_K.CGNSBase_ts ): return 0
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
def newDataClass(parent,value=CG_K.UserDefined_s):
  """-DataClass node creation -DataClass
  
  'newNode:N='*newDataClass*'(parent:N,value:A)'
  
  If a parent is given, the new <node> is added to the parent children list.
  The value argument is a DataClass enumerate. No child allowed.
  Returns a new <node> representing a DataClass_t sub-tree."""
  checkDuplicatedName(parent,CG_K.DataClass_s)
  node=newNode(CG_K.DataClass_s,setStringAsArray(value),[],
               CG_K.DataClass_ts,parent)
  return checkDataClass(node)

def updateDataClass(node,value):
  checkNode(node)  
  if (value!=None): node[1]=value
  return checkDataClass(node)

def checkDataClass(node,parent=None):
  checkNode(node)
  checkName(node[0])
  if (node[0] != CG_K.DataClass_s):  raise CG_E.cgnsException(26,node[0])
  if (node[3] != CG_K.DataClass_ts): raise CG_E.cgnsException(27,node[3])
  if (len(node[2]) != 0):         raise CG_E.cgnsException(28,node[0])
  value=getValue(node).tostring()
  if (value not in CG_K.DataClass_l):  raise CG_E.cgnsException(207,value)
  if (parent != None):
     checkTypeList(parent,[CG_K.DataArray_ts,CG_K.CGNSBase_ts,CG_K.Zone_ts,
                           CG_K.GridCoordinates_ts,CG_K.Axisymmetry_ts,
                           CG_K.RotatingCoordinates_ts,CG_K.FlowSolution_ts,
                           CG_K.Periodic_ts,CG_K.ZoneBC_ts,CG_K.BC_ts,
                           CG_K.BCDataSet_ts,
                           CG_K.BCData_ts,CG_K.FlowEquationSet_ts,
                           CG_K.GasModel_ts,
                           CG_K.ViscosityModel_ts,
                           CG_K.ThermalConductivityModel_ts,
                           CG_K.TurbulenceClosure_ts,CG_K.TurbulenceModel_ts,
                           CG_K.ThermalRelaxationModel_ts,
                           CG_K.ChemicalKineticsModel_ts,
                           CG_K.EMElectricFieldModel_ts,
                           CG_K.EMMagneticFieldModel_ts,
                           CG_K.EMConductivityModel_ts,
                           CG_K.BaseIterativeData_ts,
                           CG_K.ZoneIterativeData_ts,CG_K.RigidGridMotion_ts,
                           CG_K.ArbitraryGridMotion_ts,CG_K.ReferenceState_ts,
                           CG_K.ConvergenceHistory_ts,
                           CG_K.DiscreteData_ts,CG_K.IntegralData_ts,
                           CG_K.UserDefinedData_ts,CG_K.Gravity_ts]
                   ,CG_K.DataClass_s)
  return node
  
# -----------------------------------------------------------------------------
def newDescriptor(parent,name,value=NPY.array([''])):
  """-Descriptor node creation -Descriptor
  
  'newNode:N='*newDescriptor*'(parent:N,name:S,text:A)'
  
  No child allowed.
  Returns a new <node> representing a Descriptor_t sub-tree."""
  checkDuplicatedName(parent,name)
  node=newNode(name,setStringAsArray(value),[],CG_K.Descriptor_ts,parent)
  return checkDescriptor(node)
  
def checkDescriptor(node,parent=None):
  checkNode(node)
  checkName(node[0])
  if (node[3] != CG_K.Descriptor_ts): raise CG_E.cgnsException(27,node[3])
  if (len(node[2]) != 0):             raise CG_E.cgnsException(28,node[0])
  value=getValue(node)
  if (getValueType(value) != CG_K.Character_s):
                                      raise CG_E.cgnsException(110,node[0])
  if (parent != None):
     checkTypeList(parent,[CG_K.DataArray_ts,CG_K.CGNSBase_ts,CG_K.Zone_ts,
                           CG_K.GridCoordinates_ts,CG_K.Elements_ts,
                           CG_K.Axisymmetry_ts,
                           CG_K.RotatingCoordinates_ts,CG_K.FlowSolution_ts,
                           CG_K.ZoneGridConnectivity_ts,
                           CG_K.GridConnectivity1to1_ts,
                           CG_K.GridConnectivity_ts,
                           CG_K.GridConnectivityProperty_ts,
                           CG_K.AverageInterface_ts,CG_K.OversetHoles_ts,
                           CG_K.Periodic_ts,CG_K.ZoneBC_ts,CG_K.BC_ts,
                           CG_K.BCDataSet_ts,
                           CG_K.BCData_ts,CG_K.FlowEquationSet_ts,
                           CG_K.GasModel_ts,
                           CG_K.BCProperty_ts,CG_K.WallFunction_ts,
                           CG_K.Area_ts,
                           CG_K.GoverningEquations_ts,
                           CG_K.ViscosityModel_ts,
                           CG_K.ThermalConductivityModel_ts,
                           CG_K.TurbulenceClosure_ts,CG_K.TurbulenceModel_ts,
                           CG_K.ThermalRelaxationModel_ts,
                           CG_K.ChemicalKineticsModel_ts,
                           CG_K.EMElectricFieldModel_ts,
                           CG_K.EMMagneticFieldModel_ts,
                           CG_K.EMConductivityModel_ts,
                           CG_K.BaseIterativeData_ts,
                           CG_K.ZoneIterativeData_ts,CG_K.RigidGridMotion_ts,
                           CG_K.ArbitraryGridMotion_ts,CG_K.ReferenceState_ts,
                           CG_K.ConvergenceHistory_ts,
                           CG_K.DiscreteData_ts,CG_K.IntegralData_ts,
                           CG_K.Family_ts,CG_K.GeometryReference_ts,
                           CG_K.UserDefinedData_ts,CG_K.Gravity_ts]
                   ,CG_K.DataClass_s)
  return node

# -----------------------------------------------------------------------------
def newDimensionalUnits(parent,value=[CG_K.Meter_s,CG_K.Kelvin_s,
                                      CG_K.Second_s,CG_K.Radian_s,
                                      CG_K.Kilogram_s]):
  """-DimensionalUnits node creation -DimensionalUnits
  
  'newNode:N='*newDimensionalUnits*'(parent:N,value=[CG_K.MassUnits,CG_K.LengthUnits,
                                     CG_K.TimeUnits,CG_K.TemperatureUnits,
                                     CG_K.AngleUnits])'
                                      
  If a parent is given, the new <node> is added to the parent children list.
  new <node> is composed of a set of enumeration types : MassUnits,LengthUnits,
  TimeUnits,TemperatureUnits,AngleUnits are required
  Returns a new <node> representing a DimensionalUnits_t sub-tree.
  chapter 4.3 
  """
  if (len(value) != 5): raise CG_E.cgnsException(202)
  checkDuplicatedName(parent,CG_K.DimensionalUnits_s)
  # --- loop over values to find all required units
  vunit=[CG_K.Null_s,CG_K.Null_s,CG_K.Null_s,CG_K.Null_s,CG_K.Null_s]
  for v in value:
    if (v not in CG_K.AllUnits_l): raise CG_E.cgnsException(203,v)
    if ((v in CG_K.MassUnits_l)
        and (v not in [CG_K.Null_s,CG_K.UserDefined_s])):
      if (v in vunit): raise CG_E.cgnsException(204,v)
      else:            vunit[0]=v
    if ((v in CG_K.LengthUnits_l)
        and (v not in [CG_K.Null_s,CG_K.UserDefined_s])):
      if (v in vunit): raise CG_E.cgnsException(204,v)
      else:            vunit[1]=v
    if ((v in CG_K.TimeUnits_l)
        and (v not in [CG_K.Null_s,CG_K.UserDefined_s])):
      if (v in vunit): raise CG_E.cgnsException(204,v)
      else:            vunit[2]=v
    if ((v in CG_K.TemperatureUnits_l)
        and (v not in [CG_K.Null_s,CG_K.UserDefined_s])):
      if (v in vunit): raise CG_E.cgnsException(204,v)
      else:            vunit[3]=v
    if ((v in CG_K.AngleUnits_l)
        and (v not in [CG_K.Null_s,CG_K.UserDefined_s])):
      if (v in vunit): raise CG_E.cgnsException(204,v)
      else:            vunit[4]=v
  node=newNode(CG_K.DimensionalUnits_s,concatenateForArrayChar(vunit),[],
               CG_K.DimensionalUnits_ts,parent)
  snode=newNode(CG_K.AdditionalUnits_s,
                concatenateForArrayChar([CG_K.Null_s,CG_K.Null_s,CG_K.Null_s]),
                [],
                CG_K.AdditionalUnits_ts,node)
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
  checkDuplicatedName(parent,CG_K.DimensionalExponents_s)
  node=newNode(CG_K.DimensionalExponents_s,
               NPY.array([MassExponent,
                          LengthExponent,
                          TimeExponent,
                          TemperatureExponent,
                          AngleExponent],dtype='Float64',order='Fortran'),
               [],CG_K.DimensionalExponents_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newGridLocation(parent,value=CG_K.CellCenter_s):
  """-GridLocation node creation -GridLocation
  
  'newNode:N='*newGridLocation*'(parent:N,value:CG_K.GridLocation)'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a GridLocation_t sub-tree.
  chapter 4.5
  """
  checkDuplicatedName(parent,CG_K.GridLocation_s)
  if (value not in CG_K.GridLocation_l): raise CG_E.cgnsException(200,value)
  ## code correction: Modify value string into NPY string array
  node=newNode(CG_K.GridLocation_s,setStringAsArray(value),[],CG_K.GridLocation_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newIndexArray(parent,name,value=[]):
  checkDuplicatedName(parent,name)
  node=newNode(name,value,[],CG_K.IndexArray_ts,parent)
  return node
  
def newPointList(parent,name=CG_K.PointList_s,value=[]):
  """-PointList node creation -PointList
  
  'newNode:N='*newPointList*'(parent:N,name:S,value:[])'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a IndexArray_t sub-tree.
  chapter 4.6
  """
  checkDuplicatedName(parent,name)
  node=newNode(name,value,[],CG_K.IndexArray_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newPointRange(parent,name=CG_K.PointRange_s,value=[]):
  """-PointRange node creation -PointRange
  
  'newNode:N='*newPointRange*'(parent:N,name:S,value:[])'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a IndexRange_t sub-tree.
  chapter 4.7
  """
  checkDuplicatedName(parent,name)
  node=newNode(name,value,[],CG_K.IndexRange_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newRind(parent,value):                                    
  """-Rind node creation -Rind
  
  'newNode:N='*newRind*'(parent:N,value=A)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a Rind_t sub-tree.
  chapter 4.8
  """
  checkDuplicatedName(parent,CG_K.Rind_s)
  # check value wrt base dims
  node=newNode(CG_K.Rind_s,value,[],CG_K.Rind_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newDataConversion(parent,ConversionScale=1.0,ConversionOffset=1.0):
  """-DataConversion node creation -DataConversion
  
  'newNode:N='*newDataConversion*'(parent:N,ConversionScale:r,ConversionOffset:r)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a DataConversion_t sub-tree.
  chapter  5.1.1
  """
  checkDuplicatedName(parent,CG_K.DataConversion_s)
  node=newNode(CG_K.DataConversion_s,
               NPY.array([ConversionScale,ConversionOffset],
                         dtype='Float64',order='Fortran'),
               [],CG_K.DataConversion_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newCGNS():
  """-Tree node creation -Tree

  'newNode:N='*newCGNS*'()'

  Returns a new <node> representing a CGNS tree root.
  This is not a SIDS type.
  """
  ##code correction: Modify CGNS value type from float64 to float32.
  node=[CG_K.CGNSLibraryVersion_s,NPY.array([__CGNS_LIBRARY_VERSION__],dtype='float32'),[],
        CG_K.CGNSLibraryVersion_ts]
  badnode=[CG_K.CGNSTree_s,None,[node],CG_K.CGNSTree_ts]
  return badnode

# ----------------------------------------------------------------------------
def newSimulationType(parent,stype=NPY.array(CG_K.NonTimeAccurate_s)):
  """-SimulationType node creation -SimulationType
  
  'newNode:N='*newSimulationType*'(parent:N,stype=CG_K.SimulationType)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a SimulationType_t sub-tree.
  chapter 6.2
  """
  if (parent): checkNode(parent)
  checkDuplicatedName(parent,CG_K.SimulationType_s)
  checkType(parent,CG_K.CGNSBase_ts,CG_K.SimulationType_s)
  if (stype not in CG_K.SimulationType_l): raise CG_E.cgnsException(205,stype)
  node=newNode(CG_K.SimulationType_s,stype,[],CG_K.SimulationType_ts,parent)
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
  if (ncell not in [1,2,3]): raise CG_E.cgnsException(10,name)
  if (nphys not in [1,2,3]): raise CG_E.cgnsException(11,name)
  if (nphys < ncell):        raise CG_E.cgnsException(12,name)
  if ((tree != None) and (not checkNode(tree))):
     raise CG_E.cgnsException(6,name)
  if ((tree != None) and (tree[0] == CG_K.CGNSTree_s)): parent=tree[2]  
  else:                                              parent=tree
  checkDuplicatedName(["<root node>",None,parent],name)
  node=newNode(name,
               NPY.array([ncell,nphys],dtype=NPY.int32,order='Fortran'),
               [],CG_K.CGNSBase_ts)
  if (parent != None): parent.append(node)
  return node

def numberOfBases(tree):
  return len(hasChildrenType(tree,CG_K.CGNSBase_ts))

def readBase(tree,name):
  b=hasChildName(tree,name)
  if (b == None): raise CG_E.cgnsException(21,name)
  if (b[3] != CG_K.CGNSBase_ts):
    raise CG_E.cgnsException(20,(CG_K.CGNSBase_ts,name))
  return (b[0],b[1])
  
def updateBase(tree,name=None,ncell=None,nphys=None):
  if (ncell not in [1,2,3]): raise CG_E.cgnsException(10,name)
  if (nphys not in [1,2,3]): raise CG_E.cgnsException(11,name)
  if (nphys < ncell):        raise CG_E.cgnsException(12,name)
  
  if (tree): checkNode(tree)

  if (tree[3] != CG_K.CGNSBase_ts):
    raise CG_E.cgnsException(20,(CG_K.CGNSBase_ts,name))
  if(name!=None): tree[0]=name
  if(ncell!=None and nphys!=None and tree):
    tree[1]=NPY.array([ncell,nphys],dtype=NPY.int32,order='Fortran')
  else: raise CG_E.cgnsException(12)  
  
 
# -----------------------------------------------------------------------------
def newOrdinal(parent,value=0):
  """-Ordinal node creation -Ordinal
  
  'newNode:N='*newOrdinal*'(parent:N,value=i)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a Ordinal_t sub-tree.
  chapter 6.3
  """
  checkDuplicatedName(parent,CG_K.Ordinal_s)
  node=newNode(CG_K.Ordinal_s,value,[],CG_K.Ordinal_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newZone(parent,name,size=(2,2,2),
            ztype=CG_K.Structured_s,
            family=''):
  """-Zone node creation -Zone
  
  'newNode:N='*newZone*'(parent:N,name:S,size:(I*),ztype:CG_K.ZoneType)'
  
  Returns a new <node> representing a Zone_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list.
  Maps the 'cg_zone_write' MLL
  chapter 6.3
  """
  asize=None
  if (ztype not in CG_K.ZoneType_l): raise CG_E.cgnsException(206,ztype)
  if ((len(size) == 3) and (ztype == CG_K.Structured_s)):
    ##size=[[size[0],size[1],size[2]],[size[0]-1,size[1]-1,size[2]-1],[0,0,0]]
    ## code correction: Modify array dimensions:
    size=[[size[0],size[0]-1,0],[size[1],size[1]-1,0],[size[2],size[2]-1,0]]    
    asize=NPY.array(size,dtype=NPY.int32,order='Fortran')
  if ((len(size) == 2) and (ztype == CG_K.Structured_s)):
    size=[[size[0],size[1]],[size[0]-1,size[1]-1],[0,0]]
    asize=NPY.array(size,dtype=NPY.int32,order='Fortran')
  if ((len(size) == 1) and (ztype == CG_K.Structured_s)):
    size=[[size[0][1]],[size[0]-1],[0]]
    asize=NPY.array(size,dtype=NPY.int32,order='Fortran')
  if (ztype == CG_K.Unstructured_s):
    asize=NPY.array(size,dtype=NPY.int32,order='Fortran')
  if (asize == None): raise CG_E.cgnsException(999) 
  checkDuplicatedName(parent,name)
  znode=newNode(name,asize,[],CG_K.Zone_ts,parent)
  ## code correction: Modify ztype string into NPY string array
  newNode(CG_K.ZoneType_s,setStringAsArray(ztype),[],CG_K.ZoneType_ts,znode)
  if (family):
    ## code correction: Modify family string into NPY string array
    newNode(CG_K.FamilyName_s,setStringAsArray(family),[],CG_K.FamilyName_ts,znode)
  return znode

def numberOfZones(tree,basename):
  b=hasChildName(tree,basename)
  if (b == None): raise CG_E.cgnsException(21,basename)
  if (b[3] != CG_K.CGNSBase_ts): raise CG_E.cgnsException(20,(CG_K.CGNSBase_ts,name))
  return len(hasChildrenType(b,CG_K.Zone_ts))

def readZone(tree,basename,zonename,gtype=None):
  b=hasChildName(tree,basename)
  if (b == None): raise CG_E.cgnsException(21,basename)
  if (b[3] != CG_K.CGNSBase_ts): raise CG_E.cgnsException(20,(CG_K.CGNSBase_ts,name))
  z=hasChildName(b,zonename)
  if (z == None): raise CG_E.cgnsException(21,zonename)
  if (z[3] != CG_K.Zone_ts): raise CG_E.cgnsException(20,(CG_K.Zone_ts,name))
  if gtype: 
    zt=hasChildName(z,CG_K.ZoneType_s)
    if (zt == None): raise CG_E.cgnsException(21,CG_K.ZoneType_s)
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
  node=newNode(name,None,[],CG_K.GridCoordinates_ts,parent)
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
  ## code correction:  Add value type and fortran order
  ## code correction:  Add a specific array for string type
  ## code correction:  Modify array check
  if (type(value)==type(3)):
    vv=NPY.array([value],dtype=NPY.int32,order='Fortran')
    checkArray(vv)
  elif(type(value)==type(3.2)):
    vv=NPY.array([value],dtype=NPY.float32,order='Fortran')
    checkArray(vv)
  elif(type(value)==type("s")):
    vv=setStringAsArray(value)
    checkArrayChar(vv)
  else:
    vv=value
    if (vv != None): checkArray(vv)
    
  node=newNode(name,vv,[],CG_K.DataArray_ts,parent)
  return node

def numberOfDataArrays(parent):
  return len(hasChildrenType(parent,CG_K.DataArray_ts))

def readDataArray(parent,name):
  n=hasChildName(parent,name)
  if (n == None): raise CG_E.cgnsException(21,name)
  if (n[3] != CG_K.DataArray_ts): raise CG_E.cgnsException(20,(CG_K.DataArray_ts,name))
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
  node=newNode(name,None,[],CG_K.DiscreteData_ts,parent)
  return node 
  
# -----------------------------------------------------------------------------
def newElements(parent,
                elementstype=CG_K.UserDefined_s,
                elementsconnectivity=None,
                elementsrange=None):
  """-Elements node creation -Elements
  
  'newNode:N='*newAElements*'(parent:N,elementsType:CG_K.ElementType,value:CG_K.ElementConnectivity)'
  
   Returns a new <node> representing a Element_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   If the parent has already a child name Element then
   only the ElementType,IndexRange_t,ElementConnectivity are created.
   chapter 7.3 Add node :ElementType,IndexRange_t are required
               Add DataArray : ElementConnectivity is required
  """
  enode=hasChildName(parent,CG_K.Element_s)
  if (enode == None):
    enode=newNode(CG_K.Element_s,None,[],CG_K.Element_ts,parent)
  if (elementstype not in CG_K.ElementType_l):
    raise CG_E.cgnsException(250,elementstype)
  checkDuplicatedName(enode,CG_K.ElementType_s)   
  ccnode=newNode(CG_K.ElementType_s,setStringAsArray(elementstype),[],
                 CG_K.ElementType_ts,enode)
  newDataArray(enode,CG_K.ElementConnectivity_s,elementsconnectivity)
  checkDuplicatedName(enode,CG_K.ElementRange_s) 
  cnode=newNode(CG_K.ElementRange_s,elementsrange,[],CG_K.IndexRange_ts,enode)  
  return enode

# -----------------------------------------------------------------------------
def newZoneBC(parent):
  return newNode(CG_K.ZoneBC_s,None,[],CG_K.ZoneBC_ts,parent)

def newBC(parent,bname,brange=[0,0,0,0,0,0],
          btype=CG_K.Null_s,bcType=CG_K.Null_s,
          family=CG_K.Null_s,pttype=CG_K.PointRange_s):
  return newBoundary(parent,bname,brange,btype,bcType,pttype) 

def newBoundary(parent,bname,brange,
                btype=NPY.array(CG_K.Null_s),
                family=None,
                pttype=NPY.array(CG_K.PointRange_s)): 
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
  zbnode=hasChildName(parent,CG_K.ZoneBC_s)
  if (zbnode == None): zbnode=newNode(CG_K.ZoneBC_s,None,[],CG_K.ZoneBC_ts,parent)
  ## code correction: Modify btype string into NPY string array
  bnode=newNode(bname,setStringAsArray(btype),[],CG_K.BC_ts,zbnode)
  if (pttype==CG_K.PointRange_s):
    ## code correction: modify reshape size and order. Result is unchange.
    arange=NPY.array(brange,dtype=NPY.int32,order='Fortran')
    newNode(CG_K.PointRange_s,arange,[],CG_K.IndexRange_ts,bnode)
  else:
    ## code correction: Add order.
    arange=NPY.array(brange,dtype=NPY.int32,order='Fortran')
    newNode(CG_K.PointList_s,arange,[],CG_K.IndexArray_ts,bnode)
  if (family):
    ## code correction: Modify family string into NPY string array
    newNode(CG_K.FamilyName_s,setStringAsArray(family),[],CG_K.FamilyName_ts,bnode)
  return bnode
  
# -----------------------------------------------------------------------------
def newBCDataSet(parent,name,valueType=NPY.array(CG_K.Null_s)):
  """-BCDataSet node creation -BCDataSet
  
  'newNode:N='*newBCDataSet*'(parent:N,name:S,valueType:CG_K.BCTypeSimple)'
  
   If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a BCDataSet_t sub-tree.  
   chapter 9.4 Add node BCTypeSimple is required
  """
  node=hasChildName(parent,name)
  if (node == None):    
    node=newNode(name,None,[],CG_K.BCDataSet_ts,parent)
  if (valueType not in CG_K.BCTypeSimple_l):
    raise CG_E.cgnsException(252,valueType)
  checkDuplicatedName(node,CG_K.BCTypeSimple_s)    
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.BCTypeSimple_s,setStringAsArray(valueType),
                   [],CG_K.BCTypeSimple_ts,node)
  return node

# ---------------------------------------------------------------------------  
def newBCData(parent,name):
  """-BCData node creation -BCData
  
  'newNode:N='*newBCData*'(parent:N,name:S)'
  
   Returns a new <node> representing a BCData_t sub-tree. 
   chapter 9.5 
  """
  checkDuplicatedName(parent,name)    
  node=newNode(name,None,[],CG_K.BCData_ts,parent)
  return node 
  
# -----------------------------------------------------------------------------
def newBCProperty(parent,
                  wallfunction=NPY.array(CG_K.Null_s),
                  area=NPY.array(CG_K.Null_s)):
  """-BCProperty node creation -BCProperty
  
  'newNode:N='*newBCProperty*'(parent:N)'
  
   Returns a new <node> representing a BCProperty_t sub-tree.  
   If a parent is given, the new <node> is added to the parent children list.
   chapter 9.6
  """
  checkDuplicatedName(parent,CG_K.BCProperty_s)    
  node=newNode(CG_K.BCProperty_s,None,[],CG_K.BCProperty_ts,parent)
  wf=newNode(CG_K.WallFunction_s,None,[],CG_K.WallFunction_ts,node)
  newNode(CG_K.WallFunctionType_s,wallfunction,[],CG_K.WallFunctionType_ts,wf)
  ar=newNode(CG_K.Area_s,None,[],CG_K.Area_ts,node)
  newNode(CG_K.AreaType_s,area,[],CG_K.AreaType_ts,ar)
  return node 

# -----------------------------------------------------------------------------
def newCoordinates(parent,name=CG_K.GridCoordinates_s,value=None):
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
  gnode=hasChildName(parent,CG_K.GridCoordinates_s)
  if (gnode == None): gnode=newGridCoordinates(parent,CG_K.GridCoordinates_s)
  node=newDataArray(gnode,name,value)
  return node
  
# -----------------------------------------------------------------------------
def newAxisymmetry(parent,
                   refpoint=NPY.array([0.0,0.0,0.0]),
                   axisvector=NPY.array([0.0,0.0,0.0])):
  """-Axisymmetry node creation -Axisymmetry
  
  'newNode:N='*newAxisymmetry*'(parent:N,refpoint:A,axisvector:A)'
  
  refpoint,axisvector should be a real array.
  Returns a new <node> representing a CG_K.Axisymmetry_t sub-tree.   
  chapter 7.5 Add DataArray AxisymmetryAxisVector,AxisymmetryReferencePoint are required
  """
  if (parent): checkNode(parent)
  checkType(parent,CG_K.CGNSBase_ts,CG_K.Axisymmetry_s)
  checkDuplicatedName(parent,CG_K.Axisymmetry_s)
  checkArrayReal(refpoint)
  checkArrayReal(axisvector)
  node=newNode(CG_K.Axisymmetry_s,None,[],CG_K.Axisymmetry_ts,parent)
  n=hasChildName(parent,CG_K.AxisymmetryReferencePoint_s)
  if (n == None):
    n=newDataArray(node,CG_K.AxisymmetryReferencePoint_s,NPY.array(refpoint))
  n=hasChildName(parent,CG_K.AxisymmetryAxisVector_s)
  if (n == None):
    n=newDataArray(node,CG_K.AxisymmetryAxisVector_s,NPY.array(axisvector))
  return node

# -----------------------------------------------------------------------------
def newRotatingCoordinates(parent,
                           rotcenter=NPY.array([0.0,0.0,0.0]),
                           ratev=NPY.array([0.0,0.0,0.0])):
  """-RotatingCoordinates node creation -RotatingCoordinates
  
  'newNode:N='*newRotatingCoordinates*'(parent:N,rotcenter=A,ratev=A)'
  
   Returns a new <node> representing a RotatingCoordinates_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   rotcenter,ratev should be a real array.
   chapter  7.6 Add DataArray RotationRateVector,RotationCenter are required   
  """ 
  if (parent): checkNode(parent)
  checkTypeList(parent,[CG_K.CGNSBase_ts,CG_K.Zone_ts,CG_K.Family_ts],
                CG_K.RotatingCoordinates_s)
  checkDuplicatedName(parent,CG_K.RotatingCoordinates_s)
  checkArrayReal(rotcenter)
  checkArrayReal(ratev)
  node=newNode(CG_K.RotatingCoordinates_s,None,[],CG_K.RotatingCoordinates_ts,parent)
  n=hasChildName(node,CG_K.RotationCenter_s)
  if (n == None): 
    n=newDataArray(node,CG_K.RotationCenter_s,NPY.array(rotcenter))
  n=hasChildName(node,CG_K.RotationRateVector_s)
  if (n == None): 
    n=newDataArray(node,CG_K.RotationRateVector_s,NPY.array(ratev))
  return node

# -----------------------------------------------------------------------------
def newFlowSolution(parent,name='{FlowSolution}',gridlocation=None):
  """-Solution node creation -Solution
  
  'newNode:N='*newSolution*'(parent:N,name:S,gridlocation:None)'
  
  Returns a new <node> representing a FlowSolution_t sub-tree. 
  chapter 7.7
  """
  checkDuplicatedName(parent,name)
  node=newNode(name,None,[],CG_K.FlowSolution_ts,parent)
  return node  
  
# -----------------------------------------------------------------------------
def newZoneGridConnectivity(parent,name,ctype=NPY.array(CG_K.Null_s),donor=''):
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
  cnode=hasChildName(parent,CG_K.ZoneGridConnectivity_s)  
  if (cnode == None):   
    cnode=newNode(CG_K.ZoneGridConnectivity_s,
                  None,[],CG_K.ZoneGridConnectivity_ts,parent)
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
  cnode=hasChildName(parent,CG_K.ZoneGridConnectivity_s)
  if (cnode == None):
    cnode=newNode(CG_K.ZoneGridConnectivity_s,
                  None,[],CG_K.ZoneGridConnectivity_ts,parent)
  zcnode=newNode(name,dname,[],CG_K.GridConnectivity1to1_ts,cnode)
  newNode("Transform",NPY.array(list(trans),dtype=NPY.int32),[],
          "int[IndexDimension]",zcnode)
  ## code correction: Modify PointRange shape and order
  newNode(CG_K.PointRange_s,NPY.array(window,dtype=NPY.int32,order='Fortran'),[],
          CG_K.IndexRange_ts,zcnode)   
  ## code correction: Modify PointRange shape and order
  newNode(CG_K.PointRangeDonor_s,NPY.array(dwindow,dtype=NPY.int32,order='Fortran'),[],
          CG_K.IndexRange_ts,zcnode)   
  return zcnode

# -----------------------------------------------------------------------------
def newGridConnectivityProperty(parent): 
  """-GridConnectivityProperty node creation -GridConnectivityProperty
  
  'newNode:N='*newGridConnectivityProperty*'(parent:N)'
  
   Returns a new <node> representing a GridConnectivityProperty_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   chapter 8.5 
  """
  checkDuplicatedName(parent,CG_K.GridConnectivityProperty_s)   
  nodeType=newNode(CG_K.GridConnectivityProperty_s,None,[],
                   CG_K.GridConnectivityProperty_ts,parent)
  return nodeType

def  newPeriodic(parent,
                 rotcenter=NPY.array([0.0,0.0,0.0]),
                 ratev=NPY.array([0.0,0.0,0.0]),
                 trans=NPY.array([0.0,0.0,0.0])):
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
  cnode=hasChildName(parent,CG_K.Periodic_s)
  if (cnode == None):
    cnode=newNode(CG_K.Periodic_s,None,[],CG_K.Periodic_ts,parent)
  n=hasChildName(cnode,CG_K.RotationCenter_s)
  if (n == None): 
    newDataArray(cnode,CG_K.RotationCenter_s,NPY.array(rotcenter))
  n=hasChildName(cnode,CG_K.RotationAngle_s)
  if (n == None): 
    newDataArray(cnode,CG_K.RotationAngle_s,NPY.array(ratev))
  n=hasChildName(cnode,CG_K.Translation_s)
  if (n == None): 
    newDataArray(cnode,CG_K.Translation_s,NPY.array(trans)) 
  return cnode
  
# -----------------------------------------------------------------------------
def newAverageInterface(parent,valueType=NPY.array(CG_K.Null_s)):
  """-AverageInterface node creation -AverageInterface
  
  'newNode:N='*newAverageInterface*'(parent:N,valueType:CG_K.AverageInterfaceType)'
  
   Returns a new <node> representing a AverageInterface_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   If the parent has already a child name AverageInterface then
   only the AverageInterfaceType is created.
   chapter 8.5.2
  """
  node=hasChildName(parent,CG_K.AverageInterface_s)
  if (node == None):       
    node=newNode(CG_K.AverageInterface_s,None,[],
                 CG_K.AverageInterface_ts,parent)
  if (valueType not in CG_K.AverageInterfaceType_l):
    raise CG_E.cgnsException(253,valueType)
  checkDuplicatedName(node,CG_K.AverageInterfaceType_s) 
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.AverageInterfaceType_s,setStringAsArray(valueType),[],
                   CG_K.AverageInterfaceType_ts,node)
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
  cnode=hasChildName(parent,CG_K.ZoneGridConnectivity_s)
  if (cnode == None):
    cnode=newNode(CG_K.ZoneGridConnectivity_s,None,[],CG_K.ZoneGridConnectivity_ts,parent)
  checkDuplicatedName(cnode,name)   
  node=newNode(name,None,[],CG_K.OversetHoles_ts,cnode)
  #if(pname!=None and value!=None):
    #newPointList(node,pname,value)
  if hrange!=None:  
    ## code correction: Modify PointRange shape and order
   newPointRange(node,CG_K.PointRange_s,NPY.array(hrange,dtype=NPY.int32,order='Fortran'))
   #newNode(CG_K.PointRange_s,NPY.array(list(hrange),'i'),[],CG_K.IndexRange_ts,node)
  return node

# -----------------------------------------------------------------------------
def newFlowEquationSet(parent):
  """-FlowEquationSet node creation -FlowEquationSet
  
  'newNode:N='*newFlowEquationSet*'(parent:N)'
  
  If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a CG_K.FlowEquationSet_t sub-tree.  
   chapter 10.1
  """
  if (parent): checkNode(parent)
  checkDuplicatedName(parent,CG_K.FlowEquationSet_s)
  checkTypeList(parent,[CG_K.CGNSBase_ts,CG_K.Zone_ts],CG_K.FlowEquationSet_s)     
  node=newNode(CG_K.FlowEquationSet_s,None,[],CG_K.FlowEquationSet_ts,parent)  
  return node   
    
def newGoverningEquations(parent,valueType=NPY.array(CG_K.Euler_s)):
  """-GoverningEquations node creation -GoverningEquations
  
  'newNode:N='*newGoverningEquations*'(parent:N,valueType:CG_K.GoverningEquationsType)'
  
   Returns a new <node> representing a CG_K.GoverningEquations_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name GoverningEquations then
   only the GoverningEquationsType is created.
   chapter  10.2 Add node GoverningEquationsType is required   
  """
  node=hasChildName(parent,CG_K.GoverningEquations_s)
  if (node == None):    
    node=newNode(CG_K.GoverningEquations_s,None,[],CG_K.GoverningEquations_ts,parent)
  if (valueType not in CG_K.GoverningEquationsType_l):
      raise CG_E.cgnsException(221,valueType)
  checkDuplicatedName(parent,CG_K.GoverningEquationsType_s,)
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.GoverningEquationsType_s,setStringAsArray(valueType),[],
                     CG_K.GoverningEquationsType_ts,node)
  return node
  
# -----------------------------------------------------------------------------
def newGasModel(parent,valueType=NPY.array(CG_K.Ideal_s)):
  """-GasModel node creation -GasModel
  
  'newNode:N='*newGasModel*'(parent:N,valueType:CG_K.GasModelType)'
  
   Returns a new <node> representing a CG_K.GasModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name GasModel then
   only the GasModelType is created. 
   chapter 10.3 Add node GasModelType is required  
  """
  node=hasChildName(parent,CG_K.GasModel_s)
  if (node == None):       
    node=newNode(CG_K.GasModel_s,None,[],CG_K.GasModel_ts,parent)
  if (valueType not in CG_K.GasModelType_l): raise CG_E.cgnsException(224,valueType)
  checkDuplicatedName(node,CG_K.GasModelType_s)  
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.GasModelType_s,setStringAsArray(valueType),[],CG_K.GasModelType_ts,node)
  return node
  
def newThermalConductivityModel(parent,
                                valueType=NPY.array(CG_K.SutherlandLaw_s)):   
  """-ThermalConductivityModel node creation -ThermalConductivityModel
  
  'newNode:N='*newThermalConductivityModel*'(parent:N,valueType:CG_K.ThermalConductivityModelType)'
  
   Returns a new <node> representing a CG_K.ThermalConductivityModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ThermalConductivityModel then
   only the ThermalConductivityModelType is created. 
   chapter 10.5 Add node ThermalConductivityModelType is required     
  """
  node=hasChildName(parent,CG_K.ThermalConductivityModel_s)
  if (node == None):    
    node=newNode(CG_K.ThermalConductivityModel_s,None,[],
                 CG_K.ThermalConductivityModel_ts,parent)
  if (valueType not in CG_K.ThermalConductivityModelType_l):
    raise CG_E.cgnsException(227,valueType)
  checkDuplicatedName(node,CG_K.ThermalConductivityModelType_s)
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.ThermalConductivityModelType_s,setStringAsArray(valueType),[],
                   CG_K.ThermalConductivityModelType_ts,node)  
  return node

def newViscosityModel(parent,valueType=NPY.array(CG_K.SutherlandLaw_s)): 
  """-ViscosityModel node creation -ViscosityModel
  
  'newNode:N='*newViscosityModel*'(parent:N,valueType:CG_K.ViscosityModelType)'
  
   Returns a new <node> representing a CG_K.ViscosityModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ViscosityModel then
   only the ViscosityModelType is created. 
   chapter 10.4 Add node ViscosityModelType is (r)       
  """  
  node=hasChildName(parent,CG_K.ViscosityModel_s)
  if (node == None):    
    node=newNode(CG_K.ViscosityModel_s,None,[],CG_K.ViscosityModel_ts,parent)    
  if (valueType not in CG_K.ViscosityModelType_l):
    raise CG_E.cgnsException(230,valueType) 
  checkDuplicatedName(node,CG_K.ViscosityModelType_s)  
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.ViscosityModelType_s,setStringAsArray(valueType),[],
                     CG_K.ViscosityModelType_ts,node)  
  return node

def newTurbulenceClosure(parent,valueType=NPY.array(CG_K.EddyViscosity_s)):   
  """-TurbulenceClosure node creation -TurbulenceClosure
  
  'newNode:N='*newTurbulenceClosure*'(parent:N,valueType:CG_K.TurbulenceClosureType)'  
   Returns a new <node> representing a CG_K.TurbulenceClosure_t sub-tree.  
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name TurbulenceClosure then
   only the ViscosityModelType is created.
   chapter 10.5 Add node TurbulenceClosureType is (r)       
  """
  node=hasChildName(parent,CG_K.TurbulenceClosure_s)
  if (node == None):    
    node=newNode(CG_K.TurbulenceClosure_s,None,[],CG_K.TurbulenceClosure_ts,parent)
  if (valueType not in CG_K.TurbulenceClosureType_l):
    raise CG_E.cgnsException(233,valueType)
  checkDuplicatedName(node,CG_K.TurbulenceClosureType_s)
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.TurbulenceClosureType_s,setStringAsArray(valueType),[],
                     CG_K.TurbulenceClosure_ts,node)  
  return node

def newTurbulenceModel(parent,
                       valueType=NPY.array(CG_K.OneEquation_SpalartAllmaras_s)): 
  """-TurbulenceModel node creation -TurbulenceModel
  
  'newNode:N='*newTurbulenceModel*'(parent:N,valueType:CG_K.TurbulenceModelType)'
  
   Returns a new <node> representing a CG_K.TurbulenceModel_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name TurbulenceModel then
   only the TurbulenceModelType is created.
   chapter 10.6.2 Add node TurbulenceModelType is (r)  
  """ 
  node=hasChildName(parent,CG_K.TurbulenceModel_s)
  if (node == None):
    node=newNode(CG_K.TurbulenceModel_s,None,[],CG_K.TurbulenceModel_ts,parent)
  if (valueType not in CG_K.TurbulenceModelType_l):
    raise CG_E.cgnsException(236,valueType)  
  checkDuplicatedName(node,CG_K.TurbulenceModelType_s)
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.TurbulenceModelType_s,setStringAsArray(valueType),[],
                     CG_K.TurbulenceModelType_ts,node)
  return node

def newThermalRelaxationModel(parent,valueType):
  """-ThermalRelaxationModel node creation -ThermalRelaxationModel
  
  'newNode:N='*newThermalRelaxationModel*'(parent:N,valueType:CG_K.ThermalRelaxationModelType)'
  
   Returns a new <node> representing a CG_K.ThermalRelaxationModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ThermalRelaxationModel then
   only the ThermalRelaxationModelType is created.  
   chapter 10.7 Add node ThermalRelaxationModelType is (r)
  """
  node=hasChildName(parent,CG_K.ThermalRelaxationModel_s) 
  if (node == None):          
    node=newNode(CG_K.ThermalRelaxationModel_s,None,[],
                 CG_K.ThermalRelaxationModel_ts,parent)
  if (valueType not in CG_K.ThermalRelaxationModelType_l):
    raise CG_E.cgnsException(239,valueType) 
  checkDuplicatedName(node,CG_K.ThermalRelaxationModelType_s)   
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.ThermalRelaxationModelType_s,setStringAsArray(valueType),[],
                   CG_K.ThermalRelaxationModelType_ts,node)
  return node

def newChemicalKineticsModel(parent,valueType=NPY.array(CG_K.Null_s)):
  """-ChemicalKineticsModel node creation -ChemicalKineticsModel
  
  'newNode:N='*newChemicalKineticsModel*'(parent:N,valueType:CG_K.ChemicalKineticsModelType)'
  
   Returns a new <node> representing a CG_K.ChemicalKineticsModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ChemicalKineticsModel then
   only the ChemicalKineticsModelType is created. 
   chapter 10.8 Add node ChemicalKineticsModelType is (r)  
  """
  node=hasChildName(parent,CG_K.ChemicalKineticsModel_s) 
  if (node == None):             
    node=newNode(CG_K.ChemicalKineticsModel_s,None,[],
                 CG_K.ChemicalKineticsModel_ts,parent)
  if (valueType not in CG_K.ChemicalKineticsModelType_l):
    raise CG_E.cgnsException(242,valueType)
  checkDuplicatedName(node,CG_K.ChemicalKineticsModelType_s)     
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.ChemicalKineticsModelType_s,setStringAsArray(valueType),[],
                     CG_K.ChemicalKineticsModelType_ts,node)
  return node

def newEMElectricFieldModel(parent,valueType=CG_K.UserDefined_s):
  """-EMElectricFieldModel node creation -EMElectricFieldModel
  
  'newNode:N='*newEMElectricFieldModel*'(parent:N,valueType:CG_K.EMElectricFieldModelType)'
  
   Returns a new <node> representing a CG_K.EMElectricFieldModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
    If the parent has already a child name EMElectricFieldModel then
   only the EMElectricFieldModelType is created. 
   chapter 10.9 Add node EMElectricFieldModelType is (r)  
  """
  node=hasChildName(parent,CG_K.EMElectricFieldModel_s)   
  if (node == None):           
    node=newNode(CG_K.EMElectricFieldModel_s,None,[],
                 CG_K.EMElectricFieldModel_ts,parent)
  if (valueType not in CG_K.EMElectricFieldModelType_l):
    raise CG_E.cgnsException(245,valueType)
  checkDuplicatedName(node,CG_K.EMElectricFieldModelType_s)  
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.EMElectricFieldModelType_s,setStringAsArray(valueType),[],
                   CG_K.EMElectricFieldModelType_ts,node)
  return node

def newEMMagneticFieldModel(parent,valueType=CG_K.UserDefined_s):
  """-EMMagneticFieldModel node creation -EMMagneticFieldModel
  
  'newNode:N='*newEMMagneticFieldModel*'(parent:N,valueType:CG_K.EMMagneticFieldModelType)'
  
   Returns a new <node> representing a CG_K.EMMagneticFieldModel_t sub-tree.  
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name EMMagneticFieldModel_s then
   only the EMMagneticFieldModelType is created. 
   chapter 10.9.2 Add node EMMagneticFieldModelType is (r)  
  """
  node=hasChildName(parent,CG_K.EMMagneticFieldModel_s)   
  if (node == None):            
    node=newNode(CG_K.EMMagneticFieldModel_s,None,[],
                 CG_K.EMMagneticFieldModel_ts,parent)
  if (valueType not in CG_K.EMMagneticFieldModelType_l):
    raise CG_E.cgnsException(248,valueType)  
  checkDuplicatedName(node,CG_K.EMMagneticFieldModelType_s) 
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.EMMagneticFieldModelType_s,setStringAsArray(valueType),[],
                   CG_K.EMMagneticFieldModelType_ts,node)
  return node

def newEMConductivityModel(parent,valueType=CG_K.UserDefined_s):
  """-EMConductivityModel node creation -EMConductivityModel
  
  'newNode:N='*newEMConductivityModel*'(parent:N,valueType:CG_K.EMConductivityModelType)'
  
   Returns a new <node> representing a CG_K.EMConductivityModel_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name EMConductivityModel then
   only the EMConductivityModelType is created. 
   chapter 10.9.3 Add node EMConductivityModelType is (r)  
  """
  node=hasChildName(parent,CG_K.EMConductivityModel_s)  
  if (node == None):             
    node=newNode(CG_K.EMConductivityModel_s,None,[],
                 CG_K.EMConductivityModel_ts,parent)
  if (valueType not in CG_K.EMConductivityModelType_l):
    raise CG_E.cgnsException(218,stype)  
  checkDuplicatedName(node,CG_K.EMConductivityModelType_s)  
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.EMConductivityModelType_s,setStringAsArray(valueType),[],
                   CG_K.EMConductivityModelType_ts,node)
  return node

# -----------------------------------------------------------------------------
def newBaseIterativeData(parent,nsteps=0,
                         itype=CG_K.IterationValues_s):
  """-BaseIterativeData node creation -BaseIterativeData
  
   'newNode:N='*newBaseIterativeData*'(parent:N,nsteps:I,itype:E)'
  
   Returns a new <node> representing a BaseIterativeData_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter 11.1.1
   NumberOfSteps is required, TimeValues or IterationValues are required
  """ 
  
  if (parent): checkNode(parent)
  checkDuplicatedName(parent,CG_K.BaseIterativeData_s)
  checkType(parent,CG_K.CGNSBase_ts,CG_K.BaseIterativeData_ts)
  if ((type(nsteps) != type(1)) or (nsteps < 0)): raise CG_E.cgnsException(209)
  node=newNode(CG_K.BaseIterativeData_s,NPY.array(nsteps,dtype='i'),[],CG_K.BaseIterativeData_ts,parent)
  if (itype not in [CG_K.IterationValues_s, CG_K.TimeValues_s]):
    raise CG_E.cgnsException(210,(CG_K.IterationValues_s, CG_K.TimeValues_s))
  newNode(itype,None,[],CG_K.DataArray_ts,node)  
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
  node=newNode(name,None,[],CG_K.ZoneIterativeData_ts,parent)
  return node

# ---------------------------------------------------------------------------  
def newRigidGridMotion(parent,name,
                       valueType=CG_K.Null_s,
                       vector=NPY.array([0.0,0.0,0.0])):
  """-RigidGridMotion node creation -RigidGridMotion
  
  'newNode:N='*newRigidGridMotion*'(parent:N,name:S,valueType:CG_K.RigidGridMotionType,vector:A)'
  
  If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a CG_K.RigidGridMotion_t sub-tree.  
   If the parent has already a child name RigidGridMotion then
   only the RigidGridMotionType is created and OriginLocation is created
   chapter 11.2 Add Node RigidGridMotionType and add DataArray OriginLocation are the only required
  """
  if (parent): checkNode(parent)  
  checkDuplicatedName(parent,name)
  node=newNode(name,None,[],CG_K.RigidGridMotion_ts,parent)
  
  if (valueType not in CG_K.RigidGridMotionType_l):
      raise CG_E.cgnsException(254,valueType)
  checkDuplicatedName(parent,CG_K.RigidGridMotionType_s,)
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.RigidGridMotionType_s,setStringAsArray(valueType),[],
                   CG_K.RigidGridMotionType_ts,node)
  n=hasChildName(parent,CG_K.OriginLocation_s)
  if (n == None): 
    n=newDataArray(node,CG_K.OriginLocation_s,NPY.array(vector))
  return node
  
#-----------------------------------------------------------------------------
def newReferenceState(parent,name=CG_K.ReferenceState_s):
  """-ReferenceState node creation -ReferenceState
  
  'newNode:N='*newReferenceState*'(parent:N,name:S)'
  
   Returns a new <node> representing a ReferenceState_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.1  """   
  if (parent): checkNode(parent)
  node=hasChildName(parent,name)
  if (node == None):
    checkDuplicatedName(parent,name)
    node=newNode(name,None,[],CG_K.ReferenceState_ts,parent)
  return node

#-----------------------------------------------------------------------------
def newConvergenceHistory(parent,name=CG_K.GlobalConvergenceHistory_s,
			  iterations=0):
  """-ConvergenceHistory node creation -ConvergenceHistory
  
  'newNode:N='*newConvergenceHistory*'(parent:N,name:S,iterations:i)'
  
   Returns a new <node> representing a ConvergenceHistory_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.3  """   
  if (name not in CG_K.ConvergenceHistory_l): raise CG_E.cgnsException(201,name)
  if (parent):
    checkNode(parent)
    checkTypeList(parent,[CG_K.CGNSBase_ts,CG_K.Zone_ts],name)
  if (name == CG_K.GlobalConvergenceHistory_s):
    checkType(parent,CG_K.CGNSBase_ts,name)
  if (name == CG_K.ZoneConvergenceHistory_s):
    checkType(parent,CG_K.Zone_ts,name)
  checkDuplicatedName(parent,name)
  node=newNode(name,NPY.array(iterations,dtype='i'),[],CG_K.ConvergenceHistory_ts,parent)
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
  node=newNode(name,None,[],CG_K.IntegralData_ts,parent)
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
  checkType(parent,CG_K.CGNSBase_ts,name)
  checkDuplicatedName(parent,name)
  node=newNode(name,None,[],CG_K.Family_ts,parent)
  return node

def newFamilyName(parent,family=None):
  ## code correction: Modify family string into NPY string array
  return newNode(CG_K.FamilyName_s,setStringAsArray(family),[],CG_K.FamilyName_ts,parent)

# -----------------------------------------------------------------------------
def newGeometryReference(parent,name='{GeometryReference}',
                         valueType=CG_K.UserDefined_s):
  """-GeometryReference node creation -GeometryReference
  
  'newNode:N='*newGeometryReference*'(parent:N,name:S,valueType:CG_K.GeometryFormat)'
  
   Returns a new <node> representing a CG_K.GeometryFormat_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name CG_K.GeometryReference then
   only the .GeometryFormat is created
   chapter  12.7 Add node CG_K.GeometryFormat_t is (r) and GeometryFile_t definition not find but is required (CAD file)
  """
  node=hasChildName(parent,CG_K.GeometryReference_s)
  if (node == None):    
    node=newNode(name,None,[],CG_K.GeometryReference_ts,parent)
  if (valueType not in CG_K.GeometryFormat_l):
      raise CG_E.cgnsException(256,valueType)
  checkDuplicatedName(node,CG_K.GeometryFormat_s)
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.GeometryFormat_s,setStringAsArray(valueType),[],
                   CG_K.GeometryFormat_ts,node)
  return node
  
# -----------------------------------------------------------------------------
def newFamilyBC(parent,valueType=CG_K.UserDefined_s): 
  """-FamilyBC node creation -FamilyBC
  
  'newNode:N='*newFamilyBC*'(parent:N,valueType:CG_K.BCTypeSimple/CG_K.BCTypeCompound)'
  
   Returns a new <node> representing a CG_K.FamilyBC_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name FamilyBC then
   only the BCType is created
   chapter  12.8 Add node BCType is required   
  """ 
  node=hasChildName(parent,CG_K.FamilyBC_s)
  if (node == None):    
    node=newNode(CG_K.FamilyBC_s,None,[],CG_K.FamilyBC_ts,parent)
  if (    valueType not in CG_K.BCTypeSimple_l
      and valueType not in CG_K.BCTypeCompound_l):
      raise CG_E.cgnsException(257,valueType)
  checkDuplicatedName(node,CG_K.BCType_s)
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.BCType_s,setStringAsArray(valueType),[],
                     CG_K.BCType_ts,node)
  return node

# -----------------------------------------------------------------------------
def newArbitraryGridMotion(parent,name,valuetype=CG_K.Null_s):
  """
  .. index:: ArbitraryGridMotion,ArbitraryGridMotionType
  .. index:: RigidGridMotion,RigidGridMotionType
  Returns a **new node** representing a ``ArbitraryGridMotionType_t``
  sub-tree `(chapter 11.3) <http://www.grc.nasa.gov/WWW/cgns/sids/timedep.html#ArbitraryGridMotion>`_

  :param parent: CGNS/Python node
  :param name: String
  :param valuetype: String (``CGNS.PAT.cgnskeywords.ArbitraryGridMotionType``)


  If a *parent* is not ``None``, the **new node** is added to the parent
  children list. If the *parent* has already a child with
  name ``RigidGridMotion`` then only the ``RigidGridMotionType`` is created.

  """
  node=None
  if (parent): node=hasChildName(parent,name)
  if (node == None):
    node=newNode(name,None,[],CG_K.ArbitraryGridMotion_ts,parent)      
  if (valuetype not in CG_K.ArbitraryGridMotionType_l):
    raise CG_E.cgnsException(255,valuetype) 
  checkDuplicatedName(node,CG_K.ArbitraryGridMotionType_s)     
  ## code correction: Modify valueType string into NPY string array
  nodeType=newNode(CG_K.ArbitraryGridMotionType_s,
                   setStringAsArray(valuetype),[],
                   CG_K.ArbitraryGridMotionType_ts,node)
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
  node=newNode(name,None,[],CG_K.UserDefinedData_ts,parent)
  return node
 
# -----------------------------------------------------------------------------
def newGravity(parent,gvector=NPY.array([0.0,0.0,0.0])):
  """-Gravity node creation -Gravity
  
  'newNode:N='*newGravity*'(parent:N,gvector:A)'
  
   Returns a new <node> representing a Gravity_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   gvector should be a real array
   chapter  12.10 Add DataArray GravityVector is required   
  """ 
  if (parent): checkNode(parent)
  checkType(parent,CG_K.CGNSBase_ts,CG_K.Gravity_s)
  checkDuplicatedName(parent,CG_K.Gravity_s)
  checkArrayReal(gvector)
  node=newNode(CG_K.Gravity_s,None,[],CG_K.Gravity_ts,parent)
  n=hasChildName(parent,CG_K.GravityVector_s)
  if (n == None): 
    n=newDataArray(node,CG_K.GravityVector_s,NPY.array(gvector))
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
  checkDuplicatedName(parent,CG_K.DiffusionModel_s)
  node=newNode(CG_K.DiffusionModel_s,None,[],CG_K.DiffusionModel_ts,parent)
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

