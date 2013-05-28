#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#  
import CGNS
import CGNS.PAT.cgnskeywords as CK
import CGNS.PAT.cgnstypes    as CT
import CGNS.PAT.cgnserrors   as CE
import CGNS.PAT.cgnsutils    as CU

import numpy as NPY

__CGNS_LIBRARY_VERSION__=3.2

# =============================================================================
# MLL-like calls
# - every call takes a parent as argument. If the parent is present, the new
#   sub-tree is inserted in the parent child list. In all cases the call
#   returns the created sub-tree
# - some function are not MLL based
# - function patterns
#   newXXX :   creates a new XXX_t type node
#   updateXXX: updates fields in the XXX_t node
#   checkXXX:  check if node is ok for SIDS and for CGNS/Python
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def newCGNSTree():
  """Top CGNS/Python tree node creation::

   T=newCGNSTree()

  - Return:
   * The new :ref:`XCGNSTree_t` node

  - Remarks:
   * You *should* keep the returned node in a variable or reference to it in
     any other way, this tree root is a Python object that would be garbagged
     if its reference count reaches zero.
   * The `CGNSTree` node is a CGNS/Python node which has no existence in a
     disk HDF5 file.
   
  - Children:
   * :py:func:`newCGNSBase`

  """
  return newCGNS()

def newCGNSBase(tree,name,ncell,nphys):
  """*CGNSBase* node creation::

    # The base is put in the `T` children list
    T=newCGNSTree()
    newBase(T,'Box-1',3,3)

    # No parent, you should fetch the new node using a variable
    B=newCGNSBase(None,'Box-2',3,3)
  
  - Args:
   * `tree`: the parent node (`<node>` or `None`)
   * `name`: base name (`string`)
   * `cdim`: cell dimensions (`int`)
   * `pdim`: physical dimensions (`int`)

  - Return:
   * The new :ref:`XCGNSBase_t` node

  - Remarks:
   * If a parent is given, the new node is added to the parent children list.

  - Children:
   * :py:func:`newZone`
  """
  return newBase(tree,name,ncell,nphys)

def newCGNS():
  ##code correction: Modify CGNS value type from float64 to float32.
  node=[CK.CGNSLibraryVersion_s,
        NPY.array([__CGNS_LIBRARY_VERSION__],dtype='float32'),
        [],
        CK.CGNSLibraryVersion_ts]
  badnode=[CK.CGNSTree_s,None,[node],CK.CGNSTree_ts]
  return badnode

# -----------------------------------------------------------------------------
def newDataClass(parent,value=CK.UserDefined_s):
  """-DataClass node creation -DataClass
  
  'newNode:N='*newDataClass*'(parent:N,value:A)'
  
  If a parent is given, the new <node> is added to the parent children list.
  The value argument is a DataClass enumerate. No child allowed.
  Returns a new <node> representing a DataClass_t sub-tree."""
  CU.checkDuplicatedName(parent,CK.DataClass_s)
  node=CU.newNode(CK.DataClass_s,CU.setStringAsArray(value),[],
               CK.DataClass_ts,parent)
  return checkDataClass(node)

def updateDataClass(node,value):
  CU.checkNode(node)  
  if (value!=None): node[1]=value
  return checkDataClass(node)

def checkDataClass(node,parent=None):
  CU.checkNode(node)
  CU.checkName(node[0])
  if (node[0] != CK.DataClass_s):  raise CE.cgnsException(26,node[0])
  if (node[3] != CK.DataClass_ts): raise CE.cgnsException(27,node[3])
  if (len(node[2]) != 0):         raise CE.cgnsException(28,node[0])
  value=CU.getValue(node).tostring()
  if (value not in CK.DataClass_l):  raise CE.cgnsException(207,value)
  if (parent != None):
     CU.checkTypeList(parent,[CK.DataArray_ts,CK.CGNSBase_ts,CK.Zone_ts,
                           CK.GridCoordinates_ts,CK.Axisymmetry_ts,
                           CK.RotatingCoordinates_ts,CK.FlowSolution_ts,
                           CK.Periodic_ts,CK.ZoneBC_ts,CK.BC_ts,
                           CK.BCDataSet_ts,
                           CK.BCData_ts,CK.FlowEquationSet_ts,
                           CK.GasModel_ts,
                           CK.ViscosityModel_ts,
                           CK.ThermalConductivityModel_ts,
                           CK.TurbulenceClosure_ts,CK.TurbulenceModel_ts,
                           CK.ThermalRelaxationModel_ts,
                           CK.ChemicalKineticsModel_ts,
                           CK.EMElectricFieldModel_ts,
                           CK.EMMagneticFieldModel_ts,
                           CK.EMConductivityModel_ts,
                           CK.BaseIterativeData_ts,
                           CK.ZoneIterativeData_ts,CK.RigidGridMotion_ts,
                           CK.ArbitraryGridMotion_ts,CK.ReferenceState_ts,
                           CK.ConvergenceHistory_ts,
                           CK.DiscreteData_ts,CK.IntegralData_ts,
                           CK.UserDefinedData_ts,CK.Gravity_ts]
                   ,CK.DataClass_s)
  return node
  
# -----------------------------------------------------------------------------
def newDescriptor(parent,name,value=''):
  """-Descriptor node creation -Descriptor
  
  'newNode:N='*newDescriptor*'(parent:N,name:S,text:A)'
  
  No child allowed.
  Returns a new <node> representing a Descriptor_t sub-tree."""
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,CU.setStringAsArray(value),[],CK.Descriptor_ts,parent)
  return checkDescriptor(node)
  
def checkDescriptor(node,parent=None):
  CU.checkNode(node)
  CU.checkName(node[0])
  if (node[3] != CK.Descriptor_ts): raise CE.cgnsException(27,node[3])
  if (len(node[2]) != 0):             raise CE.cgnsException(28,node[0])
  value=CU.getValue(node)
  if (CU.getValueType(value) != CK.Character_s):
                                      raise CE.cgnsException(110,node[0])
  if (parent != None):
     CU.checkTypeList(parent,[CK.DataArray_ts,CK.CGNSBase_ts,CK.Zone_ts,
                           CK.GridCoordinates_ts,CK.Elements_ts,
                           CK.Axisymmetry_ts,
                           CK.RotatingCoordinates_ts,CK.FlowSolution_ts,
                           CK.ZoneGridConnectivity_ts,
                           CK.GridConnectivity1to1_ts,
                           CK.GridConnectivity_ts,
                           CK.GridConnectivityProperty_ts,
                           CK.AverageInterface_ts,CK.OversetHoles_ts,
                           CK.Periodic_ts,CK.ZoneBC_ts,CK.BC_ts,
                           CK.BCDataSet_ts,
                           CK.BCData_ts,CK.FlowEquationSet_ts,
                           CK.GasModel_ts,
                           CK.BCProperty_ts,CK.WallFunction_ts,
                           CK.Area_ts,
                           CK.GoverningEquations_ts,
                           CK.ViscosityModel_ts,
                           CK.ThermalConductivityModel_ts,
                           CK.TurbulenceClosure_ts,CK.TurbulenceModel_ts,
                           CK.ThermalRelaxationModel_ts,
                           CK.ChemicalKineticsModel_ts,
                           CK.EMElectricFieldModel_ts,
                           CK.EMMagneticFieldModel_ts,
                           CK.EMConductivityModel_ts,
                           CK.BaseIterativeData_ts,
                           CK.ZoneIterativeData_ts,CK.RigidGridMotion_ts,
                           CK.ArbitraryGridMotion_ts,CK.ReferenceState_ts,
                           CK.ConvergenceHistory_ts,
                           CK.DiscreteData_ts,CK.IntegralData_ts,
                           CK.Family_ts,CK.GeometryReference_ts,
                           CK.UserDefinedData_ts,CK.Gravity_ts]
                   ,CK.DataClass_s)
  return node

# -----------------------------------------------------------------------------
def newDimensionalUnits(parent,value=[CK.Meter_s,CK.Kelvin_s,
                                      CK.Second_s,CK.Radian_s,
                                      CK.Kilogram_s]):
  """DimensionalUnits node creation::

  'newNode:N='*newDimensionalUnits*'(parent:N,value=[CK.MassUnits,CK.LengthUnits,CK.TimeUnits,CK.TemperatureUnits,CK.AngleUnits])'
                                      
  If a parent is given, the new <node> is added to the parent children list.
  new <node> is composed of a set of enumeration types : `MassUnits`,`LengthUnits`,
  TimeUnits,TemperatureUnits,AngleUnits are required
  Returns a new <node> representing a DimensionalUnits_t sub-tree.
  chapter 4.3 
  """
  if (len(value) != 5): raise CE.cgnsException(202)
  CU.checkDuplicatedName(parent,CK.DimensionalUnits_s)
  # --- loop over values to find all required units
  vunit=[CK.Null_s,CK.Null_s,CK.Null_s,CK.Null_s,CK.Null_s]
  for v in value:
    if (v not in CK.AllUnits_l): raise CE.cgnsException(203,v)
    if ((v in CK.MassUnits_l)
        and (v not in [CK.Null_s,CK.UserDefined_s])):
      if (v in vunit): raise CE.cgnsException(204,v)
      else:            vunit[0]=v
    if ((v in CK.LengthUnits_l)
        and (v not in [CK.Null_s,CK.UserDefined_s])):
      if (v in vunit): raise CE.cgnsException(204,v)
      else:            vunit[1]=v
    if ((v in CK.TimeUnits_l)
        and (v not in [CK.Null_s,CK.UserDefined_s])):
      if (v in vunit): raise CE.cgnsException(204,v)
      else:            vunit[2]=v
    if ((v in CK.TemperatureUnits_l)
        and (v not in [CK.Null_s,CK.UserDefined_s])):
      if (v in vunit): raise CE.cgnsException(204,v)
      else:            vunit[3]=v
    if ((v in CK.AngleUnits_l)
        and (v not in [CK.Null_s,CK.UserDefined_s])):
      if (v in vunit): raise CE.cgnsException(204,v)
      else:            vunit[4]=v
  node=CU.newNode(CK.DimensionalUnits_s,CU.concatenateForArrayChar(vunit),[],
               CK.DimensionalUnits_ts,parent)
  snode=CU.newNode(CK.AdditionalUnits_s,
                CU.concatenateForArrayChar([CK.Null_s,CK.Null_s,CK.Null_s]),
                [],
                CK.AdditionalUnits_ts,node)
  return node

# -----------------------------------------------------------------------------
def newDimensionalExponents(parent,
                            MassExponent=0,LengthExponent=0,TimeExponent=0,
                            TemperatureExponent=0,AngleExponent=0):
  """-DimensionalExponents node creation -DimensionalExponents::
  
  'newNode:N='*newDimensionalExponents*'(parent:N,MassExponent:r,LengthExponent:r,TimeExponent:r,TemperatureExponent:r,AngleExponent:r)'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a DimensionalExponents_t sub-tree.
  chapter 4.4
  """
  CU.checkDuplicatedName(parent,CK.DimensionalExponents_s)
  node=CU.newNode(CK.DimensionalExponents_s,
               NPY.array([MassExponent,
                          LengthExponent,
                          TimeExponent,
                          TemperatureExponent,
                          AngleExponent],dtype='Float64',order='Fortran'),
               [],CK.DimensionalExponents_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newGridLocation(parent,value=CK.CellCenter_s):
  """-GridLocation node creation -GridLocation::
  
  'newNode:N='*newGridLocation*'(parent:N,value:CK.GridLocation)'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a GridLocation_t sub-tree.
  chapter 4.5
  """
  CU.checkDuplicatedName(parent,CK.GridLocation_s)
  if (value not in CK.GridLocation_l): raise CE.cgnsException(200,value)
  ## code correction: Modify value string into NPY string array
  node=CU.newNode(CK.GridLocation_s,CU.setStringAsArray(value),[],CK.GridLocation_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newIndexArray(parent,name,value=None):
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,value,[],CK.IndexArray_ts,parent)
  return node
  
def newPointList(parent,name=CK.PointList_s,value=None):
  """-PointList node creation -PointList
  
  'newNode:N='*newPointList*'(parent:N,name:S,value:[])'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a IndexArray_t sub-tree.
  chapter 4.6
  """
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,value,[],CK.IndexArray_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newPointRange(parent,name=CK.PointRange_s,value=[]):
  """-PointRange node creation -PointRange
  
  'newNode:N='*newPointRange*'(parent:N,name:S,value:[])'
  
  If a parent is given, the new <node> is added to the parent children list.
  Returns a new <node> representing a IndexRange_t sub-tree.
  chapter 4.7
  """
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,value,[],CK.IndexRange_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newRind(parent,value):                                    
  """-Rind node creation -Rind
  
  'newNode:N='*newRind*'(parent:N,value=A)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a Rind_t sub-tree.
  chapter 4.8
  """
  CU.checkDuplicatedName(parent,CK.Rind_s)
  # check value wrt base dims
  node=CU.newNode(CK.Rind_s,value,[],CK.Rind_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newDataConversion(parent,ConversionScale=1.0,ConversionOffset=1.0):
  """-DataConversion node creation -DataConversion
  
  'newNode:N='*newDataConversion*'(parent:N,ConversionScale:r,ConversionOffset:r)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a DataConversion_t sub-tree.
  chapter  5.1.1
  """
  CU.checkDuplicatedName(parent,CK.DataConversion_s)
  node=CU.newNode(CK.DataConversion_s,
               NPY.array([ConversionScale,ConversionOffset],
                         dtype='Float64',order='Fortran'),
               [],CK.DataConversion_ts,parent)
  return node

# ----------------------------------------------------------------------------
def newSimulationType(parent,stype=CK.NonTimeAccurate_s):
  """-SimulationType node creation -SimulationType
  
  'newNode:N='*newSimulationType*'(parent:N,stype=CK.SimulationType)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a SimulationType_t sub-tree.
  chapter 6.2
  """
  if (parent): CU.checkNode(parent)
  CU.checkDuplicatedName(parent,CK.SimulationType_s)
  CU.checkType(parent,CK.CGNSBase_ts,CK.SimulationType_s)
  if (stype not in CK.SimulationType_l): raise CE.cgnsException(205,stype)
  node=CU.newNode(CK.SimulationType_s,CU.setStringAsArray(stype),[],CK.SimulationType_ts,parent)
  return node
  
# ----------------------------------------------------------------------------
def newBase(tree,name,ncell,nphys):
  if (ncell not in [1,2,3]): raise CE.cgnsException(10,name)
  if (nphys not in [1,2,3]): raise CE.cgnsException(11,name)
  if (nphys < ncell):        raise CE.cgnsException(12,name)
  if ((tree != None) and (not CU.checkNode(tree))):
     raise CE.cgnsException(6,name)
  if ((tree != None) and (tree[0] == CK.CGNSTree_s)): parent=tree[2]  
  else:                                              parent=tree
  CU.checkDuplicatedName(["<root node>",None,parent],name)
  node=CU.newNode(name,
               NPY.array([ncell,nphys],dtype=NPY.int32,order='Fortran'),
               [],CK.CGNSBase_ts)
  if (parent != None): parent.append(node)
  return node

def numberOfBases(tree):
  return len(CU.hasChildrenType(tree,CK.CGNSBase_ts))

def readBase(tree,name):
  b=CU.hasChildName(tree,name)
  if (b == None): raise CE.cgnsException(21,name)
  if (b[3] != CK.CGNSBase_ts):
    raise CE.cgnsException(20,(CK.CGNSBase_ts,name))
  return (b[0],b[1])
  
def updateBase(tree,name=None,ncell=None,nphys=None):
  if (ncell not in [1,2,3]): raise CE.cgnsException(10,name)
  if (nphys not in [1,2,3]): raise CE.cgnsException(11,name)
  if (nphys < ncell):        raise CE.cgnsException(12,name)
  
  if (tree): CU.checkNode(tree)

  if (tree[3] != CK.CGNSBase_ts):
    raise CE.cgnsException(20,(CK.CGNSBase_ts,name))
  if(name!=None): tree[0]=name
  if(ncell!=None and nphys!=None and tree):
    tree[1]=NPY.array([ncell,nphys],dtype=NPY.int32,order='Fortran')
  else: raise CE.cgnsException(12)  
  
 
# -----------------------------------------------------------------------------
def newOrdinal(parent,value=0):
  """-Ordinal node creation -Ordinal
  
  'newNode:N='*newOrdinal*'(parent:N,value=i)'
  
  If a parent is given, the new <node> is added to the parent children list.  
  Returns a new <node> representing a Ordinal_t sub-tree.
  chapter 6.3
  """
  CU.checkDuplicatedName(parent,CK.Ordinal_s)
  node=CU.newNode(CK.Ordinal_s,NPY.array([value],dtype=NPY.int32),[],CK.Ordinal_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newZone(parent,name,zsize=None,
            ztype=CK.Structured_s,
            family=''):
  """*Zone* node creation::

    s=NPY.array([[10],[2],[0]],dtype='i')
    
    T=newCGNSTree()
    B=newBase(T,'Box-1',3,3)
    Z=newZone(B,name,s,CK.Unstructured_s,'Wing')
  
  - Args:
   * `parent`: the parent node (`<node>` or `None`)
   * `name`: zone name (`string`)
   * `zsize`: array of dimensions (`numpy.ndarray`)
   * `ztype`: zone type (`string`)
   * `family`: zone family (`string`)

  - Return:
   * The new :ref:`XZone_t` node

  - Remarks:
   * The zone size has dimensions [IndexDimensions][3]
  
  - Children:
   * :py:func:`newElements`
   
  """
  asize=None
  if (ztype not in CK.ZoneType_l): raise CE.cgnsException(206,ztype)
  if (zsize == None): raise CE.cgnsException(300) 
  CU.checkDuplicatedName(parent,name)
  CU.checkArray(zsize,dienow=True)
  znode=CU.newNode(name,zsize,[],CK.Zone_ts,parent)
  CU.newNode(CK.ZoneType_s,CU.setStringAsArray(ztype),[],CK.ZoneType_ts,znode)
  if (family):
    CU.newNode(CK.FamilyName_s,
               CU.setStringAsArray(family),[],CK.FamilyName_ts,znode)
  return znode

def numberOfZones(tree,basename):
  b=CU.hasChildName(tree,basename)
  if (b == None): raise CE.cgnsException(21,basename)
  if (b[3] != CK.CGNSBase_ts): raise CE.cgnsException(20,(CK.CGNSBase_ts,name))
  return len(CU.hasChildrenType(b,CK.Zone_ts))

def readZone(tree,basename,zonename,gtype=None):
  b=CU.hasChildName(tree,basename)
  if (b == None): raise CE.cgnsException(21,basename)
  if (b[3] != CK.CGNSBase_ts): raise CE.cgnsException(20,(CK.CGNSBase_ts,name))
  z=CU.hasChildName(b,zonename)
  if (z == None): raise CE.cgnsException(21,zonename)
  if (z[3] != CK.Zone_ts): raise CE.cgnsException(20,(CK.Zone_ts,name))
  if gtype: 
    zt=CU.hasChildName(z,CK.ZoneType_s)
    if (zt == None): raise CE.cgnsException(21,CK.ZoneType_s)
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
  node=CU.newNode(name,None,[],CK.GridCoordinates_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newDataArray(parent,name,value=None):
  """-DataArray node creation -Global
  
  'newNode:N='*newDataArray*'(parent:N,name:S,value:A)'
  
  Returns a new <node> representing a DataArray_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list.
  chapter 5.1
  """
  CU.checkDuplicatedName(parent,name)
  ## code correction:  Add value type and fortran order
  ## code correction:  Add a specific array for string type
  ## code correction:  Modify array check
  if (type(value)==type(3)):
    vv=NPY.array([value],dtype=NPY.int32,order='Fortran')
    CU.checkArray(vv)
  elif(type(value)==type(3.2)):
    vv=NPY.array([value],dtype=NPY.float32,order='Fortran')
    CU.checkArray(vv)
  elif(type(value)==type("s")):
    vv=CU.setStringAsArray(value)
    CU.checkArrayChar(vv)
  else:
    vv=value
    if (vv != None): CU.checkArray(vv)
    
  node=CU.newNode(name,vv,[],CK.DataArray_ts,parent)
  return node

def numberOfDataArrays(parent):
  return len(CU.hasChildrenType(parent,CK.DataArray_ts))

def readDataArray(parent,name):
  n=CU.hasChildName(parent,name)
  if (n == None): raise CE.cgnsException(21,name)
  if (n[3] != CK.DataArray_ts): raise CE.cgnsException(20,(CK.DataArray_ts,name))
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
  CU.checkDuplicatedName(parent,name)    
  node=CU.newNode(name,None,[],CK.DiscreteData_ts,parent)
  return node 
  
# -----------------------------------------------------------------------------
def newElements(parent,name,
                etype=CK.UserDefined_s,
                econnectivity=None,
                erange=None,
                eboundary=0
                ):
  """*Elements_t* node creation::

   quads=newElements(None,'QUADS',CGK.QUAD_4,quad_array,NPY.array([start,end]))'
  
  - Args:
   * `parent`: the parent node (`<node>` or `None`)
   * `name`: element node name (`string`)
   * `etype`: the type of element (`string` or 'int')
   * `econnectivity`: actual array of point connectivities (`numpy.ndarray`)
   * `erange`: the first and last index of the connectivity (`numpy.ndarray`)
   * `eboundary`: number of boundary elements (`int`)

  - Return:
   * The new :ref:`XElements_t` node

  - Remarks:
   * If a parent is given, the new node is added to the parent children list.
   * The `elementsrange` *should* insure a unique and continuous index for
     all elements nodes in the same parent zone.
   * Element type can be set as `int` such as `CGK.QUAD_4` or 7, or
     as `string` such as `CGK.QUAD_4_s` or `"QUAD_4"`
     
  - Children:
   * :py:func:`newDescriptor`
  """
  CU.checkDuplicatedName(parent,name)
  if (type(etype) in [int]):
    if (etype not in CK.ElementType.values()):
      raise CE.cgnsException(250,etype)
    etp=etype
  if (type(etype) in [str]):
    if (etype not in CK.ElementType_l):
      raise CE.cgnsException(250,etype)
    etp=CK.ElementType[etype]
  v=NPY.array([etp,eboundary],dtype='i')
  enode=CU.newNode(name,v,[],CK.Elements_ts,parent)
  newDataArray(enode,CK.ElementConnectivity_s,econnectivity)
  newPointRange(enode,CK.ElementRange_s,erange)
  return enode

# -----------------------------------------------------------------------------
def newZoneBC(parent):
  return CU.newNode(CK.ZoneBC_s,None,[],CK.ZoneBC_ts,parent)

def newBC(parent,bname,brange=[0,0,0,0,0,0],
          btype=CK.Null_s,bcType=CK.Null_s,
          family=CK.Null_s,pttype=CK.PointRange_s):
  return newBoundary(parent,bname,brange,btype,family,pttype) 

def newBoundary(parent,bname,brange,
                btype=CK.Null_s,
                family=None,
                pttype=CK.PointRange_s): 
  """-BC node creation -BC
  
  'newNode:N='*newBoundary*'(parent:N,bname:S,brange:[*i],btype:S)'
  
  Returns a new <node> representing a BC_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list.
  Parent should be Zone_t, returned node is parent.
  If the parent has already a child name ZoneBC then
  only the BC_t,IndexRange_t are created.
  chapter 9.3 Add IndexRange_t required
  """
  CU.checkDuplicatedName(parent,bname)
  zbnode=parent
  if ((zbnode is not None) and
      (zbnode[0]!=CK.ZoneBC_s) and (zbnode[3]!=CK.ZoneBC_ts)):
    zbnode=CU.newNode(CK.ZoneBC_s,None,[],CK.ZoneBC_ts,parent)
  bnode=CU.newNode(bname,CU.setStringAsArray(btype),[],CK.BC_ts,zbnode)
  if (pttype==CK.PointRange_s):
    arange=NPY.array(brange,dtype=NPY.int32,order='Fortran')
    CU.newNode(CK.PointRange_s,arange,[],CK.IndexRange_ts,bnode)
  else:
    arange=NPY.array(brange,dtype=NPY.int32,order='Fortran')
    CU.newNode(CK.PointList_s,arange,[],CK.IndexArray_ts,bnode)
  if (family):
    CU.newNode(CK.FamilyName_s,CU.setStringAsArray(family),[],
            CK.FamilyName_ts,bnode)
  return bnode
  
# -----------------------------------------------------------------------------
def newBCDataSet(parent,name,valueType=CK.Null_s):
  """-BCDataSet node creation -BCDataSet
  
  'newNode:N='*newBCDataSet*'(parent:N,name:S,valueType:CK.BCTypeSimple)'
  
   If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a BCDataSet_t sub-tree.  
   chapter 9.4 Add node BCTypeSimple is required
  """
  node=CU.hasChildName(parent,name)
  if (node == None):    
    node=CU.newNode(name,None,[],CK.BCDataSet_ts,parent)
  if (valueType not in CK.BCTypeSimple_l):
    raise CE.cgnsException(252,valueType)
  node[1]=CU.setStringAsArray(valueType)
  return node

# ---------------------------------------------------------------------------  
def newBCData(parent,name):
  """-BCData node creation -BCData
  
  'newNode:N='*newBCData*'(parent:N,name:S)'
  
   Returns a new <node> representing a BCData_t sub-tree. 
   chapter 9.5 
  """
  CU.checkDuplicatedName(parent,name)    
  node=CU.newNode(name,None,[],CK.BCData_ts,parent)
  return node 
  
# -----------------------------------------------------------------------------
def newBCProperty(parent,wallfunction=CK.Null_s,area=CK.Null_s):
  """-BCProperty node creation -BCProperty
  
  'newNode:N='*newBCProperty*'(parent:N)'
  
   Returns a new <node> representing a BCProperty_t sub-tree.  
   If a parent is given, the new <node> is added to the parent children list.
   chapter 9.6
  """
  CU.checkDuplicatedName(parent,CK.BCProperty_s)    
  node=CU.newNode(CK.BCProperty_s,None,[],CK.BCProperty_ts,parent)
  wf=CU.newNode(CK.WallFunction_s,None,[],CK.WallFunction_ts,node)
  CU.newNode(CK.WallFunctionType_s,CU.setStringAsArray(wallfunction),[],CK.WallFunctionType_ts,wf)
  ar=CU.newNode(CK.Area_s,None,[],CK.Area_ts,node)
  CU.newNode(CK.AreaType_s,CU.setStringAsArray(area),[],CK.AreaType_ts,ar)
  return node 

# -----------------------------------------------------------------------------
def newCoordinates(parent,name=CK.GridCoordinates_s,value=None):
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
  CU.checkDuplicatedName(parent,name)
  gnode=CU.hasChildName(parent,CK.GridCoordinates_s)
  if (gnode == None): gnode=newGridCoordinates(parent,CK.GridCoordinates_s)
  node=newDataArray(gnode,name,value)
  return node
  
# -----------------------------------------------------------------------------
def newAxisymmetry(parent,
                   refpoint=NPY.array([0.0,0.0,0.0]),
                   axisvector=NPY.array([0.0,0.0,0.0])):
  """-Axisymmetry node creation -Axisymmetry
  
  'newNode:N='*newAxisymmetry*'(parent:N,refpoint:A,axisvector:A)'
  
  refpoint,axisvector should be a real array.
  Returns a new <node> representing a CK.Axisymmetry_t sub-tree.   
  chapter 7.5 Add DataArray AxisymmetryAxisVector,AxisymmetryReferencePoint are required
  """
  if (parent): CU.checkNode(parent)
  CU.checkType(parent,CK.CGNSBase_ts,CK.Axisymmetry_s)
  CU.checkDuplicatedName(parent,CK.Axisymmetry_s)
  CU.checkArrayReal(refpoint)
  CU.checkArrayReal(axisvector)
  node=CU.newNode(CK.Axisymmetry_s,None,[],CK.Axisymmetry_ts,parent)
  n=CU.hasChildName(parent,CK.AxisymmetryReferencePoint_s)
  if (n == None):
    n=newDataArray(node,CK.AxisymmetryReferencePoint_s,NPY.array(refpoint))
  n=CU.hasChildName(parent,CK.AxisymmetryAxisVector_s)
  if (n == None):
    n=newDataArray(node,CK.AxisymmetryAxisVector_s,NPY.array(axisvector))
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
  if (parent): CU.checkNode(parent)
  CU.checkTypeList(parent,[CK.CGNSBase_ts,CK.Zone_ts,CK.Family_ts],
                CK.RotatingCoordinates_s)
  CU.checkDuplicatedName(parent,CK.RotatingCoordinates_s)
  CU.checkArrayReal(rotcenter)
  CU.checkArrayReal(ratev)
  node=CU.newNode(CK.RotatingCoordinates_s,None,[],CK.RotatingCoordinates_ts,parent)
  n=CU.hasChildName(node,CK.RotationCenter_s)
  if (n == None): 
    n=newDataArray(node,CK.RotationCenter_s,NPY.array(rotcenter))
  n=CU.hasChildName(node,CK.RotationRateVector_s)
  if (n == None): 
    n=newDataArray(node,CK.RotationRateVector_s,NPY.array(ratev))
  return node

# -----------------------------------------------------------------------------
def newFlowSolution(parent,name='{FlowSolution}',gridlocation=None):
  """-Solution node creation -Solution
  
  'newNode:N='*newSolution*'(parent:N,name:S,gridlocation:None)'
  
  Returns a new <node> representing a FlowSolution_t sub-tree. 
  chapter 7.7
  """
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,None,[],CK.FlowSolution_ts,parent)
  return node  
  
# -----------------------------------------------------------------------------
def newZoneGridConnectivity(parent,name=CK.ZoneGridConnectivity_s):
  """-GridConnectivity node creation -Grid
  
  'newNode:N='*newZoneGridConnectivity*'(parent:N,name:S)'

  Creates a ZoneGridConnectivity_t sub-tree
  This sub-node is returned.
  If a parent is given, the new <node> is added to the parent children list,
  the parent should be a Zone_t.
  chapter 8.1
  """
  CU.checkDuplicatedName(parent,name)
  cnode=CU.hasChildName(parent,CK.ZoneGridConnectivity_s)  
  if (cnode is None):   
    cnode=CU.newNode(CK.ZoneGridConnectivity_s,
                     None,[],CK.ZoneGridConnectivity_ts,parent)
  return cnode
  
# -----------------------------------------------------------------------------
def newGridConnectivity1to1(parent,name,dname,window,dwindow,trans):
  """-GridConnectivity1to1 node creation -Grid
  
  'newNode:N='*newGridConnectivity1to1*'(parent:N,name:S,dname:S,window:[i*],dwindow:[i*],trans:[i*])'
  
  Creates a GridConnectivity1to1_t sub-tree.
  If a parent is given, the new <node> is added to the parent children list,
  the parent should be a Zone_t.
  The returned node is the GridConnectivity1to1_t
  chapter 8.2
  """
  zcnode=CU.newNode(name,CU.setStringAsArray(dname),[],
                    CK.GridConnectivity1to1_ts,parent)
  CU.newNode(CK.Transform_s,NPY.array(list(trans),dtype=NPY.int32),[],
             CK.Transform_ts,zcnode)
  CU.newNode(CK.PointRange_s,
             NPY.array(window,dtype=NPY.int32,order='Fortran'),[],
             CK.IndexRange_ts,zcnode)   
  CU.newNode(CK.PointRangeDonor_s,
             NPY.array(dwindow,dtype=NPY.int32,order='Fortran'),[],
             CK.IndexRange_ts,zcnode)   
  return zcnode

# -----------------------------------------------------------------------------
def newGridConnectivity(parent,name,dname,ctype=CK.Overset_s):
  """-GridConnectivity node creation -Grid
  
  'newNode:N='*newGridConnectivity*'(parent:N,name:S,dname:S,ctype:S)'
  
  Creates a GridConnectivity sub-tree.
  If a parent is given, the new <node> is added to the parent children list,
  the parent should be a ZoneGridConnectivity_t.
  The returned node is the GridConnectivity_t
  chapter 8.4
  """
  zcnode=CU.newNode(name,CU.setStringAsArray(dname),
                    [],CK.GridConnectivity_ts,parent)
  zcnode[1]=CU.setStringAsArray(ctype)
  return zcnode

# -----------------------------------------------------------------------------
def newGridConnectivityProperty(parent): 
  """-GridConnectivityProperty node creation -GridConnectivityProperty
  
  'newNode:N='*newGridConnectivityProperty*'(parent:N)'
  
   Returns a new <node> representing a GridConnectivityProperty_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   chapter 8.5 
  """
  CU.checkDuplicatedName(parent,CK.GridConnectivityProperty_s)   
  nodeType=CU.newNode(CK.GridConnectivityProperty_s,None,[],
                   CK.GridConnectivityProperty_ts,parent)
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
  if (parent): CU.checkNode(parent)  
  CU.checkArrayReal(rotcenter)
  CU.checkArrayReal(ratev)
  CU.checkArrayReal(trans)
  cnode=CU.hasChildName(parent,CK.Periodic_s)
  if (cnode == None):
    cnode=CU.newNode(CK.Periodic_s,None,[],CK.Periodic_ts,parent)
  n=CU.hasChildName(cnode,CK.RotationCenter_s)
  if (n == None): 
    newDataArray(cnode,CK.RotationCenter_s,NPY.array(rotcenter))
  n=CU.hasChildName(cnode,CK.RotationAngle_s)
  if (n == None): 
    newDataArray(cnode,CK.RotationAngle_s,NPY.array(ratev))
  n=CU.hasChildName(cnode,CK.Translation_s)
  if (n == None): 
    newDataArray(cnode,CK.Translation_s,NPY.array(trans)) 
  return cnode
  
# -----------------------------------------------------------------------------
def newAverageInterface(parent,valueType=CK.Null_s):
  """-AverageInterface node creation -AverageInterface
  
  'newNode:N='*newAverageInterface*'(parent:N,valueType:CK.AverageInterfaceType)'
  
   Returns a new <node> representing a AverageInterface_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   If the parent has already a child name AverageInterface then
   only the AverageInterfaceType is created.
   chapter 8.5.2
  """
  node=CU.hasChildName(parent,CK.AverageInterface_s)
  if (node == None):       
    node=CU.newNode(CK.AverageInterface_s,None,[],
                 CK.AverageInterface_ts,parent)
  if (valueType not in CK.AverageInterfaceType_l):
    raise CE.cgnsException(253,valueType)
  CU.checkDuplicatedName(node,CK.AverageInterfaceType_s) 
  ## code correction: Modify valueType string into NPY string array
  nodeType=CU.newNode(CK.AverageInterfaceType_s,CU.setStringAsArray(valueType),[],
                   CK.AverageInterfaceType_ts,node)
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
  cnode=CU.hasChildName(parent,CK.ZoneGridConnectivity_s)
  if (cnode == None):
    cnode=CU.newNode(CK.ZoneGridConnectivity_s,None,[],CK.ZoneGridConnectivity_ts,parent)
  CU.checkDuplicatedName(cnode,name)   
  node=CU.newNode(name,None,[],CK.OversetHoles_ts,cnode)
  #if(pname!=None and value!=None):
    #newPointList(node,pname,value)
  if hrange!=None:  
    ## code correction: Modify PointRange shape and order
   newPointRange(node,CK.PointRange_s,NPY.array(hrange,dtype=NPY.int32,order='Fortran'))
   #newNode(CK.PointRange_s,NPY.array(list(hrange),'i'),[],CK.IndexRange_ts,node)
  return node

# -----------------------------------------------------------------------------
def newFlowEquationSet(parent):
  """-FlowEquationSet node creation -FlowEquationSet
  
  'newNode:N='*newFlowEquationSet*'(parent:N)'
  
  If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a CK.FlowEquationSet_t sub-tree.  
   chapter 10.1
  """
  if (parent): CU.checkNode(parent)
  CU.checkDuplicatedName(parent,CK.FlowEquationSet_s)
  CU.checkTypeList(parent,[CK.CGNSBase_ts,CK.Zone_ts],CK.FlowEquationSet_s)     
  node=CU.newNode(CK.FlowEquationSet_s,None,[],CK.FlowEquationSet_ts,parent)  
  return node   
    
def newGoverningEquations(parent,valueType=CK.Euler_s):
  """-GoverningEquations node creation -GoverningEquations
  
  'newNode:N='*newGoverningEquations*'(parent:N,valueType:CK.GoverningEquationsType)'
  
   Returns a new <node> representing a CK.GoverningEquations_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name GoverningEquations then
   only the GoverningEquationsType is created.
   chapter  10.2 Add node GoverningEquationsType is required   
  """
  node=CU.hasChildName(parent,CK.GoverningEquations_s)
  if (node == None):    
    node=CU.newNode(CK.GoverningEquations_s,None,[],CK.GoverningEquations_ts,parent)
  if (valueType not in CK.GoverningEquationsType_l):
      raise CE.cgnsException(221,valueType)
  CU.checkDuplicatedName(parent,CK.GoverningEquationsType_s,)
  node[1]=CU.setStringAsArray(valueType)
  return node
  
# -----------------------------------------------------------------------------
def newGasModel(parent,valueType=CK.Ideal_s):
  """-GasModel node creation -GasModel
  
  'newNode:N='*newGasModel*'(parent:N,valueType:CK.GasModelType)'
  
   Returns a new <node> representing a CK.GasModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name GasModel then
   only the GasModelType is created. 
   chapter 10.3 Add node GasModelType is required  
  """
  node=CU.hasChildName(parent,CK.GasModel_s)
  if (node == None):       
    node=CU.newNode(CK.GasModel_s,None,[],CK.GasModel_ts,parent)
  if (valueType not in CK.GasModelType_l):
    raise CE.cgnsException(224,valueType)
  CU.checkDuplicatedName(node,CK.GasModelType_s)  
  node[1]=CU.setStringAsArray(valueType)
  return node
  
def newThermalConductivityModel(parent,valueType=CK.SutherlandLaw_s):   
  """-ThermalConductivityModel node creation -ThermalConductivityModel
  
  'newNode:N='*newThermalConductivityModel*'(parent:N,valueType:CK.ThermalConductivityModelType)'
  
   Returns a new <node> representing a CK.ThermalConductivityModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ThermalConductivityModel then
   only the ThermalConductivityModelType is created. 
   chapter 10.5 Add node ThermalConductivityModelType is required     
  """
  node=CU.hasChildName(parent,CK.ThermalConductivityModel_s)
  if (node == None):    
    node=CU.newNode(CK.ThermalConductivityModel_s,None,[],
                 CK.ThermalConductivityModel_ts,parent)
  if (valueType not in CK.ThermalConductivityModelType_l):
    raise CE.cgnsException(227,valueType)
  CU.checkDuplicatedName(node,CK.ThermalConductivityModelType_s)
  node[1]=CU.setStringAsArray(valueType)
  return node

def newViscosityModel(parent,valueType=CK.SutherlandLaw_s): 
  """-ViscosityModel node creation -ViscosityModel
  
  'newNode:N='*newViscosityModel*'(parent:N,valueType:CK.ViscosityModelType)'
  
   Returns a new <node> representing a CK.ViscosityModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ViscosityModel then
   only the ViscosityModelType is created. 
   chapter 10.4 Add node ViscosityModelType is (r)       
  """  
  node=CU.hasChildName(parent,CK.ViscosityModel_s)
  if (node == None):    
    node=CU.newNode(CK.ViscosityModel_s,None,[],CK.ViscosityModel_ts,parent)    
  if (valueType not in CK.ViscosityModelType_l):
    raise CE.cgnsException(230,valueType) 
  CU.checkDuplicatedName(node,CK.ViscosityModelType_s)  
  node[1]=CU.setStringAsArray(valueType)
  return node

def newTurbulenceClosure(parent,valueType=CK.Null_s):   
  """-TurbulenceClosure node creation -TurbulenceClosure
  
  'newNode:N='*newTurbulenceClosure*'(parent:N,valueType:CK.TurbulenceClosureType)'  
   Returns a new <node> representing a CK.TurbulenceClosure_t sub-tree.  
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name TurbulenceClosure then
   only the ViscosityModelType is created.
   chapter 10.5 Add node TurbulenceClosureType is (r)       
  """
  node=CU.hasChildName(parent,CK.TurbulenceClosure_s)
  if (node == None):    
    node=CU.newNode(CK.TurbulenceClosure_s,None,[],CK.TurbulenceClosure_ts,parent)
  if (valueType not in CK.TurbulenceClosureType_l):
    raise CE.cgnsException(233,valueType)
  CU.checkDuplicatedName(node,CK.TurbulenceClosureType_s)
  node[1]=CU.setStringAsArray(valueType)
  return node

def newTurbulenceModel(parent,valueType=CK.OneEquation_SpalartAllmaras_s): 
  """-TurbulenceModel node creation -TurbulenceModel
  
  'newNode:N='*newTurbulenceModel*'(parent:N,valueType:CK.TurbulenceModelType)'
  
   Returns a new <node> representing a CK.TurbulenceModel_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name TurbulenceModel then
   only the TurbulenceModelType is created.
   chapter 10.6.2 Add node TurbulenceModelType is (r)  
  """ 
  node=CU.hasChildName(parent,CK.TurbulenceModel_s)
  if (node == None):
    node=CU.newNode(CK.TurbulenceModel_s,None,[],CK.TurbulenceModel_ts,parent)
  if (valueType not in CK.TurbulenceModelType_l):
    raise CE.cgnsException(236,valueType)  
  CU.checkDuplicatedName(node,CK.TurbulenceModelType_s)
  node[1]=CU.setStringAsArray(valueType)
  return node

def newThermalRelaxationModel(parent,valueType=CK.Null_s):
  """-ThermalRelaxationModel node creation -ThermalRelaxationModel
  
  'newNode:N='*newThermalRelaxationModel*'(parent:N,valueType:CK.ThermalRelaxationModelType)'
  
   Returns a new <node> representing a CK.ThermalRelaxationModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ThermalRelaxationModel then
   only the ThermalRelaxationModelType is created.  
   chapter 10.7 Add node ThermalRelaxationModelType is (r)
  """
  node=CU.hasChildName(parent,CK.ThermalRelaxationModel_s) 
  if (node == None):          
    node=CU.newNode(CK.ThermalRelaxationModel_s,None,[],
                 CK.ThermalRelaxationModel_ts,parent)
  if (valueType not in CK.ThermalRelaxationModelType_l):
    raise CE.cgnsException(239,valueType) 
  CU.checkDuplicatedName(node,CK.ThermalRelaxationModelType_s)   
  node[1]=CU.setStringAsArray(valueType)
  return node

def newChemicalKineticsModel(parent,valueType=CK.Null_s):
  """-ChemicalKineticsModel node creation -ChemicalKineticsModel
  
  'newNode:N='*newChemicalKineticsModel*'(parent:N,valueType:CK.ChemicalKineticsModelType)'
  
   Returns a new <node> representing a CK.ChemicalKineticsModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name ChemicalKineticsModel then
   only the ChemicalKineticsModelType is created. 
   chapter 10.8 Add node ChemicalKineticsModelType is (r)  
  """
  node=CU.hasChildName(parent,CK.ChemicalKineticsModel_s) 
  if (node == None):             
    node=CU.newNode(CK.ChemicalKineticsModel_s,None,[],
                 CK.ChemicalKineticsModel_ts,parent)
  if (valueType not in CK.ChemicalKineticsModelType_l):
    raise CE.cgnsException(242,valueType)
  CU.checkDuplicatedName(node,CK.ChemicalKineticsModelType_s)     
  node[1]=CU.setStringAsArray(valueType)
  return node

def newEMElectricFieldModel(parent,valueType=CK.Null_s):
  """-EMElectricFieldModel node creation -EMElectricFieldModel
  
  'newNode:N='*newEMElectricFieldModel*'(parent:N,valueType:CK.EMElectricFieldModelType)'
  
   Returns a new <node> representing a CK.EMElectricFieldModel_t sub-tree.
   If a parent is given, the new <node> is added to the parent children list.
    If the parent has already a child name EMElectricFieldModel then
   only the EMElectricFieldModelType is created. 
   chapter 10.9 Add node EMElectricFieldModelType is (r)  
  """
  node=CU.hasChildName(parent,CK.EMElectricFieldModel_s)   
  if (node == None):           
    node=CU.newNode(CK.EMElectricFieldModel_s,None,[],
                 CK.EMElectricFieldModel_ts,parent)
  if (valueType not in CK.EMElectricFieldModelType_l):
    raise CE.cgnsException(245,valueType)
  CU.checkDuplicatedName(node,CK.EMElectricFieldModelType_s)  
  node[1]=CU.setStringAsArray(valueType)
  return node

def newEMMagneticFieldModel(parent,valueType=CK.Null_s):
  """-EMMagneticFieldModel node creation -EMMagneticFieldModel
  
  'newNode:N='*newEMMagneticFieldModel*'(parent:N,valueType:CK.EMMagneticFieldModelType)'
  
   Returns a new <node> representing a CK.EMMagneticFieldModel_t sub-tree.  
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name EMMagneticFieldModel_s then
   only the EMMagneticFieldModelType is created. 
   chapter 10.9.2 Add node EMMagneticFieldModelType is (r)  
  """
  node=CU.hasChildName(parent,CK.EMMagneticFieldModel_s)   
  if (node == None):            
    node=CU.newNode(CK.EMMagneticFieldModel_s,None,[],
                 CK.EMMagneticFieldModel_ts,parent)
  if (valueType not in CK.EMMagneticFieldModelType_l):
    raise CE.cgnsException(248,valueType)  
  CU.checkDuplicatedName(node,CK.EMMagneticFieldModelType_s) 
  node[1]=CU.setStringAsArray(valueType)
  return node

def newEMConductivityModel(parent,valueType=CK.Null_s):
  """-EMConductivityModel node creation -EMConductivityModel
  
  'newNode:N='*newEMConductivityModel*'(parent:N,valueType:CK.EMConductivityModelType)'
  
   Returns a new <node> representing a CK.EMConductivityModel_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name EMConductivityModel then
   only the EMConductivityModelType is created. 
   chapter 10.9.3 Add node EMConductivityModelType is (r)  
  """
  node=CU.hasChildName(parent,CK.EMConductivityModel_s)  
  if (node == None):             
    node=CU.newNode(CK.EMConductivityModel_s,None,[],
                 CK.EMConductivityModel_ts,parent)
  if (valueType not in CK.EMConductivityModelType_l):
    raise CE.cgnsException(218,stype)  
  CU.checkDuplicatedName(node,CK.EMConductivityModelType_s)  
  node[1]=CU.setStringAsArray(valueType)
  return node

# -----------------------------------------------------------------------------
def newBaseIterativeData(parent,name,nsteps=0,
                         itype=CK.IterationValues_s):
  """-BaseIterativeData node creation -BaseIterativeData
  
   'newNode:N='*newBaseIterativeData*'(parent:N,name:S,nsteps:I,itype:E)'
  
   Returns a new <node> representing a BaseIterativeData_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter 11.1.1
   NumberOfSteps is required, TimeValues or IterationValues are required
  """ 
  
  if (parent): CU.checkNode(parent)
  CU.checkDuplicatedName(parent,name)
  CU.checkType(parent,CK.CGNSBase_ts,CK.BaseIterativeData_ts)
  if ((type(nsteps) != type(1)) or (nsteps < 0)): raise CE.cgnsException(209)
  asteps=NPY.arange(1,nsteps+1,dtype='i')
  node=CU.newNode(name,NPY.array([nsteps],dtype='i'),[],
                  CK.BaseIterativeData_ts,parent)
  if (itype not in [CK.IterationValues_s, CK.TimeValues_s]):
    raise CE.cgnsException(210,(CK.IterationValues_s, CK.TimeValues_s))
  CU.newNode(itype,asteps,[],CK.DataArray_ts,node)  
  return node

# -----------------------------------------------------------------------------
def newZoneIterativeData(parent,name):
  """-ZoneIterativeData node creation -ZoneIterativeData
  
  'newNode:N='*newZoneIterativeData*'(parent:N,name:S)'
  
   Returns a new <node> representing a ZoneIterativeData_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  11.1.2
  """ 
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,None,[],CK.ZoneIterativeData_ts,parent)
  return node

# ---------------------------------------------------------------------------  
def newRigidGridMotion(parent,name,
                       valueType=CK.Null_s,
                       vector=NPY.array([0.0,0.0,0.0])):
  """-RigidGridMotion node creation -RigidGridMotion
  
  'newNode:N='*newRigidGridMotion*'(parent:N,name:S,valueType:CK.RigidGridMotionType,vector:A)'
  
  If a parent is given, the new <node> is added to the parent children list.
   Returns a new <node> representing a CK.RigidGridMotion_t sub-tree.  
   If the parent has already a child name RigidGridMotion then
   only the RigidGridMotionType is created and OriginLocation is created
   chapter 11.2 Add Node RigidGridMotionType and add DataArray OriginLocation are the only required
  """
  if (parent): CU.checkNode(parent)  
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,None,[],CK.RigidGridMotion_ts,parent)
  
  if (valueType not in CK.RigidGridMotionType_l):
      raise CE.cgnsException(254,valueType)
  node[1]=CU.setStringAsArray(valueType)
  n=CU.hasChildName(parent,CK.OriginLocation_s)
  if (n == None): 
    n=newDataArray(node,CK.OriginLocation_s,NPY.array(vector))
  return node
  
#-----------------------------------------------------------------------------
def newReferenceState(parent,name=CK.ReferenceState_s):
  """-ReferenceState node creation -ReferenceState
  
  'newNode:N='*newReferenceState*'(parent:N,name:S)'
  
   Returns a new <node> representing a ReferenceState_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.1  """   
  if (parent): CU.checkNode(parent)
  node=CU.hasChildName(parent,name)
  if (node == None):
    CU.checkDuplicatedName(parent,name)
    node=CU.newNode(name,None,[],CK.ReferenceState_ts,parent)
  return node

#-----------------------------------------------------------------------------
def newConvergenceHistory(parent,name=CK.GlobalConvergenceHistory_s,
			  iterations=0):
  """-ConvergenceHistory node creation -ConvergenceHistory
  
  'newNode:N='*newConvergenceHistory*'(parent:N,name:S,iterations:i)'
  
   Returns a new <node> representing a ConvergenceHistory_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.3  """   
  if (name not in CK.ConvergenceHistory_l): raise CE.cgnsException(201,name)
  if (parent):
    CU.checkNode(parent)
    CU.checkTypeList(parent,[CK.CGNSBase_ts,CK.Zone_ts],name)
  if (name == CK.GlobalConvergenceHistory_s):
    CU.checkType(parent,CK.CGNSBase_ts,name)
  if (name == CK.ZoneConvergenceHistory_s):
    CU.checkType(parent,CK.Zone_ts,name)
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,NPY.array([iterations],dtype='i'),[],CK.ConvergenceHistory_ts,parent)
  return node

#-----------------------------------------------------------------------------
def newIntegralData(parent,name):
  """-IntegralData node creation -IntegralData
  
  'newNode:N='*newIntegralData*'(parent:N,name:S)'
  
   Returns a new <node> representing a IntegralData_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.5 
  """ 
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,None,[],CK.IntegralData_ts,parent)
  return node
  
# -----------------------------------------------------------------------------
def newFamily(parent,name):
  """-Family node creation -Family

  'newNode:N='*newFamily*'(parent:N,name:S)'
  
   Returns a new <node> representing a Family_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list. 
   chapter  12.6 
  """ 
  if (parent): CU.checkNode(parent)
  CU.checkType(parent,CK.CGNSBase_ts,name)
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,None,[],CK.Family_ts,parent)
  return node

def newFamilyName(parent,family=None):
  ## code correction: Modify family string into NPY string array
  return CU.newNode(CK.FamilyName_s,CU.setStringAsArray(family),[],CK.FamilyName_ts,parent)

# -----------------------------------------------------------------------------
def newGeometryReference(parent,name='{GeometryReference}',
                         valueType=CK.UserDefined_s):
  """-GeometryReference node creation -GeometryReference
  
  'newNode:N='*newGeometryReference*'(parent:N,name:S,valueType:CK.GeometryFormat)'
  
   Returns a new <node> representing a CK.GeometryFormat_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name CK.GeometryReference then
   only the .GeometryFormat is created
   chapter  12.7 Add node CK.GeometryFormat_t is (r) and GeometryFile_t definition not find but is required (CAD file)
  """
  node=CU.hasChildName(parent,CK.GeometryReference_s)
  if (node == None):    
    node=CU.newNode(name,None,[],CK.GeometryReference_ts,parent)
  if (valueType not in CK.GeometryFormat_l):
      raise CE.cgnsException(256,valueType)
  CU.checkDuplicatedName(node,CK.GeometryFormat_s)
  ## code correction: Modify valueType string into NPY string array
  nodeType=CU.newNode(CK.GeometryFormat_s,CU.setStringAsArray(valueType),[],
                   CK.GeometryFormat_ts,node)
  return node
  
# -----------------------------------------------------------------------------
def newFamilyBC(parent,valueType=CK.UserDefined_s): 
  """-FamilyBC node creation -FamilyBC
  
  'newNode:N='*newFamilyBC*'(parent:N,valueType:CK.BCTypeSimple/CK.BCTypeCompound)'
  
   Returns a new <node> representing a CK.FamilyBC_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   If the parent has already a child name FamilyBC then
   only the BCType is created
   chapter  12.8 Add node BCType is required   
  """ 
  node=CU.hasChildName(parent,CK.FamilyBC_s)
  if (node == None):    
    node=CU.newNode(CK.FamilyBC_s,None,[],CK.FamilyBC_ts,parent)
  if (    valueType not in CK.BCTypeSimple_l
      and valueType not in CK.BCTypeCompound_l):
      raise CE.cgnsException(257,valueType)
  CU.checkDuplicatedName(node,CK.BCType_s)
  ## code correction: Modify valueType string into NPY string array
  nodeType=CU.newNode(CK.BCType_s,CU.setStringAsArray(valueType),[],
                     CK.BCType_ts,node)
  return node

# -----------------------------------------------------------------------------
def newArbitraryGridMotion(parent,name,valuetype=CK.Null_s):
  """
  Returns a **new node** representing a :ref:`XArbitraryGridMotionType_t`

  :param parent: CGNS/Python node
  :param name: String
  :param valuetype: String (``CGNS.PAT.cgnskeywords.ArbitraryGridMotionType``)


  If a *parent* is not ``None``, the **new node** is added to the parent
  children list. If the *parent* has already a child with
  name ``RigidGridMotion`` then only the ``RigidGridMotionType`` is created.

  """
  node=None
  if (parent): node=CU.hasChildName(parent,name)
  if (node == None):
    node=CU.newNode(name,None,[],CK.ArbitraryGridMotion_ts,parent)      
  if (valuetype not in CK.ArbitraryGridMotionType_l):
    raise CE.cgnsException(255,valuetype) 
  CU.checkDuplicatedName(node,CK.ArbitraryGridMotionType_s)     
  node[1]=CU.setStringAsArray(valuetype)
  return node
  
# -----------------------------------------------------------------------------
def newUserDefinedData(parent,name):
  """-UserDefinedData node creation -UserDefinedData
  
  'newNode:N='*newUserDefinedData*'(parent:N,name:S)'
  
   Returns a new <node> representing a UserDefinedData_t sub-tree. 
   If a parent is given, the new <node> is added to the parent children list.
   chapter  12.9  
  """ 
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,None,[],CK.UserDefinedData_ts,parent)
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
  if (parent): CU.checkNode(parent)
  CU.checkType(parent,CK.CGNSBase_ts,CK.Gravity_s)
  CU.checkDuplicatedName(parent,CK.Gravity_s)
  CU.checkArrayReal(gvector)
  node=CU.newNode(CK.Gravity_s,None,[],CK.Gravity_ts,parent)
  n=CU.hasChildName(parent,CK.GravityVector_s)
  if (n == None): 
    n=newDataArray(node,CK.GravityVector_s,NPY.array(gvector))
  return node

# -----------------------------------------------------------------------------
def newField(parent,name,value):
  CU.checkDuplicatedName(parent,name)
  node=newDataArray(parent,name,value)
  return node

# -----------------------------------------------------------------------------
def newModel(parent,name,label,value):
  CU.checkDuplicatedName(parent,name)
  node=CU.newNode(name,value,[],label,parent)
  return node
    
# -----------------------------------------------------------------------------
def newDiffusionModel(parent,value=None):
  # the diffusion_t doesn't exist. We use the cgnspatch file to keep
  # track of this...
  CU.checkDuplicatedName(parent,CK.DiffusionModel_s)
  node=CU.newNode(CK.DiffusionModel_s,value,[],CK.DiffusionModel_ts,parent)
  return node

# -----------------------------------------------------------------------------
#def newSection():
#  pass

# -----------------------------------------------------------------------------
def newParentElements(parent,value):
  CU.checkDuplicatedName(parent,CK.ParentElements_s)
  node=CU.newNode(CK.ParentElements_s,value,[],CK.DataArray_ts,parent)
  return node

# -----------------------------------------------------------------------------
def newParentElementsPosition(parent,value):
  CU.checkDuplicatedName(parent,CK.ParentElementsPosition_s)
  node=CU.newNode(CK.ParentElementsPosition_s,value,[],CK.DataArray_ts,parent)
  return node

# -----------------------------------------------------------------------------
#def newPart():
#  pass

# -----------------------------------------------------------------------------
def nextRange(previous,etype,earray):
  r=previous
  if (previous==None): r=NPY.array([0,0],dtype='i')
  npe=CK.ElementTypeNPE[etype]
  if (npe):
    nelems=len(earray.flat)/npe
  else:
    raise 'Oupss not implemented variable number of elems'
  start=r[1]+1
  end  =start+nelems-1
  return NPY.array([start,end],dtype='i')

# -----------------------------------------------------------------------------
def getElementTypes(level,physicaldimension):
  eltype_3D = [CK.TETRA_4_s, CK.TETRA_10_s, CK.PYRA_5_s, CK.PYRA_14_s,
               CK.PENTA_6_s, CK.PENTA_15_s, CK.PENTA_18_s,
               CK.HEXA_8_s, CK.HEXA_20_s, CK.HEXA_27_s, CK.MIXED_s, CK.PYRA_13_s,
               CK.TETRA_16_s, CK.TETRA_20_s, CK.PYRA_21_s, CK.PYRA_29_s,
               CK.PYRA_30_s, CK.PENTA_24_s, CK.PENTA_38_s, CK.PENTA_40_s, CK.HEXA_32_s,
               CK.HEXA_56_s, CK.HEXA_64_s ]
  eltype_2D = [CK.TRI_3_s, CK.TRI_6_s, CK.QUAD_4_s, CK.QUAD_8_s, CK.QUAD_9_s,
               CK.TRI_9_s, CK.TRI_10_s, CK.QUAD_12_s, CK.QUAD_16_s ]
  eltype_1D = [CK.BAR_2_s, CK.BAR_3_s, CK.BAR_4_s ]
  eltype_0D = [CK.NODE_s ]
  
  if   (level == CK.Cell_s and physicaldimension == 3):
    eltype = eltype_3D
    eltype.append(CK.NFACE_n_s)
  elif (level == CK.Cell_s and physicaldimension == 2):
    eltype = eltype_2D
    eltype.append(CK.NFACE_n_s)
  elif (level == CK.Cell_s and physicaldimension == 1):
    eltype = eltype_1D
    eltype.append(CK.NFACE_n_s)
    
  elif (level == CK.Face_s and physicaldimension == 3):
    eltype = eltype_2D
    eltype.append(CK.NGON_n_s)    
  elif (level == CK.Face_s and physicaldimension == 2):
    eltype = eltype_1D
    eltype.append(CK.NGON_n_s) 
  elif (level == CK.Face_s and physicaldimension == 1):
    eltype = eltype_0D
    eltype.append(CK.NGON_n_s) 

  elif (level == CK.Edge_s and physicaldimension == 3):
    eltype = eltype_1D 
  elif (level == CK.Edge_s and physicaldimension == 2):
    eltype = eltype_0D
  
  elif (level == CK.Vertex_s and physicaldimension == 3):
    eltype = eltype_0D   
        
  else :
    eltype = []  
  return eltype