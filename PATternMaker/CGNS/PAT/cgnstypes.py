#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.PAT.cgnskeywords as CK

tlistA=[
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    CK.DataClass_ts,
    CK.DimensionalUnits_ts,
    ]

allDT=[CK.C1,CK.MT,CK.I4,CK.I8,CK.R4,CK.R8] # LK is default

C_00='Zero/Zero'
C_01='Zero/One'
C_11='One/One'
C_0N='Zero/N'
C_1N='One/N'
C_NN='N/N'

UD='{UserDefined}'

allCARD=[C_01,C_11,C_0N,C_1N,C_NN]

# --------------------------------------------------------
class CGNStype:
  def __init__(self,ntype,dtype=[CK.MT],names=[UD]):
    self.type=ntype
    self.datatype=[CK.LK]+dtype
    self.enumerate=[]
    self.shape=()
    self.names=names
    self.children=[]
    self.parents=[]
  def hasChild(self,ctype):
    for c in self.children:
      if (c[0]==ctype): return True
    return False
  def addChild(self,ctype,cname=UD,dtype=CK.MT,card=C_0N):
    if (type(cname)!=list): lname=[cname]
    else: lname=cname
    self.children.append((ctype,lname,dtype,card))
  def addParent(self,parent):
    self.parents.append(parent)
  def cardinality(self,childtype):
    for c in self.children:
      if (c[0]==childtype): return c[3]
    return C_00
  def isReservedName(self,name):
    for c in self.children:
      if (name in c[1]): return True
    return False
  def hasReservedNameType(self,name):
    nl=[]
    for c in self.children:
      if (name in c[1]): nl.append(c[0])
    return nl

cgt={}

# --------------------------------------------------------
t=CK.CGNSLibraryVersion_ts
cgt[t]=CGNStype(t,dtype=[CK.R4],names=[CK.CGNSLibraryVersion_s])
cgt[t].shape=(1,)

# --------------------------------------------------------
t=CK.Descriptor_ts
cgt[t]=CGNStype(t,dtype=[CK.C1])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.Ordinal_ts
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.Ordinal_s])
cgt[t].shape=(1,)

# --------------------------------------------------------
t=CK.DataClass_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.DataClass_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.DataClass_l

# --------------------------------------------------------
t=CK.DimensionalUnits_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.DimensionalUnits_s])
cgt[t].shape=(32,5)
cgt[t].enumerate=CK.AllDimensionalUnits_l
cgt[t].addChild(CK.AdditionalUnits_ts,CK.AdditionalUnits_s)

# --------------------------------------------------------
t=CK.AdditionalUnits_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.AdditionalUnits_s])
cgt[t].shape=(32,3)
cgt[t].enumerate=CK.AllAdditionalUnits_l

# --------------------------------------------------------
t=CK.DataConversion_ts
cgt[t]=CGNStype(t,dtype=[CK.R4,CK.R8],names=[CK.DataConversion_s])
cgt[t].shape=(2,)

# --------------------------------------------------------
t=CK.DimensionalExponents_ts
cgt[t]=CGNStype(t,dtype=[CK.R4,CK.R8],names=[CK.DimensionalExponents_s])
cgt[t].shape=(5,)

# --------------------------------------------------------
t=CK.AdditionalExponents_ts
cgt[t]=CGNStype(t,dtype=[CK.R4,CK.R8],names=[CK.AdditionalExponents_s])
cgt[t].shape=(3,)

# --------------------------------------------------------
t=CK.DataArray_ts
cgt[t]=CGNStype(t,dtype=allDT)
cgt[t].addChild(CK.DimensionalExponents_ts,CK.DimensionalExponents_s)
cgt[t].addChild(CK.DataConversion_ts,CK.DataConversion_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)

# --------------------------------------------------------
t=CK.Transform_ts
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.Transform_s])
cgt[t].shape=(0,)
t=CK.Transform_ts2
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.Transform_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.DiffusionModel_ts
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.DiffusionModel_s])
cgt[t].shape=(0,)
t=CK.DiffusionModel_ts2
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.DiffusionModel_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.InwardNormalIndex_ts
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.InwardNormalIndex_s])
cgt[t].shape=(0,)
t=CK.InwardNormalIndex_ts2
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.InwardNormalIndex_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.EquationDimension_ts
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.EquationDimension_s])
cgt[t].shape=(1,)
t=CK.EquationDimension_ts2
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.EquationDimension_s])
cgt[t].shape=(1,)

# --------------------------------------------------------
t=CK.GridLocation_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GridLocation_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.Rind_ts
cgt[t]=CGNStype(t,dtype=[CK.I4],names=[CK.Rind_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.IndexRange_ts
cgt[t]=CGNStype(t,dtype=[CK.I4])
cgt[t].shape=(0,2)
cgt[t].names=[CK.PointRange_s,CK.PointRangeDonor_s,CK.ElementRange_s,UD]

# --------------------------------------------------------
t=CK.IndexArray_ts
cgt[t]=CGNStype(t,dtype=[CK.I4,CK.R4,CK.R8])
cgt[t].shape=(0,0)
cgt[t].names=[CK.PointList_s,CK.PointListDonor_s,CK.CellListDonor_s,
              CK.InwardNormalList_s,UD]

# --------------------------------------------------------
t=CK.ReferenceState_ts
cgt[t]=CGNStype(t,names=[CK.ReferenceState_s])
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.Descriptor_ts,CK.ReferenceStateDescription_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ConvergenceHistory_ts
cgt[t]=CGNStype(t,names=[CK.GlobalConvergenceHistory_s,
                         CK.ZoneConvergenceHistory_s],dtype=[CK.I4])
cgt[t].shape=(1,)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.Descriptor_ts,CK.NormDefinitions_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.IntegralData_ts
cgt[t]=CGNStype(t)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.UserDefinedData_ts
cgt[t]=CGNStype(t)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
cgt[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
cgt[t].addChild(CK.IndexArray_ts,CK.PointList_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.FamilyName_ts,[CK.FamilyName_s],card=C_01)
cgt[t].addChild(CK.AdditionalFamilyName_ts,card=C_0N)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)

# --------------------------------------------------------
t=CK.Gravity_ts
cgt[t]=CGNStype(t)
cgt[t].addChild(CK.DataArray_ts,CK.GravityVector_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.FlowEquationSet_ts
cgt[t]=CGNStype(t,names=[CK.FlowEquationSet_s])
cgt[t].addChild(CK.GoverningEquations_ts,CK.GoverningEquations_s)
cgt[t].addChild(CK.EquationDimension_ts,CK.EquationDimension_s)
cgt[t].addChild(CK.GasModel_ts,CK.GasModel_s)
cgt[t].addChild(CK.ViscosityModel_ts,CK.ViscosityModel_s)
cgt[t].addChild(CK.ThermalRelaxationModel_ts,CK.ThermalRelaxationModel_s)
cgt[t].addChild(CK.ThermalConductivityModel_ts,CK.ThermalConductivityModel_s)
cgt[t].addChild(CK.TurbulenceModel_ts,CK.TurbulenceModel_s)
cgt[t].addChild(CK.TurbulenceClosure_ts,CK.TurbulenceClosure_s)
cgt[t].addChild(CK.ChemicalKineticsModel_ts,CK.ChemicalKineticsModel_s)
cgt[t].addChild(CK.EMMagneticFieldModel_ts,CK.EMMagneticFieldModel_s)
cgt[t].addChild(CK.EMElectricFieldModel_ts,CK.EMElectricFieldModel_s)
cgt[t].addChild(CK.EMConductivityModel_ts,CK.EMConductivityModel_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GoverningEquations_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GoverningEquations_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.GoverningEquationsType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DiffusionModel_ts,CK.DiffusionModel_s)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GasModel_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GasModel_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.GasModelType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ViscosityModel_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ViscosityModel_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.ViscosityModelType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ThermalConductivityModel_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ThermalConductivityModel_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.ThermalConductivityModelType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.TurbulenceClosure_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.TurbulenceClosure_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.TurbulenceClosureType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.TurbulenceModel_ts
cgt[t]=CGNStype(t)
cgt[t].datatype=[CK.C1]
cgt[t].shape=(0,)
cgt[t].enumerate=CK.TurbulenceModelType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DiffusionModel_ts,CK.DiffusionModel_s)

# --------------------------------------------------------
t=CK.ThermalRelaxationModel_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ThermalRelaxationModel_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.ThermalRelaxationModelType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ChemicalKineticsModel_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ChemicalKineticsModel_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.ChemicalKineticsModelType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.EMElectricFieldModel_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.EMElectricFieldModel_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.EMElectricFieldModelType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.EMMagneticFieldModel_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.EMMagneticFieldModel_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.EMMagneticFieldModelType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.EMConductivityModel_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.EMConductivityModel_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.EMConductivityModelType_l
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ZoneType_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ZoneType_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.ZoneType_l

# --------------------------------------------------------
t=CK.SimulationType_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.SimulationType_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.SimulationType_l

# --------------------------------------------------------
t=CK.GridConnectivityType_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GridConnectivityType_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.GridConnectivityType_l

# --------------------------------------------------------
t=CK.Family_ts
cgt[t]=CGNStype(t)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)
cgt[t].addChild(CK.FamilyBC_ts)
cgt[t].addChild(CK.GeometryReference_ts)
cgt[t].addChild(CK.RotatingCoordinates_ts,CK.RotatingCoordinates_s)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.FamilyName_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.FamilyName_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.FamilyBC_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.FamilyBC_s])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.BCType_l
cgt[t].addChild(CK.BCDataSet_ts)

# --------------------------------------------------------
t=CK.GeometryReference_ts
cgt[t]=CGNStype(t)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.GeometryFile_ts,CK.GeometryFile_s)
cgt[t].addChild(CK.GeometryFormat_ts,CK.GeometryFormat_s)
cgt[t].addChild(CK.GeometryEntity_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GeometryFile_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GeometryFile_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.GeometryFormat_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GeometryFormat_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.GeometryEntity_ts
cgt[t]=CGNStype(t)

# --------------------------------------------------------
t=CK.CGNSTree_ts
cgt[t]=CGNStype(t,names=[CK.CGNSTree_s,UD])
cgt[t].addChild(CK.CGNSLibraryVersion_ts,[CK.CGNSLibraryVersion_s],card=C_11)
cgt[t].addChild(CK.CGNSBase_ts,card=C_0N)

# --------------------------------------------------------
t=CK.CGNSBase_ts
cgt[t]=CGNStype(t,dtype=[CK.I4])
cgt[t].shape=(0,0)
cgt[t].addChild(CK.Zone_ts,card=C_0N)
cgt[t].addChild(CK.SimulationType_ts,[CK.SimulationType_s],card=C_01)
cgt[t].addChild(CK.BaseIterativeData_ts,card=C_01)
cgt[t].addChild(CK.IntegralData_ts,card=C_0N)
cgt[t].addChild(CK.ConvergenceHistory_ts,[CK.GlobalConvergenceHistory_s],card=C_01)
cgt[t].addChild(CK.Family_ts,card=C_0N)
cgt[t].addChild(CK.FlowEquationSet_ts,[CK.FlowEquationSet_s],card=C_01)
cgt[t].addChild(CK.ReferenceState_ts,[CK.ReferenceState_s],card=C_01)
cgt[t].addChild(CK.Axisymmetry_ts,[CK.Axisymmetry_s],card=C_01)
cgt[t].addChild(CK.RotatingCoordinates_ts,[CK.RotatingCoordinates_s],card=C_01)
cgt[t].addChild(CK.Gravity_ts,[CK.Gravity_s],card=C_01)
cgt[t].addChild(CK.DataClass_ts,[CK.DataClass_s],card=C_01)
cgt[t].addChild(CK.DimensionalUnits_ts,[CK.DimensionalUnits_s],card=C_01)
cgt[t].addChild(CK.Descriptor_ts,card=C_0N)
cgt[t].addChild(CK.UserDefinedData_ts,card=C_0N)

# --------------------------------------------------------
t=CK.Zone_ts
cgt[t]=CGNStype(t,dtype=[CK.I4,CK.I8])
cgt[t].shape=(0,3)
cgt[t].addChild(CK.GridCoordinates_ts,card=C_0N)
cgt[t].addChild(CK.DiscreteData_ts,card=C_0N)
cgt[t].addChild(CK.Elements_ts,card=C_0N)
cgt[t].addChild(CK.ZoneBC_ts,CK.ZoneBC_s,card=C_01)
cgt[t].addChild(CK.FlowSolution_ts,card=C_0N)
cgt[t].addChild(CK.ZoneSubRegion_ts,card=C_0N)
cgt[t].addChild(CK.ZoneType_ts,CK.ZoneType_s,card=C_11)
cgt[t].addChild(CK.Ordinal_ts,CK.Ordinal_s,card=C_01)
cgt[t].addChild(CK.ZoneGridConnectivity_ts,CK.ZoneGridConnectivity_s,card=C_01)
cgt[t].addChild(CK.ZoneIterativeData_ts,card=C_01)
cgt[t].addChild(CK.RigidGridMotion_ts,card=C_0N)
cgt[t].addChild(CK.ReferenceState_ts,CK.ReferenceState_s,card=C_01)
cgt[t].addChild(CK.IntegralData_ts,card=C_0N)
cgt[t].addChild(CK.ArbitraryGridMotion_ts,card=C_0N)
cgt[t].addChild(CK.FamilyName_ts,CK.FamilyName_s,card=C_01)
cgt[t].addChild(CK.AdditionalFamilyName_ts,card=C_0N)
cgt[t].addChild(CK.FlowEquationSet_ts,CK.FlowEquationSet_s,card=C_01)
cgt[t].addChild(CK.ConvergenceHistory_ts,CK.ZoneConvergenceHistory_s,card=C_01)
cgt[t].addChild(CK.RotatingCoordinates_ts,CK.RotatingCoordinates_s,card=C_01)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s,card=C_01)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s,card=C_01)
cgt[t].addChild(CK.Descriptor_ts,card=C_0N)
cgt[t].addChild(CK.UserDefinedData_ts,card=C_0N)

# --------------------------------------------------------
t=CK.GridCoordinates_ts
cgt[t]=CGNStype(t,names=[CK.GridCoordinates_s,UD])
cgt[t].addChild(CK.DataArray_ts,card=C_0N)
cgt[t].addChild(CK.Rind_ts,CK.Rind_s,card=C_01)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s,card=C_01)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s,card=C_01)
cgt[t].addChild(CK.Descriptor_ts,card=C_0N)
cgt[t].addChild(CK.UserDefinedData_ts,card=C_0N)

# --------------------------------------------------------
t=CK.ZoneSubRegion_ts
cgt[t]=CGNStype(t,dtype=[CK.I4])
cgt[t].shape=(1,)
cgt[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
cgt[t].addChild(CK.IndexRange_ts,CK.PointRange_s,card=C_01)
cgt[t].addChild(CK.IndexArray_ts,CK.PointList_s,card=C_01)
cgt[t].addChild(CK.FamilyName_ts,CK.FamilyName_s,card=C_01)
cgt[t].addChild(CK.AdditionalFamilyName_ts,card=C_0N)
cgt[t].addChild(CK.DataArray_ts,card=C_0N)
cgt[t].addChild(CK.Rind_ts,CK.Rind_s,card=C_01)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s,card=C_01)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s,card=C_01)
cgt[t].addChild(CK.Descriptor_ts,card=C_0N)
cgt[t].addChild(CK.UserDefinedData_ts,[CK.BCRegionName_s,
                                       CK.GridConnectivityRegionName_s],
                card=C_0N)

# --------------------------------------------------------
t=CK.Elements_ts
cgt[t]=CGNStype(t,dtype=[CK.I4])
cgt[t].shape=(2,)
cgt[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
cgt[t].addChild(CK.IndexArray_ts,CK.PointList_s)
cgt[t].addChild(CK.DataArray_ts,CK.ElementConnectivity_s,card=C_0N)
cgt[t].addChild(CK.DataArray_ts,CK.ParentElements_s,card=C_01)
cgt[t].addChild(CK.DataArray_ts,CK.ParentElementsPosition_s,card=C_01)
cgt[t].addChild(CK.DataArray_ts,CK.ParentData_s,card=C_01)
cgt[t].addChild(CK.Rind_ts,CK.Rind_s,card=C_01)
cgt[t].addChild(CK.Descriptor_ts,card=C_0N)
cgt[t].addChild(CK.UserDefinedData_ts,card=C_0N)

# --------------------------------------------------------
t=CK.Axisymmetry_ts
cgt[t]=CGNStype(t,names=[CK.Axisymmetry_s])
cgt[t].addChild(CK.DataArray_ts,CK.AxisymmetryReferencePoint_s)
cgt[t].addChild(CK.DataArray_ts,CK.AxisymmetryAxisVector_s)
cgt[t].addChild(CK.DataArray_ts,CK.AxisymmetryAngle_s)
cgt[t].addChild(CK.DataArray_ts,CK.CoordinateNames_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.RotatingCoordinates_ts
cgt[t]=CGNStype(t,names=[CK.RotatingCoordinates_s])
cgt[t].addChild(CK.DataArray_ts,CK.RotationCenter_s)
cgt[t].addChild(CK.DataArray_ts,CK.RotationRateVector_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.FlowSolution_ts
cgt[t]=CGNStype(t)
cgt[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.Rind_ts,CK.Rind_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.DiscreteData_ts
cgt[t]=CGNStype(t)
cgt[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.Rind_ts,CK.Rind_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ZoneBC_ts
cgt[t]=CGNStype(t,names=[CK.ZoneBC_s])
cgt[t].addChild(CK.BC_ts)
cgt[t].addChild(CK.ReferenceState_ts,CK.ReferenceState_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.BCProperty_ts
cgt[t]=CGNStype(t,names=[CK.BCProperty_s])
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.WallFunction_ts,CK.WallFunction_s)
cgt[t].addChild(CK.Area_ts,CK.Area_s)

# --------------------------------------------------------
t=CK.BCData_ts
cgt[t]=CGNStype(t,names=[CK.DirichletData_s,CK.NeumannData_s])
cgt[t].addChild(CK.DataArray_ts)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.BCDataSet_ts
cgt[t]=CGNStype(t,dtype=[CK.C1])
cgt[t].enumerate=CK.BCTypeSimple_l
cgt[t].addChild(CK.BCData_ts,CK.NeumannData_s)
cgt[t].addChild(CK.BCData_ts,CK.DirichletData_s)
cgt[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
cgt[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
cgt[t].addChild(CK.IndexArray_ts,CK.PointList_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.ReferenceState_ts,CK.ReferenceState_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.BC_ts
cgt[t]=CGNStype(t,dtype=[CK.C1])
cgt[t].enumerate=CK.BCType_l
cgt[t].shape=(0,)
cgt[t].addChild(CK.ReferenceState_ts,CK.ReferenceState_s)
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)
cgt[t].addChild(CK.FamilyName_ts,CK.FamilyName_s)
cgt[t].addChild(CK.AdditionalFamilyName_ts,card=C_0N)
cgt[t].addChild(CK.IndexArray_ts,CK.InwardNormalList_s)
cgt[t].addChild(CK.BCDataSet_ts)
cgt[t].addChild(CK.InwardNormalIndex_ts,CK.InwardNormalIndex_s)
cgt[t].addChild(CK.IndexArray_ts,CK.ElementList_s)
cgt[t].addChild(CK.IndexArray_ts,CK.PointList_s)
cgt[t].addChild(CK.IndexRange_ts,CK.ElementRange_s)
cgt[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
cgt[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
cgt[t].addChild(CK.BCProperty_ts,CK.BCProperty_s)

# --------------------------------------------------------
t=CK.ArbitraryGridMotionType_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],
                  names=[CK.ArbitraryGridMotionType_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.RigidGridMotionType_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.RigidGridMotionType_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.WallFunctionType_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.WallFunctionType_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.WallFunction_ts
cgt[t]=CGNStype(t,names=[CK.WallFunction_s])
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.WallFunctionType_ts,CK.WallFunctionType_s)

# --------------------------------------------------------
t=CK.AreaType_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.AreaType_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.Area_ts
cgt[t]=CGNStype(t,names=[CK.Area_s])
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.AreaType_ts,CK.AreaType_s)
cgt[t].addChild(CK.DataArray_ts,CK.SurfaceArea_s)
cgt[t].addChild(CK.DataArray_ts,CK.RegionName_s)

# --------------------------------------------------------
t=CK.BaseIterativeData_ts
cgt[t]=CGNStype(t,dtype=[CK.I4])
cgt[t].shape=(1,)
cgt[t].addChild(CK.DataClass_ts,[CK.DataClass_s],card=C_01)
cgt[t].addChild(CK.DimensionalUnits_ts,[CK.DimensionalUnits_s],card=C_01)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.DataArray_ts)

# --------------------------------------------------------
t=CK.ZoneIterativeData_ts
cgt[t]=CGNStype(t)
cgt[t].addChild(CK.DataClass_ts,[CK.DataClass_s],card=C_01)
cgt[t].addChild(CK.DimensionalUnits_ts,[CK.DimensionalUnits_s],card=C_01)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.DataArray_ts,[CK.RigidGridMotionPointers_s,
                                 CK.ArbitraryGridMotionPointers_s,
                                 CK.FlowSolutionPointers_s,
                                 CK.ZoneGridConnectivityPointers_s,
                                 CK.ZoneSubRegionPointers_s])

# --------------------------------------------------------
t=CK.RigidGridMotion_ts
cgt[t]=CGNStype(t,dtype=[CK.C1])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.RigidGridMotionType_l
cgt[t].addChild(CK.DataClass_ts,[CK.DataClass_s],card=C_01)
cgt[t].addChild(CK.DimensionalUnits_ts,[CK.DimensionalUnits_s],card=C_01)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.DataArray_ts,[CK.OriginLocation_s,
                                 CK.RigidRotationAngle_s,
                                 CK.RigidRotationRate_s,
                                 CK.RigidVelocity_s])

# --------------------------------------------------------
t=CK.ArbitraryGridMotion_ts
cgt[t]=CGNStype(t,dtype=[CK.C1])
cgt[t].shape=(0,)
cgt[t].enumerate=CK.ArbitraryGridMotionType_l
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
cgt[t].addChild(CK.Rind_ts,CK.Rind_s)
cgt[t].addChild(CK.DataArray_ts)

# --------------------------------------------------------
t=CK.ZoneGridConnectivity_ts
cgt[t]=CGNStype(t,names=[CK.ZoneGridConnectivity_s])
cgt[t].addChild(CK.GridConnectivity1to1_ts)
cgt[t].addChild(CK.GridConnectivity_ts)
cgt[t].addChild(CK.OversetHoles_ts)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GridConnectivityProperty_ts
cgt[t]=CGNStype(t,names=[CK.GridConnectivityProperty_s])
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.Periodic_ts,CK.Periodic_s)
cgt[t].addChild(CK.AverageInterface_ts,CK.AverageInterface_s)

# --------------------------------------------------------
t=CK.Periodic_ts
cgt[t]=CGNStype(t,names=[CK.Periodic_s])
cgt[t].addChild(CK.DataClass_ts,CK.DataClass_s)
cgt[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.DataArray_ts,CK.RotationCenter_s)
cgt[t].addChild(CK.DataArray_ts,CK.RotationAngle_s)
cgt[t].addChild(CK.DataArray_ts,CK.Translation_s)

# --------------------------------------------------------
t=CK.AverageInterfaceType_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.AverageInterfaceType_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.AverageInterface_ts
cgt[t]=CGNStype(t,names=[CK.AverageInterface_s])
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)
cgt[t].addChild(CK.AverageInterfaceType_ts,CK.AverageInterfaceType_s)

# --------------------------------------------------------
t=CK.GridConnectivity1to1_ts
cgt[t]=CGNStype(t,dtype=[CK.C1])
cgt[t].shape=(0,)
cgt[t].addChild(CK.Transform_ts,CK.Transform_s)
cgt[t].addChild(CK.IntIndexDimension_ts,CK.Transform_s)
cgt[t].addChild(CK.Transform_ts2,CK.Transform_s)
cgt[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
cgt[t].addChild(CK.IndexRange_ts,CK.PointRangeDonor_s)
cgt[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)
cgt[t].addChild(CK.GridConnectivityProperty_ts,CK.GridConnectivityProperty_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GridConnectivityType_ts
cgt[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GridConnectivityType_s])
cgt[t].shape=(0,)

# --------------------------------------------------------
t=CK.GridConnectivity_ts
cgt[t]=CGNStype(t,dtype=[CK.C1])
cgt[t].shape=(0,)
cgt[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
cgt[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
cgt[t].addChild(CK.IndexArray_ts,CK.PointList_s)
cgt[t].addChild(CK.IndexArray_ts,CK.PointListDonor_s)
cgt[t].addChild(CK.IndexArray_ts,CK.CellListDonor_s)
cgt[t].addChild(CK.GridConnectivityProperty_ts,CK.GridConnectivityProperty_s)
cgt[t].addChild(CK.GridConnectivityType_ts,CK.GridConnectivityType_s)
cgt[t].addChild(CK.DataArray_ts,CK.InterpolantsDonor_s)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.OversetHoles_ts
cgt[t]=CGNStype(t)
cgt[t].addChild(CK.Descriptor_ts)
cgt[t].addChild(CK.IndexArray_ts,CK.PointList_s)
cgt[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
cgt[t].addChild(CK.IndexRange_ts)
cgt[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
types=cgt
tk=types.keys()
tk.sort()
for pk in tk:
  for ck in tk:
    if ((ck!=pk) and (types[pk].hasChild(ck))):
        types[ck].addParent(pk)
  
# --- last line
