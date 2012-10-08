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
    self.children.append((ctype,cname,dtype,card))
  def addParent(self,parent):
    self.parents.append(parent)
  def cardinality(self,childtype):
    for c in self.children:
      if (c[0]==childtype): return c[3]
    return C_00
types={}

# --------------------------------------------------------
t=CK.CGNSLibraryVersion_ts
types[t]=CGNStype(t,dtype=[CK.R4],names=[CK.CGNSLibraryVersion_s])
types[t].shape=(1,)

# --------------------------------------------------------
t=CK.Descriptor_ts
types[t]=CGNStype(t,dtype=[CK.C1])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.Ordinal_ts
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.Ordinal_s])
types[t].shape=(1,)

# --------------------------------------------------------
t=CK.DataClass_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.DataClass_s])
types[t].shape=(0,)
types[t].enumerate=CK.DataClass_l

# --------------------------------------------------------
t=CK.DimensionalUnits_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.DimensionalUnits_s])
types[t].shape=(32,5)
types[t].enumerate=CK.AllDimensionalUnits_l
types[t].addChild(CK.AdditionalUnits_ts,CK.AdditionalUnits_s)

# --------------------------------------------------------
t=CK.AdditionalUnits_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.AdditionalUnits_s])
types[t].shape=(32,3)
types[t].enumerate=CK.AllAdditionalUnits_l

# --------------------------------------------------------
t=CK.DataConversion_ts
types[t]=CGNStype(t,dtype=[CK.R4,CK.R8],names=[CK.DataConversion_s])
types[t].shape=(2,)

# --------------------------------------------------------
t=CK.DimensionalExponents_ts
types[t]=CGNStype(t,dtype=[CK.R4,CK.R8],names=[CK.DimensionalExponents_s])
types[t].shape=(5,)

# --------------------------------------------------------
t=CK.AdditionalExponents_ts
types[t]=CGNStype(t,dtype=[CK.R4,CK.R8],names=[CK.AdditionalExponents_s])
types[t].shape=(3,)

# --------------------------------------------------------
t=CK.DataArray_ts
types[t]=CGNStype(t,dtype=allDT)
types[t].addChild(CK.DimensionalExponents_ts,CK.DimensionalExponents_s)
types[t].addChild(CK.DataConversion_ts,CK.DataConversion_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)

# --------------------------------------------------------
t=CK.Transform_ts
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.Transform_s])
types[t].shape=(0,)
t=CK.Transform_ts2
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.Transform_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.DiffusionModel_ts
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.DiffusionModel_s])
types[t].shape=(0,)
t=CK.DiffusionModel_ts2
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.DiffusionModel_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.InwardNormalIndex_ts
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.InwardNormalIndex_s])
types[t].shape=(0,)
t=CK.InwardNormalIndex_ts2
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.InwardNormalIndex_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.EquationDimension_ts
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.EquationDimension_s])
types[t].shape=(1,)
t=CK.EquationDimension_ts2
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.EquationDimension_s])
types[t].shape=(1,)

# --------------------------------------------------------
t=CK.GridLocation_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GridLocation_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.Rind_ts
types[t]=CGNStype(t,dtype=[CK.I4],names=[CK.Rind_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.IndexRange_ts
types[t]=CGNStype(t,dtype=[CK.I4])
types[t].shape=(0,2)
types[t].names=[CK.PointRange_s,CK.PointRangeDonor_s,CK.ElementRange_s,UD]

# --------------------------------------------------------
t=CK.IndexArray_ts
types[t]=CGNStype(t,dtype=[CK.I4,CK.R4,CK.R8])
types[t].shape=(0,0)
types[t].names=[CK.PointList_s,CK.PointListDonor_s,CK.CellListDonor_s,
                CK.InwardNormalList_s,UD]

# --------------------------------------------------------
t=CK.ReferenceState_ts
types[t]=CGNStype(t,names=[CK.ReferenceState_s])
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.Descriptor_ts,CK.ReferenceStateDescription_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ConvergenceHistory_ts
types[t]=CGNStype(t,names=[CK.GlobalConvergenceHistory_s,
                           CK.ZoneConvergenceHistory_s],dtype=[CK.I4])
types[t].shape=(1,)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.Descriptor_ts,CK.NormDefinitions_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.IntegralData_ts
types[t]=CGNStype(t)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.UserDefinedData_ts
types[t]=CGNStype(t)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
types[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
types[t].addChild(CK.IndexArray_ts,CK.PointList_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.FamilyName_ts,CK.FamilyName_s)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)

# --------------------------------------------------------
t=CK.Gravity_ts
types[t]=CGNStype(t)
types[t].addChild(CK.DataArray_ts,CK.GravityVector_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.FlowEquationSet_ts
types[t]=CGNStype(t,names=[CK.FlowEquationSet_s])
types[t].addChild(CK.GoverningEquations_ts,CK.GoverningEquations_s)
types[t].addChild(CK.EquationDimension_ts,CK.EquationDimension_s)
types[t].addChild(CK.GasModel_ts,CK.GasModel_s)
types[t].addChild(CK.ViscosityModel_ts,CK.ViscosityModel_s)
types[t].addChild(CK.ThermalRelaxationModel_ts,CK.ThermalRelaxationModel_s)
types[t].addChild(CK.ThermalConductivityModel_ts,CK.ThermalConductivityModel_s)
types[t].addChild(CK.TurbulenceModel_ts,CK.TurbulenceModel_s)
types[t].addChild(CK.TurbulenceClosure_ts,CK.TurbulenceClosure_s)
types[t].addChild(CK.ChemicalKineticsModel_ts,CK.ChemicalKineticsModel_s)
types[t].addChild(CK.EMMagneticFieldModel_ts,CK.EMMagneticFieldModel_s)
types[t].addChild(CK.EMElectricFieldModel_ts,CK.EMElectricFieldModel_s)
types[t].addChild(CK.EMConductivityModel_ts,CK.EMConductivityModel_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GoverningEquations_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GoverningEquations_s])
types[t].shape=(0,)
types[t].enumerate=CK.GoverningEquationsType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DiffusionModel_ts,CK.DiffusionModel_s)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GasModel_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GasModel_s])
types[t].shape=(0,)
types[t].enumerate=CK.GasModelType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ViscosityModel_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ViscosityModel_s])
types[t].shape=(0,)
types[t].enumerate=CK.ViscosityModelType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ThermalConductivityModel_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ThermalConductivityModel_s])
types[t].shape=(0,)
types[t].enumerate=CK.ThermalConductivityModelType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.TurbulenceClosure_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.TurbulenceClosure_s])
types[t].shape=(0,)
types[t].enumerate=CK.TurbulenceClosureType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.TurbulenceModel_ts
types[t]=CGNStype(t)
types[t].datatype=[CK.C1]
types[t].cardinality=C_01
types[t].shape=(0,)
types[t].enumerate=CK.TurbulenceModelType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DiffusionModel_ts,CK.DiffusionModel_s)

# --------------------------------------------------------
t=CK.ThermalRelaxationModel_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ThermalRelaxationModel_s])
types[t].shape=(0,)
types[t].enumerate=CK.ThermalRelaxationModelType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ChemicalKineticsModel_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ChemicalKineticsModel_s])
types[t].shape=(0,)
types[t].enumerate=CK.ChemicalKineticsModelType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.EMElectricFieldModel_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.EMElectricFieldModel_s])
types[t].shape=(0,)
types[t].enumerate=CK.EMElectricFieldModelType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.EMMagneticFieldModel_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.EMMagneticFieldModel_s])
types[t].shape=(0,)
types[t].enumerate=CK.EMMagneticFieldModelType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.EMConductivityModel_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.EMConductivityModel_s])
types[t].shape=(0,)
types[t].enumerate=CK.EMConductivityModelType_l
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ZoneType_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.ZoneType_s])
types[t].shape=(0,)
types[t].enumerate=CK.ZoneType_l

# --------------------------------------------------------
t=CK.SimulationType_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.SimulationType_s])
types[t].shape=(0,)
types[t].enumerate=CK.SimulationType_l

# --------------------------------------------------------
t=CK.GridConnectivityType_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GridConnectivityType_s])
types[t].shape=(0,)
types[t].enumerate=CK.GridConnectivityType_l

# --------------------------------------------------------
t=CK.Family_ts
types[t]=CGNStype(t)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)
types[t].addChild(CK.FamilyBC_ts)
types[t].addChild(CK.GeometryReference_ts)
types[t].addChild(CK.RotatingCoordinates_ts,CK.RotatingCoordinates_s)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.FamilyName_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.FamilyName_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.FamilyBC_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.FamilyBC_s])
types[t].shape=(0,)
types[t].enumerate=CK.BCType_l
types[t].addChild(CK.BCDataSet_ts)

# --------------------------------------------------------
t=CK.GeometryReference_ts
types[t]=CGNStype(t)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.GeometryFile_ts,CK.GeometryFile_s)
types[t].addChild(CK.GeometryFormat_ts,CK.GeometryFormat_s)
types[t].addChild(CK.GeometryEntity_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GeometryFile_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GeometryFile_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.GeometryFormat_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GeometryFormat_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.GeometryEntity_ts
types[t]=CGNStype(t)

# --------------------------------------------------------
t=CK.CGNSTree_ts
types[t]=CGNStype(t,names=[CK.CGNSTree_s,UD])
types[t].addChild(CK.CGNSLibraryVersion_ts,CK.CGNSLibraryVersion_s,card=C_11)
types[t].addChild(CK.CGNSBase_ts,card=C_0N)

# --------------------------------------------------------
t=CK.CGNSBase_ts
types[t]=CGNStype(t,dtype=[CK.I4])
types[t].shape=(0,0)
types[t].addChild(CK.Zone_ts,card=C_0N)
types[t].addChild(CK.SimulationType_ts,CK.SimulationType_s)
types[t].addChild(CK.BaseIterativeData_ts)
types[t].addChild(CK.IntegralData_ts)
types[t].addChild(CK.ConvergenceHistory_ts,CK.GlobalConvergenceHistory_s)
types[t].addChild(CK.Family_ts)
types[t].addChild(CK.FlowEquationSet_ts,CK.FlowEquationSet_s)
types[t].addChild(CK.ReferenceState_ts,CK.ReferenceState_s)
types[t].addChild(CK.Axisymmetry_ts,CK.Axisymmetry_s)
types[t].addChild(CK.RotatingCoordinates_ts,CK.RotatingCoordinates_s)
types[t].addChild(CK.Gravity_ts,CK.Gravity_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.Zone_ts
types[t]=CGNStype(t,dtype=[CK.I4])
types[t].shape=(0,3)
types[t].addChild(CK.GridCoordinates_ts)
types[t].addChild(CK.DiscreteData_ts)
types[t].addChild(CK.Elements_ts)
types[t].addChild(CK.ZoneBC_ts,CK.ZoneBC_s,card=C_01)
types[t].addChild(CK.FlowSolution_ts)
types[t].addChild(CK.ZoneType_ts,CK.ZoneType_s,card=C_11)
types[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)
types[t].addChild(CK.ZoneGridConnectivity_ts,CK.ZoneGridConnectivity_s)
types[t].addChild(CK.ZoneIterativeData_ts)
types[t].addChild(CK.RigidGridMotion_ts)
types[t].addChild(CK.ReferenceState_ts,CK.ReferenceState_s)
types[t].addChild(CK.IntegralData_ts)
types[t].addChild(CK.ArbitraryGridMotion_ts)
types[t].addChild(CK.FamilyName_ts,CK.FamilyName_s,card=C_01)
types[t].addChild(CK.FlowEquationSet_ts,CK.FlowEquationSet_s)
types[t].addChild(CK.ConvergenceHistory_ts,CK.ZoneConvergenceHistory_s)
types[t].addChild(CK.RotatingCoordinates_ts,CK.RotatingCoordinates_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GridCoordinates_ts
types[t]=CGNStype(t,names=[CK.GridCoordinates_s,UD])
types[t].addChild(CK.DataArray_ts,card=C_0N)
types[t].addChild(CK.Rind_ts,CK.Rind_s,card=C_01)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s,card=C_01)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s,card=C_01)
types[t].addChild(CK.Descriptor_ts,card=C_0N)
types[t].addChild(CK.UserDefinedData_ts,card=C_0N)

# --------------------------------------------------------
t=CK.Elements_ts
types[t]=CGNStype(t,dtype=[CK.I4])
types[t].shape=(2,)
types[t].addChild(CK.IndexRange_ts,CK.ElementRange_s,card=C_11)
types[t].addChild(CK.DataArray_ts,CK.ElementConnectivity_s)
types[t].addChild(CK.DataArray_ts,CK.ParentData_s)
types[t].addChild(CK.Rind_ts,CK.Rind_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.Axisymmetry_ts
types[t]=CGNStype(t,names=[CK.Axisymmetry_s])
types[t].addChild(CK.DataArray_ts,CK.AxisymmetryReferencePoint_s)
types[t].addChild(CK.DataArray_ts,CK.AxisymmetryAxisVector_s)
types[t].addChild(CK.DataArray_ts,CK.AxisymmetryAngle_s)
types[t].addChild(CK.DataArray_ts,CK.CoordinateNames_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.RotatingCoordinates_ts
types[t]=CGNStype(t,names=[CK.RotatingCoordinates_s])
types[t].addChild(CK.DataArray_ts,CK.RotationCenter_s)
types[t].addChild(CK.DataArray_ts,CK.RotationRateVector_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.FlowSolution_ts
types[t]=CGNStype(t)
types[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.Rind_ts,CK.Rind_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.DiscreteData_ts
types[t]=CGNStype(t)
types[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.Rind_ts,CK.Rind_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.ZoneBC_ts
types[t]=CGNStype(t,names=[CK.ZoneBC_s])
types[t].addChild(CK.BC_ts)
types[t].addChild(CK.ReferenceState_ts,CK.ReferenceState_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.BCProperty_ts
types[t]=CGNStype(t,names=[CK.BCProperty_s])
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.WallFunction_ts,CK.WallFunction_s)
types[t].addChild(CK.Area_ts,CK.Area_s)

# --------------------------------------------------------
t=CK.BCData_ts
types[t]=CGNStype(t,names=[CK.DirichletData_s,CK.NeumannData_s])
types[t].addChild(CK.DataArray_ts)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.BCDataSet_ts
types[t]=CGNStype(t,dtype=[CK.C1])
types[t].enumerate=CK.BCTypeSimple_l
types[t].addChild(CK.BCData_ts,CK.NeumannData_s)
types[t].addChild(CK.BCData_ts,CK.DirichletData_s)
types[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
types[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
types[t].addChild(CK.IndexArray_ts,CK.PointList_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.ReferenceState_ts,CK.ReferenceState_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.BC_ts
types[t]=CGNStype(t,dtype=[CK.C1])
types[t].enumerate=CK.BCType_l
types[t].shape=(0,)
types[t].addChild(CK.ReferenceState_ts,CK.ReferenceState_s)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)
types[t].addChild(CK.FamilyName_ts,CK.FamilyName_s)
types[t].addChild(CK.IndexArray_ts,CK.InwardNormalList_s)
types[t].addChild(CK.BCDataSet_ts)
types[t].addChild(CK.InwardNormalIndex_ts,CK.InwardNormalIndex_s)
types[t].addChild(CK.IndexArray_ts,CK.ElementList_s)
types[t].addChild(CK.IndexArray_ts,CK.PointList_s)
types[t].addChild(CK.IndexRange_ts,CK.ElementRange_s)
types[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
types[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
types[t].addChild(CK.BCProperty_ts,CK.BCProperty_s)

# --------------------------------------------------------
t=CK.ArbitraryGridMotionType_ts
types[t]=CGNStype(t,dtype=[CK.C1],
                  names=[CK.ArbitraryGridMotionType_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.RigidGridMotionType_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.RigidGridMotionType_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.WallFunctionType_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.WallFunctionType_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.WallFunction_ts
types[t]=CGNStype(t,names=[CK.WallFunction_s])
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.WallFunctionType_ts,CK.WallFunctionType_s)

# --------------------------------------------------------
t=CK.AreaType_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.AreaType_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.Area_ts
types[t]=CGNStype(t,names=[CK.Area_s])
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.AreaType_ts,CK.AreaType_s)
types[t].addChild(CK.DataArray_ts,CK.SurfaceArea_s)
types[t].addChild(CK.DataArray_ts,CK.RegionName_s)

# --------------------------------------------------------
t=CK.BaseIterativeData_ts
types[t]=CGNStype(t,dtype=[CK.I4])
types[t].shape=(1,)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.DataArray_ts)

# --------------------------------------------------------
t=CK.ZoneIterativeData_ts
types[t]=CGNStype(t)
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.DataArray_ts)

# --------------------------------------------------------
t=CK.RigidGridMotion_ts
types[t]=CGNStype(t,dtype=[CK.C1])
types[t].shape=(0,)
types[t].enumerate=CK.RigidGridMotionType_l
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.DataArray_ts)

# --------------------------------------------------------
t=CK.ArbitraryGridMotion_ts
types[t]=CGNStype(t,dtype=[CK.C1])
types[t].shape=(0,)
types[t].enumerate=CK.ArbitraryGridMotionType_l
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
types[t].addChild(CK.Rind_ts,CK.Rind_s)
types[t].addChild(CK.DataArray_ts)

# --------------------------------------------------------
t=CK.ZoneGridConnectivity_ts
types[t]=CGNStype(t,names=[CK.ZoneGridConnectivity_s])
types[t].addChild(CK.GridConnectivity1to1_ts)
types[t].addChild(CK.GridConnectivity_ts)
types[t].addChild(CK.OversetHoles_ts)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GridConnectivityProperty_ts
types[t]=CGNStype(t,names=[CK.GridConnectivityProperty_s])
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.Periodic_ts,CK.Periodic_s)
types[t].addChild(CK.AverageInterface_ts,CK.AverageInterface_s)

# --------------------------------------------------------
t=CK.Periodic_ts
types[t]=CGNStype(t,names=[CK.Periodic_s])
types[t].addChild(CK.DataClass_ts,CK.DataClass_s)
types[t].addChild(CK.DimensionalUnits_ts,CK.DimensionalUnits_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.DataArray_ts,CK.RotationCenter_s)
types[t].addChild(CK.DataArray_ts,CK.RotationAngle_s)
types[t].addChild(CK.DataArray_ts,CK.Translation_s)

# --------------------------------------------------------
t=CK.AverageInterfaceType_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.AverageInterfaceType_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.AverageInterface_ts
types[t]=CGNStype(t,names=[CK.AverageInterface_s])
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)
types[t].addChild(CK.AverageInterfaceType_ts,CK.AverageInterfaceType_s)

# --------------------------------------------------------
t=CK.GridConnectivity1to1_ts
types[t]=CGNStype(t,dtype=[CK.C1])
types[t].shape=(0,)
types[t].addChild(CK.Transform_ts,CK.Transform_s)
types[t].addChild(CK.IntIndexDimension_ts,CK.Transform_s)
types[t].addChild(CK.Transform_ts2,CK.Transform_s)
types[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
types[t].addChild(CK.IndexRange_ts,CK.PointRangeDonor_s)
types[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)
types[t].addChild(CK.GridConnectivityProperty_ts,CK.GridConnectivityProperty_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
t=CK.GridConnectivityType_ts
types[t]=CGNStype(t,dtype=[CK.C1],names=[CK.GridConnectivityType_s])
types[t].shape=(0,)

# --------------------------------------------------------
t=CK.GridConnectivity_ts
types[t]=CGNStype(t,dtype=[CK.C1])
types[t].shape=(0,)
types[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
types[t].addChild(CK.Ordinal_ts,CK.Ordinal_s)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.IndexRange_ts,CK.PointRange_s)
types[t].addChild(CK.IndexArray_ts,CK.PointList_s)
types[t].addChild(CK.IndexArray_ts,CK.PointListDonor_s)
types[t].addChild(CK.IndexArray_ts,CK.CellListDonor_s)
types[t].addChild(CK.GridConnectivityProperty_ts,CK.GridConnectivityProperty_s)
types[t].addChild(CK.GridConnectivityType_ts,CK.GridConnectivityType_s)
types[t].addChild(CK.DataArray_ts,CK.InterpolantsDonor_s)

# --------------------------------------------------------
t=CK.OversetHoles_ts
types[t]=CGNStype(t)
types[t].addChild(CK.Descriptor_ts)
types[t].addChild(CK.IndexArray_ts,CK.PointList_s)
types[t].addChild(CK.GridLocation_ts,CK.GridLocation_s)
types[t].addChild(CK.IndexRange_ts)
types[t].addChild(CK.UserDefinedData_ts)

# --------------------------------------------------------
tk=types.keys()
tk.sort()
for pk in tk:
  for ck in tk:
    if ((ck!=pk) and (types[pk].hasChild(ck))):
        types[ck].addParent(pk)
  
# --- last line
