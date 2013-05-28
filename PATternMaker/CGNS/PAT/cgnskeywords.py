#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#  
"""
 TYPES, ENUMERATES, CONSTANTS, NAMES from CGNS/MLL

 Conventions:

 [1] A CGNS/SIDS string constant is postfixed with _s
 'ZoneType' is ZoneType_s

 [2] A CGNS/SIDS string constant naming a type has _ts
 'ZoneType_t' is ZoneType_ts

 [3] A list of possible values for a given type has _l
 ZoneType_l is [Null_s,UserDefined_s,Structured_s,Unstructured_s]
 which is same as ["Null","UserDefined","Structured","Unstructured"]
 List should be ordered wrt the actual enumerate

 [4] An enumerate mapping of a list of values is not prefixed
 ZoneType is {'Unstructured':3,'Null':0,'Structured':2,'UserDefined':1}

 [5] The reverse dictionnary of the previous one is postfixed with _
 ZoneType_ is {0:'Null',1:'UserDefined',2:'Structured',3:'Unstructured'}

 [6] The variables are declared with an integer value (not enumerates)
 wrt their position in the _l list, for example:
 (Null,UserDefined,Structured,Unstructured)=ZoneType_.keys()

 [7] The _t type names are reserved for Cython, enums are then used as int:
 ctypedef int DataType_t
 int cg_array_read_as(int A, DataType_t type, void *Data)
 
"""

# ----------------------------------------------------------------------------
import CGNS.pyCGNSconfig

def stringAsKeyDict(l):
  return dict(zip(l,range(len(l))))

def enumAsKeyDict(l):
  return dict(zip(range(len(l)),l))

# --------------------------------------------------
# --- ADF-level Datatypes
#
adftypes=('C1','I4','I8','R4','R8','MT','LK')
(C1,I4,I8,R4,R8,MT,LK)=adftypes

# --------------------------------------------------
# --- ADF-level Constants
#
ADF_DATA_TYPE_LENGTH     =   32
ADF_DATE_LENGTH          =   32
ADF_FILENAME_LENGTH      = 1024
ADF_FORMAT_LENGTH        =   20
ADF_LABEL_LENGTH         =   32
ADF_MAXIMUM_LINK_DEPTH   =  100
ADF_MAX_DIMENSIONS       =   12
ADF_MAX_ERROR_STR_LENGTH =   80
ADF_MAX_LINK_DATA_SIZE   = 4096
ADF_NAME_LENGTH          =   32
ADF_STATUS_LENGTH        =   32
ADF_VERSION_LENGTH       =   32

ADF_ROOT_NODE_NAME  = "HDF5 MotherNode"
ADF_ROOT_NODE_LABEL = "Root Node of HDF5 File"

CGNSHDF5ROOT_s = ADF_ROOT_NODE_NAME

# -------------------------------------------------- (NOT SIDS)
# --- CGNS/Python mapping extensions
#
CGNSTree_ts           = 'CGNSTree_t'
CGNSTree_s            = 'CGNSTree'

# --- Type with weird (coming from outer space) names
#
Transform_ts          = 'Transform_ts'
DiffusionModel_ts     = 'DiffusionModel_t'
EquationDimension_ts  = 'EquationDimension_t'
InwardNormalIndex_ts  = 'InwardNormalIndex_t'
IntIndexDimension_ts  = 'IntIndexDimension_ts'

# --- Add legacy strings for translation tools
#
Transform_ts2         = '"int[IndexDimension]"'
DiffusionModel_ts2    = '"int[1+...+IndexDimension]"'
EquationDimension_ts2 = '"int"'
InwardNormalIndex_ts2 = '"int[IndexDimension]"'

weirdSIDStypes={
  Transform_ts2:         IntIndexDimension_ts,
  DiffusionModel_ts2:    DiffusionModel_ts,
  EquationDimension_ts2: EquationDimension_ts,
  InwardNormalIndex_ts2: IntIndexDimension_ts,
  }

weirdSIDStypes_={
  Transform_ts:         Transform_ts2,
  DiffusionModel_ts:    DiffusionModel_ts2,
  EquationDimension_ts: EquationDimension_ts2,
  InwardNormalIndex_ts: InwardNormalIndex_ts2,
  }

# -------------------------------------------------- (SIDS)
# SIDS
#
Null_s        = "Null"
UserDefined_s = "UserDefined"

# --------------------------------------------------
Kilogram_s  = "Kilogram"
Gram_s      = "Gram"
Slug_s      = "Slug"
PoundMass_s = "PoundMass"
MassUnits_l = [Null_s,UserDefined_s,
               Kilogram_s,Gram_s,Slug_s,PoundMass_s]

MassUnits   = stringAsKeyDict(MassUnits_l)
MassUnits_  = enumAsKeyDict(MassUnits_l)
(MassUnitsNull,MassUnitsUserDefined,
 Kilogram,Gram,Slug,PoundMass)=MassUnits_.keys()

# --------------------------------------------------
Meter_s       = "Meter"
Centimeter_s  = "Centimeter"
Millimeter_s  = "Millimeter"
Foot_s        = "Foot"
Inch_s        = "Inch"
LengthUnits_l = [Null_s,UserDefined_s,
                 Meter_s,Centimeter_s,Millimeter_s,Foot_s,Inch_s]

LengthUnits   = stringAsKeyDict(LengthUnits_l)
LengthUnits_  = enumAsKeyDict(LengthUnits_l)
(LengthUnitsNull,LengthUnitsUserDefined,
 Meter,Centimeter,Millimeter,Foot,Inch)=LengthUnits_.keys()

# --------------------------------------------------
Second_s    = "Second"
TimeUnits_l = [Null_s,UserDefined_s,Second_s]

TimeUnits   = stringAsKeyDict(TimeUnits_l)
TimeUnits_  = enumAsKeyDict(TimeUnits_l)
(TimeUnitsNull,TimeUnitsUserDefined,Second)=TimeUnits_.keys()

# --------------------------------------------------
Kelvin_s           = "Kelvin"
Celcius_s          = "Celcius"
Rankine_s          = "Rankine"
Fahrenheit_s       = "Fahrenheit"
TemperatureUnits_l = [Null_s,UserDefined_s,
                      Kelvin_s,Celcius_s,Rankine_s,Fahrenheit_s]

TemperatureUnits   = stringAsKeyDict(TemperatureUnits_l)
TemperatureUnits_  = enumAsKeyDict(TemperatureUnits_l)
(TemperatureUnitsNull,TemperatureUnitsUserDefined,
 Kelvin,Celcius,Rankine,Fahrenheit)=TemperatureUnits_.keys()

# --------------------------------------------------
Degree_s     = "Degree"
Radian_s     = "Radian"
AngleUnits_l = [Null_s,UserDefined_s,Degree_s,Radian_s]

AngleUnits   = stringAsKeyDict(AngleUnits_l)
AngleUnits_  = enumAsKeyDict(AngleUnits_l)
(AngleUnitsNull,AngleUnitsUserDefined,Degree,Radian)=AngleUnits_.keys()

# --------------------------------------------------
Ampere_s               = "Ampere"
Abampere_s             = "Abampere"
Statampere_s           = "Statampere"
Edison_s               = "Edison"
auCurrent_s            = "auCurrent"
ElectricCurrentUnits_l = [Null_s,UserDefined_s,
                          Ampere_s,Abampere_s,Statampere_s,
                          Edison_s,auCurrent_s]

ElectricCurrentUnits   = stringAsKeyDict(ElectricCurrentUnits_l)
ElectricCurrentUnits_  = enumAsKeyDict(ElectricCurrentUnits_l)
(ElectricCurrentUnitsNull,ElectricCurrentUnitsUserDefined,
 Ampere,Abampere,Statampere,
 Edison,auCurrent)=ElectricCurrentUnits_.keys()

# --------------------------------------------------
Mole_s                 = "Mole"
Entities_s             = "Entities"
StandardCubicFoot_s    = "StandardCubicFoot"
StandardCubicMeter_s   = "StandardCubicMeter"
SubstanceAmountUnits_l =[Null_s,UserDefined_s,
                         Mole_s,Entities_s,
                         StandardCubicFoot_s,StandardCubicMeter_s]

SubstanceAmountUnits   = stringAsKeyDict(SubstanceAmountUnits_l)
SubstanceAmountUnits_  = enumAsKeyDict(SubstanceAmountUnits_l)
(SubstanceAmountUnitsNull,SubstanceAmountUnitsUserDefined,
 Mole,Entities,
 StandardCubicFoot,StandardCubicMeter)=SubstanceAmountUnits_.keys()

# --------------------------------------------------
Candela_s = "Candela"
Candle_s  = "Candle"
Carcel_s  = "Carcel"
Hefner_s  = "Hefner"
Violle_s  = "Violle"     
LuminousIntensityUnits_l = [Null_s,UserDefined_s,
                            Candela_s,Candle_s,Carcel_s,Hefner_s,Violle_s]

LuminousIntensityUnits   = stringAsKeyDict(LuminousIntensityUnits_l)
LuminousIntensityUnits_  = enumAsKeyDict(LuminousIntensityUnits_l)
(LuminousIntensityUnitsNull,LuminousIntensityUnitsUserDefined,
 Candela,Candle,Carcel,Hefner,Violle)=LuminousIntensityUnits_.keys()


# --------------------------------------------------
DimensionalUnits_s    = "DimensionalUnits"
AdditionalUnits_s     = "AdditionalUnits"
AdditionalExponents_s = "AdditionalExponents"

AllDimensionalUnits_l = TimeUnits_l+MassUnits_l+LengthUnits_l\
                        +TemperatureUnits_l+AngleUnits_l
AllAdditionalUnits_l  = LuminousIntensityUnits_l+SubstanceAmountUnits_l\
                        +ElectricCurrentUnits_l
AllUnits_l            = AllDimensionalUnits_l+AllAdditionalUnits_l

# --------------------------------------------------
DataClass_ts = "DataClass_t"
DataClass_s  = "DataClass"

Dimensional_s                    = "Dimensional"
NormalizedByDimensional_s        = "NormalizedByDimensional"
NormalizedByUnknownDimensional_s = "NormalizedByUnknownDimensional"
NondimensionalParameter_s        = "NondimensionalParameter"
DimensionlessConstant_s          = "DimensionlessConstant"

DataClass_l=[Null_s,UserDefined_s,Dimensional_s,NormalizedByDimensional_s,
             NormalizedByUnknownDimensional_s,NondimensionalParameter_s,
             DimensionlessConstant_s]
DataClass  = stringAsKeyDict(DataClass_l)
DataClass_ = enumAsKeyDict(DataClass_l)
(Null,UserDefined,Dimensional,NormalizedByDimensional,
 NormalizedByUnknownDimensional,NondimensionalParameter,
 DimensionlessConstant)=DataClass_.keys()

# ------------------------------------------------------------
GridLocation_ts= "GridLocation_t"
GridLocation_s = "GridLocation"

Vertex_s       = "Vertex"
Edge_s         = "Edge"
Face_s         = "Face"
Cell_s         = "Cell"
CellCenter_s   = "CellCenter"
FaceCenter_s   = "FaceCenter"
IFaceCenter_s  = "IFaceCenter"
JFaceCenter_s  = "JFaceCenter"
KFaceCenter_s  = "KFaceCenter"
EdgeCenter_s   = "EdgeCenter"

GridLocation_l = [Null_s,UserDefined_s,Vertex_s,CellCenter_s,FaceCenter_s,
                  IFaceCenter_s,JFaceCenter_s,KFaceCenter_s,
                  EdgeCenter_s]
GridLocation   = stringAsKeyDict(GridLocation_l)
GridLocation_  = enumAsKeyDict(GridLocation_l)
(Null,UserDefined,Vertex,CellCenter,FaceCenter,
 IFaceCenter,JFaceCenter,KFaceCenter,EdgeCenter)=GridLocation_.keys()

VertexSize_s         = "VertexSize"
CellSize_s           = "CellSize"
VertexSizeBoundary_s = "VertexSizeBoundary"

# ------------------------------------------------------------
PointSetType_ts = "PointSetType_t"

PointList_s       = "PointList"
PointListDonor_s  = "PointListDonor"
PointRange_s      = "PointRange"
PointRangeDonor_s = "PointRangeDonor"
ElementRange_s    = "ElementRange"
ElementList_s     = "ElementList"
CellListDonor_s   = "CellListDonor"

PointSetType_l = [Null_s,UserDefined_s,
                  PointList_s,PointListDonor_s,PointRange_s,PointRangeDonor_s,
                  ElementRange_s,ElementList_s,CellListDonor_s]
PointSetType  = stringAsKeyDict(PointSetType_l)
PointSetType_ = enumAsKeyDict(PointSetType_l)
(Null,UserDefined,PointList,PointListDonor,PointRange,PointRangeDonor,
 ElementRange,ElementList,CellListDonor)=PointSetType_.keys()

# ------------------------------------------------------------
BCDataType_ts = "BCDataType_t"
BCDataType_s  = "BCDataType"

DirichletData_s = "DirichletData"
NeumannData_s   = "NeumannData"
Dirichlet_s     = "Dirichlet"
Neumann_s       = "Neumann"

BCDataType_l=[Null_s,UserDefined_s,Dirichlet_s,Neumann_s]
BCDataType   = stringAsKeyDict(BCDataType_l)
BCDataType_  = enumAsKeyDict(BCDataType_l)
(BCDataTypeNull,BCDataTypeUserDefined,Dirichlet,Neumann)=BCDataType_.keys()

FullPotential_s               = "FullPotential"
Euler_s                       = "Euler"
NSLaminar_s                   = "NSLaminar"
NSTurbulent_s                 = "NSTurbulent"
NSLaminarIncompressible_s     = "NSLaminarIncompressible"
NSTurbulentIncompressible_s   = "NSTurbulentIncompressible"

Ideal_s                       = "Ideal"
VanderWaals_s                 = "VanderWaals"
Constant_s                    = "Constant"
PowerLaw_s                    = "PowerLaw"
SutherlandLaw_s               = "SutherlandLaw"
ConstantPrandtl_s             = "ConstantPrandtl"
EddyViscosity_s               = "EddyViscosity"
ReynoldsStress_s              = "ReynoldsStress"
Algebraic_s                   = "Algebraic"
BaldwinLomax_s                = "BaldwinLomax"
ReynoldsStressAlgebraic_s     = "ReynoldsStressAlgebraic"
Algebraic_BaldwinLomax_s      = "Algebraic_BaldwinLomax"
Algebraic_CebeciSmith_s       = "Algebraic_CebeciSmith"
HalfEquation_JohnsonKing_s    = "HalfEquation_JohnsonKing"
OneEquation_BaldwinBarth_s    = "OneEquation_BaldwinBarth"
OneEquation_SpalartAllmaras_s = "OneEquation_SpalartAllmaras"
TwoEquation_JonesLaunder_s    = "TwoEquation_JonesLaunder"
TwoEquation_MenterSST_s       = "TwoEquation_MenterSST"
TwoEquation_Wilcox_s          = "TwoEquation_Wilcox"
CaloricallyPerfect_s          = "CaloricallyPerfect"
ThermallyPerfect_s            = "ThermallyPerfect"
ConstantDensity_s             = "ConstantDensity"
RedlichKwong_s                = "RedlichKwong"
Frozen_s                      = "Frozen"
ThermalEquilib_s              = "ThermalEquilib"
ThermalNonequilib_s           = "ThermalNonequilib"
ChemicalEquilibCurveFit_s     = "ChemicalEquilibCurveFit"
ChemicalEquilibMinimization_s = "ChemicalEquilibMinimization"
ChemicalNonequilib_s          = "ChemicalNonequilib"
EMElectricField_s             = "EMElectricField"
EMMagneticField_s             = "EMMagneticField"
EMConductivity_s              = "EMConductivity"
Voltage_s                     = "Voltage"
Interpolated_s                = "Interpolated"
Equilibrium_LinRessler_s      = "Equilibrium_LinRessler"
Chemistry_LinRessler_s        = "Chemistry_LinRessler"

FamilySpecified_s             = "FamilySpecified"

# --------------------------------------------------
DataType_ts    = "DataType_t"
DataType_s     = "DataType"

Integer_s      = "Integer"
LongInteger_s  = "LongInteger"
RealSingle_s   = "RealSingle"
RealDouble_s   = "RealDouble"
Character_s    = "Character"

DataType_l = [Null_s,UserDefined_s,
              Integer_s,RealSingle_s,RealDouble_s,Character_s,LongInteger_s]
DataType   = stringAsKeyDict(DataType_l)
DataType_  = enumAsKeyDict(DataType_l)

(DataTypeNull,DataTypeUserDefined,\
 Integer,RealSingle,RealDouble,Character,LongInteger)=DataType_.keys()

# --------------------------------------------------
GridConnectivityType_ts = "GridConnectivityType_t"
GridConnectivityType_s  = "GridConnectivityType"
GridConnectivity_ts     = "GridConnectivity_t"
ZoneGridConnectivity_ts = "ZoneGridConnectivity_t"
ZoneGridConnectivity_s  = "ZoneGridConnectivity"

Overset_s       = "Overset"
Abutting_s      = "Abutting"
Abutting1to1_s  = "Abutting1to1"

GridConnectivityType_l = [Null_s,UserDefined_s,
                          Overset_s,Abutting_s,Abutting1to1_s]
GridConnectivityType   = stringAsKeyDict(GridConnectivityType_l)
GridConnectivityType_  = enumAsKeyDict(GridConnectivityType_l)

(Null,UserDefined,
 Overset,Abutting,Abutting1to1)=GridConnectivityType_.keys()

# --------------------------------------------------
ZoneType_ts    = "ZoneType_t"
ZoneType_s     = "ZoneType"
Zone_ts        = "Zone_t"

Structured_s   = "Structured"
Unstructured_s = "Unstructured"

ZoneType_l     = [Null_s,UserDefined_s,Structured_s,Unstructured_s]
ZoneType       = stringAsKeyDict(ZoneType_l)
ZoneType_      = enumAsKeyDict(ZoneType_l)

(ZoneTypeNull,ZoneTypeUserdefined,Structured,Unstructured)=ZoneType_.keys()

ZoneSubRegion_ts = "ZoneSubRegion_t"
BCRegionName_s   = "BCRegionName"
GridConnectivityRegionName_s = "GridConnectivityRegionName"

# --------------------------------------------------
SimulationType_ts = "SimulationType_t"
SimulationType_s  = "SimulationType"

TimeAccurate_s    = "TimeAccurate"
NonTimeAccurate_s = "NonTimeAccurate"

SimulationType_l  = [Null_s,UserDefined_s,TimeAccurate_s,NonTimeAccurate_s]
SimulationType    = stringAsKeyDict(SimulationType_l)
SimulationType_   = enumAsKeyDict(SimulationType_l)

(Null,UserDefined,TimeAccurate,NonTimeAccurate)=SimulationType_.keys()

# --------------------------------------------------
ConstantRate_s        = "ConstantRate"
VariableRate_s        = "VariableRate"
NonDeformingGrid_s    = "NonDeformingGrid"
DeformingGrid_s       = "DeformingGrid"
RigidGridMotionType_l = [Null_s,ConstantRate_s,VariableRate_s,UserDefined_s]

RigidGridMotionType_s="RigidGridMotionType"
RigidGridMotionType_ts="RigidGridMotionType_t"

Generic_s                     = "Generic"
BleedArea_s                   = "BleedArea"
CaptureArea_s                 = "CaptureArea"
AverageAll_s                  = "AverageAll"
AverageCircumferential_s      = "AverageCircumferential"
AverageRadial_s               = "AverageRadial"
AverageI_s                    = "AverageI"
AverageJ_s                    = "AverageJ"
AverageK_s                    = "AverageK"
CGNSLibraryVersion_s          = "CGNSLibraryVersion"
CellDimension_s               = "CellDimension"
IndexDimension_s              = "IndexDimension"
PhysicalDimension_s           = "PhysicalDimension"
GridCoordinates_s             = "GridCoordinates"
CoordinateNames_s             = "CoordinateNames"
CoordinateX_s                 = "CoordinateX"
CoordinateY_s                 = "CoordinateY"
CoordinateZ_s                 = "CoordinateZ"
CoordinateR_s                 = "CoordinateR"
CoordinateTheta_s             = "CoordinateTheta"
CoordinatePhi_s               = "CoordinatePhi"
CoordinateNormal_s            = "CoordinateNormal"
CoordinateTangential_s        = "CoordinateTangential"
CoordinateXi_s                = "CoordinateXi"
CoordinateEta_s               = "CoordinateEta"
CoordinateZeta_s              = "CoordinateZeta"
CoordinateTransform_s         = "CoordinateTransform"
InterpolantsDonor_s           = "InterpolantsDonor"
ElementConnectivity_s         = "ElementConnectivity"
ParentData_s                  = "ParentData"
ParentElements_s              = "ParentElements"
ParentElementsPosition_s      = "ParentElementsPosition"
ElementSizeBoundary_s         = "ElementSizeBoundary"
VectorX_ps                    = "%sX"
VectorY_ps                    = "%sY"
VectorZ_ps                    = "%sZ"
VectorTheta_ps                = "%sTheta"
VectorPhi_ps                  = "%sPhi"
VectorMagnitude_ps            = "%sMagnitude"
VectorNormal_ps               = "%sNormal"
VectorTangential_ps           = "%sTangential"
Potential_s                   = "Potential"
StreamFunction_s              = "StreamFunction"
Density_s                     = "Density"
Pressure_s                    = "Pressure"
Temperature_s                 = "Temperature"
EnergyInternal_s              = "EnergyInternal"
Enthalpy_s                    = "Enthalpy"
Entropy_s                     = "Entropy"
EntropyApprox_s               = "EntropyApprox"
DensityStagnation_s           = "DensityStagnation"
PressureStagnation_s          = "PressureStagnation"
TemperatureStagnation_s       = "TemperatureStagnation"
EnergyStagnation_s            = "EnergyStagnation"
EnthalpyStagnation_s          = "EnthalpyStagnation"
EnergyStagnationDensity_s     = "EnergyStagnationDensity"
VelocityX_s                   = "VelocityX"
VelocityY_s                   = "VelocityY"
VelocityZ_s                   = "VelocityZ"
VelocityR_s                   = "VelocityR"
VelocityTheta_s               = "VelocityTheta"
VelocityPhi_s                 = "VelocityPhi"
VelocityMagnitude_s           = "VelocityMagnitude"
VelocityNormal_s              = "VelocityNormal"
VelocityTangential_s          = "VelocityTangential"
VelocitySound_s               = "VelocitySound"
VelocitySoundStagnation_s     = "VelocitySoundStagnation"
MomentumX_s                   = "MomentumX"
MomentumY_s                   = "MomentumY"
MomentumZ_s                   = "MomentumZ"
MomentumMagnitude_s           = "MomentumMagnitude"
RotatingVelocityX_s           = "RotatingVelocityX"
RotatingVelocityY_s           = "RotatingVelocityY"
RotatingVelocityZ_s           = "RotatingVelocityZ"
RotatingMomentumX_s           = "RotatingMomentumX"
RotatingMomentumY_s           = "RotatingMomentumY"
RotatingMomentumZ_s           = "RotatingMomentumZ"
RotatingVelocityMagnitude_s   = "RotatingVelocityMagnitude"
RotatingPressureStagnation_s  = "RotatingPressureStagnation"
RotatingEnergyStagnation_s    = "RotatingEnergyStagnation"
RotatingEnergyStagnationDensity_s = "RotatingEnergyStagnationDensity"
RotatingEnthalpyStagnation_s  = "RotatingEnthalpyStagnation"
EnergyKinetic_s               = "EnergyKinetic"
PressureDynamic_s             = "PressureDynamic"
SoundIntensityDB_s            = "SoundIntensityDB"
SoundIntensity_s              = "SoundIntensity"
VorticityX_s                  = "VorticityX"
VorticityY_s                  = "VorticityY"
VorticityZ_s                  = "VorticityZ"
VorticityMagnitude_s          = "VorticityMagnitude"
SkinFrictionX_s               = "SkinFrictionX"
SkinFrictionY_s               = "SkinFrictionY"
SkinFrictionZ_s               = "SkinFrictionZ"
SkinFrictionMagnitude_s       = "SkinFrictionMagnitude"
VelocityAngleX_s              = "VelocityAngleX"
VelocityAngleY_s              = "VelocityAngleY"
VelocityAngleZ_s              = "VelocityAngleZ"
VelocityUnitVectorX_s         = "VelocityUnitVectorX"
VelocityUnitVectorY_s         = "VelocityUnitVectorY"
VelocityUnitVectorZ_s         = "VelocityUnitVectorZ"
MassFlow_s                    = "MassFlow"
ViscosityKinematic_s          = "ViscosityKinematic"
ViscosityMolecular_s          = "ViscosityMolecular"
ViscosityEddyDynamic_s        = "ViscosityEddyDynamic"
ViscosityEddy_s               = "ViscosityEddy"
ThermalConductivity_s         = "ThermalConductivity"
PowerLawExponent_s            = "PowerLawExponent"
SutherlandLawConstant_s       = "SutherlandLawConstant"
TemperatureReference_s        = "TemperatureReference"
ViscosityMolecularReference_s = "ViscosityMolecularReference"
ThermalConductivityReference_s = "ThermalConductivityReference"
IdealGasConstant_s            = "IdealGasConstant"
SpecificHeatPressure_s        = "SpecificHeatPressure"
SpecificHeatVolume_s          = "SpecificHeatVolume"
ReynoldsStressXX_s            = "ReynoldsStressXX"
ReynoldsStressXY_s            = "ReynoldsStressXY"
ReynoldsStressXZ_s            = "ReynoldsStressXZ"
ReynoldsStressYY_s            = "ReynoldsStressYY"
ReynoldsStressYZ_s            = "ReynoldsStressYZ"
ReynoldsStressZZ_s            = "ReynoldsStressZZ"
LengthReference_s             = "LengthReference"
MolecularWeight_s             = "MolecularWeight"
MolecularWeight_ps            = "MolecularWeight%s"
HeatOfFormation_s             = "HeatOfFormation"
HeatOfFormation_ps            = "HeatOfFormation%s"
FuelAirRatio_s                = "FuelAirRatio"
ReferenceTemperatureHOF_s     = "ReferenceTemperatureHOF"
MassFraction_s                = "MassFraction"
MassFraction_ps               = "MassFraction%s"
LaminarViscosity_s            = "LaminarViscosity"
LaminarViscosity_ps           = "LaminarViscosity%s"
ThermalConductivity_ps        = "ThermalConductivity%s"
EnthalpyEnergyRatio_s         = "EnthalpyEnergyRatio"
CompressibilityFactor_s       = "CompressibilityFactor"
VibrationalElectronEnergy_s   = "VibrationalElectronEnergy"
VibrationalElectronTemperature_s = "VibrationalElectronTemperature"
SpeciesDensity_s              = "SpeciesDensity"
SpeciesDensity_ps             = "SpeciesDensity%s"
MoleFraction_s                = "MoleFraction"
MoleFraction_ps               = "MoleFraction%s"
ElectricFieldX_s              = "ElectricFieldX"
ElectricFieldY_s              = "ElectricFieldY"
ElectricFieldZ_s              = "ElectricFieldZ"
MagneticFieldX_s              = "MagneticFieldX"
MagneticFieldY_s              = "MagneticFieldY"
MagneticFieldZ_s              = "MagneticFieldZ"
CurrentDensityX_s             = "CurrentDensityX"
CurrentDensityY_s             = "CurrentDensityY"
CurrentDensityZ_s             = "CurrentDensityZ"
LorentzForceX_s               = "LorentzForceX"
LorentzForceY_s               = "LorentzForceY"
LorentzForceZ_s               = "LorentzForceZ"
ElectricConductivity_s        = "ElectricConductivity"
JouleHeating_s                = "JouleHeating"
TurbulentDistance_s           = "TurbulentDistance"
TurbulentEnergyKinetic_s      = "TurbulentEnergyKinetic"
TurbulentDissipation_s        = "TurbulentDissipation"
TurbulentDissipationRate_s    = "TurbulentDissipationRate"
TurbulentBBReynolds_s         = "TurbulentBBReynolds"
TurbulentSANuTilde_s          = "TurbulentSANuTilde"
Mach_s                        = "Mach"
Mach_Velocity_s               = "Mach_Velocity"
Mach_VelocitySound_s          = "Mach_VelocitySound"
Reynolds_s                    = "Reynolds"
Reynolds_Velocity_s           = "Reynolds_Velocity"
Reynolds_Length_s             = "Reynolds_Length"
Reynolds_ViscosityKinematic_s = "Reynolds_ViscosityKinematic"
Prandtl_s                     = "Prandtl"
Prandtl_ThermalConductivity_s = "Prandtl_ThermalConductivity"
Prandtl_ViscosityMolecular_s  = "Prandtl_ViscosityMolecular"
Prandtl_SpecificHeatPressure_s = "Prandtl_SpecificHeatPressure"
PrandtlTurbulent_s            = "PrandtlTurbulent"
SpecificHeatRatio_s           = "SpecificHeatRatio"
SpecificHeatRatio_Pressure_s  = "SpecificHeatRatio_Pressure"
SpecificHeatRatio_Volume_s    = "SpecificHeatRatio_Volume"
CoefPressure_s                = "CoefPressure"
CoefSkinFrictionX_s           = "CoefSkinFrictionX"
CoefSkinFrictionY_s           = "CoefSkinFrictionY"
CoefSkinFrictionZ_s           = "CoefSkinFrictionZ"
Coef_PressureDynamic_s        = "Coef_PressureDynamic"
Coef_PressureReference_s      = "Coef_PressureReference"
Vorticity_s                   = "Vorticity"
Acoustic_s                    = "Acoustic"
RiemannInvariantPlus_s        = "RiemannInvariantPlus"
RiemannInvariantMinus_s       = "RiemannInvariantMinus"
CharacteristicEntropy_s       = "CharacteristicEntropy"
CharacteristicVorticity1_s    = "CharacteristicVorticity1"
CharacteristicVorticity2_s    = "CharacteristicVorticity2"
CharacteristicAcousticPlus_s  = "CharacteristicAcousticPlus"
CharacteristicAcousticMinus_s = "CharacteristicAcousticMinus"
ForceX_s                      = "ForceX"
ForceY_s                      = "ForceY"
ForceZ_s                      = "ForceZ"
ForceR_s                      = "ForceR"
ForceTheta_s                  = "ForceTheta"
ForcePhi_s                    = "ForcePhi"
Lift_s                        = "Lift"
Drag_s                        = "Drag"
MomentX_s                     = "MomentX"
MomentY_s                     = "MomentY"
MomentZ_s                     = "MomentZ"
MomentR_s                     = "MomentR"
MomentTheta_s                 = "MomentTheta"
MomentPhi_s                   = "MomentPhi"
MomentXi_s                    = "MomentXi"
MomentEta_s                   = "MomentEta"
MomentZeta_s                  = "MomentZeta"
Moment_CenterX_s              = "Moment_CenterX"
Moment_CenterY_s              = "Moment_CenterY"
Moment_CenterZ_s              = "Moment_CenterZ"
CoefLift_s                    = "CoefLift"
CoefDrag_s                    = "CoefDrag"
CoefMomentX_s                 = "CoefMomentX"
CoefMomentY_s                 = "CoefMomentY"
CoefMomentZ_s                 = "CoefMomentZ"
CoefMomentR_s                 = "CoefMomentR"
CoefMomentTheta_s             = "CoefMomentTheta"
CoefMomentPhi_s               = "CoefMomentPhi"
CoefMomentXi_s                = "CoefMomentXi"
CoefMomentEta_s               = "CoefMomentEta"
CoefMomentZeta_s              = "CoefMomentZeta"
Coef_PressureDynamic_s        = "Coef_PressureDynamic"
Coef_Area_s                   = "Coef_Area"
Coef_Length_s                 = "Coef_Length"
TimeValues_s                  = "TimeValues"
IterationValues_s             = "IterationValues"
NumberOfZones_s               = "NumberOfZones"
NumberOfFamilies_s            = "NumberOfFamilies"
NumberOfSteps_s               = "NumberOfSteps"
DataConversion_s              = "DataConversion"

ZonePointers_s                = "ZonePointers"
FamilyPointers_s              = "FamilyPointers"
RigidGridMotionPointers_s     = "RigidGridMotionPointers"
ArbitraryGridMotionPointers_s = "ArbitraryGridMotionPointers"
GridCoordinatesPointers_s     = "GridCoordinatesPointers"
FlowSolutionPointers_s        = "FlowSolutionPointers"
ZoneGridConnectivityPointers_s= "ZoneGridConnectivityPointers"
ZoneSubRegionPointers_s       = "ZoneSubRegionPointers"
PointerNames_l = [ZonePointers_s,FamilyPointers_s,RigidGridMotionPointers_s,
                  ArbitraryGridMotionPointers_s,GridCoordinatesPointers_s,
                  FlowSolutionPointers_s,
                  ZoneGridConnectivityPointers_s,ZoneSubRegionPointers_s]

OriginLocation_s              = "OriginLocation"
RigidRotationAngle_s          = "RigidRotationAngle"
Translation_s                 = "Translation"
RotationAngle_s               = "RotationAngle"
RigidVelocity_s               = "RigidVelocity"
RigidRotationRate_s           = "RigidRotationRate"
GridVelocityX_s               = "GridVelocityX"
GridVelocityY_s               = "GridVelocityY"
GridVelocityZ_s               = "GridVelocityZ"
GridVelocityR_s               = "GridVelocityR"
GridVelocityTheta_s           = "GridVelocityTheta"
GridVelocityPhi_s             = "GridVelocityPhi"
GridVelocityXi_s              = "GridVelocityXi"
GridVelocityEta_s             = "GridVelocityEta"
GridVelocityZeta_s            = "GridVelocityZeta"

ArbitraryGridMotion_ts        = "ArbitraryGridMotion_t"
ArbitraryGridMotion_s         = "ArbitraryGridMotion"
ArbitraryGridMotionType_l     = [Null_s,NonDeformingGrid_s,
                                DeformingGrid_s,UserDefined_s]

ArbitraryGridMotionType_s     ="ArbitraryGridMotionType"
ArbitraryGridMotionType_ts    ="ArbitraryGridMotionType_t"

Area_ts                       = "Area_t"
Area_s                        = "Area"
AreaType_ts                   = "AreaType_t"
AreaType_s                    = "AreaType"
SurfaceArea_s                 = "SurfaceArea"
RegionName_s                  = "RegionName"
AverageInterface_ts           = "AverageInterface_t"
Axisymmetry_ts                = "Axisymmetry_t"
Axisymmetry_s                 = "Axisymmetry"
AxisymmetryReferencePoint_s   = "AxisymmetryReferencePoint"
AxisymmetryAxisVector_s       = "AxisymmetryAxisVector"
AxisymmetryAngle_s            = "AxisymmetryAngle"
BCDataSet_ts                  = "BCDataSet_t"
BCData_ts                     = "BCData_t"
BCData_s                      = "BCData"

BCProperty_ts                 = "BCProperty_t"
BCProperty_s                  = "BCProperty"
BC_ts                         = "BC_t"

BaseIterativeData_ts          = "BaseIterativeData_t"
BaseIterativeData_s           = "BaseIterativeData"

CGNSBase_ts                   = "CGNSBase_t"
CGNSLibraryVersion_ts         = "CGNSLibraryVersion_t"

# --------------------------------------------------
ConvergenceHistory_ts         = "ConvergenceHistory_t"
ZoneConvergenceHistory_s      = "ZoneConvergenceHistory"
GlobalConvergenceHistory_s    = "GlobalConvergenceHistory"

ConvergenceHistory_l          = [ZoneConvergenceHistory_s,
                                 GlobalConvergenceHistory_s]

NormDefinitions_s             ="NormDefinitions"

DataArray_ts                  = "DataArray_t"
DataConversion_ts             = "DataConversion_t"
Descriptor_ts                 = "Descriptor_t"

# --------------------------------------------------
DimensionalExponents_ts       = "DimensionalExponents_t"
DimensionalExponents_s        = "DimensionalExponents"
DimensionalUnits_ts           = "DimensionalUnits_t"
AdditionalUnits_ts            = "AdditionalUnits_t"
AdditionalExponents_ts        = "AdditionalExponents_t"

DiscreteData_ts               = "DiscreteData_t"
DiscreteData_s                = "DiscreteData"

FamilyBC_s                    = "FamilyBC"
FamilyBC_ts                   = "FamilyBC_t"
FamilyBCDataSet_ts            = "FamilyBCDataSet_t"

FamilyName_ts                 = "FamilyName_t"
FamilyName_s                  = "FamilyName"
AdditionalFamilyName_ts       = "AdditionalFamilyName_t"
AdditionalFamilyName_s        = "AdditionalFamilyName"
Family_ts                     = "Family_t"
Family_s                      = "Family"
FlowEquationSet_ts            = "FlowEquationSet_t"
FlowEquationSet_s             = "FlowEquationSet"
FlowSolution_ts               = "FlowSolution_t"
GasModel_ts                   = "GasModel_t"
GasModel_s                    = "GasModel"

GeometryEntity_ts             = "GeometryEntity_t"
GeometryFile_ts               = "GeometryFile_t"
GeometryFile_s                = "GeometryFile"

GeometryFormat_s              = "GeometryFormat"
GeometryFormat_ts             = "GeometryFormat_t"

# not supported '-'
NASAIGES_s                    ="NASA-IGES"
ICEMCFD_s                     ="ICEM-CFD"

SDRC_s                        ="SDRC"
Unigraphics_s                 ="Unigraphics"
ProEngineer_s                 ="ProEngineer"
GeometryFormat_l              =[Null_s,NASAIGES_s,SDRC_s,Unigraphics_s,
                                ProEngineer_s,ICEMCFD_s,UserDefined_s]
GeometryReference_ts          = "GeometryReference_t"
GeometryReference_s           = "GeometryReference"

Gravity_ts                    = "Gravity_t"
Gravity_s                     = "Gravity"
GravityVector_s               = "GravityVector"

GridConnectivity1to1_ts       = "GridConnectivity1to1_t"
GridConnectivityProperty_ts   = "GridConnectivityProperty_t"
GridConnectivityProperty_s    = "GridConnectivityProperty"

GridCoordinates_ts            = "GridCoordinates_t"
IndexArray_ts                 = "IndexArray_t"
IndexRange_ts                 = "IndexRange_t"
IntegralData_ts               = "IntegralData_t"
InwardNormalList_ts           = "InwardNormalList_t"
InwardNormalList_s            = "InwardNormalList"
InwardNormalIndex_s           = "InwardNormalIndex"
Ordinal_ts                    = "Ordinal_t"
Ordinal_s                     = "Ordinal"
Transform_s                   = "Transform"
OversetHoles_ts               = "OversetHoles_t"
OversetHoles_s                = "OversetHoles"
Periodic_ts                   = "Periodic_t"
Periodic_s                    = "Periodic"

ReferenceState_ts             = "ReferenceState_t"
ReferenceState_s              = "ReferenceState"
ReferenceStateDescription_s   = "ReferenceStateDescription"

RigidGridMotion_ts            = "RigidGridMotion_t"
RigidGridMotion_s             = "RigidGridMotion"

Rind_s                        = "Rind"
Rind_ts                       = "Rind_t"

RotatingCoordinates_s         = "RotatingCoordinates"
RotatingCoordinates_ts        = "RotatingCoordinates_t"
RotationRateVector_s          = "RotationRateVector"
RotationCenter_s              = "RotationCenter"

GoverningEquations_s          = "GoverningEquations"
GoverningEquations_ts         = "GoverningEquations_t"

GoverningEquationsType_l      = [Null_s, UserDefined_s,
                                 FullPotential_s,Euler_s,NSLaminar_s,
                                 NSTurbulent_s,NSLaminarIncompressible_s,
                                 NSTurbulentIncompressible_s]
GoverningEquationsType_s      = "GoverningEquationsType"
GoverningEquationsType_ts     = "GoverningEquationsType_t"
GoverningEquationsType        = stringAsKeyDict(GoverningEquationsType_l)
GoverningEquationsType_       = enumAsKeyDict(GoverningEquationsType_l)
(GoverningEquationsNull, GoverningEquationUserDefined,
 FullPotential,Euler,NSLaminar,
 NSTurbulent,NSLaminarIncompressible,
 NSTurbulentIncompressible_s)=GoverningEquationsType_.keys()

# --------------------------------------------------
BCType_s                   = "BCType"
BCType_ts                  = "BCType_t"
BCTypeSimple_s             = "BCTypeSimple"
BCTypeSimple_ts            = "BCTypeSimple_t"

BCAxisymmetricWedge_s      = "BCAxisymmetricWedge"
BCDegenerateLine_s         = "BCDegenerateLine"
BCDegeneratePoint_s        = "BCDegeneratePoint"
BCDirichlet_s              = "BCDirichlet"
BCExtrapolate_s            = "BCExtrapolate"
BCFarfield_s               = "BCFarfield"
BCGeneral_s                = "BCGeneral"
BCInflow_s                 = "BCInflow"
BCInflowSubsonic_s         = "BCInflowSubsonic"
BCInflowSupersonic_s       = "BCInflowSupersonic"
BCNeumann_s                = "BCNeumann"
BCOutflow_s                = "BCOutflow"
BCOutflowSubsonic_s        = "BCOutflowSubsonic"
BCOutflowSupersonic_s      = "BCOutflowSupersonic"
BCSymmetryPlane_s          = "BCSymmetryPlane"
BCSymmetryPolar_s          = "BCSymmetryPolar"
BCTunnelInflow_s           = "BCTunnelInflow"
BCTunnelOutflow_s          = "BCTunnelOutflow"
BCWall_s                   = "BCWall"
BCWallInviscid_s           = "BCWallInviscid"
BCWallViscous_s            = "BCWallViscous"
BCWallViscousHeatFlux_s    = "BCWallViscousHeatFlux"
BCWallViscousIsothermal_s  = "BCWallViscousIsothermal"

BCType_l = [Null_s,UserDefined_s,
           BCAxisymmetricWedge_s,BCDegenerateLine_s,BCDegeneratePoint_s,
           BCDirichlet_s,BCExtrapolate_s,BCFarfield_s,
           BCGeneral_s,BCInflow_s,BCInflowSubsonic_s,BCInflowSupersonic_s,
           BCNeumann_s,BCOutflow_s,BCOutflowSubsonic_s,BCOutflowSupersonic_s,
           BCSymmetryPlane_s,BCSymmetryPolar_s,
           BCTunnelInflow_s,BCTunnelOutflow_s,
           BCWall_s,BCWallInviscid_s,BCWallViscous_s,
           BCWallViscousHeatFlux_s,BCWallViscousIsothermal_s,
           FamilySpecified_s]
BCType   = stringAsKeyDict(BCType_l)
BCType_  = enumAsKeyDict(BCType_l)
(Null,UserDefined,
 BCAxisymmetricWedge,BCDegenerateLine,BCDegeneratePoint,
 BCDirichlet,BCExtrapolate,BCFarfield,
 BCGeneral,BCInflow,BCInflowSubsonic,BCInflowSupersonic,
 BCNeumann,BCOutflow,BCOutflowSubsonic,BCOutflowSupersonic,
 BCSymmetryPlane,BCSymmetryPolar,
 BCTunnelInflow,BCTunnelOutflow,
 BCWall,BCWallInviscid,BCWallViscous,
 BCWallViscousHeatFlux,BCWallViscousIsothermal,
 FamilySpecified)=BCType_.keys()

FamilyBC_l = BCType_l
FamilyBC   = BCType
FamilyBC_  = BCType_

# CAUTION, index of values in the lists below cannot be used as enumerate,
# the lists are subset of the global list and some index are missing.
BCTypeSimple_l   =[Null_s,UserDefined_s,BCGeneral_s,BCDirichlet_s,BCNeumann_s,
                   BCExtrapolate_s,BCWallInviscid_s,BCWallViscousHeatFlux_s,
                   BCWallViscousIsothermal_s,BCWallViscous_s,BCWall_s,
                   BCInflowSubsonic_s,BCInflowSupersonic_s,BCOutflowSubsonic_s,
                   BCOutflowSupersonic_s,BCTunnelInflow_s,BCTunnelOutflow_s,
                   BCDegenerateLine_s,BCDegeneratePoint_s,BCSymmetryPlane_s,
                   BCSymmetryPolar_s,BCAxisymmetricWedge_s,FamilySpecified_s]
BCTypeCompound_l = [Null_s,UserDefined_s,
                    BCInflow_s,BCOutflow_s,BCFarfield_s]

# --------------------------------------------------
# WARNING -
# - the model sub-list CANNOT be used to define the corresponding enumerate
#   because the new values are add at the end of a global list (ModelType)
#   then the ONLY enumerate to use is the ModelType one which has its own list
#
GasModelType_l       = [Null_s,UserDefined_s,
                        Ideal_s,VanderWaals_s,CaloricallyPerfect_s,
                        ThermallyPerfect_s,ConstantDensity_s,RedlichKwong_s]
GasModelType_s       = "GasModelType"
GasModelType_ts      = "GasModelType_t"
GasModelIdentifier_l = [IdealGasConstant_s,SpecificHeatRatio_s,
                        SpecificHeatVolume_s,SpecificHeatPressure_s]

ViscosityModel_ts          = "ViscosityModel_t"
ViscosityModel_s           = "ViscosityModel"
ViscosityModelType_l       = [Null_s,UserDefined_s,
                              Constant_s,PowerLaw_s,SutherlandLaw_s]
ViscosityModelType_s       = "ViscosityModelType"
ViscosityModelType_ts      = "ViscosityModelType_t"
ViscosityModelIdentifier_l = [PowerLawExponent_s,SutherlandLawConstant_s,
                              TemperatureReference_s,
                              ViscosityMolecularReference_s]

ThermalConductivityModel_ts          = "ThermalConductivityModel_t"
ThermalConductivityModel_s           = "ThermalConductivityModel"
ThermalConductivityModelType_l       = [Null_s,UserDefined_s,
                                        ConstantPrandtl_s,PowerLaw_s,
                                        SutherlandLaw_s]
ThermalConductivityModelType_s       = "ThermalConductivityModelType"
ThermalConductivityModelType_ts      = "ThermalConductivityModelType_t"
ThermalConductivityModelIdentifier_l = [Prandtl_s,PowerLawExponent_s,
                                        SutherlandLawConstant_s,
                                        TemperatureReference_s,
                                        ThermalConductivityReference_s]

TurbulenceModel_ts     = "TurbulenceModel_t"
TurbulenceModel_s      = "TurbulenceModel"
TurbulenceModelType_l  = [Null_s,UserDefined_s,
                          Algebraic_BaldwinLomax_s,
                          Algebraic_CebeciSmith_s,
                          HalfEquation_JohnsonKing_s,
                          OneEquation_BaldwinBarth_s,
                          OneEquation_SpalartAllmaras_s,
                          TwoEquation_JonesLaunder_s,
                          TwoEquation_MenterSST_s,TwoEquation_Wilcox_s]
TurbulenceModelType_s  = "TurbulenceModelType"
TurbulenceModelType_ts = "TurbulenceModelType_t"

TurbulenceClosure_ts          = "TurbulenceClosure_t"
TurbulenceClosure_s           = "TurbulenceClosure"
TurbulenceClosureType_l       = [Null_s,UserDefined_s,
                                 EddyViscosity_s,ReynoldsStress_s,
                                 ReynoldsStressAlgebraic_s]
TurbulenceClosureType_s       = "TurbulenceClosureType"
TurbulenceClosureType_ts      = "TurbulenceClosureType_t"
TurbulenceClosureIdentifier_l = [PrandtlTurbulent_s]

ThermalRelaxationModel_ts     = "ThermalRelaxationModel_t"
ThermalRelaxationModel_s      = "ThermalRelaxationModel"
ThermalRelaxationModelType_l  = [Null_s,UserDefined_s,
                                 Frozen_s,ThermalEquilib_s,
                                 ThermalNonequilib_s]
ThermalRelaxationModelType_s  = "ThermalRelaxationModelType"
ThermalRelaxationModelType_ts = "ThermalRelaxationModelType_t"

ChemicalKineticsModel_ts          = "ChemicalKineticsModel_t"
ChemicalKineticsModel_s           = "ChemicalKineticsModel"
ChemicalKineticsModelType_l       = [Null_s,UserDefined_s,
                                     Frozen_s,ChemicalEquilibCurveFit_s,
                                     ChemicalEquilibMinimization_s,
                                     ChemicalNonequilib_s]
ChemicalKineticsModelType_s       = "ChemicalKineticsModelType"
ChemicalKineticsModelType_ts      = "ChemicalKineticsModelType_t"
ChemicalKineticsModelIdentifier_l = [FuelAirRatio_s,ReferenceTemperatureHOF_s]

EMElectricFieldModel_s      = "EMElectricFieldModel"
EMElectricFieldModel_ts     = "EMElectricFieldModel_t"
EMElectricFieldModelType_l  = [Null_s,UserDefined_s,
                               Constant_s,Frozen_s,
                               Interpolated_s,Voltage_s]
EMElectricFieldModelType_s  = "EMElectricFieldModelType"
EMElectricFieldModelType_ts = "EMElectricFieldModelType_t"

EMMagneticFieldModel_s      = "EMMagneticFieldModel"
EMMagneticFieldModel_ts     = "EMMagneticFieldModel_t"
EMMagneticFieldModelType_l  = [Null_s,UserDefined_s,
                               Constant_s,Frozen_s,Interpolated_s]
EMMagneticFieldModelType_s  = "EMMagneticFieldModelType"
EMMagneticFieldModelType_ts = "EMMagneticFieldModelType_t"

EMConductivityModel_s           = "EMConductivityModel"
EMConductivityModel_ts          = "EMConductivityModel_t"
EMConductivityModelType_l       = [Null_s,UserDefined_s,
                                   Constant_s,Frozen_s,
                                   Equilibrium_LinRessler_s,
                                   Chemistry_LinRessler_s]
EMConductivityModelType_s       = "EMConductivityModelType"
EMConductivityModelType_ts      = "EMConductivityModelType_t"
EMConductivityModelIdentifier_l = [Null_s,UserDefined_s,
                                   ElectricFieldX_s,ElectricFieldY_s,
                                   ElectricFieldZ_s,MagneticFieldX_s,
                                   MagneticFieldY_s,MagneticFieldZ_s,
                                   CurrentDensityX_s,CurrentDensityY_s,
                                   CurrentDensityZ_s,ElectricConductivity_s,
                                   LorentzForceX_s,LorentzForceY_s,
                                   LorentzForceZ_s,JouleHeating_s]

ModelType_l = [Null_s,UserDefined_s,
               Ideal_s,VanderWaals_s,Constant_s,PowerLaw_s,SutherlandLaw_s,
               ConstantPrandtl_s,EddyViscosity_s,ReynoldsStress_s,
               ReynoldsStressAlgebraic_s,Algebraic_BaldwinLomax_s,
               Algebraic_CebeciSmith_s,HalfEquation_JohnsonKing_s,
               OneEquation_BaldwinBarth_s,OneEquation_SpalartAllmaras_s,
               TwoEquation_JonesLaunder_s,TwoEquation_MenterSST_s,
               TwoEquation_Wilcox_s,CaloricallyPerfect_s,ThermallyPerfect_s,
               ConstantDensity_s,RedlichKwong_s,Frozen_s,ThermalEquilib_s,
               ThermalNonequilib_s,ChemicalEquilibCurveFit_s,
               ChemicalEquilibMinimization_s,ChemicalNonequilib_s,
               EMElectricField_s,EMMagneticField_s,EMConductivity_s,
               Voltage_s,Interpolated_s,Equilibrium_LinRessler_s,
               Chemistry_LinRessler_s]
ModelType   = stringAsKeyDict(ModelType_l)
ModelType_  = enumAsKeyDict(ModelType_l)
(Null,UserDefined,
 Ideal,VanderWaals,Constant,PowerLaw,SutherlandLaw,
 ConstantPrandtl,EddyViscosity,ReynoldsStress,
 ReynoldsStressAlgebraic,Algebraic_BaldwinLomax,
 Algebraic_CebeciSmith,HalfEquation_JohnsonKing,
 OneEquation_BaldwinBarth,OneEquationpalartAllmaras,
 TwoEquation_JonesLaunder,TwoEquation_MenterSST,
 TwoEquation_Wilcox,CaloricallyPerfect,ThermallyPerfect,
 ConstantDensity,RedlichKwong,Frozen,ThermalEquilib,
 ThermalNonequilib,ChemicalEquilibCurveFit,
 ChemicalEquilibMinimization,ChemicalNonequilib,
 EMElectricField,EMMagneticField,EMConductivity,
 Voltage,Interpolated,Equilibrium_LinRessler,
 Chemistry_LinRessler)=ModelType_.keys()

DiffusionModel_s    = 'DiffusionModel'
EquationDimension_s = 'EquationDimension'

# --------------------------------------------------

AverageInterfaceType_s  = "AverageInterfaceType"
AverageInterfaceType_ts = "AverageInterfaceType_t"
AverageInterfaceType_l  = [Null_s,AverageAll_s,AverageCircumferential_s,
                           AverageRadial_s,AverageI_s,AverageJ_s,AverageK_s,
                           UserDefined_s]
AverageInterface_s      = "AverageInterface"
AverageInterface_ts     = "AverageInterface_t"

NODE_s     = "NODE"
BAR_2_s    = "BAR_2"
BAR_3_s    = "BAR_3"
TRI_3_s    = "TRI_3"
TRI_6_s    = "TRI_6"
QUAD_4_s   = "QUAD_4"
QUAD_8_s   = "QUAD_8"
QUAD_9_s   = "QUAD_9"
TETRA_4_s  = "TETRA_4"
TETRA_10_s = "TETRA_10"
PYRA_5_s   = "PYRA_5"
PYRA_13_s  = "PYRA_13"
PYRA_14_s  = "PYRA_14"
PENTA_6_s  = "PENTA_6"
PENTA_15_s = "PENTA_15"
PENTA_18_s = "PENTA_18"
HEXA_8_s   = "HEXA_8"
HEXA_20_s  = "HEXA_20"
HEXA_27_s  = "HEXA_27"
MIXED_s    = "MIXED"
NGON_n_s   = "NGON_n"
NFACE_n_s  = "NFACE_n"
BAR_4_s    = "BAR_4"
TRI_9_s    = "TRI_9"
TRI_10_s   = "TRI_10"
QUAD_12_s  = "QUAD_12"
QUAD_16_s  = "QUAD_16"
TETRA_16_s = "TETRA_16"
TETRA_20_s = "TETRA_20"
PYRA_21_s  = "PYRA_21"
PYRA_29_s  = "PYRA_29"
PYRA_30_s  = "PYRA_30"
PENTA_24_s = "PENTA_24"
PENTA_38_s = "PENTA_38"
PENTA_40_s = "PENTA_40"
HEXA_32_s  = "HEXA_32"
HEXA_56_s  = "HEXA_56"
HEXA_64_s  = "HEXA_64"

Null_npe        = 0
UserDefined_npe = 0

NODE_npe     =  1
BAR_2_npe    =  2
BAR_3_npe    =  3
TRI_3_npe    =  3
TRI_6_npe    =  6
QUAD_4_npe   =  4
QUAD_8_npe   =  8
QUAD_9_npe   =  9
TETRA_4_npe  =  4
TETRA_10_npe = 10
PYRA_5_npe   =  5
PYRA_13_npe  = 13
PYRA_14_npe  = 14
PENTA_6_npe  =  6
PENTA_15_npe = 15
PENTA_18_npe = 18
HEXA_8_npe   =  8
HEXA_20_npe  = 20
HEXA_27_npe  = 27
MIXED_npe    =  0
NGON_n_npe   =  0
NFACE_n_npe  =  0
BAR_4_npe    =  4
TRI_9_npe    =  9
TRI_10_npe   = 10
QUAD_12_npe  = 12
QUAD_16_npe  = 16
TETRA_16_npe = 16
TETRA_20_npe = 20
PYRA_21_npe  = 21
PYRA_29_npe  = 29
PYRA_30_npe  = 30
PENTA_24_npe = 24
PENTA_38_npe = 38
PENTA_40_npe = 40
HEXA_32_npe  = 32
HEXA_56_npe  = 56
HEXA_64_npe  = 64

Elements_ts    = "Elements_t"
ElementType_ts = "ElementType_t"
ElementType_s  = "ElementType"
Elements_s     = "Elements"
ElementType_l  = [Null_s, UserDefined_s, NODE_s, BAR_2_s, BAR_3_s,
                 TRI_3_s, TRI_6_s, QUAD_4_s, QUAD_8_s, QUAD_9_s,
                 TETRA_4_s, TETRA_10_s, PYRA_5_s, PYRA_14_s,
                 PENTA_6_s, PENTA_15_s, PENTA_18_s,
                 HEXA_8_s, HEXA_20_s, HEXA_27_s, MIXED_s, PYRA_13_s,
                 NGON_n_s, NFACE_n_s, BAR_4_s, TRI_9_s, TRI_10_s, QUAD_12_s,
                 QUAD_16_s, TETRA_16_s, TETRA_20_s, PYRA_21_s, PYRA_29_s,
                 PYRA_30_s, PENTA_24_s, PENTA_38_s, PENTA_40_s, HEXA_32_s,
                 HEXA_56_s, HEXA_64_s ]
ElementTypeNPE_l = [Null_npe, UserDefined_npe, NODE_npe, BAR_2_npe, BAR_3_npe,
                   TRI_3_npe, TRI_6_npe, QUAD_4_npe, QUAD_8_npe, QUAD_9_npe,
                   TETRA_4_npe, TETRA_10_npe, PYRA_5_npe, PYRA_14_npe,
                   PENTA_6_npe, PENTA_15_npe, PENTA_18_npe,
                   HEXA_8_npe, HEXA_20_npe, HEXA_27_npe, MIXED_npe, 
                   PYRA_13_npe, NGON_n_npe, NFACE_n_npe, BAR_4_npe, TRI_9_npe,
                   TRI_10_npe, QUAD_12_npe, QUAD_16_npe, TETRA_16_npe,
                   TETRA_20_npe, PYRA_21_npe, PYRA_29_npe,
                   PYRA_30_npe, PENTA_24_npe, PENTA_38_npe, PENTA_40_npe,
                   HEXA_32_npe, HEXA_56_npe, HEXA_64_npe]
ElementType    = stringAsKeyDict(ElementType_l)
ElementType_   = enumAsKeyDict(ElementType_l)
ElementTypeNPE = dict(zip(ElementType_l,ElementTypeNPE_l))
(Null, UserDefined, NODE, BAR_2, BAR_3,
 TRI_3, TRI_6, QUAD_4, QUAD_8, QUAD_9,
 TETRA_4, TETRA_10, PYRA_5, PYRA_14,
 PENTA_6, PENTA_15, PENTA_18,
 HEXA_8, HEXA_20, HEXA_27, MIXED, PYRA_13,
 NGON_n, NFACE_n,
 BAR_4, TRI_9, TRI_10, QUAD_12,
 QUAD_16, TETRA_16, TETRA_20, PYRA_21, PYRA_29,
 PYRA_30, PENTA_24, PENTA_38, PENTA_40, HEXA_32,
 HEXA_56, HEXA_64)=ElementType_.keys()
 
ElementRangeList_s = "ElementRangeList"
#

WallFunction_ts               = "WallFunction_t"
WallFunction_s                = "WallFunction"
WallFunctionType_ts           = "WallFunctionType_t"
WallFunctionType_s            = "WallFunctionType"
ZoneBC_ts                     = "ZoneBC_t"
ZoneBC_s                      = "ZoneBC"
ZoneIterativeData_ts          = "ZoneIterativeData_t"
ZoneIterativeData_s           = "ZoneIterativeData"

UserDefinedData_ts            = "UserDefinedData_t"

# ---
cgnsnames=[globals()[k] for k in dir() if (k[-2:]=='_s')]
cgnstypes=[globals()[k] for k in dir() if (k[-3:]=='_ts')]
cgnsenums={}
for k in dir():
  if (k[-2:]=='_l'): cgnsenums[k[:-1]+'t']=locals()[k]
#
cgnsnames.sort()
cgnstypes.sort()
#
# Compatibility
# - some constants have no integer/string correspondance, the legacy wrapper
#   had its own defines, try to set them as long as it doens't collide with
#   existing constants.
#
CoordinateX=CoordinateX_s
CoordinateY=CoordinateY_s
CoordinateZ=CoordinateZ_s

Zone_t=Zone_ts
ReferenceState_t=ReferenceState_ts
FlowEquationSet_t=FlowEquationSet_ts
GoverningEquations_t=GoverningEquations_ts
GasModel_t=GasModel_ts
ViscosityModel_t=ViscosityModel_ts
TurbulenceModel_t=TurbulenceModel_ts
UserDefinedData_t=UserDefinedData_ts
FlowSolution_t=FlowSolution_ts
DataArray_t=DataArray_ts
Family_t=Family_ts
DiscreteData_t=DiscreteData_ts
#
# --- last line

