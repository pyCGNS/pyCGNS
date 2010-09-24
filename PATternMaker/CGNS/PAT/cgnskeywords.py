#  ---------------------------------------------------------------------------
#  pyCGNS.PAT - Python package for CFD General Notation System - PATternMaker
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#  $Release$
#  ---------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# All enumerates, constants, magic values from CGNS SIDS, MLL, ADF and more...
#
# TYPES, ENUMERATES, CONSTANTS, NAMES from CGNS/SIDS v2.5.3
#
#
# READ THIS PLEASE:
#
# [1] A CGNS/SIDS string constant is postfixed with _s
# 'ZoneType' is ZoneType_s
#
# [2] A CGNS/SIDS string constant repersenting a type has _ts
# 'ZoneType_t' is ZoneType_ts
#
# [3] A list of possible values for a given type has _l
# ZoneType_l is [Structured_s,Unstructured_s,Null_s,UserDefined_s]
# which is same as ["Structured","Unstructured","Null","UserDefined"]
#
# [4] An enumerate mapping of a list of values is not prefixed
# ZoneType is {'Unstructured':3,'Null':0,'Structured':2,'UserDefined':1}
#
# [5] The reverse dictionnary of the previous one is postfixed with _
# ZoneType_ is {0:'Null',1:'UserDefined',2:'Structured',3:'Unstructured'}
#
# ----------------------------------------------------------------------------
#
import CGNS.pyCGNSconfig

try:
  CGNS_VERSION = int(float(CGNS.pyCGNSconfig.MLL_VERSION))
  CGNS_DOTVERS = CGNS_VERSION/1000.
except TypeError:
  CGNS_VERSION = 2520
  CGNS_DOTVERS = 2.52

MODE_READ = 0
MODE_WRITE = 1

# version switch to add here
if (CGNS_VERSION<3000):
  MODE_MODIFY = 3
  MODE_CLOSED = 2
else:
  MODE_MODIFY = 2
  MODE_CLOSED = 3

CG_OK = 0
CG_ERROR = 1
CG_NODE_NOT_FOUND = 2
CG_INCORRECT_PATH = 3
CG_NO_INDEX_DIM = 4
Null = 0
UserDefined = 1
CG_FILE_NONE=0
CG_FILE_ADF =1
CG_FILE_HDF5=2
CG_FILE_XML =3

# --------------------------------------------------
# NOT SIDS
#
CGNSTree_ts       = 'CGNSTree_t'
CGNSTree_s        = 'CGNSTree'
DiffusionModel_ts = '"int[1+...+IndexDimension]"'
DiffusionModel_s  = 'DiffusionModel'

# --------------------------------------------------
# SIDS
#
Null_s = "Null"
UserDefined_s = "Userdefined"

# --------------------------------------------------
Kilogram_s = "Kilogram"
Gram_s = "Gram"
Slug_s = "Slug"
PoundMass_s = "PoundMass"
MassUnits_l=[Kilogram_s,Gram_s,Slug_s,PoundMass_s,Null_s,UserDefined_s]

# --------------------------------------------------
Meter_s = "Meter"
Centimeter_s = "Centimeter"
Millimeter_s = "Millimeter"
Foot_s = "Foot"
Inch_s = "Inch"
LengthUnits_l=[Meter_s,Centimeter_s,Millimeter_s,Foot_s,Inch_s,
               Null_s,UserDefined_s]

# --------------------------------------------------
Second_s = "Second"
TimeUnits_l=[Second_s,Null_s,UserDefined_s]

# --------------------------------------------------
Kelvin_s = "Kelvin"
Celcius_s = "Celcius"
Rankine_s = "Rankine"
Fahrenheit_s = "Fahrenheit"
TemperatureUnits_l=[Kelvin_s,Celcius_s,Rankine_s,Fahrenheit_s,
                    Null_s,UserDefined_s]

# --------------------------------------------------
Degree_s = "Degree"
Radian_s = "Radian"
AngleUnits_l=[Degree_s,Radian_s,Null_s,UserDefined_s]

# --------------------------------------------------
Ampere_s = "Ampere"
Abampere_s = "Abampere"
Statampere_s = "Statampere"
Edison_s = "Edison"
auCurrent_s = "auCurrent"

# --------------------------------------------------
Mole_s = "Mole"
Entities_s = "Entities"
StandardCubicFoot_s = "StandardCubicFoot"
StandardCubicMeter_s = "StandardCubicMeter"
  
# --------------------------------------------------
Candela_s = "Candela"
Candle_s = "Candle"
Carcel_s = "Carcel"
Hefner_s = "Hefner"
Violle_s = "Violle"
LuminousIntensityUnits_l=[Candela_s,Candle_s,Carcel_s,Hefner_s,
                          Violle_s,Null_s,UserDefined_s]

DimensionalUnits_s="DimensionalUnits"
AdditionalUnits_s="AdditionalUnits"
AllUnits_l=TimeUnits_l+MassUnits_l+LengthUnits_l+TemperatureUnits_l+AngleUnits_l

# --------------------------------------------------
Dimensional_s = "Dimensional"
NormalizedByDimensional_s = "NormalizedByDimensional"
NormalizedByUnknownDimensional_s = "NormalizedByUnknownDimensional"
NondimensionalParameter_s = "NondimensionalParameter"
DimensionlessConstant_s = "DimensionlessConstant"
DataClass_l=[Dimensional_s,NormalizedByDimensional_s,
             NormalizedByUnknownDimensional_s,NondimensionalParameter_s,
             DimensionlessConstant_s,Null_s,UserDefined_s]
DataClass_ts = "DataClass_t"
DataClass_s = "DataClass"

# ------------------------------------------------------------
NofValidGridLocation = 9
Vertex_s      = "Vertex"
CellCenter_s  = "CellCenter"
FaceCenter_s  = "FaceCenter"
IFaceCenter_s = "IFaceCenter"
JFaceCenter_s = "JFaceCenter"
KFaceCenter_s = "KFaceCenter"
EdgeCenter_s  = "EdgeCenter"

GridLocation_s= "GridLocation"

GridLocation_l=[CellCenter_s,Vertex_s,FaceCenter_s,
                IFaceCenter_s,JFaceCenter_s,KFaceCenter_s,
                EdgeCenter_s,Null_s,UserDefined_s]

# ------------------------------------------------------------
DirichletData_s = "DirichletData"
NeumannData_s = "NeumannData"
NofValidBCDataTypes = 4
Dirichlet_s = "Dirichlet"
Neumann_s = "Neumann"
NofValidGridConnectivityTypes = 5
Overset_s = "Overset"
Abutting_s = "Abutting"
Abutting1to1_s = "Abutting1to1"

GridConnectivityType_l=[Overset_s,Abutting_s,Abutting1to1_s,Null_s,UserDefined_s]


NofValidPointSetTypes = 9

PointList_s = "PointList"
PointListDonor_s = "PointListDonor"
PointRange_s = "PointRange"
PointRangeDonor_s = "PointRangeDonor"
ElementRange_s = "ElementRange"
ElementList_s = "ElementList"
CellListDonor_s = "CellListDonor"

NofValidGoverningEquationsTypes = 8
FullPotential_s = "FullPotential"
Euler_s = "Euler"
NSLaminar_s = "NSLaminar"
NSTurbulent_s = "NSTurbulent"
NSLaminarIncompressible_s = "NSLaminarIncompressible"
NSTurbulentIncompressible_s = "NSTurbulentIncompressible"
NofValidModelTypes = 36
Ideal_s = "Ideal"
VanderWaals_s = "VanderWaals"
Constant_s = "Constant"
PowerLaw_s = "PowerLaw"
SutherlandLaw_s = "SutherlandLaw"
ConstantPrandtl_s = "ConstantPrandtl"
EddyViscosity_s = "EddyViscosity"
ReynoldsStress_s = "ReynoldsStress"
Algebraic_s = "Algebraic"
BaldwinLomax_s = "BaldwinLomax"
ReynoldsStressAlgebraic_s = "ReynoldsStressAlgebraic"
Algebraic_BaldwinLomax_s = "Algebraic_BaldwinLomax"
Algebraic_CebeciSmith_s = "Algebraic_CebeciSmith"
HalfEquation_JohnsonKing_s = "HalfEquation_JohnsonKing"
OneEquation_BaldwinBarth_s = "OneEquation_BaldwinBarth"
OneEquation_SpalartAllmaras_s = "OneEquation_SpalartAllmaras"
TwoEquation_JonesLaunder_s = "TwoEquation_JonesLaunder"
TwoEquation_MenterSST_s = "TwoEquation_MenterSST"
TwoEquation_Wilcox_s = "TwoEquation_Wilcox"
CaloricallyPerfect_s = "CaloricallyPerfect"
ThermallyPerfect_s = "ThermallyPerfect"
ConstantDensity_s = "ConstantDensity"
RedlichKwong_s = "RedlichKwong"
Frozen_s = "Frozen"
ThermalEquilib_s = "ThermalEquilib"
ThermalNonequilib_s = "ThermalNonequilib"
ChemicalEquilibCurveFit_s = "ChemicalEquilibCurveFit"
ChemicalEquilibMinimization_s = "ChemicalEquilibMinimization"
ChemicalNonequilib_s = "ChemicalNonequilib"
EMElectricField_s = "EMElectricField"
EMMagneticField_s = "EMMagneticField"
EMConductivity_s = "EMConductivity"
Voltage_s = "Voltage"
Interpolated_s = "Interpolated"
Equilibrium_LinRessler_s = "Equilibrium_LinRessler"
Chemistry_LinRessler_s = "Chemistry_LinRessler"
NofValidBCTypes = 26
BCAxisymmetricWedge_s = "BCAxisymmetricWedge"
BCDegenerateLine_s = "BCDegenerateLine"
BCDegeneratePoint_s = "BCDegeneratePoint"
BCDirichlet_s = "BCDirichlet"
BCExtrapolate_s = "BCExtrapolate"
BCFarfield_s = "BCFarfield"
BCGeneral_s = "BCGeneral"
BCInflow_s = "BCInflow"
BCInflowSubsonic_s = "BCInflowSubsonic"
BCInflowSupersonic_s = "BCInflowSupersonic"
BCNeumann_s = "BCNeumann"
BCOutflow_s = "BCOutflow"
BCOutflowSubsonic_s = "BCOutflowSubsonic"
BCOutflowSupersonic_s = "BCOutflowSupersonic"
BCSymmetryPlane_s = "BCSymmetryPlane"
BCSymmetryPolar_s = "BCSymmetryPolar"
BCTunnelInflow_s = "BCTunnelInflow"
BCTunnelOutflow_s = "BCTunnelOutflow"
BCWall_s = "BCWall"
BCWallInviscid_s = "BCWallInviscid"
BCWallViscous_s = "BCWallViscous"
BCWallViscousHeatFlux_s = "BCWallViscousHeatFlux"
BCWallViscousIsothermal_s = "BCWallViscousIsothermal"
FamilySpecified_s = "FamilySpecified"
NofValidDataTypes = 6
Integer_s = "Integer"
RealSingle_s = "RealSingle"
RealDouble_s = "RealDouble"
Character_s = "Character"
NofValidElementTypes = 22
NODE_s = "NODE"
BAR_2_s = "BAR_2"
BAR_3_s = "BAR_3"
TRI_3_s = "TRI_3"
TRI_6_s = "TRI_6"
QUAD_4_s = "QUAD_4"
QUAD_8_s = "QUAD_8"
QUAD_9_s = "QUAD_9"
TETRA_4_s = "TETRA_4"
TETRA_10_s = "TETRA_10"
PYRA_5_s = "PYRA_5"
PYRA_14_s = "PYRA_14"
PENTA_6_s = "PENTA_6"
PENTA_15_s = "PENTA_15"
PENTA_18_s = "PENTA_18"
HEXA_8_s = "HEXA_8"
HEXA_20_s = "HEXA_20"
HEXA_27_s = "HEXA_27"
MIXED_s = "MIXED"
NGON_n_s = "NGON_n"

# --------------------------------------------------
Structured = Structured_s = "Structured"
Unstructured_s = "Unstructured"
ZoneType_s = "ZoneType"
ZoneType_l=[Structured_s,Unstructured_s,Null_s,UserDefined_s]

# --------------------------------------------------
TimeAccurate_s = "TimeAccurate"
NonTimeAccurate_s = "NonTimeAccurate"
SimulationType_ts = "SimulationType_t"
SimulationType_s = "SimulationType"
SimulationType_l=[TimeAccurate_s,NonTimeAccurate_s,Null_s,UserDefined_s]



# --------------------------------------------------
NofValidRigidGridMotionTypes = 4
ConstantRate_s = "ConstantRate"
VariableRate_s = "VariableRate"
NofValidArbitraryGridMotionTypes = 4
NonDeformingGrid_s = "NonDeformingGrid"
DeformingGrid_s = "DeformingGrid"


RigidGridMotionType_l = [Null_s,ConstantRate_s,VariableRate_s,UserDefined_s]
RigidGridMotionType_s="RigidGridMotionType"
RigidGridMotionType_ts="RigidGridMotionType_t"

NofValidWallFunctionTypes = 3
Generic_s = "Generic"
NofValidAreaTypes = 4
BleedArea_s = "BleedArea"
CaptureArea_s = "CaptureArea"
NofValidAverageInterfaceTypes = 8
AverageAll_s = "AverageAll"
AverageCircumferential_s = "AverageCircumferential"
AverageRadial_s = "AverageRadial"
AverageI_s = "AverageI"
AverageJ_s = "AverageJ"
AverageK_s = "AverageK"
CGNSLibraryVersion_s = "CGNSLibraryVersion"
GridCoordinates_s = "GridCoordinates"
ZoneGridConnectivity_s = "ZoneGridConnectivity"
CoordinateNames_s = "CoordinateNames"
CoordinateX_s = "CoordinateX"
CoordinateY_s = "CoordinateY"
CoordinateZ_s = "CoordinateZ"
CoordinateR_s = "CoordinateR"
CoordinateTheta_s = "CoordinateTheta"
CoordinatePhi_s = "CoordinatePhi"
CoordinateNormal_s = "CoordinateNormal"
CoordinateTangential_s = "CoordinateTangential"
CoordinateXi_s = "CoordinateXi"
CoordinateEta_s = "CoordinateEta"
CoordinateZeta_s = "CoordinateZeta"
CoordinateTransform_s = "CoordinateTransform"
InterpolantsDonor_s = "InterpolantsDonor"
ElementConnectivity_s = "ElementConnectivity"
ParentData_s = "ParentData"
VectorX_ps = "%sX"
VectorY_ps = "%sY"
VectorZ_ps = "%sZ"
VectorTheta_ps = "%sTheta"
VectorPhi_ps = "%sPhi"
VectorMagnitude_ps = "%sMagnitude"
VectorNormal_ps = "%sNormal"
VectorTangential_ps = "%sTangential"
Potential_s = "Potential"
StreamFunction_s = "StreamFunction"
Density_s = "Density"
Pressure_s = "Pressure"
Temperature_s = "Temperature"
EnergyInternal_s = "EnergyInternal"
Enthalpy_s = "Enthalpy"
Entropy_s = "Entropy"
EntropyApprox_s = "EntropyApprox"
DensityStagnation_s = "DensityStagnation"
PressureStagnation_s = "PressureStagnation"
TemperatureStagnation_s = "TemperatureStagnation"
EnergyStagnation_s = "EnergyStagnation"
EnthalpyStagnation_s = "EnthalpyStagnation"
EnergyStagnationDensity_s = "EnergyStagnationDensity"
VelocityX_s = "VelocityX"
VelocityY_s = "VelocityY"
VelocityZ_s = "VelocityZ"
VelocityR_s = "VelocityR"
VelocityTheta_s = "VelocityTheta"
VelocityPhi_s = "VelocityPhi"
VelocityMagnitude_s = "VelocityMagnitude"
VelocityNormal_s = "VelocityNormal"
VelocityTangential_s = "VelocityTangential"
VelocitySound_s = "VelocitySound"
VelocitySoundStagnation_s = "VelocitySoundStagnation"
MomentumX_s = "MomentumX"
MomentumY_s = "MomentumY"
MomentumZ_s = "MomentumZ"
MomentumMagnitude_s = "MomentumMagnitude"
RotatingVelocityX_s = "RotatingVelocityX"
RotatingVelocityY_s = "RotatingVelocityY"
RotatingVelocityZ_s = "RotatingVelocityZ"
RotatingMomentumX_s = "RotatingMomentumX"
RotatingMomentumY_s = "RotatingMomentumY"
RotatingMomentumZ_s = "RotatingMomentumZ"
RotatingVelocityMagnitude_s = "RotatingVelocityMagnitude"
RotatingPressureStagnation_s = "RotatingPressureStagnation"
RotatingEnergyStagnation_s = "RotatingEnergyStagnation"
RotatingEnergyStagnationDensity_s = "RotatingEnergyStagnationDensity"
RotatingEnthalpyStagnation_s = "RotatingEnthalpyStagnation"
EnergyKinetic_s = "EnergyKinetic"
PressureDynamic_s = "PressureDynamic"
SoundIntensityDB_s = "SoundIntensityDB"
SoundIntensity_s = "SoundIntensity"
VorticityX_s = "VorticityX"
VorticityY_s = "VorticityY"
VorticityZ_s = "VorticityZ"
VorticityMagnitude_s = "VorticityMagnitude"
SkinFrictionX_s = "SkinFrictionX"
SkinFrictionY_s = "SkinFrictionY"
SkinFrictionZ_s = "SkinFrictionZ"
SkinFrictionMagnitude_s = "SkinFrictionMagnitude"
VelocityAngleX_s = "VelocityAngleX"
VelocityAngleY_s = "VelocityAngleY"
VelocityAngleZ_s = "VelocityAngleZ"
VelocityUnitVectorX_s = "VelocityUnitVectorX"
VelocityUnitVectorY_s = "VelocityUnitVectorY"
VelocityUnitVectorZ_s = "VelocityUnitVectorZ"
MassFlow_s = "MassFlow"
ViscosityKinematic_s = "ViscosityKinematic"
ViscosityMolecular_s = "ViscosityMolecular"
ViscosityEddyDynamic_s = "ViscosityEddyDynamic"
ViscosityEddy_s = "ViscosityEddy"
ThermalConductivity_s = "ThermalConductivity"
PowerLawExponent_s = "PowerLawExponent"
SutherlandLawConstant_s = "SutherlandLawConstant"
TemperatureReference_s = "TemperatureReference"
ViscosityMolecularReference_s = "ViscosityMolecularReference"
ThermalConductivityReference_s = "ThermalConductivityReference"
IdealGasConstant_s = "IdealGasConstant"
SpecificHeatPressure_s = "SpecificHeatPressure"
SpecificHeatVolume_s = "SpecificHeatVolume"
ReynoldsStressXX_s = "ReynoldsStressXX"
ReynoldsStressXY_s = "ReynoldsStressXY"
ReynoldsStressXZ_s = "ReynoldsStressXZ"
ReynoldsStressYY_s = "ReynoldsStressYY"
ReynoldsStressYZ_s = "ReynoldsStressYZ"
ReynoldsStressZZ_s = "ReynoldsStressZZ"
LengthReference_s = "LengthReference"
MolecularWeight_s = "MolecularWeight"
MolecularWeight_ps = "MolecularWeight%s"
HeatOfFormation_s = "HeatOfFormation"
HeatOfFormation_ps = "HeatOfFormation%s"
FuelAirRatio_s = "FuelAirRatio"
ReferenceTemperatureHOF_s = "ReferenceTemperatureHOF"
MassFraction_s = "MassFraction"
MassFraction_ps = "MassFraction%s"
LaminarViscosity_s = "LaminarViscosity"
LaminarViscosity_ps = "LaminarViscosity%s"
ThermalConductivity_ps = "ThermalConductivity%s"
EnthalpyEnergyRatio_s = "EnthalpyEnergyRatio"
CompressibilityFactor_s = "CompressibilityFactor"
VibrationalElectronEnergy_s = "VibrationalElectronEnergy"
VibrationalElectronTemperature_s = "VibrationalElectronTemperature"
SpeciesDensity_s = "SpeciesDensity"
SpeciesDensity_ps = "SpeciesDensity%s"
MoleFraction_s = "MoleFraction"
MoleFraction_ps = "MoleFraction%s"
ElectricFieldX_s = "ElectricFieldX"
ElectricFieldY_s = "ElectricFieldY"
ElectricFieldZ_s = "ElectricFieldZ"
MagneticFieldX_s = "MagneticFieldX"
MagneticFieldY_s = "MagneticFieldY"
MagneticFieldZ_s = "MagneticFieldZ"
CurrentDensityX_s = "CurrentDensityX"
CurrentDensityY_s = "CurrentDensityY"
CurrentDensityZ_s = "CurrentDensityZ"
LorentzForceX_s = "LorentzForceX"
LorentzForceY_s = "LorentzForceY"
LorentzForceZ_s = "LorentzForceZ"
ElectricConductivity_s = "ElectricConductivity"
JouleHeating_s = "JouleHeating"
TurbulentDistance_s = "TurbulentDistance"
TurbulentEnergyKinetic_s = "TurbulentEnergyKinetic"
TurbulentDissipation_s = "TurbulentDissipation"
TurbulentDissipationRate_s = "TurbulentDissipationRate"
TurbulentBBReynolds_s = "TurbulentBBReynolds"
TurbulentSANuTilde_s = "TurbulentSANuTilde"
Mach_s = "Mach"
Mach_Velocity_s = "Mach_Velocity"
Mach_VelocitySound_s = "Mach_VelocitySound"
Reynolds_s = "Reynolds"
Reynolds_Velocity_s = "Reynolds_Velocity"
Reynolds_Length_s = "Reynolds_Length"
Reynolds_ViscosityKinematic_s = "Reynolds_ViscosityKinematic"
Prandtl_s = "Prandtl"
Prandtl_ThermalConductivity_s = "Prandtl_ThermalConductivity"
Prandtl_ViscosityMolecular_s = "Prandtl_ViscosityMolecular"
Prandtl_SpecificHeatPressure_s = "Prandtl_SpecificHeatPressure"
PrandtlTurbulent_s = "PrandtlTurbulent"
SpecificHeatRatio_s = "SpecificHeatRatio"
SpecificHeatRatio_Pressure_s = "SpecificHeatRatio_Pressure"
SpecificHeatRatio_Volume_s = "SpecificHeatRatio_Volume"
CoefPressure_s = "CoefPressure"
CoefSkinFrictionX_s = "CoefSkinFrictionX"
CoefSkinFrictionY_s = "CoefSkinFrictionY"
CoefSkinFrictionZ_s = "CoefSkinFrictionZ"
Coef_PressureDynamic_s = "Coef_PressureDynamic"
Coef_PressureReference_s = "Coef_PressureReference"
Vorticity_s = "Vorticity"
Acoustic_s = "Acoustic"
RiemannInvariantPlus_s = "RiemannInvariantPlus"
RiemannInvariantMinus_s = "RiemannInvariantMinus"
CharacteristicEntropy_s = "CharacteristicEntropy"
CharacteristicVorticity1_s = "CharacteristicVorticity1"
CharacteristicVorticity2_s = "CharacteristicVorticity2"
CharacteristicAcousticPlus_s = "CharacteristicAcousticPlus"
CharacteristicAcousticMinus_s = "CharacteristicAcousticMinus"
ForceX_s = "ForceX"
ForceY_s = "ForceY"
ForceZ_s = "ForceZ"
ForceR_s = "ForceR"
ForceTheta_s = "ForceTheta"
ForcePhi_s = "ForcePhi"
Lift_s = "Lift"
Drag_s = "Drag"
MomentX_s = "MomentX"
MomentY_s = "MomentY"
MomentZ_s = "MomentZ"
MomentR_s = "MomentR"
MomentTheta_s = "MomentTheta"
MomentPhi_s = "MomentPhi"
MomentXi_s = "MomentXi"
MomentEta_s = "MomentEta"
MomentZeta_s = "MomentZeta"
Moment_CenterX_s = "Moment_CenterX"
Moment_CenterY_s = "Moment_CenterY"
Moment_CenterZ_s = "Moment_CenterZ"
CoefLift_s = "CoefLift"
CoefDrag_s = "CoefDrag"
CoefMomentX_s = "CoefMomentX"
CoefMomentY_s = "CoefMomentY"
CoefMomentZ_s = "CoefMomentZ"
CoefMomentR_s = "CoefMomentR"
CoefMomentTheta_s = "CoefMomentTheta"
CoefMomentPhi_s = "CoefMomentPhi"
CoefMomentXi_s = "CoefMomentXi"
CoefMomentEta_s = "CoefMomentEta"
CoefMomentZeta_s = "CoefMomentZeta"
Coef_PressureDynamic_s = "Coef_PressureDynamic"
Coef_Area_s = "Coef_Area"
Coef_Length_s = "Coef_Length"
TimeValues_s = "TimeValues"
IterationValues_s = "IterationValues"
NumberOfZones_s = "NumberOfZones"
NumberOfFamilies_s = "NumberOfFamilies"
DataConversion_s ="DataConversion"

ZonePointers_s = "ZonePointers"
FamilyPointers_s = "FamilyPointers"
RigidGridMotionPointers_s = "RigidGridMotionPointers"
ArbitraryGridMotionPointers_s = "ArbitraryGridMotionPointers"
GridCoordinatesPointers_s = "GridCoordinatesPointers"
FlowSolutionsPointers_s = "FlowSolutionsPointers"
PointerNames_l=[ZonePointers_s,FamilyPointers_s,RigidGridMotionPointers_s,
                ArbitraryGridMotionPointers_s,GridCoordinatesPointers_s,
                FlowSolutionsPointers_s]

OriginLocation_s = "OriginLocation"
RigidRotationAngle_s = "RigidRotationAngle"
Translation_s = "Translation"
RotationAngle_s = "RotationAngle"
RigidVelocity_s = "RigidVelocity"
RigidRotationRate_s = "RigidRotationRate"
GridVelocityX_s = "GridVelocityX"
GridVelocityY_s = "GridVelocityY"
GridVelocityZ_s = "GridVelocityZ"
GridVelocityR_s = "GridVelocityR"
GridVelocityTheta_s = "GridVelocityTheta"
GridVelocityPhi_s = "GridVelocityPhi"
GridVelocityXi_s = "GridVelocityXi"
GridVelocityEta_s = "GridVelocityEta"
GridVelocityZeta_s = "GridVelocityZeta"

#chapter 11.3
ArbitraryGridMotion_ts = "ArbitraryGridMotion_t"
ArbitraryGridMotion_s = "ArbitraryGridMotion"
ArbitraryGridMotionType_l=[Null_s,NonDeformingGrid_s,
                           DeformingGrid_s,UserDefined_s]
ArbitraryGridMotionType_s="ArbitraryGridMotionType"
ArbitraryGridMotionType_ts="ArbitraryGridMotionType_t"
#

Area_ts = "Area_t"
Area_s = "Area"
AreaType_ts = "AreaType_t"
AreaType_s = "AreaType"
SurfaceArea_s="SurfaceArea"
RegionName_s="RegionName"
AverageInterface_ts = "AverageInterface_t"
Axisymmetry_ts = "Axisymmetry_t"
Axisymmetry_s = "Axisymmetry"
AxisymmetryReferencePoint_s = "AxisymmetryReferencePoint"
AxisymmetryAxisVector_s = "AxisymmetryAxisVector"
AxisymmetryAngle_s = "AxisymmetryAngle"
BCDataSet_ts = "BCDataSet_t"
BCData_ts = "BCData_t"
BCData_s = "BCData"

BCProperty_ts = "BCProperty_t"
BCProperty_s = "BCProperty"
BC_ts = "BC_t"

BaseIterativeData_ts = "BaseIterativeData_t"
BaseIterativeData_s = "BaseIterativeData"

CGNSBase_ts = "CGNSBase_t"
CGNSLibraryVersion_ts = "CGNSLibraryVersion_t"


# --------------------------------------------------
ConvergenceHistory_ts = "ConvergenceHistory_t"
ZoneConvergenceHistory_s  = "ZoneConvergenceHistory"
GlobalConvergenceHistory_s  = "GlobalConvergenceHistory"

ConvergenceHistory_l=[ZoneConvergenceHistory_s,GlobalConvergenceHistory_s]
NormDefinitions_s="NormDefinitions"

DataArray_ts = "DataArray_t"
DataConversion_ts = "DataConversion_t"
Descriptor_ts = "Descriptor_t"

# --------------------------------------------------
DimensionalExponents_ts = "DimensionalExponents_t"
DimensionalExponents_s = "DimensionalExponents"
DimensionalUnits_ts = "DimensionalUnits_t"
AdditionalUnits_ts = "AdditionalUnits_t"

DiscreteData_ts = "DiscreteData_t"
DiscreteData_s= "DiscreteData"
Elements_ts = "Elements_t"

FamilyBC_s = "FamilyBC"
FamilyBC_ts = "FamilyBC_t"

FamilyName_ts = "FamilyName_t"
FamilyName_s = "FamilyName"
Family_ts = "Family_t"
Family_s = "Family"
FlowEquationSet_ts = "FlowEquationSet_t"
FlowEquationSet_s = "FlowEquationSet"
FlowSolution_ts = "FlowSolution_t"
GasModel_ts = "GasModel_t"
GasModel_s = "GasModel"
#
GeometryEntity_ts = "GeometryEntity_t"
GeometryFile_ts = "GeometryFile_t"

#chapter 12.7
GeometryFormat_s = "GeometryFormat"
GeometryFormat_ts = "GeometryFormat_t"
# not supported '-'
NASAIGES_s="NASA-IGES"
SDRC_s="SDRC"
Unigraphics_s="Unigraphics"
ProEngineer_s="ProEngineer"
ICEMCFD_s="ICEM-CFD"
GeometryFormat_l=[Null_s,NASAIGES_s,SDRC_s,Unigraphics_s,
                  ProEngineer_s,ICEMCFD_s,UserDefined_s]
GeometryReference_ts = "GeometryReference_t"
GeometryReference_s = "GeometryReference"
#

Gravity_ts = "Gravity_t"
Gravity_s  = "Gravity"
GravityVector_s="GravityVector"

GridConnectivity1to1_ts = "GridConnectivity1to1_t"
GridConnectivityProperty_ts = "GridConnectivityProperty_t"
GridConnectivityProperty_s="GridConnectivityProperty"
GridConnectivityType_ts = "GridConnectivityType_t"
GridConnectivityType_s = "GridConnectivityType"
GridConnectivity_ts = "GridConnectivity_t"

GridCoordinates_ts = "GridCoordinates_t"
GridLocation_ts = "GridLocation_t"
IndexArray_ts = "IndexArray_t"
IndexRange_ts = "IndexRange_t"
IntegralData_ts = "IntegralData_t"
InwardNormalList_ts = "InwardNormalList_t"
InwardNormalList_s = "InwardNormalList"
InwardNormalIndex_s = "InwardNormalIndex"
Ordinal_ts = "Ordinal_t"
Ordinal_s = "Ordinal"
Transform_s = "Transform"
OversetHoles_ts = "OversetHoles_t"
OversetHoles_s = "OversetHoles"
Periodic_ts = "Periodic_t"
Periodic_s = "Periodic"

ReferenceState_ts = "ReferenceState_t"
ReferenceState_s = "ReferenceState"
ReferenceStateDescription_s = "ReferenceStateDescription"

RigidGridMotion_ts = "RigidGridMotion_t"
RigidGridMotion_s = "RigidGridMotion"

Rind_ts = "Rind_t"
Rind_s  = "Rind"

RotatingCoordinates_ts = "RotatingCoordinates_t"
RotatingCoordinates_s = "RotatingCoordinates"
RotationRateVector_s="RotationRateVector"
RotationCenter_s="RotationCenter"


GoverningEquations_ts = "GoverningEquations_t"
#chapter 10.2
GoverningEquations_s = "GoverningEquations"
GoverningEquationsType_l=[Euler_s,NSLaminar_s,NSTurbulent_s]
GoverningEquationsType_s="GoverningEquationsType"
GoverningEquationsType_ts="GoverningEquationsType_t"
#

#chapter 9.7
BCType_s="BCType"
BCType_ts="BCType_t"
BCTypeSimple_l=[Null_s, BCGeneral_s, BCDirichlet_s, BCNeumann_s,
                BCExtrapolate_s, BCWallInviscid_s, BCWallViscousHeatFlux_s,
                BCWallViscousIsothermal_s, BCWallViscous_s, BCWall_s,
                BCInflowSubsonic_s, BCInflowSupersonic_s, BCOutflowSubsonic_s,
                BCOutflowSupersonic_s, BCTunnelInflow_s, BCTunnelOutflow_s,
                BCDegenerateLine_s, BCDegeneratePoint_s, BCSymmetryPlane_s,
                BCSymmetryPolar_s, BCAxisymmetricWedge_s, FamilySpecified_s,
                UserDefined_s]
BCTypeSimple_s="BCTypeSimple"
BCTypeSimple_ts="BCTypeSimple_t"

BCTypeCompound_l=[Null_s, BCInflow_s, BCOutflow_s, BCFarfield_s, UserDefined_s]

ThermalConductivityModel_ts = "ThermalConductivityModel_t"
# chapter 10.4
ThermalConductivityModel_s = "ThermalConductivityModel"
ThermalConductivityModelType_l=[Null_s,ConstantPrandtl_s,PowerLaw_s,
                              SutherlandLaw_s,UserDefined_s]
ThermalConductivityModelType_s="ThermalConductivityModelType"
ThermalConductivityModelType_ts="ThermalConductivityModelType_t"
ThermalConductivityModelIdentifier_l=[(Prandtl_s),(PowerLawExponent_s),(SutherlandLawConstant_s),(TemperatureReference_s),(ThermalConductivityReference_s)]
#

TurbulenceClosure_ts = "TurbulenceClosure_t"
# chapter 10.5
TurbulenceClosure_s = "TurbulenceClosure"
TurbulenceClosureType_l=[Null_s,EddyViscosity_s,ReynoldsStress_s,
                         ReynoldsStressAlgebraic_s,UserDefined_s]
TurbulenceClosureType_s="TurbulenceClosureType"
TurbulenceClosureType_ts="TurbulenceClosureType_t"
TurbulenceClosureIdentifier_l=[PrandtlTurbulent_s]
#

TurbulenceModel_ts = "TurbulenceModel_t"
# chapter 10.6
TurbulenceModel_s = "TurbulenceModel"
TurbulenceModelType_l=[Null_s,Algebraic_BaldwinLomax_s,Algebraic_CebeciSmith_s,
                       HalfEquation_JohnsonKing_s,OneEquation_BaldwinBarth_s,
                       OneEquation_SpalartAllmaras_s,TwoEquation_JonesLaunder_s,
                       TwoEquation_MenterSST_s,TwoEquation_Wilcox_s]
TurbulenceModelType_s="TurbulenceModelType"
TurbulenceModelType_ts="TurbulenceModelType_t"

ViscosityModel_ts = "ViscosityModel_t"
# chapter 10.4
ViscosityModel_s = "ViscosityModel"
ViscosityModelType_l=[Null_s,Constant_s,PowerLaw_s,SutherlandLaw_s,UserDefined_s]
ViscosityModelType_s="ViscosityModelType"
ViscosityModelType_ts="ViscosityModelType_t"
ViscosityModelIdentifier_l=[(PowerLawExponent_s),(SutherlandLawConstant_s),
                          (TemperatureReference_s),
                          (ViscosityMolecularReference_s)]
#

# chapter 10.3
GasModelType_l=[Null_s,Ideal_s,VanderWaals_s,CaloricallyPerfect_s,
                ThermallyPerfect_s,ConstantDensity_s,RedlichKwong_s,
                UserDefined_s]
GasModelType_s="GasModelType"
GasModelType_ts="GasModelType_t"
GasModelIdentifier_l=[IdealGasConstant_s,SpecificHeatRatio_s,
                      SpecificHeatVolume_s,SpecificHeatPressure_s]
#

ThermalRelaxationModel_ts = "ThermalRelaxationModel_t"
# chapter 10.7
ThermalRelaxationModel_s = "ThermalRelaxationModel"
ThermalRelaxationModelType_l=[Null_s,Frozen_s,ThermalEquilib_s,
                              ThermalNonequilib_s,UserDefined_s]
ThermalRelaxationModelType_s="ThermalRelaxationModelType"
ThermalRelaxationModelType_ts="ThermalRelaxationModelType_t"
#

ChemicalKineticsModel_ts = "ChemicalKineticsModel_t"
# chapter 10.8
ChemicalKineticsModel_s = "ChemicalKineticsModel"
ChemicalKineticsModelType_l=[Null_s,Frozen_s,ChemicalEquilibCurveFit_s,
                             ChemicalEquilibMinimization_s,ChemicalNonequilib_s,
                             UserDefined_s]
ChemicalKineticsModelType_s="ChemicalKineticsModelType"
ChemicalKineticsModelType_ts="ChemicalKineticsModelType_t"
ChemicalKineticsModelIdentifier_l=[FuelAirRatio_s,ReferenceTemperatureHOF_s]
#

# chapter 10.9.1
EMElectricFieldModel_s="EMElectricFieldModel"
EMElectricFieldModel_ts="EMElectricFieldModel_t"
EMElectricFieldModelType_l=[Null_s,Constant_s,Frozen_s,
                            Interpolated_s,Voltage_s,UserDefined_s]
EMElectricFieldModelType_s="EMElectricFieldModelType"
EMElectricFieldModelType_ts="EMElectricFieldModelType_t"
#

# chapter 10.9.2
EMMagneticFieldModel_s="EMMagneticFieldModel"
EMMagneticFieldModel_ts="EMMagneticFieldModel_t"
EMMagneticFieldModelType_l=[Null_s,Constant_s,Frozen_s,
                          Interpolated_s,UserDefined_s]
EMMagneticFieldModelType_s="EMMagneticFieldModelType"
EMMagneticFieldModelType_ts="EMMagneticFieldModelType_t"
#

# chapter 10.9.3
EMConductivityModel_s="EMConductivityModel"
EMConductivityModel_ts="EMConductivityModel_t"
EMConductivityModelType_l=[Null_s,Constant_s,Frozen_s,Equilibrium_LinRessler_s,
                         Chemistry_LinRessler_s,UserDefined_s]
EMConductivityModelType_s="EMConductivityModelType"
EMConductivityModelType_ts="EMConductivityModelType_t"
EMConductivityModelIdentifier_l=[ElectricFieldX_s,ElectricFieldY_s,
                               ElectricFieldZ_s,MagneticFieldX_s,
                               MagneticFieldY_s,MagneticFieldZ_s,
                               CurrentDensityX_s,CurrentDensityY_s,
                               CurrentDensityZ_s,ElectricConductivity_s,
                               LorentzForceX_s,LorentzForceY_s,
                               LorentzForceZ_s,JouleHeating_s]
#

# chapter 8.5.2
AverageInterfaceType_s="AverageInterfaceType"
AverageInterfaceType_ts="AverageInterfaceType_t"
AverageInterfaceType_l=[Null_s,AverageAll_s,AverageCircumferential_s,
                      AverageRadial_s,AverageI_s,AverageJ_s,AverageK_s,
                      UserDefined_s]
AverageInterface_s="AverageInterface"
AverageInterface_ts="AverageInterface_t"
#

# chapter 10.9.3
Element_s="Element"
ElementType_l=[
Null_s, NODE_s, BAR_2_s, BAR_3_s,
TRI_3_s, TRI_6_s, QUAD_4_s, QUAD_8_s, QUAD_9_s,
TETRA_4_s, TETRA_10_s, PYRA_5_s, PYRA_14_s,
PENTA_6_s, PENTA_15_s, PENTA_18_s,
HEXA_8_s, HEXA_20_s, HEXA_27_s, MIXED_s, NGON_n_s, UserDefined_s]

Element_ts="Element_t"
ElementType_ts="ElementType_t"
ElementType_s="ElementType"
#

WallFunction_ts = "WallFunction_t"
WallFunction_s = "WallFunction"
WallFunctionType_ts = "WallFunctionType_t"
WallFunctionType_s = "WallFunctionType"
ZoneBC_ts = "ZoneBC_t"
ZoneBC_s = "ZoneBC"
ZoneGridConnectivity_ts = "ZoneGridConnectivity_t"
ZoneIterativeData_ts = "ZoneIterativeData_t"
ZoneIterativeData_s = "ZoneIterativeData"
ZoneType_ts = "ZoneType_t"
Zone_ts = "Zone_t"

UserDefinedData_ts = "UserDefinedData_t"

# ---
names=[
Abampere_s,
Abutting1to1_s,
Abutting_s,
Acoustic_s,
AdditionalUnits_s,
Algebraic_BaldwinLomax_s,
Algebraic_CebeciSmith_s,
Algebraic_s,
Ampere_s,
ArbitraryGridMotionPointers_s,
ArbitraryGridMotionType_s,
ArbitraryGridMotion_s,
Area_s,
AreaType_s,
AverageAll_s,
AverageCircumferential_s,
AverageI_s,
AverageInterfaceType_s,
AverageInterface_s,
AverageJ_s,
AverageK_s,
AverageRadial_s,
AxisymmetryAngle_s,
AxisymmetryAxisVector_s,
AxisymmetryReferencePoint_s,
Axisymmetry_s,
BAR_2_s,
BAR_3_s,
BCAxisymmetricWedge_s,
BCData_s,
BCDegenerateLine_s,
BCDegeneratePoint_s,
BCDirichlet_s,
BCExtrapolate_s,
BCFarfield_s,
BCGeneral_s,
BCInflowSubsonic_s,
BCInflowSupersonic_s,
BCInflow_s,
BCNeumann_s,
BCOutflowSubsonic_s,
BCOutflowSupersonic_s,
BCOutflow_s,
BCProperty_s,
BCSymmetryPlane_s,
BCSymmetryPolar_s,
BCTunnelInflow_s,
BCTunnelOutflow_s,
BCTypeSimple_s,
BCType_s,
BCWallInviscid_s,
BCWallViscousHeatFlux_s,
BCWallViscousIsothermal_s,
BCWallViscous_s,
BCWall_s,
BaldwinLomax_s,
BaseIterativeData_s,
ZoneIterativeData_s,
BleedArea_s,
CGNSLibraryVersion_s,
CaloricallyPerfect_s,
Candela_s,
Candle_s,
CaptureArea_s,
Carcel_s,
Celcius_s,
CellCenter_s,
CellListDonor_s,
Centimeter_s,
Character_s,
CharacteristicAcousticMinus_s,
CharacteristicAcousticPlus_s,
CharacteristicEntropy_s,
CharacteristicVorticity1_s,
CharacteristicVorticity2_s,
ChemicalEquilibCurveFit_s,
ChemicalEquilibMinimization_s,
ChemicalKineticsModelType_s,
ChemicalKineticsModel_s,
ChemicalNonequilib_s,
Chemistry_LinRessler_s,
CoefDrag_s,
CoefLift_s,
CoefMomentEta_s,
CoefMomentPhi_s,
CoefMomentR_s,
CoefMomentTheta_s,
CoefMomentX_s,
CoefMomentXi_s,
CoefMomentY_s,
CoefMomentZ_s,
CoefMomentZeta_s,
CoefPressure_s,
CoefSkinFrictionX_s,
CoefSkinFrictionY_s,
CoefSkinFrictionZ_s,
Coef_Area_s,
Coef_Length_s,
Coef_PressureDynamic_s,
Coef_PressureDynamic_s,
Coef_PressureReference_s,
CompressibilityFactor_s,
ConstantDensity_s,
ConstantPrandtl_s,
ConstantRate_s,
Constant_s,
CoordinateEta_s,
CoordinateNames_s,
CoordinateNormal_s,
CoordinatePhi_s,
CoordinateR_s,
CoordinateTangential_s,
CoordinateTheta_s,
CoordinateTransform_s,
CoordinateX_s,
CoordinateXi_s,
CoordinateY_s,
CoordinateZ_s,
CoordinateZeta_s,
CurrentDensityX_s,
CurrentDensityY_s,
CurrentDensityZ_s,
DataClass_s,
DataConversion_s,
DeformingGrid_s,
Degree_s,
DensityStagnation_s,
Density_s,
DimensionalExponents_s,
DimensionalUnits_s,
Dimensional_s,
DimensionlessConstant_s,
Dirichlet_s,
DiscreteData_s,
Drag_s,
EMConductivityModelType_s,
EMConductivityModel_s,
EMConductivity_s,
EMElectricFieldModelType_s,
EMElectricFieldModel_s,
EMElectricField_s,
EMMagneticFieldModelType_s,
EMMagneticFieldModel_s,
EMMagneticField_s,
EddyViscosity_s,
EdgeCenter_s,
Edison_s,
ElectricConductivity_s,
ElectricFieldX_s,
ElectricFieldY_s,
ElectricFieldZ_s,
ElementConnectivity_s,
ElementList_s,
ElementRange_s,
ElementType_s,
Element_s,
EnergyInternal_s,
EnergyKinetic_s,
EnergyStagnationDensity_s,
EnergyStagnation_s,
EnthalpyEnergyRatio_s,
EnthalpyStagnation_s,
Enthalpy_s,
Entities_s,
EntropyApprox_s,
Entropy_s,
Equilibrium_LinRessler_s,
Euler_s,
FaceCenter_s,
Fahrenheit_s,
FamilyBC_s,
FamilyPointers_s,
FamilySpecified_s,
Family_s,
FamilyName_s,
FlowEquationSet_s,
FlowSolutionsPointers_s,
Foot_s,
ForcePhi_s,
ForceR_s,
ForceTheta_s,
ForceX_s,
ForceY_s,
ForceZ_s,
Frozen_s,
FuelAirRatio_s,
FullPotential_s,
GasModelType_s,
GasModel_s,
Generic_s,
GeometryFormat_s,
GeometryReference_s,
GlobalConvergenceHistory_s,
GoverningEquationsType_s,
GoverningEquations_s,
Gram_s,
GravityVector_s,
Gravity_s,
GridConnectivityType_s,
GridConnectivityProperty_s,
GridCoordinatesPointers_s,
GridCoordinates_s,
GridLocation_s,
GridVelocityEta_s,
GridVelocityPhi_s,
GridVelocityR_s,
GridVelocityTheta_s,
GridVelocityX_s,
GridVelocityXi_s,
GridVelocityY_s,
GridVelocityZ_s,
GridVelocityZeta_s,
HEXA_20_s,
HEXA_27_s,
HEXA_8_s,
HalfEquation_JohnsonKing_s,
HeatOfFormation_s,
Hefner_s,
ICEMCFD_s,
IFaceCenter_s,
IdealGasConstant_s,
Ideal_s,
Inch_s,
Integer_s,
InterpolantsDonor_s,
Interpolated_s,
IterationValues_s,
JFaceCenter_s,
JouleHeating_s,
KFaceCenter_s,
Kelvin_s,
Kilogram_s,
LaminarViscosity_s,
LengthReference_s,
Lift_s,
LorentzForceX_s,
LorentzForceY_s,
LorentzForceZ_s,
MIXED_s,
Mach_VelocitySound_s,
Mach_Velocity_s,
Mach_s,
MagneticFieldX_s,
MagneticFieldY_s,
MagneticFieldZ_s,
MassFlow_s,
MassFraction_s,
Meter_s,
Millimeter_s,
MoleFraction_s,
Mole_s,
MolecularWeight_s,
MomentEta_s,
MomentPhi_s,
MomentR_s,
MomentTheta_s,
MomentX_s,
MomentXi_s,
MomentY_s,
MomentZ_s,
MomentZeta_s,
Moment_CenterX_s,
Moment_CenterY_s,
Moment_CenterZ_s,
MomentumMagnitude_s,
MomentumX_s,
MomentumY_s,
MomentumZ_s,
NASAIGES_s,
NGON_n_s,
NODE_s,
NSLaminarIncompressible_s,
NSLaminar_s,
NSTurbulentIncompressible_s,
NSTurbulent_s,
Neumann_s,
NonDeformingGrid_s,
NonTimeAccurate_s,
NondimensionalParameter_s,
NormDefinitions_s,
NormalizedByDimensional_s,
NormalizedByUnknownDimensional_s,
Null_s,
NumberOfFamilies_s,
NumberOfZones_s,
OneEquation_BaldwinBarth_s,
OneEquation_SpalartAllmaras_s,
Ordinal_s,
OriginLocation_s,
Overset_s,
PENTA_15_s,
PENTA_18_s,
PENTA_6_s,
PYRA_14_s,
PYRA_5_s,
ParentData_s,
Periodic_s,
PointListDonor_s,
PointList_s,
PointRangeDonor_s,
PointRange_s,
Potential_s,
PoundMass_s,
PowerLawExponent_s,
PowerLaw_s,
PrandtlTurbulent_s,
Prandtl_SpecificHeatPressure_s,
Prandtl_ThermalConductivity_s,
Prandtl_ViscosityMolecular_s,
Prandtl_s,
PressureDynamic_s,
PressureStagnation_s,
Pressure_s,
ProEngineer_s,
QUAD_4_s,
QUAD_8_s,
QUAD_9_s,
Radian_s,
Rankine_s,
RealDouble_s,
RealSingle_s,
RedlichKwong_s,
ReferenceStateDescription_s,
ReferenceState_s,
ReferenceTemperatureHOF_s,
RegionName_s,
ReynoldsStressAlgebraic_s,
ReynoldsStressXX_s,
ReynoldsStressXY_s,
ReynoldsStressXZ_s,
ReynoldsStressYY_s,
ReynoldsStressYZ_s,
ReynoldsStressZZ_s,
ReynoldsStress_s,
Reynolds_Length_s,
Reynolds_Velocity_s,
Reynolds_ViscosityKinematic_s,
Reynolds_s,
RiemannInvariantMinus_s,
RiemannInvariantPlus_s,
RigidGridMotionPointers_s,
RigidGridMotionType_s,
RigidGridMotion_s,
RigidRotationAngle_s,
RigidRotationRate_s,
RigidVelocity_s,
Rind_s,
RotatingCoordinates_s,
RotatingEnergyStagnationDensity_s,
RotatingEnergyStagnation_s,
RotatingEnthalpyStagnation_s,
RotatingMomentumX_s,
RotatingMomentumY_s,
RotatingMomentumZ_s,
RotatingPressureStagnation_s,
RotatingVelocityMagnitude_s,
RotatingVelocityX_s,
RotatingVelocityY_s,
RotatingVelocityZ_s,
RotationAngle_s,
RotationCenter_s,
RotationRateVector_s,
SDRC_s,
Second_s,
SimulationType_s,
SkinFrictionMagnitude_s,
SkinFrictionX_s,
SkinFrictionY_s,
SkinFrictionZ_s,
Slug_s,
SoundIntensityDB_s,
SoundIntensity_s,
SpeciesDensity_s,
SpecificHeatPressure_s,
SpecificHeatRatio_Pressure_s,
SpecificHeatRatio_Volume_s,
SpecificHeatRatio_s,
SpecificHeatVolume_s,
StandardCubicFoot_s,
StandardCubicMeter_s,
Statampere_s,
StreamFunction_s,
Structured_s,
SurfaceArea_s,
NeumannData_s,
DirichletData_s,
InwardNormalIndex_s,
InwardNormalList_s,
SutherlandLawConstant_s,
SutherlandLaw_s,
TETRA_10_s,
TETRA_4_s,
TRI_3_s,
TRI_6_s,
TemperatureReference_s,
TemperatureStagnation_s,
Temperature_s,
ThermalConductivityModelType_s,
ThermalConductivityModel_s,
ThermalConductivityReference_s,
ThermalConductivity_s,
ThermalEquilib_s,
ThermalNonequilib_s,
ThermalRelaxationModelType_s,
ThermalRelaxationModel_s,
ThermallyPerfect_s,
TimeAccurate_s,
TimeValues_s,
Translation_s,
Transform_s,
TurbulenceClosureType_s,
TurbulenceClosure_s,
TurbulenceModelType_s,
TurbulenceModel_s,
TurbulentBBReynolds_s,
TurbulentDissipationRate_s,
TurbulentDissipation_s,
TurbulentDistance_s,
TurbulentEnergyKinetic_s,
TurbulentSANuTilde_s,
TwoEquation_JonesLaunder_s,
TwoEquation_MenterSST_s,
TwoEquation_Wilcox_s,
Unigraphics_s,
Unstructured_s,
UserDefined_s,
VanderWaals_s,
VariableRate_s,
VelocityAngleX_s,
VelocityAngleY_s,
VelocityAngleZ_s,
VelocityMagnitude_s,
VelocityNormal_s,
VelocityPhi_s,
VelocityR_s,
VelocitySoundStagnation_s,
VelocitySound_s,
VelocityTangential_s,
VelocityTheta_s,
VelocityUnitVectorX_s,
VelocityUnitVectorY_s,
VelocityUnitVectorZ_s,
VelocityX_s,
VelocityY_s,
VelocityZ_s,
Vertex_s,
VibrationalElectronEnergy_s,
VibrationalElectronTemperature_s,
Violle_s,
ViscosityEddyDynamic_s,
ViscosityEddy_s,
ViscosityKinematic_s,
ViscosityModelType_s,
ViscosityModel_s,
ViscosityMolecularReference_s,
ViscosityMolecular_s,
Voltage_s,
VorticityMagnitude_s,
VorticityX_s,
VorticityY_s,
VorticityZ_s,
Vorticity_s,
WallFunction_s,
WallFunctionType_s,
ZoneBC_s,
ZoneConvergenceHistory_s,
ZoneGridConnectivity_s,
ZonePointers_s,
ZoneType_s,
auCurrent_s
]

# --- other stuff

C1='C1'
MT='MT'
I4='I4'
I8='I8'
R4='R4'
R8='R8'
LK='LK'

#
# --- last line

