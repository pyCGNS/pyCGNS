
#ifndef CGNSLIB_KEYWORDS_H
#define CGNSLIB_KEYWORDS_H

#define CG_Null_s                      "Null"
#define CG_UserDefined_s               "Userdefined"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Dimensional Units                                                *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/* Mass ---*/
#define Kilogram_s                     "Kilogram"
#define Gram_s                         "Gram"
#define Slug_s                         "Slug"
#define PoundMass_s                    "PoundMass"
#define Meter_s                        "Meter"

/* Length ---*/
#define Centimeter_s                   "Centimeter"
#define Millimeter_s                   "Millimeter"
#define Foot_s                         "Foot"
#define Inch_s                         "Inch"

/* Time ---*/
#define Second_s                       "Second"

/* Temperature ---*/
#define Kelvin_s                       "Kelvin"
#define Celsius_s                      "Celsius"
#define Rankine_s                      "Rankine"
#define Fahrenheit_s                   "Fahrenheit"

/* Angle ---*/
#define Degree_s                       "Degree"
#define Radian_s                       "Radian"

/* ElectricCurrent ---*/
#define Ampere_s                       "Ampere"
#define Abampere_s                     "Abampere"
#define Statampere_s                   "Statampere"
#define Edison_s                       "Edison"
#define auCurrent_s                    "auCurrent"

/* SubstanceAmount ---*/
#define Mole_s                         "Mole"
#define Entities_s                     "Entities"
#define StandardCubicFoot_s            "StandardCubicFoot"
#define StandardCubicMeter_s           "StandardCubicMeter"

/* LuminousIntensity ---*/
#define Candela_s                      "Candela"
#define Candle_s                       "Candle"
#define Carcel_s                       "Carcel"
#define Hefner_s                       "Hefner"
#define Violle_s                       "Violle"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Data Class                                                       *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define Dimensional_s                    "Dimensional"
#define NormalizedByDimensional_s        "NormalizedByDimensional"
#define NormalizedByUnknownDimensional_s "NormalizedByUnknownDimensional"
#define NondimensionalParameter_s        "NondimensionalParameter"
#define DimensionlessConstant_s          "DimensionlessConstant"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *	Grid Location
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define Vertex_s                       "Vertex"
#define CellCenter_s                   "CellCenter"
#define FaceCenter_s                   "FaceCenter"
#define IFaceCenter_s                  "IFaceCenter"
#define JFaceCenter_s                  "JFaceCenter"
#define KFaceCenter_s                  "KFaceCenter"
#define EdgeCenter_s                   "EdgeCenter"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      BCData Types                                                     *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define Dirichlet_s                    "Dirichlet"
#define Neumann_s                      "Neumann"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *	Grid Connectivity Types 					 *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define Overset_s                      "Overset"
#define Abutting_s                     "Abutting"
#define Abutting1to1_s                 "Abutting1to1"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *	Point Set Types							 *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define PointList_s                    "PointList"
#define PointListDonor_s               "PointListDonor"
#define PointRange_s                   "PointRange"
#define PointRangeDonor_s              "PointRangeDonor"
#define ElementRange_s                 "ElementRange"
#define ElementList_s                  "ElementList"
#define CellListDonor_s                "CellListDonor"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Governing Equations and Physical Models Types                    *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define FullPotential_s                "FullPotential"
#define Euler_s                        "Euler"
#define NSLaminar_s                    "NSLaminar"
#define NSTurbulent_s                  "NSTurbulent"
#define NSLaminarIncompressible_s      "NSLaminarIncompressible"
#define NSTurbulentIncompressible_s    "NSTurbulentIncompressible"

#define Ideal_s                        "Ideal"
#define VanderWaals_s                  "VanderWaals"
#define Constant_s                     "Constant"
#define PowerLaw_s                     "PowerLaw"    
#define SutherlandLaw_s                "SutherlandLaw"
#define ConstantPrandtl_s              "ConstantPrandtl"
#define EddyViscosity_s                "EddyViscosity"
#define ReynoldsStress_s               "ReynoldsStress"
#define Algebraic_s                    "Algebraic"
#define BaldwinLomax_s                 "BaldwinLomax"
#define ReynoldsStressAlgebraic_s      "ReynoldsStressAlgebraic"
#define Algebraic_CebeciSmith_s	       "Algebraic_CebeciSmith"
#define HalfEquation_JohnsonKing_s     "HalfEquation_JohnsonKing"
#define OneEquation_BaldwinBarth_s     "OneEquation_BaldwinBarth"
#define OneEquation_SpalartAllmaras_s  "OneEquation_SpalartAllmaras"
#define TwoEquation_JonesLaunder_s     "TwoEquation_JonesLaunder"
#define TwoEquation_MenterSST_s        "TwoEquation_MenterSST"
#define TwoEquation_Wilcox_s           "TwoEquation_Wilcox"
#define	CaloricallyPerfect_s           "CaloricallyPerfect"
#define ThermallyPerfect_s             "ThermallyPerfect"
#define ConstantDensity_s              "ConstantDensity"
#define RedlichKwong_s                 "RedlichKwong"
#define Frozen_s                       "Frozen"
#define ThermalEquilib_s               "ThermalEquilib"
#define ThermalNonequilib_s            "ThermalNonequilib"
#define ChemicalEquilibCurveFit_s      "ChemicalEquilibCurveFit"
#define ChemicalEquilibMinimization_s  "ChemicalEquilibMinimization"
#define ChemicalNonequilib_s           "ChemicalNonequilib"
#define EMElectricField_s              "EMElectricField"
#define EMMagneticField_s              "EMMagneticField"
#define EMConductivity_s               "EMConductivity"
#define Voltage_s                      "Voltage"
#define Interpolated_s                 "Interpolated"
#define Equilibrium_LinRessler_s       "Equilibrium_LinRessler"
#define Chemistry_LinRessler_s         "Chemistry_LinRessler"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 * 	Boundary Condition Types					 *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define BCAxisymmetricWedge_s          "BCAxisymmetricWedge"
#define BCDegenerateLine_s             "BCDegenerateLine"
#define BCDegeneratePoint_s            "BCDegeneratePoint"
#define BCDirichlet_s                  "BCDirichlet"
#define BCExtrapolate_s                "BCExtrapolate"
#define BCFarfield_s                   "BCFarfield"
#define BCGeneral_s                    "BCGeneral"
#define BCInflow_s                     "BCInflow"
#define BCInflowSubsonic_s             "BCInflowSubsonic"
#define BCInflowSupersonic_s           "BCInflowSupersonic"
#define BCNeumann_s                    "BCNeumann"
#define BCOutflow_s                    "BCOutflow"
#define BCOutflowSubsonic_s            "BCOutflowSubsonic"
#define BCOutflowSupersonic_s          "BCOutflowSupersonic"
#define BCSymmetryPlane_s              "BCSymmetryPlane"
#define BCSymmetryPolar_s              "BCSymmetryPolar"
#define BCTunnelInflow_s               "BCTunnelInflow"
#define BCTunnelOutflow_s              "BCTunnelOutflow"
#define BCWall_s                       "BCWall"
#define BCWallInviscid_s               "BCWallInviscid"
#define BCWallViscous_s                "BCWallViscous"
#define BCWallViscousHeatFlux_s        "BCWallViscousHeatFlux"
#define BCWallViscousIsothermal_s      "BCWallViscousIsothermal"
#define FamilySpecified_s              "FamilySpecified_s"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Data types:  Can not add data types and stay forward compatible  *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define Integer_s                      "Integer"
#define RealSingle_s                   "RealSingle"
#define RealDouble_s                   "RealDouble"
#define Character_s                    "Character"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Element types                                                    *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define NODE_s                         "NODE"
#define BAR_2_s                        "BAR_2" 	 
#define BAR_3_s                        "BAR_3"
#define TRI_3_s                        "TRI_3" 	 
#define TRI_6_s                        "TRI_6"           
#define QUAD_4_s                       "QUAD_4" 	 
#define QUAD_8_s                       "QUAD_8"          
#define QUAD_9_s                       "QUAD_9"           
#define TETRA_4_s                      "TETRA_4" 	 
#define TETRA_10_s                     "TETRA_10"           
#define PYRA_5_s                       "PYRA_5" 	 
#define PYRA_13_s                      "PYRA_13"           
#define PYRA_14_s                      "PYRA_14"           
#define PENTA_6_s                      "PENTA_6" 	 
#define PENTA_15_s                     "PENTA_15"          
#define PENTA_18_s                     "PENTA_18"           
#define HEXA_8_s                       "HEXA_8" 	 
#define HEXA_20_s                      "HEXA_20"          
#define HEXA_27_s                      "HEXA_27"           
#define MIXED_s                        "MIXED" 	 
#define NGON_n_s                       "NGON_n"          
#define NFACE_n_s                      "NFACE_n"          

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Zone types                                                       *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define Structured_s                   "Structured"
#define Unstructured_s                 "Unstructured"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Rigid Grid Motion types						 *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define ConstantRate_s                 "ConstantRate"
#define VariableRate_s                 "VariableRate"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Arbitrary Grid Motion types                                      *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define NonDeformingGrid_s             "NonDeformingGrid"
#define DeformingGrid_s                "DeformingGrid"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Simulation types					         *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define TimeAccurate_s                 "TimeAccurate"
#define NonTimeAccurate_s              "NonTimeAccurate"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *	BC Property types						 *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define Generic_s                      "Generic"
#define BleedArea_s                    "BleedArea"
#define CaptureArea_s                  "CaptureArea"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Grid Connectivity Property types				 *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define AverageAll_s                   "AverageAll"
#define AverageCircumferential_s       "AverageCircumferential"
#define AverageRadial_s                "AverageRadial"
#define AverageI_s                     "AverageI"
#define AverageJ_s                     "AverageJ"
#define AverageK_s                     "AverageK"



/* The strings defined below are node names or node name patterns */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Coordinate system                                                *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define CoordinateX_s                  "CoordinateX"
#define CoordinateY_s                  "CoordinateY"
#define CoordinateZ_s                  "CoordinateZ"
#define CoordinateR_s                  "CoordinateR"
#define CoordinateTheta_s              "CoordinateTheta"
#define CoordinatePhi_s                "CoordinatePhi"
#define CoordinateNormal_s             "CoordinateNormal"
#define CoordinateTangential_s         "CoordinateTangential"
#define CoordinateXi_s                 "CoordinateXi"
#define CoordinateEta_s                "CoordinateEta"
#define CoordinateZeta_s               "CoordinateZeta"
#define CoordinateTransform_s          "CoordinateTransform"
#define InterpolantsDonor_s            "InterpolantsDonor"
#define ElementConnectivity_s          "ElementConnectivity"
#define ParentData_s                   "ParentData"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      FlowSolution Quantities                                          *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/* Patterns --- */
#define VectorX_ps                     "%sX"
#define VectorY_ps                     "%sY"
#define VectorZ_ps                     "%sZ"    
#define VectorTheta_ps                 "%sTheta"
#define VectorPhi_ps                   "%sPhi"
#define VectorMagnitude_ps             "%sMagnitude"
#define VectorNormal_ps                "%sNormal"
#define VectorTangential_ps            "%sTangential"

#define Potential_s                    "Potential"
#define StreamFunction_s               "StreamFunction"
#define Density_s                      "Density"
#define Pressure_s                     "Pressure"
#define Temperature_s                  "Temperature"
#define EnergyInternal_s               "EnergyInternal"
#define Enthalpy_s                     "Enthalpy"
#define Entropy_s                      "Entropy"
#define EntropyApprox_s                "EntropyApprox"
#define DensityStagnation_s            "DensityStagnation"
#define PressureStagnation_s           "PressureStagnation"
#define TemperatureStagnation_s        "TemperatureStagnation"
#define EnergyStagnation_s             "EnergyStagnation"
#define EnthalpyStagnation_s           "EnthalpyStagnation"
#define EnergyStagnationDensity_s      "EnergyStagnationDensity"
#define VelocityX_s                    "VelocityX"
#define VelocityY_s                    "VelocityY"
#define VelocityZ_s                    "VelocityZ"
#define VelocityR_s                    "VelocityR"
#define VelocityTheta_s                "VelocityTheta"
#define VelocityPhi_s                  "VelocityPhi"
#define VelocityMagnitude_s            "VelocityMagnitude"
#define VelocityNormal_s               "VelocityNormal"
#define VelocityTangential_s           "VelocityTangential"
#define VelocitySound_s                "VelocitySound"
#define VelocitySoundStagnation_s      "VelocitySoundStagnation"
#define MomentumX_s                    "MomentumX"
#define MomentumY_s                    "MomentumY"
#define MomentumZ_s                    "MomentumZ"
#define MomentumMagnitude_s            "MomentumMagnitude"
#define RotatingVelocityX_s            "RotatingVelocityX"
#define RotatingVelocityY_s            "RotatingVelocityY"
#define RotatingVelocityZ_s            "RotatingVelocityZ"
#define RotatingMomentumX_s            "RotatingMomentumX"
#define RotatingMomentumY_s            "RotatingMomentumY"
#define RotatingMomentumZ_s            "RotatingMomentumZ"
#define RotatingVelocityMagnitude_s    "RotatingVelocityMagnitude"
#define RotatingPressureStagnation_s   "RotatingPressureStagnation"
#define RotatingEnergyStagnation_s     "RotatingEnergyStagnation"
#define RotatingEnergyStagnationDensity_s  "RotatingEnergyStagnationDensity"
#define RotatingEnthalpyStagnation_s     "RotatingEnthalpyStagnation"
#define EnergyKinetic_s                "EnergyKinetic"
#define PressureDynamic_s              "PressureDynamic"
#define SoundIntensityDB_s             "SoundIntensityDB"
#define SoundIntensity_s               "SoundIntensity"

#define VorticityX_s                   "VorticityX"
#define VorticityY_s                   "VorticityY"
#define VorticityZ_s                   "VorticityZ"
#define VorticityMagnitude_s           "VorticityMagnitude"
#define SkinFrictionX_s                "SkinFrictionX"
#define SkinFrictionY_s                "SkinFrictionY"
#define SkinFrictionZ_s                "SkinFrictionZ"
#define SkinFrictionMagnitude_s        "SkinFrictionMagnitude"
#define VelocityAngleX_s               "VelocityAngleX"
#define VelocityAngleY_s               "VelocityAngleY"
#define VelocityAngleZ_s               "VelocityAngleZ"
#define VelocityUnitVectorX_s          "VelocityUnitVectorX"
#define VelocityUnitVectorY_s          "VelocityUnitVectorY"
#define VelocityUnitVectorZ_s          "VelocityUnitVectorZ"
#define MassFlow_s                     "MassFlow"
#define ViscosityKinematic_s           "ViscosityKinematic"
#define ViscosityMolecular_s           "ViscosityMolecular"
#define ViscosityEddyDynamic_s         "ViscosityEddyDynamic"
#define ViscosityEddy_s                "ViscosityEddy"
#define ThermalConductivity_s          "ThermalConductivity"
#define PowerLawExponent_s             "PowerLawExponent"
#define SutherlandLawConstant_s        "SutherlandLawConstant"
#define TemperatureReference_s         "TemperatureReference"
#define ViscosityMolecularReference_s  "ViscosityMolecularReference"
#define ThermalConductivityReference_s "ThermalConductivityReference"
#define IdealGasConstant_s             "IdealGasConstant"
#define SpecificHeatPressure_s         "SpecificHeatPressure"
#define SpecificHeatVolume_s           "SpecificHeatVolume"
#define ReynoldsStressXX_s             "ReynoldsStressXX"
#define ReynoldsStressXY_s             "ReynoldsStressXY"
#define ReynoldsStressXZ_s             "ReynoldsStressXZ"
#define ReynoldsStressYY_s             "ReynoldsStressYY"
#define ReynoldsStressYZ_s             "ReynoldsStressYZ"
#define ReynoldsStressZZ_s             "ReynoldsStressZZ"
#define LengthReference_s              "LengthReference"

#define MolecularWeight_s              "MolecularWeight"
#define MolecularWeight_ps             "MolecularWeight%s"
#define HeatOfFormation_s              "HeatOfFormation"
#define HeatOfFormation_ps             "HeatOfFormation%s"
#define FuelAirRatio_s                 "FuelAirRatio"
#define ReferenceTemperatureHOF_s      "ReferenceTemperatureHOF"
#define MassFraction_s                 "MassFraction"
#define MassFraction_ps                "MassFraction%s"
#define LaminarViscosity_s             "LaminarViscosity"
#define LaminarViscosity_ps            "LaminarViscosity%s"
#define ThermalConductivity_ps         "ThermalConductivity%s"
#define EnthalpyEnergyRatio_s          "EnthalpyEnergyRatio"
#define CompressibilityFactor_s        "CompressibilityFactor"
#define VibrationalElectronEnergy_s    "VibrationalElectronEnergy"
#define VibrationalElectronTemperature_s  "VibrationalElectronTemperature"
#define SpeciesDensity_s               "SpeciesDensity"
#define SpeciesDensity_ps              "SpeciesDensity%s"
#define MoleFraction_s                 "MoleFraction"
#define MoleFraction_ps                "MoleFraction%s"

#define ElectricFieldX_s               "ElectricFieldX"
#define ElectricFieldY_s               "ElectricFieldY"
#define ElectricFieldZ_s               "ElectricFieldZ"
#define MagneticFieldX_s               "MagneticFieldX"
#define MagneticFieldY_s               "MagneticFieldY"
#define MagneticFieldZ_s               "MagneticFieldZ"
#define CurrentDensityX_s              "CurrentDensityX"
#define CurrentDensityY_s              "CurrentDensityY"
#define CurrentDensityZ_s              "CurrentDensityZ"
#define LorentzForceX_s                "LorentzForceX"
#define LorentzForceY_s                "LorentzForceY"
#define LorentzForceZ_s                "LorentzForceZ"
#define ElectricConductivity_s         "ElectricConductivity"
#define JouleHeating_s                 "JouleHeating"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Typical Turbulence Models                                        *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define TurbulentDistance_s            "TurbulentDistance"
#define TurbulentEnergyKinetic_s       "TurbulentEnergyKinetic"
#define TurbulentDissipation_s         "TurbulentDissipation"
#define TurbulentDissipationRate_s     "TurbulentDissipationRate"
#define TurbulentBBReynolds_s          "TurbulentBBReynolds"
#define TurbulentSANuTilde_s           "TurbulentSANuTilde"


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Nondimensional Parameters                                        *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define Mach_s                         "Mach"
#define Mach_Velocity_s                "Mach_Velocity"
#define Mach_VelocitySound_s           "Mach_VelocitySound"
#define Reynolds_s                     "Reynolds"
#define Reynolds_Velocity_s            "Reynolds_Velocity"
#define Reynolds_Length_s              "Reynolds_Length"
#define Reynolds_ViscosityKinematic_s  "Reynolds_ViscosityKinematic"
#define Prandtl_s                      "Prandtl"
#define Prandtl_ThermalConductivity_s  "Prandtl_ThermalConductivity"
#define Prandtl_ViscosityMolecular_s   "Prandtl_ViscosityMolecular"
#define Prandtl_SpecificHeatPressure_s "Prandtl_SpecificHeatPressure"
#define PrandtlTurbulent_s             "PrandtlTurbulent"
#define SpecificHeatRatio_s            "SpecificHeatRatio"
#define SpecificHeatRatio_Pressure_s   "SpecificHeatRatio_Pressure"
#define SpecificHeatRatio_Volume_s     "SpecificHeatRatio_Volume"
#define CoefPressure_s                 "CoefPressure"
#define CoefSkinFrictionX_s            "CoefSkinFrictionX"
#define CoefSkinFrictionY_s            "CoefSkinFrictionY"
#define CoefSkinFrictionZ_s            "CoefSkinFrictionZ"
#define Coef_PressureDynamic_s         "Coef_PressureDynamic"
#define Coef_PressureReference_s       "Coef_PressureReference"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Characteristics and Riemann invariant                            *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define Vorticity_s                    "Vorticity"
#define Acoustic_s                     "Acoustic"

#define RiemannInvariantPlus_s         "RiemannInvariantPlus"
#define RiemannInvariantMinus_s        "RiemannInvariantMinus"
#define CharacteristicEntropy_s        "CharacteristicEntropy"
#define CharacteristicVorticity1_s     "CharacteristicVorticity1"
#define CharacteristicVorticity2_s     "CharacteristicVorticity2"
#define CharacteristicAcousticPlus_s   "CharacteristicAcousticPlus"
#define CharacteristicAcousticMinus_s  "CharacteristicAcousticMinus"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      Forces and Moments                                               *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define ForceX_s                       "ForceX"
#define ForceY_s                       "ForceY"
#define ForceZ_s                       "ForceZ"
#define ForceR_s                       "ForceR"
#define ForceTheta_s                   "ForceTheta"
#define ForcePhi_s                     "ForcePhi"
#define Lift_s                         "Lift"
#define Drag_s                         "Drag"
#define MomentX_s                      "MomentX"
#define MomentY_s                      "MomentY"
#define MomentZ_s                      "MomentZ"
#define MomentR_s                      "MomentR"
#define MomentTheta_s                  "MomentTheta"
#define MomentPhi_s                    "MomentPhi"
#define MomentXi_s                     "MomentXi"
#define MomentEta_s                    "MomentEta"
#define MomentZeta_s                   "MomentZeta"
#define Moment_CenterX_s               "Moment_CenterX"
#define Moment_CenterY_s               "Moment_CenterY"
#define Moment_CenterZ_s               "Moment_CenterZ"
#define CoefLift_s                     "CoefLift"
#define CoefDrag_s                     "CoefDrag"
#define CoefMomentX_s                  "CoefMomentX"
#define CoefMomentY_s                  "CoefMomentY"
#define CoefMomentZ_s                  "CoefMomentZ"
#define CoefMomentR_s                  "CoefMomentR"
#define CoefMomentTheta_s              "CoefMomentTheta"
#define CoefMomentPhi_s                "CoefMomentPhi"
#define CoefMomentXi_s                 "CoefMomentXi"
#define CoefMomentEta_s                "CoefMomentEta"
#define CoefMomentZeta_s               "CoefMomentZeta"
#define Coef_PressureDynamic_s         "Coef_PressureDynamic"
#define Coef_Area_s                    "Coef_Area"
#define Coef_Length_s                  "Coef_Length"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *       Time dependent flow                                             *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define TimeValues_s                   "TimeValues"
#define IterationValues_s              "IterationValues"
#define NumberOfZones_s                "NumberOfZones"
#define NumberOfFamilies_s             "NumberOfFamilies"
#define ZonePointers_s                 "ZonePointers"
#define FamilyPointers_s               "FamilyPointers"
#define RigidGridMotionPointers_s      "RigidGridMotionPointers"
#define ArbitraryGridMotionPointers_s  "ArbitraryGridMotionPointers"
#define GridCoordinatesPointers_s      "GridCoordinatesPointers"
#define FlowSolutionPointers_s         "FlowSolutionPointers"
#define OriginLocation_s               "OriginLocation"
#define RigidRotationAngle_s           "RigidRotationAngle"
#define RigidVelocity_s                "RigidVelocity"
#define RigidRotationRate_s            "RigidRotationRate"
#define GridVelocityX_s                "GridVelocityX"
#define GridVelocityY_s                "GridVelocityY"
#define GridVelocityZ_s                "GridVelocityZ"
#define GridVelocityR_s                "GridVelocityR"
#define GridVelocityTheta_s            "GridVelocityTheta"
#define GridVelocityPhi_s              "GridVelocityPhi"
#define GridVelocityXi_s               "GridVelocityXi"
#define GridVelocityEta_s              "GridVelocityEta"
#define GridVelocityZeta_s             "GridVelocityZeta"


/* The strings defined below are type names used for node labels */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *       Types as strings                                                *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define ArbitraryGridMotion_ts         "ArbitraryGridMotion_t"
#define Area_ts                        "Area_t"
#define AverageInterface_ts            "AverageInterface_t"
#define Axisymmetry_ts                 "Axisymmetry_t"
#define BCDataSet_ts                   "BCDataSet_t"
#define BCData_ts                      "BCData_t"
#define BCProperty_ts                  "BCProperty_t"
#define BC_ts                          "BC_t"
#define BaseIterativeData_ts           "BaseIterativeData_t"
#define CGNSBase_ts                    "CGNSBase_t"
#define CGNSLibraryVersion_ts          "CGNSLibraryVersion_t"
#define ChemicalKineticsModel_ts       "ChemicalKineticsModel_t"
#define ConvergenceHistory_ts          "ConvergenceHistory_t"
#define DataArray_ts                   "DataArray_t"
#define DataClass_ts                   "DataClass_t"
#define DataConversion_ts              "DataConversion_t"
#define Descriptor_ts                  "Descriptor_t"
#define DimensionalExponents_ts        "DimensionalExponents_t"
#define DimensionalUnits_ts            "DimensionalUnits_t"   
#define DiscreteData_ts                "DiscreteData_t"
#define Elements_ts                    "Elements_t"
#define FamilyBC_ts                    "FamilyBC_t"
#define FamilyName_ts                  "FamilyName_t"
#define Family_ts                      "Family_t"
#define FlowEquationSet_ts             "FlowEquationSet_t"
#define FlowSolution_ts                "FlowSolution_t"
#define GasModel_ts                    "GasModel_t"
#define GeometryEntity_ts              "GeometryEntity_t"
#define GeometryFile_ts                "GeometryFile_t"
#define GeometryFormat_ts              "GeometryFormat_t"
#define GeometryReference_ts           "GeometryReference_t"
#define GoverningEquations_ts          "GoverningEquations_t"
#define Gravity_ts                     "Gravity_t"
#define GridConnectivity1to1_ts        "GridConnectivity1to1_t"
#define GridConnectivityProperty_ts    "GridConnectivityProperty_t"
#define GridConnectivityType_ts        "GridConnectivityType_t"
#define GridConnectivity_ts            "GridConnectivity_t"
#define GridCoordinates_ts             "GridCoordinates_t"
#define GridLocation_ts                "GridLocation_t"
#define IndexArray_ts                  "IndexArray_t"
#define IndexRange_ts                  "IndexRange_t"   
#define IntegralData_ts                "IntegralData_t"
#define InwardNormalList_ts            "InwardNormalList_t"
#define Ordinal_ts                     "Ordinal_t"
#define OversetHoles_ts                "OversetHoles_t"
#define Periodic_ts                    "Periodic_t"
#define ReferenceState_ts              "ReferenceState_t"
#define RigidGridMotion_ts             "RigidGridMotion_t"
#define Rind_ts                        "Rind_t"   
#define RotatingCoordinates_ts         "RotatingCoordinates_t"
#define SimulationType_ts              "SimulationType_t"
#define ThermalConductivityModel_ts    "ThermalConductivityModel_t"
#define ThermalRelaxationModel_ts      "ThermalRelaxationModel_t"
#define TurbulenceClosure_ts           "TurbulenceClosure_t"
#define TurbulenceModel_ts             "TurbulenceModel_t"
#define UserDefinedData_ts             "UserDefinedData_t"
#define ViscosityModel_ts              "ViscosityModel_t"
#define WallFunction_ts                "WallFunction_t"
#define ZoneBC_ts                      "ZoneBC_t"
#define ZoneGridConnectivity_ts        "ZoneGridConnectivity_t"
#define ZoneIterativeData_ts           "ZoneIterativeData_t"
#define ZoneType_ts                    "ZoneType_t"
#define Zone_ts                        "Zone_t"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *       No line after this comment -                                    *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#endif
