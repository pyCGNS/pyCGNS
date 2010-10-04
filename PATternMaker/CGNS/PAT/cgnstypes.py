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

allDT=[CK.C1,CK.MT,CK.I4,CK.I8,CK.R4,CK.R8,CK.LK] # LK declared as data type

C_01='Zero/One'
C_11='One/One'
C_0N='Zero/N'
C_1N='One/N'
C_NN='N/N'

cardE=[]

# <type description>      := <typename>:[<list of constraints>]
# <list of constraints>   := [<list of children types>,<attribute constraints>]
# <attribute constraints> := [<list of data types>,
#                             <cardinality>,
#                             <names>,
#                             <enumerates>]
#
types={
    # --------------------------------------------------
    CK.Rind_ts:[[
    ],[[CK.C1],C_01,[],[]]],
    # --------------------------------------------------
    CK.AdditionalUnits_ts:[[
    ],[[CK.C1],C_01,[],[]]],
    # --------------------------------------------------
    CK.TurbulenceModelType_ts:[[
    ],[[CK.C1],C_01,[],[]]],
    # --------------------------------------------------
    CK.DataConversion_ts:[[
    ],[[CK.C1],C_01,[],[]]],
    # --------------------------------------------------
    CK.DataClass_ts:[[
    ],[[CK.C1],C_01,[],[]]],
    # --------------------------------------------------
    CK.Descriptor_ts:[[
    ],[[CK.C1],'0/N',[],[]]],
    # --------------------------------------------------
    CK.GridLocation_ts:[[
    ],[[CK.C1],C_11,[],[]]],
    # --------------------------------------------------
    CK.CGNSBase_ts:[[
    CK.Zone_ts,
    CK.Family_ts,
    CK.ReferenceState_ts,
    CK.SimulationType_ts,
    CK.BaseIterativeData_ts,
    CK.ConvergenceHistory_ts,
    CK.FlowEquationSet_ts,
    CK.Gravity_ts,
    CK.IntegralData_ts,
    CK.RotatingCoordinates_ts,
    CK.Axisymmetry_ts,
    ]+tlistA,[[CK.I4],C_0N,[],[]]],
    # --------------------------------------------------
    CK.Zone_ts:[[
    CK.GridCoordinates_ts,
    CK.DiscreteData_ts,
    CK.Elements_ts,
    CK.ZoneBC_ts,
    CK.FlowSolution_ts,
    CK.Ordinal_ts,
    CK.ZoneGridConnectivity_ts,
    CK.ZoneIterativeData_ts,
    CK.RigidGridMotion_ts,
    CK.ReferenceState_ts,
    CK.IntegralData_ts,
    CK.ArbitraryGridMotion_ts,
    CK.FamilyName_ts,
    CK.FlowEquationSet_ts,
    CK.ConvergenceHistory_ts,
    CK.RotatingCoordinates_ts,
    CK.ZoneType_ts,
    ]+tlistA,[[CK.I4],C_0N,[],[]]],
    # --------------------------------------------------
    CK.GridCoordinates_ts:[[
    CK.DataArray_ts,
    CK.Rind_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.Elements_ts:[[
    CK.IndexRange_ts,
    CK.DataArray_ts,
    CK.Rind_ts,
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.Axisymmetry_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.RotatingCoordinates_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.FlowSolution_ts:[[
    CK.GridLocation_ts,
    CK.DataArray_ts,
    CK.Rind_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.DiscreteData_ts:[[
    CK.GridLocation_ts,
    CK.DataArray_ts,
    CK.Rind_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.ZoneGridConnectivity_ts:[[
    CK.GridConnectivity1to1_ts,
    CK.GridConnectivity_ts,
    CK.OversetHoles_ts,
    CK.Descriptor_ts,
    CK.UserDefinedData_ts
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.GridConnectivity1to1_ts:[[
    '"int[IndexDimension]"',
    CK.IndexRange_ts,
    CK.Ordinal_ts,
    CK.GridConnectivityProperty_ts,
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.GridConnectivity_ts:[[
    CK.IndexRange_ts,
    CK.GridLocation_ts,
    CK.Ordinal_ts,
    CK.IndexArray_ts,
    CK.GridConnectivityProperty_ts,
    CK.GridConnectivityType_ts,
    CK.DataArray_ts,
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.GridConnectivityProperty_ts:[[
    CK.Periodic_ts,
    CK.AverageInterface_ts,
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.Periodic_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.AverageInterface_ts:[[
    CK.AverageInterfaceType_ts,
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.OversetHoles_ts:[[
    CK.Descriptor_ts,
    CK.IndexArray_ts,
    CK.GridLocation_ts,
    CK.IndexRange_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.ZoneBC_ts:[[
    CK.BC_ts,
    CK.ReferenceState_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.BC_ts:[[
    CK.IndexArray_ts,
    '"int[IndexDimension]"',
    CK.IndexRange_ts,
    CK.GridLocation_ts,
    CK.BCProperty_ts,
    CK.ReferenceState_ts,
    CK.Ordinal_ts,
    CK.FamilyName_ts,
    CK.BCDataSet_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.BCDataSet_ts:[[
    CK.BCData_ts,
    CK.GridLocation_ts,
    CK.IndexRange_ts,
    CK.IndexArray_ts,
    CK.ReferenceState_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.BCData_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.BCProperty_ts:[[
    CK.WallFunction_ts,
    CK.Area_ts,
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.WallFunction_ts:[[
    CK.WallFunctionType_ts,
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.Area_ts:[[
    CK.AreaType_ts,
    CK.DataArray_ts,
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.FlowEquationSet_ts:[[
    CK.GoverningEquations_ts,
    CK.GasModel_ts,
    CK.ViscosityModel_ts,
    CK.ThermalRelaxationModel_ts,
    '"int"',
    CK.ThermalConductivityModel_ts,
    CK.TurbulenceModel_ts,
    CK.TurbulenceClosure_ts,
    CK.ChemicalKineticsModel_ts,
    CK.EMMagneticFieldModel_ts,
    CK.EMElectricFieldModel_ts,    
    CK.EMConductivityModel_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.GoverningEquations_ts:[[
    '"int[1 + ... + IndexDimension]"',   # '"int[1+ ...+IndexDimension]"'
    CK.Descriptor_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.GasModel_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.ViscosityModel_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.ThermalRelaxationModel_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.ThermalConductivityModel_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.TurbulenceModel_ts:[[
    '"int[1 + ... + IndexDimension]"',   # '"int[1+ ...+IndexDimension]"'
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.TurbulenceClosure_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.ChemicalKineticsModel_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.EMMagneticFieldModel_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.EMElectricFieldModel_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.EMConductivityModel_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.ConvergenceHistory_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------        
    CK.IntegralData_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.ReferenceState_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.DataArray_ts:[[
    CK.DimensionalExponents_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.DimensionalUnits_ts:[[
    CK.AdditionalUnits_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.DimensionalExponents_ts:[[
    CK.AdditionalExponents_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.Family_ts:[[
    CK.Descriptor_ts,
    CK.Ordinal_ts,
    CK.FamilyBC_ts,
    CK.GeometryReference_ts,
    CK.RotatingCoordinates_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.FamilyBC_ts:[[
    CK.BCDataSet_ts,    
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.GeometryReference_ts:[[
    CK.Descriptor_ts,
    CK.GeometryFile_ts,
    CK.GeometryFormat_ts,
    CK.GeometryEntity_ts,
    CK.UserDefinedData_ts,
    ],[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.BaseIterativeData_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.ZoneIterativeData_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.RigidGridMotion_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.ArbitraryGridMotion_ts:[[
    CK.DataArray_ts,
    CK.GridLocation_ts,
    CK.Rind_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.UserDefinedData_ts:[[
    CK.GridLocation_ts,
    CK.IndexRange_ts,
    CK.IndexArray_ts,
    CK.DataArray_ts,
    CK.FamilyName_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.Gravity_ts:[[
    CK.DataArray_ts,
    ]+tlistA,[allDT,C_0N,[],[]]],
    # --------------------------------------------------
    CK.SimulationType_ts:[[
    ],[[CK.C1],C_01,[],CK.SimulationType_ts]],
    # --------------------------------------------------
    CK.ZoneType_ts:[[
    ],[[CK.C1],C_11,[],CK.ZoneType_ts]],
    # --------------------------------------------------
    CK.CGNSLibraryVersion_ts:[[
    ],[[CK.R4],C_11,[],[]]],
    # --------------------------------------------------
    CK.IndexRange_ts:[[
    ],[[CK.I4],C_0N,[],[]]],
    # --------------------------------------------------
    CK.IndexArray_ts:[[
    ],[[CK.I4],C_0N,[],[]]],
    # --------------------------------------------------
    CK.FamilyName_ts:[[
    ],[[CK.C1],C_0N,[],[]]],
    # --------------------------------------------------
    CK.GridConnectivityType_ts:[[
    ],[[CK.C1],C_0N,[],CK.GridConnectivityType_ts]],

    # --------------------------------------------------
    CK.DiffusionModel_ts2:[[
    ],[[CK.I4],C_0N,[],[]]],
    # --------------------------------------------------
    CK.Transform_ts2:[[
    ],[[CK.I4],C_0N,[],[]]],
    # --------------------------------------------------
    # --------------------------------------------------
    CK.DiffusionModel_ts:[[
    ],[[CK.I4],C_0N,[],[]]],
    # --------------------------------------------------
    CK.Transform_ts:[[
    ],[[CK.I4],C_0N,[],[]]],
    # --------------------------------------------------
}
