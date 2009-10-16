# -----------------------------------------------------------------------------
# pyS7 - CGNS/SIDS editor
# ONERA/DSNA - marc.poinot@onera.fr
# pyS7 - $Rev: 70 $ $Date: 2009-01-30 11:49:10 +0100 (Fri, 30 Jan 2009) $
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source
# tree for license information.

import CGNS.PAT.cgnskeywords as CK

tlistA=[
    'Descriptor_t',
    'UserDefinedData_t',
    'DataClass_t',
    'DimensionalUnits_t',
    ]

allDT=['C1','MT','I4','I8','R4','R8','LK'] # LK declared as data type

# <type description>      := <typename>:[<list of constraints>]
# <list of constraints>   := [<list of children types>,<attribute constraints>]
# <attribute constraints> := [<list of data types>,
#                             <cardinality>,
#                             <names>,
#                             <enumerates>]
#
types={
    # --------------------------------------------------
    'GridLocation_t':[[
    ],[['C1'],'1/1',[],[]]],
    # --------------------------------------------------
    'CGNSBase_t':[[
    'Zone_t',
    'Family_t',
    'ReferenceState_t',
    'SimulationType_t',
    'BaseIterativeData_t',
    'ConvergenceHistory_t',
    'FlowEquationSet_t',
    'Gravity_t',
    'IntegralData_t',
    'RotatingCoordinates_t',
    'Axisymmetry_t',
    ]+tlistA,[['I4'],'zero/N',[],[]]],
    # --------------------------------------------------
    'Zone_t':[[
    'GridCoordinates_t',
    'DiscreteData_t',
    'Elements_t',
    'ZoneBC_t',
    'FlowSolution_t',
    'Ordinal_t',
    'ZoneGridConnectivity_t',
    'ZoneIterativeData_t',
    'RigidGridMotion_t',
    'ReferenceState_t',
    'IntegralData_t',
    'ArbitraryGridMotion_t',
    'FamilyName_t',
    'FlowEquationSet_t',
    'ConvergenceHistory_t',
    'RotatingCoordinates_t',
    'ZoneType_t',
    ]+tlistA,[['I4'],'zero/N',[],[]]],
    # --------------------------------------------------
    'GridCoordinates_t':[[
    'DataArray_t',
    'Rind_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'Elements_t':[[
    'IndexRange_t',
    'DataArray_t',
    'Rind_t',
    'Descriptor_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'Axisymmetry_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'RotatingCoordinates_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'FlowSolution_t':[[
    'GridLocation_t',
    'DataArray_t',
    'Rind_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'DiscreteData_t':[[
    'GridLocation_t',
    'DataArray_t',
    'Rind_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'ZoneGridConnectivity_t':[[
    'GridConnectivity1to1_t',
    'GridConnectivity_t',
    'OversetHoles_t',
    'Descriptor_t',
    'UserDefinedData_t'
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'GridConnectivity1to1_t':[[
    '"int[IndexDimension]"',
    'IndexRange_t',
    'Ordinal_t',
    'GridConnectivityProperty_t',
    'Descriptor_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'GridConnectivity_t':[[
    'IndexRange_t',
    'GridLocation_t',
    'Ordinal_t',
    'IndexArray_t',
    'GridConnectivityProperty_t',
    'GridConnectivityType_t',
    'DataArray_t',
    'Descriptor_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'GridConnectivityProperty_t':[[
    'Periodic_t',
    'AverageInterface_t',
    'Descriptor_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'Periodic_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'AverageInterface_t':[[
    'AverageInterfaceType_t',
    'Descriptor_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'OversetHoles_t':[[
    'Descriptor_t',
    'IndexArray_t',
    'GridLocation_t',
    'IndexRange_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'ZoneBC_t':[[
    'BC_t',
    'ReferenceState_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'BC_t':[[
    'IndexArray_t',
    '"int[IndexDimension]"',
    'IndexRange_t',
    'GridLocation_t',
    'BCProperty_t',
    'ReferenceState_t',
    'Ordinal_t',
    'FamilyName_t',
    'BCDataSet_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'BCDataSet_t':[[
    'BCData_t',
    'GridLocation_t',
    'IndexRange_t',
    'IndexArray_t',
    'ReferenceState_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'BCData_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'BCProperty_t':[[
    'WallFunction_t',
    'Area_t',
    'Descriptor_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'WallFunction_t':[[
    'WallFunctionType_t',
    'Descriptor_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'Area_t':[[
    'AreaType_t',
    'DataArray_t',
    'Descriptor_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'FlowEquationSet_t':[[
    'GoverningEquations_t',
    'GasModel_t',
    'ViscosityModel_t',
    'ThermalRelaxationModel_t',
    '"int"',
    'ThermalConductivityModel_t',
    'TurbulenceModel_t',
    'TurbulenceClosure_t',
    'ChemicalKineticsModel_t',
    'EMMagneticFieldModel_t',
    'EMElectricFieldModel_t',    
    'EMConductivityModel_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'GoverningEquations_t':[[
    '"int[1 + ... + IndexDimension]"',   # '"int[1+ ...+IndexDimension]"'
    'Descriptor_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'GasModel_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'ViscosityModel_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'ThermalRelaxationModel_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'ThermalConductivityModel_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'TurbulenceModel_t':[[
    '"int[1 + ... + IndexDimension]"',   # '"int[1+ ...+IndexDimension]"'
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'TurbulenceClosure_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'ChemicalKineticsModel_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'EMMagneticFieldModel_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'EMElectricFieldModel_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'EMConductivityModel_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'ConvergenceHistory_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------        
    'IntegralData_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'ReferenceState_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'DataArray_t':[[
    'DimensionalExponents_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'DimensionalUnits_t':[[
    'AdditionalUnits_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'DimensionalExponents_t':[[
    'AdditionalExponents_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'Family_t':[[
    'Descriptor_t',
    'Ordinal_t',
    'FamilyBC_t',
    'GeometryReference_t',
    'RotatingCoordinates_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'FamilyBC_t':[[
    'BCDataSet_t',    
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'GeometryReference_t':[[
    'Descriptor_t',
    'GeometryFile_t',
    'GeometryFormat_t',
    'GeometryEntity_t',
    'UserDefinedData_t',
    ],[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'BaseIterativeData_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'ZoneIterativeData_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'RigidGridMotion_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'ArbitraryGridMotion_t':[[
    'DataArray_t',
    'GridLocation_t',
    'Rind_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'UserDefinedData_t':[[
    'GridLocation_t',
    'IndexRange_t',
    'IndexArray_t',
    'DataArray_t',
    'FamilyName_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'Gravity_t':[[
    'DataArray_t',
    ]+tlistA,[allDT,'zero/N',[],[]]],
    # --------------------------------------------------
    'SimulationType_t':[[
    ],[['C1'],'1',[],CK.SimulationType_ts]],
    # --------------------------------------------------
    'ZoneType_t':[[
    ],[['C1'],'1',[],CK.ZoneType_ts]],
    # --------------------------------------------------
    'CGNSLibraryVersion_t':[[
    ],[['R4'],'1',[],[]]],
    # --------------------------------------------------
    'IndexRange_t':[[
    ],[['I4'],'0/N',[],[]]],
    # --------------------------------------------------
    'IndexArray_t':[[
    ],[['I4'],'0/N',[],[]]],
    # --------------------------------------------------
    'FamilyName_t':[[
    ],[['C1'],'0/N',[],[]]],
    # --------------------------------------------------
    'GridConnectivityType_t':[[
    ],[['C1'],'0/N',[],CK.GridConnectivityType_ts]],
    # --------------------------------------------------
    '"int[IndexDimension]"':[[
    ],[['I4'],'0/N',[],[]]],
    # --------------------------------------------------
}
