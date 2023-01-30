#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

from . import cgnskeywords as CGK


tlistA = [
    CGK.Descriptor_ts,
    CGK.UserDefinedData_ts,
    CGK.DataClass_ts,
    CGK.DimensionalUnits_ts,
]

allDT = [CGK.C1, CGK.MT, CGK.I4, CGK.I8, CGK.R4, CGK.R8]  # LK is default

C_00 = "Zero/Zero"
C_01 = "Zero/One"
C_11 = "One/One"
C_0N = "Zero/N"
C_1N = "One/N"
C_NN = "N/N"

UD = "{UserDefined}"

allCARD = [C_01, C_11, C_0N, C_1N, C_NN]


# --------------------------------------------------------
class CGNStype:
    def __init__(self, ntype, dtype=None, names=None):
        if dtype is None:
            dtype = [CGK.MT]
        if names is None:
            names = [UD]
        self.type = ntype
        self.datatype = [CGK.LK] + dtype
        self.enumerate = []
        self.shape = ()
        self.names = names
        self.children = []
        self.parents = []

    def hasChild(self, ctype):
        for c in self.children:
            if c[0] == ctype:
                return True
        return False

    def addChild(self, ctype, cname=UD, dtype=CGK.MT, card=C_0N):
        if not isinstance(cname, list):
            lname = [cname]
        else:
            lname = cname
        self.children.append((ctype, lname, dtype, card))

    def addParent(self, parent):
        self.parents.append(parent)

    def cardinality(self, childtype):
        for c in self.children:
            if c[0] == childtype:
                return c[3]
        return C_00

    def isReservedName(self, name):
        for c in self.children:
            if name in c[1]:
                return True
        return False

    def hasReservedNameType(self, name):
        nl = []
        for c in self.children:
            if name in c[1]:
                nl.append(c[0])
        return nl


cgt = {}

# --------------------------------------------------------
t = CGK.CGNSLibraryVersion_ts
cgt[t] = CGNStype(t, dtype=[CGK.R4], names=[CGK.CGNSLibraryVersion_s])
cgt[t].shape = (1,)

# (CGK.CGNSLibraryVersion_ts,   # SIDS type
# [CGK.CGNSLibraryVersion_s],  # Names
# [CGK.R4],                    # Datatypes
# (1,),                        # Shape
# [2.3,2.4,3.2,3.3],           # Values
# )

# --------------------------------------------------------
t = CGK.Descriptor_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.Ordinal_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4], names=[CGK.Ordinal_s])
cgt[t].shape = (1,)

# --------------------------------------------------------
t = CGK.DataClass_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.DataClass_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.DataClass_l

# --------------------------------------------------------
t = CGK.DimensionalUnits_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.DimensionalUnits_s])
cgt[t].shape = (32, 5)
cgt[t].enumerate = CGK.AllDimensionalUnits_l
cgt[t].addChild(CGK.AdditionalUnits_ts, CGK.AdditionalUnits_s)

# --------------------------------------------------------
t = CGK.AdditionalUnits_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.AdditionalUnits_s])
cgt[t].shape = (32, 3)
cgt[t].enumerate = CGK.AllAdditionalUnits_l

# --------------------------------------------------------
t = CGK.DataConversion_ts
cgt[t] = CGNStype(t, dtype=[CGK.R4, CGK.R8], names=[CGK.DataConversion_s])
cgt[t].shape = (2,)

# --------------------------------------------------------
t = CGK.DimensionalExponents_ts
cgt[t] = CGNStype(t, dtype=[CGK.R4, CGK.R8], names=[CGK.DimensionalExponents_s])
cgt[t].shape = (5,)

# --------------------------------------------------------
t = CGK.AdditionalExponents_ts
cgt[t] = CGNStype(t, dtype=[CGK.R4, CGK.R8], names=[CGK.AdditionalExponents_s])
cgt[t].shape = (3,)

# --------------------------------------------------------
t = CGK.DataArray_ts
cgt[t] = CGNStype(t, dtype=allDT)
cgt[t].addChild(CGK.DimensionalExponents_ts, CGK.DimensionalExponents_s)
cgt[t].addChild(CGK.DataConversion_ts, CGK.DataConversion_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)

# --------------------------------------------------------
t = CGK.Transform_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4], names=[CGK.Transform_s])
cgt[t].shape = (0,)
t = CGK.Transform_ts2
cgt[t] = CGNStype(t, dtype=[CGK.I4], names=[CGK.Transform_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.DiffusionModel_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4], names=[CGK.DiffusionModel_s])
cgt[t].shape = (0,)
t = CGK.DiffusionModel_ts2
cgt[t] = CGNStype(t, dtype=[CGK.I4], names=[CGK.DiffusionModel_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.InwardNormalIndex_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4], names=[CGK.InwardNormalIndex_s])
cgt[t].shape = (0,)
t = CGK.InwardNormalIndex_ts2
cgt[t] = CGNStype(t, dtype=[CGK.I4], names=[CGK.InwardNormalIndex_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.EquationDimension_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4], names=[CGK.EquationDimension_s])
cgt[t].shape = (1,)
t = CGK.EquationDimension_ts2
cgt[t] = CGNStype(t, dtype=[CGK.I4], names=[CGK.EquationDimension_s])
cgt[t].shape = (1,)

# --------------------------------------------------------
t = CGK.GridLocation_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.GridLocation_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.Rind_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4, CGK.I8], names=[CGK.Rind_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.IndexRange_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4, CGK.I8])
cgt[t].shape = (0, 2)
cgt[t].names = [CGK.PointRange_s, CGK.PointRangeDonor_s, CGK.ElementRange_s, UD]

# --------------------------------------------------------
t = CGK.IndexArray_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4, CGK.I8, CGK.R4, CGK.R8])
cgt[t].shape = (0, 0)
cgt[t].names = [
    CGK.PointList_s,
    CGK.PointListDonor_s,
    CGK.CellListDonor_s,
    CGK.InwardNormalList_s,
    UD,
]

# --------------------------------------------------------
t = CGK.ReferenceState_ts
cgt[t] = CGNStype(t, names=[CGK.ReferenceState_s])
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.Descriptor_ts, CGK.ReferenceStateDescription_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.ConvergenceHistory_ts
cgt[t] = CGNStype(
    t,
    names=[CGK.GlobalConvergenceHistory_s, CGK.ZoneConvergenceHistory_s],
    dtype=[CGK.I4],
)
cgt[t].shape = (1,)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.Descriptor_ts, CGK.NormDefinitions_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.IntegralData_ts
cgt[t] = CGNStype(t)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.UserDefinedData_ts
cgt[t] = CGNStype(t)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.IndexRange_ts, CGK.PointRange_s)
cgt[t].addChild(CGK.IndexArray_ts, CGK.PointList_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.FamilyName_ts, [CGK.FamilyName_s], card=C_01)
cgt[t].addChild(CGK.AdditionalFamilyName_ts, card=C_0N)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.Ordinal_ts, CGK.Ordinal_s)

# --------------------------------------------------------
t = CGK.Gravity_ts
cgt[t] = CGNStype(t)
cgt[t].addChild(CGK.DataArray_ts, CGK.GravityVector_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.FlowEquationSet_ts
cgt[t] = CGNStype(t, names=[CGK.FlowEquationSet_s])
cgt[t].addChild(CGK.GoverningEquations_ts, CGK.GoverningEquations_s)
cgt[t].addChild(CGK.EquationDimension_ts, CGK.EquationDimension_s)
cgt[t].addChild(CGK.GasModel_ts, CGK.GasModel_s)
cgt[t].addChild(CGK.ViscosityModel_ts, CGK.ViscosityModel_s)
cgt[t].addChild(CGK.ThermalRelaxationModel_ts, CGK.ThermalRelaxationModel_s)
cgt[t].addChild(CGK.ThermalConductivityModel_ts, CGK.ThermalConductivityModel_s)
cgt[t].addChild(CGK.TurbulenceModel_ts, CGK.TurbulenceModel_s)
cgt[t].addChild(CGK.TurbulenceClosure_ts, CGK.TurbulenceClosure_s)
cgt[t].addChild(CGK.ChemicalKineticsModel_ts, CGK.ChemicalKineticsModel_s)
cgt[t].addChild(CGK.EMMagneticFieldModel_ts, CGK.EMMagneticFieldModel_s)
cgt[t].addChild(CGK.EMElectricFieldModel_ts, CGK.EMElectricFieldModel_s)
cgt[t].addChild(CGK.EMConductivityModel_ts, CGK.EMConductivityModel_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.GoverningEquations_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.GoverningEquations_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.GoverningEquationsType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DiffusionModel_ts, CGK.DiffusionModel_s)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.GasModel_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.GasModel_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.GasModelType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.ViscosityModel_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.ViscosityModel_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.ViscosityModelType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.ThermalConductivityModel_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.ThermalConductivityModel_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.ThermalConductivityModelType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.TurbulenceClosure_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.TurbulenceClosure_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.TurbulenceClosureType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.TurbulenceModel_ts
cgt[t] = CGNStype(t, names=[CGK.TurbulenceModel_s])
cgt[t].datatype = [CGK.C1]
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.TurbulenceModelType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DiffusionModel_ts, CGK.DiffusionModel_s)

# --------------------------------------------------------
t = CGK.ThermalRelaxationModel_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.ThermalRelaxationModel_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.ThermalRelaxationModelType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.ChemicalKineticsModel_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.ChemicalKineticsModel_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.ChemicalKineticsModelType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.EMElectricFieldModel_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.EMElectricFieldModel_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.EMElectricFieldModelType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.EMMagneticFieldModel_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.EMMagneticFieldModel_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.EMMagneticFieldModelType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.EMConductivityModel_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.EMConductivityModel_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.EMConductivityModelType_l
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.ZoneType_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.ZoneType_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.ZoneType_l

# --------------------------------------------------------
t = CGK.SimulationType_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.SimulationType_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.SimulationType_l

# --------------------------------------------------------
t = CGK.GridConnectivityType_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.GridConnectivityType_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.GridConnectivityType_l

# --------------------------------------------------------
t = CGK.Family_ts
cgt[t] = CGNStype(t)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.Ordinal_ts, CGK.Ordinal_s)
cgt[t].addChild(CGK.FamilyBC_ts, card=C_01)
cgt[t].addChild(CGK.GeometryReference_ts)
cgt[t].addChild(CGK.RotatingCoordinates_ts, CGK.RotatingCoordinates_s)
cgt[t].addChild(CGK.FamilyName_ts, card=C_0N)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.FamilyName_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.FamilyName_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.AdditionalFamilyName_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.FamilyBC_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.FamilyBC_s])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.BCType_l
cgt[t].addChild(CGK.FamilyBCDataSet_ts)

# --------------------------------------------------------
t = CGK.FamilyBCDataSet_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1])
cgt[t].enumerate = CGK.BCTypeSimple_l
cgt[t].addChild(CGK.BCData_ts, CGK.NeumannData_s)
cgt[t].addChild(CGK.BCData_ts, CGK.DirichletData_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.ReferenceState_ts, CGK.ReferenceState_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.GeometryReference_ts
cgt[t] = CGNStype(t)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.GeometryFile_ts, CGK.GeometryFile_s)
cgt[t].addChild(CGK.GeometryFormat_ts, CGK.GeometryFormat_s)
cgt[t].addChild(CGK.GeometryEntity_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.GeometryFile_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.GeometryFile_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.GeometryFormat_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.GeometryFormat_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.GeometryEntity_ts
cgt[t] = CGNStype(t)

# --------------------------------------------------------
t = CGK.CGNSTree_ts
cgt[t] = CGNStype(t, names=[CGK.CGNSTree_s, UD])
cgt[t].addChild(CGK.CGNSLibraryVersion_ts, [CGK.CGNSLibraryVersion_s], card=C_11)
cgt[t].addChild(CGK.CGNSBase_ts, card=C_0N)

# --------------------------------------------------------
t = CGK.CGNSBase_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4])
cgt[t].shape = (0, 0)
cgt[t].addChild(CGK.Zone_ts, card=C_0N)
cgt[t].addChild(CGK.SimulationType_ts, [CGK.SimulationType_s], card=C_01)
cgt[t].addChild(CGK.BaseIterativeData_ts, card=C_01)
cgt[t].addChild(CGK.IntegralData_ts, card=C_0N)
cgt[t].addChild(CGK.ConvergenceHistory_ts, [CGK.GlobalConvergenceHistory_s], card=C_01)
cgt[t].addChild(CGK.Family_ts, card=C_0N)
cgt[t].addChild(CGK.FlowEquationSet_ts, [CGK.FlowEquationSet_s], card=C_01)
cgt[t].addChild(CGK.ReferenceState_ts, [CGK.ReferenceState_s], card=C_01)
cgt[t].addChild(CGK.Axisymmetry_ts, [CGK.Axisymmetry_s], card=C_01)
cgt[t].addChild(CGK.RotatingCoordinates_ts, [CGK.RotatingCoordinates_s], card=C_01)
cgt[t].addChild(CGK.Gravity_ts, [CGK.Gravity_s], card=C_01)
cgt[t].addChild(CGK.DataClass_ts, [CGK.DataClass_s], card=C_01)
cgt[t].addChild(CGK.DimensionalUnits_ts, [CGK.DimensionalUnits_s], card=C_01)
cgt[t].addChild(CGK.Descriptor_ts, card=C_0N)
cgt[t].addChild(CGK.UserDefinedData_ts, card=C_0N)

# --------------------------------------------------------
t = CGK.Zone_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4, CGK.I8])
cgt[t].shape = (0, 3)
cgt[t].addChild(CGK.GridCoordinates_ts, card=C_0N)
cgt[t].addChild(CGK.DiscreteData_ts, card=C_0N)
cgt[t].addChild(CGK.Elements_ts, card=C_0N)
cgt[t].addChild(CGK.ZoneBC_ts, CGK.ZoneBC_s, card=C_01)
cgt[t].addChild(CGK.FlowSolution_ts, card=C_0N)
cgt[t].addChild(CGK.ZoneSubRegion_ts, card=C_0N)
cgt[t].addChild(CGK.ZoneType_ts, CGK.ZoneType_s, card=C_11)
cgt[t].addChild(CGK.Ordinal_ts, CGK.Ordinal_s, card=C_01)
cgt[t].addChild(CGK.ZoneGridConnectivity_ts, CGK.ZoneGridConnectivity_s, card=C_01)
cgt[t].addChild(CGK.ZoneIterativeData_ts, card=C_01)
cgt[t].addChild(CGK.RigidGridMotion_ts, card=C_0N)
cgt[t].addChild(CGK.ReferenceState_ts, CGK.ReferenceState_s, card=C_01)
cgt[t].addChild(CGK.IntegralData_ts, card=C_0N)
cgt[t].addChild(CGK.ArbitraryGridMotion_ts, card=C_0N)
cgt[t].addChild(CGK.FamilyName_ts, CGK.FamilyName_s, card=C_01)
cgt[t].addChild(CGK.AdditionalFamilyName_ts, card=C_0N)
cgt[t].addChild(CGK.FlowEquationSet_ts, CGK.FlowEquationSet_s, card=C_01)
cgt[t].addChild(CGK.ConvergenceHistory_ts, CGK.ZoneConvergenceHistory_s, card=C_01)
cgt[t].addChild(CGK.RotatingCoordinates_ts, CGK.RotatingCoordinates_s, card=C_01)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s, card=C_01)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s, card=C_01)
cgt[t].addChild(CGK.Descriptor_ts, card=C_0N)
cgt[t].addChild(CGK.UserDefinedData_ts, card=C_0N)

# --------------------------------------------------------
t = CGK.GridCoordinates_ts
cgt[t] = CGNStype(t, names=[CGK.GridCoordinates_s, UD])
cgt[t].addChild(CGK.DataArray_ts, card=C_0N)
cgt[t].addChild(CGK.Rind_ts, CGK.Rind_s, card=C_01)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s, card=C_01)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s, card=C_01)
cgt[t].addChild(CGK.Descriptor_ts, card=C_0N)
cgt[t].addChild(CGK.UserDefinedData_ts, card=C_0N)

# --------------------------------------------------------
t = CGK.ZoneSubRegion_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4])
cgt[t].shape = (1,)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.IndexRange_ts, CGK.PointRange_s, card=C_01)
cgt[t].addChild(CGK.IndexArray_ts, CGK.PointList_s, card=C_01)
cgt[t].addChild(CGK.FamilyName_ts, CGK.FamilyName_s, card=C_01)
cgt[t].addChild(CGK.AdditionalFamilyName_ts, card=C_0N)
cgt[t].addChild(CGK.DataArray_ts, card=C_0N)
cgt[t].addChild(CGK.Rind_ts, CGK.Rind_s, card=C_01)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s, card=C_01)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s, card=C_01)
cgt[t].addChild(
    CGK.Descriptor_ts, [CGK.BCRegionName_s, CGK.GridConnectivityRegionName_s], card=C_0N
)
cgt[t].addChild(CGK.UserDefinedData_ts, card=C_0N)

# --------------------------------------------------------
t = CGK.Elements_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4])
cgt[t].shape = (2,)
cgt[t].addChild(CGK.IndexRange_ts, CGK.PointRange_s)
cgt[t].addChild(CGK.IndexArray_ts, CGK.PointList_s)
cgt[t].addChild(CGK.DataArray_ts, CGK.ElementConnectivity_s, card=C_01)
cgt[t].addChild(CGK.DataArray_ts, CGK.ElementStartOffset_s, card=C_01)
cgt[t].addChild(CGK.DataArray_ts, CGK.ParentElements_s, card=C_01)
cgt[t].addChild(CGK.DataArray_ts, CGK.ParentElementsPosition_s, card=C_01)
cgt[t].addChild(CGK.DataArray_ts, CGK.ParentData_s, card=C_01)
cgt[t].addChild(CGK.Rind_ts, CGK.Rind_s, card=C_01)
cgt[t].addChild(CGK.Descriptor_ts, card=C_0N)
cgt[t].addChild(CGK.UserDefinedData_ts, card=C_0N)

# --------------------------------------------------------
t = CGK.Axisymmetry_ts
cgt[t] = CGNStype(t, names=[CGK.Axisymmetry_s])
cgt[t].addChild(CGK.DataArray_ts, CGK.AxisymmetryReferencePoint_s)
cgt[t].addChild(CGK.DataArray_ts, CGK.AxisymmetryAxisVector_s)
cgt[t].addChild(CGK.DataArray_ts, CGK.AxisymmetryAngle_s)
cgt[t].addChild(CGK.DataArray_ts, CGK.CoordinateNames_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.RotatingCoordinates_ts
cgt[t] = CGNStype(t, names=[CGK.RotatingCoordinates_s])
cgt[t].addChild(CGK.DataArray_ts, CGK.RotationCenter_s)
cgt[t].addChild(CGK.DataArray_ts, CGK.RotationRateVector_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.FlowSolution_ts
cgt[t] = CGNStype(t)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.Rind_ts, CGK.Rind_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.DiscreteData_ts
cgt[t] = CGNStype(t)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.Rind_ts, CGK.Rind_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.ZoneBC_ts
cgt[t] = CGNStype(t, names=[CGK.ZoneBC_s])
cgt[t].addChild(CGK.BC_ts)
cgt[t].addChild(CGK.ReferenceState_ts, CGK.ReferenceState_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.BCProperty_ts
cgt[t] = CGNStype(t, names=[CGK.BCProperty_s])
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.WallFunction_ts, CGK.WallFunction_s)
cgt[t].addChild(CGK.Area_ts, CGK.Area_s)

# --------------------------------------------------------
t = CGK.BCData_ts
cgt[t] = CGNStype(t, names=[CGK.DirichletData_s, CGK.NeumannData_s])
cgt[t].addChild(CGK.DataArray_ts)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.BCDataSet_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1])
cgt[t].enumerate = CGK.BCTypeSimple_l
cgt[t].addChild(CGK.BCData_ts, CGK.NeumannData_s)
cgt[t].addChild(CGK.BCData_ts, CGK.DirichletData_s)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.IndexRange_ts, CGK.PointRange_s)
cgt[t].addChild(CGK.IndexArray_ts, CGK.PointList_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.ReferenceState_ts, CGK.ReferenceState_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.BC_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1])
cgt[t].enumerate = CGK.BCType_l
cgt[t].shape = (0,)
cgt[t].addChild(CGK.ReferenceState_ts, CGK.ReferenceState_s)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.Ordinal_ts, CGK.Ordinal_s)
cgt[t].addChild(CGK.FamilyName_ts, CGK.FamilyName_s)
cgt[t].addChild(CGK.AdditionalFamilyName_ts, card=C_0N)
cgt[t].addChild(CGK.IndexArray_ts, CGK.InwardNormalList_s)
cgt[t].addChild(CGK.BCDataSet_ts)
cgt[t].addChild(CGK.InwardNormalIndex_ts, CGK.InwardNormalIndex_s)
cgt[t].addChild(CGK.IndexArray_ts, CGK.ElementList_s)
cgt[t].addChild(CGK.IndexArray_ts, CGK.PointList_s)
cgt[t].addChild(CGK.IndexRange_ts, CGK.ElementRange_s)
cgt[t].addChild(CGK.IndexRange_ts, CGK.PointRange_s)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.BCProperty_ts, CGK.BCProperty_s)

# --------------------------------------------------------
t = CGK.ArbitraryGridMotionType_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.ArbitraryGridMotionType_s])
cgt[t].shape = (0,)
cgt[t].addChild(CGK.DataArray_ts, card=C_0N)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.Rind_ts, CGK.Rind_s, card=C_01)
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s, card=C_01)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s, card=C_01)
cgt[t].addChild(CGK.Descriptor_ts, card=C_0N)
cgt[t].addChild(CGK.UserDefinedData_ts, card=C_0N)

# --------------------------------------------------------
t = CGK.RigidGridMotionType_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.RigidGridMotionType_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.WallFunctionType_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.WallFunctionType_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.WallFunction_ts
cgt[t] = CGNStype(t, names=[CGK.WallFunction_s])
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.WallFunctionType_ts, CGK.WallFunctionType_s)

# --------------------------------------------------------
t = CGK.AreaType_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.AreaType_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.Area_ts
cgt[t] = CGNStype(t, names=[CGK.Area_s])
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.AreaType_ts, CGK.AreaType_s, card=C_11)
cgt[t].addChild(CGK.DataArray_ts, CGK.SurfaceArea_s, dtype=CGK.C1, card=C_11)
cgt[t].addChild(CGK.DataArray_ts, CGK.RegionName_s, dtype=CGK.C1, card=C_11)

# --------------------------------------------------------
t = CGK.BaseIterativeData_ts
cgt[t] = CGNStype(t, dtype=[CGK.I4])
cgt[t].shape = (1,)
cgt[t].addChild(CGK.DataClass_ts, [CGK.DataClass_s], card=C_01)
cgt[t].addChild(CGK.DimensionalUnits_ts, [CGK.DimensionalUnits_s], card=C_01)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.DataArray_ts)

# --------------------------------------------------------
t = CGK.ZoneIterativeData_ts
cgt[t] = CGNStype(t)
cgt[t].addChild(CGK.DataClass_ts, [CGK.DataClass_s], card=C_01)
cgt[t].addChild(CGK.DimensionalUnits_ts, [CGK.DimensionalUnits_s], card=C_01)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(
    CGK.DataArray_ts,
    [
        CGK.RigidGridMotionPointers_s,
        CGK.ArbitraryGridMotionPointers_s,
        CGK.FlowSolutionPointers_s,
        CGK.ZoneGridConnectivityPointers_s,
        CGK.ZoneSubRegionPointers_s,
    ],
)

# --------------------------------------------------------
t = CGK.RigidGridMotion_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.RigidGridMotionType_l
cgt[t].addChild(CGK.DataClass_ts, [CGK.DataClass_s], card=C_01)
cgt[t].addChild(CGK.DimensionalUnits_ts, [CGK.DimensionalUnits_s], card=C_01)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(
    CGK.DataArray_ts,
    [
        CGK.OriginLocation_s,
        CGK.RigidRotationAngle_s,
        CGK.RigidRotationRate_s,
        CGK.RigidVelocity_s,
    ],
)

# --------------------------------------------------------
t = CGK.ArbitraryGridMotion_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1])
cgt[t].shape = (0,)
cgt[t].enumerate = CGK.ArbitraryGridMotionType_l
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.Rind_ts, CGK.Rind_s)
cgt[t].addChild(CGK.DataArray_ts)

# --------------------------------------------------------
t = CGK.ZoneGridConnectivity_ts
cgt[t] = CGNStype(t, names=[CGK.ZoneGridConnectivity_s])
cgt[t].addChild(CGK.GridConnectivity1to1_ts)
cgt[t].addChild(CGK.GridConnectivity_ts)
cgt[t].addChild(CGK.OversetHoles_ts)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.GridConnectivityProperty_ts
cgt[t] = CGNStype(t, names=[CGK.GridConnectivityProperty_s])
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.Periodic_ts, CGK.Periodic_s)
cgt[t].addChild(CGK.AverageInterface_ts, CGK.AverageInterface_s)

# --------------------------------------------------------
t = CGK.Periodic_ts
cgt[t] = CGNStype(t, names=[CGK.Periodic_s])
cgt[t].addChild(CGK.DataClass_ts, CGK.DataClass_s)
cgt[t].addChild(CGK.DimensionalUnits_ts, CGK.DimensionalUnits_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.DataArray_ts, CGK.RotationCenter_s)
cgt[t].addChild(CGK.DataArray_ts, CGK.RotationAngle_s)
cgt[t].addChild(CGK.DataArray_ts, CGK.Translation_s)

# --------------------------------------------------------
t = CGK.AverageInterfaceType_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.AverageInterfaceType_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.AverageInterface_ts
cgt[t] = CGNStype(t, names=[CGK.AverageInterface_s])
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)
cgt[t].addChild(CGK.AverageInterfaceType_ts, CGK.AverageInterfaceType_s)

# --------------------------------------------------------
t = CGK.GridConnectivity1to1_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1])
cgt[t].shape = (0,)
cgt[t].addChild(CGK.Transform_ts, CGK.Transform_s)
cgt[t].addChild(CGK.IntIndexDimension_ts, CGK.Transform_s)
cgt[t].addChild(CGK.Transform_ts2, CGK.Transform_s)
cgt[t].addChild(CGK.IndexRange_ts, CGK.PointRange_s)
cgt[t].addChild(CGK.IndexRange_ts, CGK.PointRangeDonor_s)
cgt[t].addChild(CGK.Ordinal_ts, CGK.Ordinal_s)
cgt[t].addChild(CGK.GridConnectivityProperty_ts, CGK.GridConnectivityProperty_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.GridConnectivityType_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1], names=[CGK.GridConnectivityType_s])
cgt[t].shape = (0,)

# --------------------------------------------------------
t = CGK.GridConnectivity_ts
cgt[t] = CGNStype(t, dtype=[CGK.C1])
cgt[t].shape = (0,)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.Ordinal_ts, CGK.Ordinal_s)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.IndexRange_ts, CGK.PointRange_s)
cgt[t].addChild(CGK.IndexArray_ts, CGK.PointList_s)
cgt[t].addChild(CGK.IndexArray_ts, CGK.PointListDonor_s)
cgt[t].addChild(CGK.IndexArray_ts, CGK.CellListDonor_s)
cgt[t].addChild(CGK.GridConnectivityProperty_ts, CGK.GridConnectivityProperty_s)
cgt[t].addChild(CGK.GridConnectivityType_ts, CGK.GridConnectivityType_s)
cgt[t].addChild(CGK.DataArray_ts, CGK.InterpolantsDonor_s)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
t = CGK.OversetHoles_ts
cgt[t] = CGNStype(t)
cgt[t].addChild(CGK.Descriptor_ts)
cgt[t].addChild(CGK.IndexArray_ts, CGK.PointList_s)
cgt[t].addChild(CGK.GridLocation_ts, CGK.GridLocation_s)
cgt[t].addChild(CGK.IndexRange_ts)
cgt[t].addChild(CGK.UserDefinedData_ts)

# --------------------------------------------------------
types = cgt
tk = list(types)
tk.sort()
for pk in tk:
    for ck in tk:
        if (ck != pk) and (types[pk].hasChild(ck)):
            types[ck].addParent(pk)

# --- reserved names / SIDS types
cgnsnametypes = {
    CGK.Density_s: (CGK.DataArray_ts, None),
    CGK.Pressure_s: (CGK.DataArray_ts, None),
    CGK.Temperature_s: (CGK.DataArray_ts, None),
    CGK.EnergyInternal_s: (CGK.DataArray_ts, None),
    CGK.Enthalpy_s: (CGK.DataArray_ts, None),
    CGK.Entropy_s: (CGK.DataArray_ts, None),
    CGK.EntropyApprox_s: (CGK.DataArray_ts, None),
    CGK.DensityStagnation_s: (CGK.DataArray_ts, None),
    CGK.PressureStagnation_s: (CGK.DataArray_ts, None),
    CGK.TemperatureStagnation_s: (CGK.DataArray_ts, None),
    CGK.EnergyStagnation_s: (CGK.DataArray_ts, None),
    CGK.EnthalpyStagnation_s: (CGK.DataArray_ts, None),
    CGK.EnergyStagnationDensity_s: (CGK.DataArray_ts, None),
}

for pk in tk:
    pkn = types[pk].names
    if pkn != [UD]:
        for curname in pkn:
            if pkn != UD:
                cgnsnametypes[curname] = (types[pk].type, types[pk].enumerate)

# --- last line
