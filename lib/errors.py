#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
TAG = "\n# pyCGNS:"


class cgnsException(Exception):
    def __init__(self, value, msg=None):
        self.value = value
        self.msg = msg

    def set(self, msg):
        self.msg = msg

    def __perr(self, error_id, *tp):
        try:
            msg = TAG + " ERROR [%.3d]- %s" % (error_id, errorTable[error_id])
        except (TypeError, KeyError):
            msg = TAG + " ERROR [%.3d]- %s" % (error_id, errorTable[999])
        ret = msg
        if tp:
            if isinstance(tp, tuple) and (len(tp) > 0):
                ret = msg % tp
            else:
                ret = msg % tp
        return ret

    def __str__(self):
        if self.msg:
            ret = self.__perr(self.value, self.msg)
        else:
            ret = self.__perr(self.value)
        return ret


# -----------------------------------------------------------------------------
errorTable = {
    0: "No error",
    1: "Node is empty !",
    2: "Node should be a list of <name, value, children, type>",
    3: "Node name should be a string",
    4: "Node [%s] children list should be a list",
    5: "More than one CGNSLibraryVersion node found",
    6: "Parent node for [%s] is node a CGNS node",
    7: "Failed to add a badly formed child node into [%s]",
    8: "Failed to add root node as child node into [%s]",
    9: "Link chaser returns a badly formed node as place of [%s]",
    10: "CGNSBase [%s] bad cell dimensions",
    11: "CGNSBase [%s] bad physical dimensions",
    12: "CGNSBase [%s] bad cell/physical dimensions values",
    20: "No node of type [%s] with name [%s]",
    21: "No node with name [%s]",
    22: "Node name should have type string",
    23: "Empty string is not allowed for a node name",
    24: "Node name should not contain a '/'",
    25: "Node name length should not be greater than 32 chars",
    26: "Bad name [%s] for this node",
    27: "Bad type [%s] for this node",
    28: "Node [%s] should have no child",
    100: "Absolute path implies a root node with CGNSLibraryVersion node",
    101: "No child on CGNS tree first level",
    102: "Duplicated child name [%s] in [%s]",
    103: "Parent node of [%s] should have type [%s]",
    104: "Parent node of [%s] should have type in %s",
    105: "String data should have type array or string",
    106: "Value should be a real array",
    107: "Value should be an integer array",
    108: "Parent node should have type [%s]",
    109: "Value should be an array",
    110: "Value should be a string in [%s]",
    200: "Bad GridLocation value [%s]",
    201: "Bad ConvergenceHistory name [%s]",
    202: "Units specification list should have 5 values (one per unit)",
    203: "Bad units specification [%s]",
    204: "Units specifications duplicated [%s]",
    205: "Bad SimulationType value [%s]",
    206: "Bad ZoneType value [%s]",
    207: "Bad DataClass value [%s]",
    208: "Exponents specification list should have 5 values (one per unit)",
    209: "BaseIterativeData number of steps should be an integer value",
    210: "BaseIterativeData steps should in %s",
    211: "BaseIterativeData value has greater length than number of steps",
    212: "BaseIterativeData is missing sub-node",
    213: "Parent node should be BaseIterativeData",
    214: "BaseIterativeData bad Pointer name [%s]",
    215: "BaseIterativeData Pointer list [%s] has not the right length [%s]",
    216: "BaseIterativeData Pointer list [%s] contents has bad length [%s]",
    217: "Parent node should be EMConductivityModel",
    218: "Bad EMConductivityModelType value [%s]",
    219: "EMConductivityModelType is missing sub-node",
    220: "Parent node should be GoverningEquations",
    221: "Bad GoverningEquationsType value [%s]",
    222: "GoverningEquationsType is missing sub-node",
    223: "Parent node should be GasModel",
    224: "Bad GasModelType value [%s]",
    225: "GasModelType is missing sub-node",
    226: "Parent node should be ThermalConductivityModel",
    227: "Bad ThermalConductivityModelType value [%s]",
    228: "ThermalConductivityModelType is missing sub-node",
    229: "Parent node should be ViscosityModel",
    230: "Bad ViscosityModelType value [%s]",
    231: "ViscosityModelType is missing sub-node",
    232: "Parent node should be TurbulenceClosure",
    233: "Bad TurbulenceClosureType value [%s]",
    234: "TurbulenceClosureType is missing sub-node",
    235: "Parent node should be TurbulenceModel",
    236: "Bad TurbulenceModelType value [%s]",
    237: "TurbulenceModelType is missing sub-node",
    238: "Parent node should be ThermalRelaxationModel",
    239: "Bad ThermalRelaxationModelType value [%s]",
    240: "ThermalRelaxationModelType is missing sub-node",
    241: "Parent node should be ChemicalKineticsModel",
    242: "Bad ChemicalKineticsModelType value [%s]",
    243: "ChemicalKineticsModelType is missing sub-node",
    244: "Parent node should be EMElectricFieldModel",
    245: "Bad EMElectricFieldModelType value [%s]",
    246: "EMElectricFieldModelType is missing sub-node",
    247: "Parent node should be EMMagneticFieldModel",
    248: "Bad EMMagneticFieldModelType value [%s]",
    249: "EMMagneticFieldModelType is missing sub-node",
    250: "Bad ElementType value [%s]",
    251: "Bad ElementConnectivity value [%s]",
    252: "Bad BCTypeSimple value [%s]",
    253: "Bad AverageInterfaceType value [%s]",
    254: "Bad RigidGridMotionType value [%s]",
    255: "Bad ArbitraryGridMotionType value [%s]",
    256: "Bad GeometryFormat value [%s]",
    257: "Bad BCTypeSimple or BCTypeCompound value [%s]",
    258: "Parent node should be ArbitraryGridMotion_t",
    259: "Parent node should be ReferenceState_t",
    260: "Parent node should be IntegralData_t",
    261: "Parent node should be UserDefinedData_t",
    262: "Parent node should be Family_t",
    263: "Parent node should be GridCoordinates_t",
    264: "Parent node should be DiscreteData_t",
    265: "Parent node should be BCData_t",
    266: "Parent node should be ZoneIterativeData_t",
    267: "Parent node should be AverageInterface_t",
    268: "Bad AverageInterfaceType value [%s]",
    269: "AverageInterfaceType is missing sub-node",
    270: "Parent node should be ZoneGridConnectivity_t",
    271: "Bad ZoneGridConnectivityType value [%s]",
    272: "ZoneGridConnectivityType is missing sub-node",
    273: "Parent node should be DimensionalExponents_t",
    274: "Bad SimulationType value [%s]",
    275: "Parent node should be BCDataSet_t",
    276: "BCDataSetType is missing sub-node",
    277: "Parent node should be ZoneBC_t",
    278: "Parent node should be GridCoordinates_t",
    279: "Parent node should be RotatingCoordinates_t",
    280: "Parent node should be Axisymmetry_t",
    281: "Parent node should be FlowSolution_t ",
    282: "Parent node should be Periodic_s ",
    283: "Parent node should be OversetHoles_t ",
    284: "Parent node should be GridConnectivity1to1_t ",
    285: "Parent node should be GridCoordinates",
    286: "Parent node should be DiscreteData_t",
    287: "Parent node should be DataArray_t",
    288: "Parent node should be Zone_t",
    289: "Parent node should be DataConversion_t",
    290: "Parent node should be GridLocation_t",
    291: "Parent node should be Descriptor_t",
    292: "Parent node should be GeometryReference_t",
    293: "GeometryReferenceType is missing sub-node",
    294: "Parent node should be FamilyBC_t",
    295: "FamilyBCType is missing sub-node",

    800: "adf.database_open No such open status [%s]",
    801: "adf.database_open No such open format [%s]",
    802: "adf.database_open Empty file name",

    900: "Cannot find pyCGNS config ?",
    901: "No information about numeric library in pyCGNS config",
    902: "Bad information about numeric library in pyCGNS config [%s]",

    999: "Unknow error code",

    1001: "No such file",
    1002: "Open file failed",
    1003: "File already exists",
    1004: "No file name",
    1005: "Bad ADF status at open",
    1006: "Bad ADF format at open",
    1100: "Not yet connected to a CGNS database",
    1101: "No such base name",
    1102: "Absolute path required",
    1103: "No such zone name",
    1104: "No such array name",
    1105: "No such user data node name",
    1106: "No such flow solution node name",
    1107: "No such boundary condition node name",
    1110: "File should be readable",
    1901: "No such goto path in CGNS tree",
    1200: "Bad type",
    1201: "Bad OpenMode value",
    1202: "Bad ErrorCode type",
    1203: "Bad MassUnit type",
    1204: "Bad LengthUnit type",
    1205: "Bad TimeUnit type",
    1206: "Bad TemperatureUnit type",
    1207: "Bad AngleUnit type",
    1208: "Bad DataClass type",
    1209: "Bad GridLocation type",
    1210: "Bad BC data type",
    1211: "Bad GridConnectivityType type",
    1212: "Bad PointSetType type",
    1213: "Bad GoverningEquation type",
    1214: "Bad Model type",
    1215: "Bad BC type",
    1216: "Bad DataType type",
    1217: "Bad ElementType type",
    1218: "Bad ZoneType type",
    1219: "Bad RigidGridMotionType type",
    1220: "Bad ArbitraryGridMotionType type",
    1221: "Bad SimulationType type",
    1222: "Bad WallFunctionType type",
    1223: "Bad AreaType type",
    1224: "Bad AverageInterfaceType type",

}

CGNS_NoSuchFile = cgnsException(1001)
CGNS_OpenFileFailed = cgnsException(1002)
CGNS_FileAlreadyExists = cgnsException(1003)
CGNS_NoFileName = cgnsException(1004)
CGNS_BadADFstatus = cgnsException(1005)
CGNS_BadADFformat = cgnsException(1006)
CGNS_NotConnected = cgnsException(1100)
CGNS_NoSuchBase = cgnsException(1101)
CGNS_AbsolutePathRequired = cgnsException(1102)
CGNS_NoSuchZone = cgnsException(1103)
CGNS_NoSuchArray = cgnsException(1104)
CGNS_NoSuchUserData = cgnsException(1105)
CGNS_NoSuchFlowSolution = cgnsException(1106)
CGNS_NoSuchBoundary = cgnsException(1107)
CGNS_FileShouldBeReadable = cgnsException(1110)
CGNS_GotoPathNotFound = cgnsException(1901)
CGNS_BadType = cgnsException(1200)
CGNS_BadOpenMode = cgnsException(1201)
CGNS_BadErrorCode = cgnsException(1202)
CGNS_BadMassUnit = cgnsException(1203)
CGNS_BadLengthUnit = cgnsException(1204)
CGNS_BadTimeUnit = cgnsException(1205)
CGNS_BadTemperatureUnit = cgnsException(1206)
CGNS_BadAngleUnit = cgnsException(1207)
CGNS_BadDataClass = cgnsException(1208)
CGNS_BadGridLocation = cgnsException(1209)
CGNS_BadBCDataType = cgnsException(1210)
CGNS_BadGridConnectivityType = cgnsException(1211)
CGNS_BadPointSetType = cgnsException(1212)
CGNS_BadGoverningType = cgnsException(1213)
CGNS_BadModelType = cgnsException(1214)
CGNS_BadBCType = cgnsException(1215)
CGNS_BadDataType = cgnsException(1216)
CGNS_BadElementType = cgnsException(1217)
CGNS_BadZoneType = cgnsException(1218)
CGNS_BadRigidGridMotionType = cgnsException(1219)
CGNS_BadArbitraryGridMotionType = cgnsException(1220)
CGNS_BadSimulationType = cgnsException(1221)
CGNS_BadWallFunctionType = cgnsException(1222)
CGNS_BadAreaType = cgnsException(1223)
CGNS_BadAverageInterfaceType = cgnsException(1224)

# -----------------------------------------------------------------------------
