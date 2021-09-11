#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
TAG = "\n### pyCGNS:"
TAG_ERROR = " ERROR "


def perr(error_id, *tp):
    try:
        msg = TAG + TAG_ERROR + "[%.3d] %s" % (error_id, errorTable[error_id])
    except (TypeError, KeyError):
        msg = TAG + TAG_ERROR + "[%.3d] %s" % (error_id, errorTable[999])
    ret = msg
    if tp:
        if isinstance(tp, tuple) and (len(tp) > 0):
            ret = msg % tp[0]
        else:
            ret = msg % tp
    return ret


class CGNSException(Exception):
    def __init__(self, code, args=None):
        self.code = code
        self.value = args

    def __str__(self):
        if self.value is not None:
            ret = perr(self.code, self.value)
        else:
            ret = perr(self.code)
        return ret


cgnsException = CGNSException


class cgnsNodeError(cgnsException):
    pass


class cgnsNameError(cgnsNodeError):
    pass


class cgnsTypeError(cgnsNodeError):
    pass


# -----------------------------------------------------------------------------
errorTable = {
    -1: "Stop on check",
    0: "No error",
    1: "Node is empty !",
    2: "Node should be a list of <name, value, children, type>",
    3: "Node name should be a string",
    4: "Node [%s] children list should be a list",
    5: "Node [%s] bad value: should be a numpy object",
    6: "Parent node for [%s] is not a CGNS node",
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
    24: "Node name should not contain a '/' (you have [%s])",
    25: "Node name length should not be greater than 32 chars: %s",
    26: "Bad name [%s] for this node",
    27: "Bad type [%s] for this node",
    28: "Node [%s] should have no child",
    29: "Node name cannot be '.' or '..'",
    30: "Nodes [%s] and [%s] are different",
    31: "Node name cannot be only ' '",
    32: "Node name should not have prefix or suffix ' 's",
    33: "Node name contains unsafe chars",
    34: "Node name contains two consecutive spaces",
    40: "Node [%s] has type [%s] which is not a CGNS/SIDS type",
    90: "Node is None!",
    91: "Root node can have only CGNSBase_t or CGNSLibraryVersion_t children",
    99: "More than one CGNSLibraryVersion node found",
    100: "Absolute path implies a root node with CGNSLibraryVersion node",
    101: "No child on CGNS tree first level",
    102: "Duplicated child name [%s] in [%s]",
    103: "Node type of [%s] not [%s]",
    104: "Node type of [%s] not in %s",
    105: "String data should have type array or string",
    106: "Value should be a real array",
    107: "Value should be an integer array",
    108: "Parent node should have type [%s]",
    109: "Value should be an array",
    110: "Value should be a string in [%s]",
    111: "Bad value type",
    112: "Empty args is not allowed as value",
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
    300: "Zone creation requires a dimension tuple",
    710: "Numpy array should have a 'Fortran' order",
    800: "Cannot find pyCGNS config ?",
    801: "No information about numeric library in pyCGNS config",
    802: "Bad information about numeric library in pyCGNS config [%s]",
    900: "No such file [{}]",
    901: "File already exists [{}]",
    902: "File name should be a string",
    903: "Cannot propagate save to non-existant linked-to file [{}]",
    904: "Cannot propagate save to unreadable linked-to file [{}]",
    905: "Cannot propagate save to non-existant node in linked-to file [{}]",
    906: "Bad tree structure (root should be list of 4 items, refer to doc)",
    907: "Invalid combination of flags (refer to doc)",
    908: "Using a maxdata implies FNODATA flag",
    909: "Cannot have attributes {} at the same time",
    910: "Keyword argument [{}] unknown",
    911: "Bad [flag] argument (should be int)",
    912: "Bad [depth] argument (should be int)",
    913: "Bad [path] argument (should be str)",
    914: "Bad [lksearch] argument (should be list of str)",
    915: "Bad [maxdata] argument (should be int)",
    916: "Bad [skip] argument (should be list of str)",
    917: "Bad [links] argument (refer to doc)",
    920: "Cannot write in file [{}]",
    930: "Filter dict requires a path (str) as key",
    931: "Filter dict value should be a tuple with int as first arg",
    932: 'SPARTIAL Filter requires a "list of 8 lists of integers" as value"',
    933: "SPARTIAL Filter bad parameters",
    934: "SCONTIGUOUS requires a tuple of index,rank (integers) as value",
    999: "Unknow error code",
}

CGNS_NoSuchFile = "pyCGNS[001] No such file"
CGNS_OpenFileFailed = "pyCGNS[002] Open file failed"
CGNS_FileAlreadyExists = "pyCGNS[003] File already exists"
CGNS_NotConnected = "pyCGNS[100] Not yet connected to a CGNS database"
CGNS_NoSuchBase = "pyCGNS[101] No such base name"
CGNS_AbsolutePathRequired = "pyCGNS[102] Absolute path required"
CGNS_NoSuchZone = "pyCGNS[103] No such zone name"
CGNS_NoSuchArray = "pyCGNS[104] No such array name"
CGNS_NoSuchUserData = "pyCGNS[105] No such user data node name"
CGNS_NoSuchFlowSolution = "pyCGNS[106] No such flow solution node name"
CGNS_NoSuchBoundary = "pyCGNS[107] No such boundary condition node name"
CGNS_FileShouldBeReadable = "pyCGNS[110] File should be readable"
CGNS_GotoPathNotFound = "pyCGNS[901] No such goto path in CGNS tree"
CGNS_BadType = "pyCGNS[200] Bad type"
CGNS_BadOpenMode = "pyCGNS[201] Bad OpenMode value"
CGNS_BadErrorCode = "pyCGNS[202] Bad ErrorCode type"
CGNS_BadMassUnit = "pyCGNS[203] Bad MassUnit type"
CGNS_BadLengthUnit = "pyCGNS[204] Bad LengthUnit type"
CGNS_BadTimeUnit = "pyCGNS[205] Bad TimeUnit type"
CGNS_BadTemperatureUnit = "pyCGNS[206] Bad TemperatureUnit type"
CGNS_BadAngleUnit = "pyCGNS[207] Bad AngleUnit type"
CGNS_BadDataClass = "pyCGNS[208] Bad DataClass type"
CGNS_BadGridLocation = "pyCGNS[209] Bad GridLocation type"
CGNS_BadBCDataType = "pyCGNS[210] Bad BC data type"
CGNS_BadGridConnectivityType = "pyCGNS[211] Bad GridConnectivityType type"
CGNS_BadPointSetType = "pyCGNS[212] Bad PointSetType type"
CGNS_BadGoverningType = "pyCGNS[213] Bad GoverningEquation type"
CGNS_BadModelType = "pyCGNS[214] Bad Model type"
CGNS_BadBCType = "pyCGNS[215] Bad BC type"
CGNS_BadDataType = "pyCGNS[216] Bad DataType type"
CGNS_BadElementType = "pyCGNS[217] Bad ElementType type"
CGNS_BadZoneType = "pyCGNS[218] Bad ZoneType type"
CGNS_BadRigidGridMotionType = "pyCGNS[219] Bad RigidGridMotionType type"
CGNS_BadArbitraryGridMotionType = "pyCGNS[220] Bad ArbitraryGridMotionType type"
CGNS_BadSimulationType = "pyCGNS[221] Bad SimulationType type"
CGNS_BadWallFunctionType = "pyCGNS[222] Bad WallFunctionType type"
CGNS_BadAreaType = "pyCGNS[223] Bad AreaType type"
CGNS_BadAverageInterfaceType = "pyCGNS[224] Bad AverageInterfaceType type"

# -----------------------------------------------------------------------------
