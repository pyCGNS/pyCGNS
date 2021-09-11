#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
from . import CGNSBase_t
from . import Zone_t
from . import ZoneBC_t
from . import BC_t
from . import DataClass_t
from . import Descriptor_t
from . import DimensionalUnits_t
from . import DimensionalExponents_t
from . import GridLocation_t
from . import PointList_t
from . import PointRange_t
from . import Rind_t
from . import DataConversion_t
from . import SimulationType_t
from . import Ordinal_t
from . import GridCoordinates_t
from . import DataArray_t
from . import DiscreteData_t
from . import Elements_t
from . import BCDataSet_t
from . import BCData_t
from . import BCProperty_t
from . import RotatingCoordinates_t
from . import ZoneGridConnectivity_t
from . import GridConnectivity1to1_t
from . import GridConnectivity_t
from . import GridConnectivityProperty_t
from . import AverageInterface_t
from . import OversetHoles_t
from . import FlowSolution_t
from . import FlowEquationSet_t
from . import GoverningEquations_t
from . import GasModel_t
from . import ThermalConductivityModel_t
from . import ThermalRelaxationModel_t
from . import ChemicalKineticModel_t
from . import EMElectricFieldModel_t
from . import EMMagneticFieldModel_t
from . import EMConductivityModel_t
from . import ViscosityModel_t
from . import TurbulenceClosure_t
from . import TurbulenceModel_t
from . import AxiSymmetry_t
from . import BaseIterativeData_t
from . import ZoneIterativeData_t
from . import RigidGridMotion_t
from . import ArbitraryGridMotion_t
from . import ReferenceState_t
from . import ConvergenceHistory_t
from . import IntegralData_t
from . import UserDefinedData_t
from . import Gravity_t
from . import Family_t
from . import FamilyName_t
from . import FamilyBC_t
from . import GeometryReference_t
from . import ZoneSubRegion_t

profile = {
    "CGNSBase_t": CGNSBase_t.pattern,
    "Zone_t": Zone_t.pattern,
    "ZoneBC_t": ZoneBC_t.pattern,
    "BC_t": BC_t.pattern,
    "DataClass_t": DataClass_t.pattern,
    "Descriptor_t": Descriptor_t.pattern,
    "DimensionalUnits_t": DimensionalUnits_t.pattern,
    "DimensionalExponents_t": DimensionalExponents_t.pattern,
    "GridLocation_t": GridLocation_t.pattern,
    "PointList_t": PointList_t.pattern,
    "PointRange_t": PointRange_t.pattern,
    "Rind_t": Rind_t.pattern,
    "DataConversion_t": DataConversion_t.pattern,
    "SimulationType_t": SimulationType_t.pattern,
    "Ordinal_t": Ordinal_t.pattern,
    "GridCoordinates_t": GridCoordinates_t.pattern,
    "DataArray_t": DataArray_t.pattern,
    "DiscreteData_t": DiscreteData_t.pattern,
    "Elements_t": Elements_t.pattern,
    "BCDataSet_t": BCDataSet_t.pattern,
    "BCData_t": BCData_t.pattern,
    "BCProperty_t": BCProperty_t.pattern,
    "RotatingCoordinates_t": RotatingCoordinates_t.pattern,
    "ZoneGridConnectivity_t": ZoneGridConnectivity_t.pattern,
    "GridConnectivity1to1_t": GridConnectivity1to1_t.pattern,
    "GridConnectivity_t": GridConnectivity_t.pattern,
    "GridConnectivityProperty_t": GridConnectivityProperty_t.pattern,
    "AverageInterface_t": AverageInterface_t.pattern,
    "OversetHoles_t": OversetHoles_t.pattern,
    "FlowSolution_t": FlowSolution_t.pattern,
    "FlowEquationSet_t": FlowEquationSet_t.pattern,
    "GoverningEquations_t": GoverningEquations_t.pattern,
    "GasModel_t": GasModel_t.pattern,
    "ThermalConductivityModel_t": ThermalConductivityModel_t.pattern,
    "ThermalRelaxationModel_t": ThermalRelaxationModel_t.pattern,
    "ChemicalKineticModel_t": ChemicalKineticModel_t.pattern,
    "EMElectricFieldModel_t": EMElectricFieldModel_t.pattern,
    "EMMagneticFieldModel_t": EMMagneticFieldModel_t.pattern,
    "EMConductivityModel_t": EMConductivityModel_t.pattern,
    "ViscosityModel_t": ViscosityModel_t.pattern,
    "TurbulenceClosure_t": TurbulenceClosure_t.pattern,
    "TurbulenceModel_t": TurbulenceModel_t.pattern,
    "AxiSymmetry_t": AxiSymmetry_t.pattern,
    "BaseIterativeData_t": BaseIterativeData_t.pattern,
    "ZoneIterativeData_t": ZoneIterativeData_t.pattern,
    "RigidGridMotion_t": RigidGridMotion_t.pattern,
    "ArbitraryGridMotion_t": ArbitraryGridMotion_t.pattern,
    "ReferenceState_t": ReferenceState_t.pattern,
    "ConvergenceHistory_t": ConvergenceHistory_t.pattern,
    "IntegralData_t": IntegralData_t.pattern,
    "UserDefinedData_t": UserDefinedData_t.pattern,
    "Gravity_t": Gravity_t.pattern,
    "Family_t": Family_t.pattern,
    "FamilyName_t": FamilyName_t.pattern,
    "FamilyBC_t": FamilyBC_t.pattern,
    "GeometryReference_t": GeometryReference_t.pattern,
    "ZoneSubRegion_t": ZoneSubRegion_t.pattern,
}
#
