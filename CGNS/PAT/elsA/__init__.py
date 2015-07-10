#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
from . import CGNSBase_t
from . import ReferenceState_t
from . import SolverBC
from . import SolverBC_injmfr1

profile={ 
'CGNSBase':          CGNSBase_t.pattern,
'Solver#Compute':    SolverCompute.pattern,
'Solver#Compute 01': SolverCompute_01.pattern,
'Solver#BC':         SolverBC.pattern,
'Solver#BC injmfr1': SolverBC_injmfr1.pattern,
'ReferenceState':    ReferenceState_t.pattern
}

# -- last line
