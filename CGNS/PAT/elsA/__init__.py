#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
from . import CGNSBase_t
from . import ReferenceState_t
from . import SolverBC

profile={ 
'CGNSBase':       CGNSBase_t.pattern,
'Solver#BC':      SolverBC.pattern,
'ReferenceState': ReferenceState_t.pattern
}

# -- last line
