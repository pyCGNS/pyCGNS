#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import unicode_literals
from __future__ import print_function


def run():
    import CGNS.VAL.test.cgu


def run2():
    print('### pyCGNS TEST: starting VAL tests')
    import CGNS.VAL.suite.run
    CGNS.VAL.suite.run.runall()

# --- last line
