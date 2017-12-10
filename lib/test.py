#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

from CGNS.MAP.test import run as MAP_test

MAP_test()

from CGNS.PAT.test import run as PAT_test

PAT_test()

try:
    from CGNS.VAL.test import run as VAL_test

    VAL_test()
except ImportError:
    pass

try:
    from CGNS.NAV.test import run as NAV_test

    NAV_test()
except ImportError:
    pass

from CGNS.APP.test import run as APP_test

APP_test()

# --- last line
