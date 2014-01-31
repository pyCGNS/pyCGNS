#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

from CGNS.MAP.test import run as MAP_test
from CGNS.PAT.test import run as PAT_test
from CGNS.WRA.test import run as WRA_test
from CGNS.VAL.test import run as VAL_test
from CGNS.NAV.test import run as NAV_test
from CGNS.DAT.test import run as DAT_test
from CGNS.APP.test import run as APP_test

MAP_test()
PAT_test()
WRA_test()
VAL_test()
NAV_test()
APP_test()
DAT_test()

# --- last line

