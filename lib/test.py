# =============================================================================
# pyCGNS - CFD General Notation System - ONERA - marc.poinot@onera.fr
# $Rev: 79 $ $Date: 2009-03-13 10:19:54 +0100 (Fri, 13 Mar 2009) $
# See file 'license' in the root directory of this Python module source 
# tree for license information. 
# =============================================================================

from CGNS.MAP.test import run as MAP_test
from CGNS.PAT.test import run as PAT_test
from CGNS.WRA.test import run as WRA_test
from CGNS.VAL.test import run as VAL_test
from CGNS.NAV.test import run as NAV_test
# DAT
from CGNS.APP.test import run as APP_test

MAP_test()
PAT_test()
WRA_test()
VAL_test()
NAV_test()
APP_test()

# --- last line

