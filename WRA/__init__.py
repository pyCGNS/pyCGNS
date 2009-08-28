# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 79 $ $Date: 2009-03-13 10:19:54 +0100 (Fri, 13 Mar 2009) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
import CGNS.pyCGNSconfig

if (CGNS.pyCGNSconfig.HAS_MLL != False):
  from CGNS.WRA.midlevel  import *
  from CGNS.WRA.wrap      import *
  from CGNS.WRA.utils     import *

from CGNS.WRA.cgnslib      import *
from CGNS.WRA.cgnskeywords import *
from CGNS.WRA.version      import *

config="# pyCGNS v%s - The Python CGNS API\n"%version
config+="# produced %s on %s %s %s"%(CGNS.pyCGNSconfig.DATE,
                            CGNS.pyCGNSconfig.PLATFORM[0],
                            CGNS.pyCGNSconfig.PLATFORM[2],
                            CGNS.pyCGNSconfig.PLATFORM[4])

if (CGNS.pyCGNSconfig.HAS_MLL == True):
  config+="""#
# using CGNS Librarie v%s found in %s
"""%(CGNS.pyCGNSconfig.MLL_VERSION,CGNS.pyCGNSconfig.PATH_MLL)
else:
  config+="""# Built without CGNS library"""

if (CGNS.pyCGNSconfig.HAS_CHLONE == True):
  config+="""# using %s found in %s\n"""%(CGNS.pyCGNSconfig.CHLONE_VERSION,
                                          CGNS.pyCGNSconfig.PATH_CHLONE)

vnum=""
if (CGNS.pyCGNSconfig.NUM_LIBRARY == "numpy"):
  import numpy
  vnum="v"+numpy.version.version

config+="# using module %s %s\n"%(CGNS.pyCGNSconfig.NUM_LIBRARY,vnum)

def test():
  import CGNS.test
  CGNS.test.showConfig()
  CGNS.test.run()
#
