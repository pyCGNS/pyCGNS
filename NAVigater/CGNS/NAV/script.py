#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys

from PySide.QtGui import QApplication
from CGNS.NAV.wcontrol import Q7Main

# -----------------------------------------------------------------
def run(args=[],files=[],flags=(False,False,False,False),ppath=None):
  app=QApplication(args)
  wcontrol=Q7Main()
  wcontrol._application=app
  wcontrol.setOptionValue('NAVTrace',flags[2])
  wcontrol._T('start')
  if (flags[1]): wcontrol.loadlast()
  for f in files:
    wcontrol.transientRecurse=flags[0]
    wcontrol.transientVTK=flags[3]
    wcontrol.loadfile(f)
  wcontrol.show()
  app.exec_()
  wcontrol._T('leave')
  sys.exit()

# -----------------------------------------------------------------
