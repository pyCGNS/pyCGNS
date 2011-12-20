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
def run(args,*tpargs):
  app=QApplication(args)
  frame=Q7Main()
  if (len(args)>1):
      fname=args[1]
      frame.load(fname)
  frame.show()
  app.exec_()
  sys.exit()

# -----------------------------------------------------------------
