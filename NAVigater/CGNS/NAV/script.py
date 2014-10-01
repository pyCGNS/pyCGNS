#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import sys

from PySide.QtGui import QApplication
from CGNS.NAV.wcontrol import Q7Main

# -----------------------------------------------------------------
def run(args=[],files=[],
        flags=(False,False,False,False,False),
        ppath=None,query=None):
  if (flags[4]):
    from CGNS.NAV.wquery import Q7Query
    Q7Query.loadUserQueries()
    Q7Query.fillQueries()
    ql=Q7Query.queriesNamesList()
    print '# CGNS.NAV available queries:'
    for q in ql:
      print ' ',q
  else:    
    app=QApplication(args)
    Q7Main.verbose=flags[2]
    wcontrol=Q7Main()
    wcontrol._application=app
    wcontrol.setOptionValue('NAVTrace',flags[2])
    wcontrol.transientRecurse=flags[0]
    wcontrol.transientVTK=flags[3]
    wcontrol.query=query
    wcontrol._T('start')
    if (flags[1]):  wcontrol.loadlast()
    wcontrol.loadfile(files[0]) # loop on file list broken in wfingerprint
    wcontrol.show()
    app.exec_()
    wcontrol._T('leave')
  sys.exit()

# -----------------------------------------------------------------
