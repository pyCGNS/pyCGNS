#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from CGNS.NAV.moption import Q7OptionContext as OCTXT

import sys
import time

from PyQt4.QtCore import *
from PyQt4.QtGui import QApplication, QPixmap, QSplashScreen
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
    pixmap = QPixmap(":/images/splash.png")
    splash = QSplashScreen(pixmap,Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("Release v%s"%OCTXT._ToolVersion,
                       Qt.AlignHCenter|Qt.AlignBottom)
    app.processEvents()
    t1=time.time()
    Q7Main.verbose=flags[2]
    wcontrol=Q7Main()
    wcontrol._application=app
    wcontrol.setOptionValue('NAVTrace',flags[2])
    wcontrol.transientRecurse=flags[0]
    wcontrol.transientVTK=flags[3]
    wcontrol.query=query
    wcontrol._T('start')
    wcontrol.setDefaults()
    if (flags[1]):  wcontrol.loadlast()
    if files:
      wcontrol.loadfile(files[0]) # loop on file list broken in wfingerprint
    t2=time.time()
    if (t2-t1<2.0): time.sleep(2)
    wcontrol.show()
    splash.finish(wcontrol)
    app.exec_()
    wcontrol._T('leave')
  sys.exit()

# -----------------------------------------------------------------
