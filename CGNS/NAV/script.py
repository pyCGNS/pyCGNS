#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from __future__ import unicode_literals
from __future__ import print_function
try:
  from builtins import (str, bytes, range, dict)
except ImportError:
  from __builtin__ import (str, bytes, range, dict)
import sys
import time

if sys.platform == "win32":
    import ctypes
    myappid = u'pycgns.pycgns.cglook.version' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

from CGNS.NAV.moption import Q7OptionContext as OCTXT

from qtpy.QtCore import *
from qtpy.QtWidgets import QApplication, QSplashScreen
from qtpy.QtGui import QPixmap
from qtpy.QtGui import *
from CGNS.NAV.wcontrol import Q7Main


# -----------------------------------------------------------------
def run(args=[], files=[], datasets=[],
        flags=(False, False, False, False, False, False, False),
        ppath=None, query=None):
    hidecontrol = False
    if flags[4]:
        from CGNS.NAV.wquery import Q7Query
        Q7Query.loadUserQueries()
        Q7Query.fillQueries()
        ql = Q7Query.queriesNamesList()
        print('# CGNS.NAV available queries:')
        for q in ql:
            print(' ', q)
    else:
        app = QApplication(args)
        if not flags[5]:
            pixmap = QPixmap(":/images/splash.png")
            splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
            splash.show()
            splash.showMessage("Release v%s" % OCTXT._ToolVersion,
                               Qt.AlignHCenter | Qt.AlignBottom)
        app.processEvents()
        t1 = time.time()
        Q7Main.verbose = flags[2]
        wcontrol = Q7Main()
        wcontrol._application = app
        wcontrol.setOptionValue('NAVTrace', flags[2])
        wcontrol.transientRecurse = flags[0]
        wcontrol.transientVTK = flags[3]
        wcontrol.query = query
        wcontrol._T('start')
        wcontrol.setDefaults()
        if flags[1]:
            wcontrol.loadlast()
            hidecontrol = flags[6]
        if files:
            for ff in files:
                wcontrol.loadfile(ff)
            hidecontrol = flags[6]
        if datasets:
            for dd in datasets:
                wcontrol.loadCompleted(dataset_name='FFFF',
                                       dataset_base='BASE',
                                       dataset_tree=dd,
                                       dataset_references=[],
                                       dataset_paths=[])
            hidecontrol = flags[6]
        wcontrol.show()
        if hidecontrol:
            wcontrol.hide()
        if not flags[5]:
            t2 = time.time()
            if t2 - t1 < 2.0:
                time.sleep(2)
            splash.finish(wcontrol)
        app.exec_()
        wcontrol._T('leave')
    sys.exit()


run()
# -----------------------------------------------------------------
