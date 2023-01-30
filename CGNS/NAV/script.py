#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

import sys
import time

from CGNS.pyCGNSconfig import HAS_MSW

if HAS_MSW:
    import ctypes

    myappid = "pycgns.pycgns.cglook.version"  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

from .moption import Q7OptionContext as OCTXT

from qtpy import QtCore
from qtpy.QtWidgets import QApplication, QSplashScreen
from qtpy.QtGui import QPixmap

# from qtpy.QtGui import *
from .wcontrol import Q7Main

splash = None
wcontrol = None


def closeSplash():
    global splash
    global wcontrol
    splash.finish(wcontrol)


# -----------------------------------------------------------------
def run(
    args=[],
    files=[],
    datasets=[],
    flags=(False, False, False, False, False, False, False),
    ppath=None,
    query=None,
):
    hidecontrol = False
    global splash
    global wcontrol

    if flags[4]:
        from CGNS.NAV.wquery import Q7Query

        Q7Query.loadUserQueries()
        Q7Query.fillQueries()
        ql = Q7Query.queriesNamesList()
        print("# CGNS.NAV available queries:")
        for q in ql:
            print(" ", q)
    else:
        app = QApplication(args)
        if not flags[5]:
            pixmap = QPixmap(":/images/splash.png")
            splash = QSplashScreen(pixmap, QtCore.Qt.WindowStaysOnTopHint)
            splash.show()
            splash.showMessage(
                "Release v%s" % OCTXT._ToolVersion,
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
            )
            timer = QtCore.QTimer.singleShot(3000, closeSplash)
        app.processEvents()
        Q7Main.verbose = flags[2]
        wcontrol = Q7Main()
        wcontrol._application = app
        wcontrol.setOptionValue("NAVTrace", flags[2])
        wcontrol.transientRecurse = flags[0]
        wcontrol.transientVTK = flags[3]
        wcontrol.query = query
        wcontrol._T("start")
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
                wcontrol.loadCompleted(
                    dataset_name="FFFF",
                    dataset_base="BASE",
                    dataset_tree=dd,
                    dataset_references=[],
                    dataset_paths=[],
                )
            hidecontrol = flags[6]
        wcontrol.show()
        if hidecontrol:
            wcontrol.hide()
        app.exec_()
        wcontrol._T("leave")
    sys.exit()


# run()

# -----------------------------------------------------------------
