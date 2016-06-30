#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# for windows port see windows.txt file

import sys
from cx_Freeze import setup, Executable
#sys.path.append("d:\poinot\AppData\Local\pyCGNS\dist\Lib\site-packages")
# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
"packages": ["os","sys","numpy","vtk","PyQt4.QtCore","PyQt4.QtGui",
             "CGNS.NAV","CGNS.MAP","CGNS.PAT","CGNS.VAL","CGNS.APP",
], 
"excludes": ["tkinter","tcl","tk8.5","PySide","PySide.QtCore","PySide.QtGui",
],
"include_files": ["lib/pyCGNS.ico","lib/pyCGNS-wizard.bmp",
                  "lib/pyCGNS-small.ico","lib/pyCGNS-wizard-small.bmp",
                  "license.txt",
                  "demo/SquaredNozzle.cgns",
                  "demo/124Disk/124Disk_FamilyName.hdf",
                  "demo/124Disk/124Disk_ReferenceState.hdf",
                  "demo/124Disk/124Disk_zone1.hdf",
                  "demo/124Disk/124Disk_zone1_GridCoordinates.hdf",
                  "demo/124Disk/124Disk_zone1_GridCoordinates_X.hdf",
                  "demo/124Disk/124Disk_zone1_ZoneType.hdf",
                  
],
}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if (sys.platform == "win32"):
    base = "Win32GUI"

# --- pyCGNSconfig search
import os
sys.path=[os.path.abspath(os.path.normpath('./lib'))]+sys.path
import setuputils

# ---

setup(  
name = "cg_look",
version = '4.6',
description  = "pyCGNS - CGNS/Python trees navigator and editor",
author       = "Marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
options = {"build_exe": build_exe_options},
    executables = [Executable("CGNS/App/tools/cg_look",base=base)])

# --- last line
