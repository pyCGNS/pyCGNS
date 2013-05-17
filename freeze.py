#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#
import sys
from cx_Freeze import setup, Executable
sys.path.append("f:\localadmin\Bureau\pyCGNS\dist\Lib\site-packages")
# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
"packages": ["os","sys","numpy","vtk","PySide.QtCore","PySide.QtGui",
             "CGNS.NAV","CGNS.MAP","CGNS.PAT","CGNS.VAL","CGNS.APP",
], 
"excludes": ["tkinter","tcl","tk8.5","PyQt4","PyQt4.QtCore","PyQt4.QtGui",
],
"include_files": ["lib/pyCGNS.ico","lib/pyCGNS.bmp",
                  "lib/pyCGNS-small.ico","lib/pyCGNS-small.bmp",
                  "license.txt",
                  "demo/001Disk.hdf","demo/sqnz_unstruct_2dom.hdf",
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
(pyCGNSconfig,installprocess)=setuputils.search('NAV',['numpy'])
# ---

setup(  
name = "CGNS.NAV",
version = pyCGNSconfig.NAV_VERSION,
description  = "pyCGNS NAVigator - CGNS/Python trees navigator and editor",
author       = "marc Poinot",
author_email = "marc.poinot@onera.fr",
license      = "LGPL 2",
options = {"build_exe": build_exe_options},
    executables = [Executable("NAVigater/CGNS/NAV/CGNSNAV.py",base=base)])

# --- last line
