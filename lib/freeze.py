#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# Posted by Cacasodo (http://www.techanswerguy.com)
# creating a transparent favicon.ico with Gimp
#I'm always forgetting the steps to do this properly. Here you go:
#1) open a favorite image that you'd like to turn into a favicon to appear in your browser's location bar
#2) click Image->Mode->RGB
#3) using the erase tool, erase any areas of the graphic that you'd like to be transparent
#4) using Image->Scale, resize the image to 32x32, the proper size for a favicon
#5) first save the file as a GIF
#- Gimp will popup a dialog that says "GIF can only handle grayscale or indexed images"
#6) select * Convert to Indexed using default settings
#7) click Export
#8) save a copy as .ico. It will give you a number of choices for bits/pixel:
#1 bpp, 1 bit alpha, 2-slot palette
#4 bpp, 1 bit alpha, 16-slot palette
#8 bpp, 1 bit alpha, 256-slot palette
#32 bpp, 8 bit alpha, no palette
#
#Choose 4bpp. However, I have seen with some images that 8bpp needs to be used.
#9) upload to your website and enjoy!
import sys
import glob

from cx_Freeze import setup, Executable
import cx_Freeze
print(cx_Freeze.__file__)

df=glob.glob('demo/*')+glob.glob('demo/124Disk-PASS/*')+glob.glob('demo/124Disk-FAIL/*')+glob.glob('demo/DPW5/*')
demo_files=[(f,f) for f in df]

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
"packages": ["os","sys","numpy","vtk","qtpy","qtpy.QtCore","qtpy.QtGui","qtpy.QtWidgets",
             "CGNS.NAV","CGNS.MAP","CGNS.PAT","CGNS.VAL","CGNS.APP",
             "numpy.core.multiarray"
], 
"excludes": ["tkinter","tcl","tk8.5","PySide","PySide.QtCore","PySide.QtGui",
],
"include_files": ["license.txt",
r"C:\Users\Public\Python3.6\Lib\site-packages\pyCGNS-5.3.0-py3.6-win-amd64.egg",
#r"D:\poinot\AppData\Local\Continuum\Anaconda\Lib\site-packages\pyCGNS-5.3.0-py3.6-win-amd64.egg",
#r"D:\poinot\AppData\Local\Continuum\Anaconda\Lib",
#r"D:\poinot\AppData\Local\Continuum\Anaconda\DLLs",
"lib/pyCGNS.ico",
"lib/pyCGNS-wizard.bmp",
"lib/pyCGNS-small.ico",
"lib/pyCGNS-wizard-small.bmp",                  
]+demo_files,
"add_to_path":True,
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
name = "pyCGNS",
version = '4.6',
description  = "pyCGNS - CGNS/Python trees navigator and editor",
author       = "Marc Poinot",
author_email = "marc.poinot@safrangroup.com",
license      = "LGPL 2",
options = {"build_exe": build_exe_options,"build_msi": build_exe_options},
           executables = [Executable("CGNS/App/tools/cg_look",base=base)])

# --- last line
