#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# Trick to avoid window identification in Windows/Qt
# See https://bugreports.qt-project.org/browse/PYSIDE-46
#
from PySide.QtCore import *
from PySide.QtGui import *
import ctypes

_QWidget_winId = QWidget.winId

def _winId(self):
  ctypes.pythonapi.PyCObject_AsVoidPtr.restype = ctypes.c_void_p
  ctypes.pythonapi.PyCObject_AsVoidPtr.argtypes = [ctypes.py_object]
  return ctypes.pythonapi.PyCObject_AsVoidPtr(_QWidget_winId(self))

QWidget.winId = _winId

# ---
