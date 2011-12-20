#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

from PySide.QtCore       import *
from PySide.QtGui        import *

(INFO,YESNO,QUESTION2,QUESTION2,ERROR,WARNING)=(0,1,2,3,99,100)

def message(text,info,btype=INFO):
  msgBox = QMessageBox()
  msgBox.setText(text)
  msgBox.setInformativeText(info)
  msgBox.setFixedWidth(375)
  if (btype == INFO):
      msgBox.setStandardButtons(QMessageBox.Ok)
      msgBox.setDefaultButton(QMessageBox.Ok)
      msgBox.setIcon(QMessageBox.Information)
  if (btype == WARNING):
      msgBox.setStandardButtons(QMessageBox.Ok)
      msgBox.setDefaultButton(QMessageBox.Ok)
      msgBox.setIcon(QMessageBox.Warning)
  if (btype == YESNO):
      msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
      msgBox.setDefaultButton(QMessageBox.No)
      msgBox.setIcon(QMessageBox.Question)
  return msgBox.exec_()
  
