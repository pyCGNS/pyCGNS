#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from PySide.QtCore       import *
from PySide.QtGui        import *

(INFO,YESNO,QUESTION2,QUESTION2,ERROR,WARNING)=(0,1,2,3,99,100)

def message(text,info,btype=INFO,fixed=False):
  msgBox = QMessageBox()
  msgBox.setText(text)
  msgBox.setInformativeText(info)
  msgBox.setFixedWidth(675)
  if (btype == INFO):
      msgBox.setStandardButtons(QMessageBox.Ok)
      msgBox.setDefaultButton(QMessageBox.Ok)
      msgBox.setIcon(QMessageBox.Information)
  if (btype == WARNING):
      msgBox.setStandardButtons(QMessageBox.Ok)
      msgBox.setDefaultButton(QMessageBox.Ok)
      msgBox.setIcon(QMessageBox.Warning)
  if (btype == ERROR):
      msgBox.setStandardButtons(QMessageBox.Ok)
      msgBox.setDefaultButton(QMessageBox.Ok)
      msgBox.setIconPixmap(QPixmap(":/images/icons/cgSpy.gif"))
  if (btype == YESNO):
      msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
      msgBox.setDefaultButton(QMessageBox.No)
      msgBox.setIcon(QMessageBox.Question)
  if (fixed):
    msgBox.setFont(QFont('Courier'))
  return msgBox.exec_()
  
def wError(code,info,error):
  filler=160*'-'
  error="<b>%s</b><br>"%error
  message("<big>ERROR #%d</big><br>%s%s"%(code,filler,info),error,ERROR)

# --- last line
