#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import sys
import string
from PySide.QtCore       import *
from PySide.QtGui        import *
from CGNS.NAV.Q7MessageWindow import Ui_Q7MessageWindow
from CGNS.NAV.moption import Q7OptionContext  as OCTXT

(INFO,QUESTION,ERROR,WARNING)=(0,1,2,3)

# globals
OK=True
CANCEL=False
ANSWER=True

# -----------------------------------------------------------------
class Q7MessageBox(QDialog,Ui_Q7MessageWindow):
    def __init__(self,control):
        QDialog.__init__(self,None)
        self.setupUi(self)
        self.bOK.clicked.connect(self.runOK)
        self.bCANCEL.clicked.connect(self.runCANCEL)
        self.bInfo.clicked.connect(self.infoMessageView)
        self._text=''
        self._result=None
        self._control=control
    def setMode(self,cancel=False,again=False):
        if (not again): self.cNotAgain.hide()
        if (not cancel): self.bCANCEL.hide()
    def setLayout(self,text,btype=INFO,
                  cancel=False,again=False,buttons=('OK','Cancel')):
        self.bOK.setText(buttons[0])
        if (len(buttons)>1):
          self.bCANCEL.setText(buttons[1])
        self.eMessage.setText(text)
        self._text=text
        self.eMessage.setReadOnly(True)
        self.setMode(cancel,again)
    def runOK(self,*arg):
        self._result=True
        self.close()
    def runCANCEL(self,*arg):
        self._result=False
        self.close()
    def infoMessageView(self):
        self._control.helpWindow('Message')
    def showAndWait(self):
        self.exec_()
  
def wError(control,code,info,error):
  txt="""<img source=":/images/icons/user-G.png">  <big>ERROR #%d</big><hr>
         %s<br>%s"""%(code,error,info)
  msg=Q7MessageBox(control)
  msg.setLayout(txt,btype=ERROR,cancel=False,again=False,buttons=('Close',))
  msg.showAndWait()
  return msg._result

def wQuestion(control,title,question,again=True,buttons=('OK','Cancel')):
  txt="""<img source=":/images/icons/user-M.png">
         <b> <big>%s</big></b><hr>%s"""%(title,question)
  msg=Q7MessageBox(control)
  msg.setLayout(txt,btype=QUESTION,cancel=True,again=again,buttons=buttons)
  msg.showAndWait()
  return msg._result

def wInfo(control,title,info,again=True,buttons=('Close',)):
  txt="""<img source=":/images/icons/user-S.png">
         <b> <big>%s</big></b><hr>%s"""%(title,info)
  msg=Q7MessageBox(control)
  msg.setLayout(txt,btype=INFO,cancel=False,again=again,buttons=buttons)
  msg.showAndWait()
  return msg._result
  
  
# --- last line
