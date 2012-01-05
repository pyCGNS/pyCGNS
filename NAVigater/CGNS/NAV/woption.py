#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
from PySide.QtCore  import *
from PySide.QtGui   import *
from CGNS.NAV.Q7OptionsWindow import Ui_Q7OptionsWindow
from CGNS.NAV.wfingerprint import Q7Window

# -----------------------------------------------------------------
class Q7Option(Q7Window,Ui_Q7OptionsWindow):
    def __init__(self,parent):
        Q7Window.__init__(self,Q7Window.VIEW_OPTION,parent,None,None)
        self.bApply.clicked.connect(self.accept)
        self.bClose.clicked.connect(self.reject)
        self.bReset.clicked.connect(self.reset)
    def getopt(self,name):
        return getattr(self,'__O_'+name)
    def reset(self):
        self.getOptions()
        data=self._options
        for k in data:
          if (type(data[k]) is bool):
            if (data[k]): self.getopt(k).setCheckState(Qt.Checked)
            else: self.getopt(k).setCheckState(Qt.Unchecked)
          if (type(data[k]) is int):
            self.getopt(k).setValue(data[k])
          if (type(data[k]) is str):
            self.getopt(k).setText(data[k])
          if (type(data[k]) is list):
            s=''
            for l in data[k]:
                s+='%s\n'%l
            self.getopt(k).setPlainText(s)
    def show(self):
        self.reset()
        super(Q7Option, self).show()
    def accept(self):
        data=self._options
        for k in data:
            if (type(data[k]) is bool):
                if (self.getopt(k).isChecked()): data[k]=True
                else: data[k]=False
            if (type(data[k]) is int):
                v=self.getopt(k).value()
                if (self.validateOption(k,v)):
                    data[k]=self.getopt(k).value()
            if (type(data[k]) is str):
                v=self.getopt(k).text()
                if (self.validateOption(k,v)):
                    data[k]=self.getopt(k).text()
            if (type(data[k]) is list):
                s=self.getopt(k).toPlainText()
                v=[]
                for l in s.split('\n'):
                    if (l): v.append(l)
                if (self.validateOption(k,v)): data[k]=v
        self.setOptions()
    def reject(self):
        self.hide()
    
# -----------------------------------------------------------------
