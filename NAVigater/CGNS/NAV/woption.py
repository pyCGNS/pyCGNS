#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
import string
from PySide.QtCore  import *
from PySide.QtGui   import *
from CGNS.NAV.Q7OptionsWindow import Ui_Q7OptionsWindow
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.moption import Q7OptionContext as OCTXT

# -----------------------------------------------------------------
class Q7Option(Q7Window,Ui_Q7OptionsWindow):
    def __init__(self,parent):
        Q7Window.__init__(self,Q7Window.VIEW_OPTION,parent,None,None)
        self.bApply.clicked.connect(self.accept)
        self.bInfo.clicked.connect(self.infoOptionView)
        self.bClose.clicked.connect(self.reject)
        self.bReset.clicked.connect(self.reset)
        self.getOptions()
    def infoOptionView(self):
        self._control.helpWindow('Option')
    def getopt(self,name):
        if (name[0]=='_'): return None
        try:
            a=getattr(self,'_Ui_Q7OptionsWindow__O_'+string.lower(name))
        except AttributeError:
            return None
        return a
    def checkDeps(self):
        for dep in OCTXT._depends:
            for chks in OCTXT._depends[dep]:
                if (self.getopt(chks) is not None):
                  if (not self.getopt(chks).isChecked()):
                      self.getopt(dep).setDisabled(True)
                  else:
                      self.getopt(dep).setEnabled(True)
                else:
                    print 'CGNS.NAV (debug) NO OPTION :',chks
    def reset(self):
        self.getOptions()
        data=self._options
        for k in data:
          if ((k[0]!='_') and (self.getopt(k) is not None)):
            setattr(OCTXT,k,data[k])
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
        self.checkDeps()
    def show(self):
        self.reset()
        super(Q7Option, self).show()
    def accept(self):
        data=self._options
        for k in data:
          if ((k[0]!='_') and (self.getopt(k) is not None)):
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
        self.reset()
    def reject(self):
        self.hide()
    
# -----------------------------------------------------------------
