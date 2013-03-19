#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  
import sys
import string
from PySide.QtCore  import *
from PySide.QtGui   import *
from CGNS.NAV.Q7OptionsWindow import Ui_Q7OptionsWindow
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.moption import Q7OptionContext as OCTXT

combonames=[]
for tn in ['label','edit','table','button','rname','nname']:
  combonames+=['__O_%s_family'%tn,'__O_%s_size'%tn,
               '__O_%s_bold'%tn,'__O_%s_italic'%tn]

# -----------------------------------------------------------------
class Q7Option(Q7Window,Ui_Q7OptionsWindow):
    labdict={'Label':['QLabel','QTabWidget','QGroupBox','QCheckBox'],
             'Button':['QPushButton'],
             'Edit':['QLineEdit','QSpinBox','QComboBox'],
             'RName':[],
             'NName':[],
             'Table':[]}
    combos=combonames
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
            if (self.getopt(k).objectName() in Q7Option.combos):
                pass
            elif (type(data[k]) is bool):
              if (data[k]): self.getopt(k).setCheckState(Qt.Checked)
              else: self.getopt(k).setCheckState(Qt.Unchecked)
            elif (type(data[k]) is int):
              self.getopt(k).setValue(data[k])
            elif (type(data[k]) is str):
              self.getopt(k).setText(data[k])
            elif (type(data[k]) is list):
              s=''
              for l in data[k]:
                  s+='%s\n'%l
              self.getopt(k).setPlainText(s)
        self.checkDeps()
        self.updateFonts()
    def show(self):
        self.reset()
        super(Q7Option, self).show()
    def accept(self):
        data=self._options
        for k in data:
          if ((k[0]!='_') and (self.getopt(k) is not None)):
            if (self.getopt(k).objectName() in Q7Option.combos):
              pass
            elif (type(data[k]) is bool):
                if (self.getopt(k).isChecked()): data[k]=True
                else: data[k]=False
            elif (type(data[k]) is int):
                v=self.getopt(k).value()
                if (self.validateOption(k,v)):
                    data[k]=self.getopt(k).value()
            elif (type(data[k]) is str):
                v=self.getopt(k).text()
                if (self.validateOption(k,v)):
                    data[k]=self.getopt(k).text()
            elif (type(data[k]) is list):
                s=self.getopt(k).toPlainText()
                cset=[]
                for l in s.split('\n'):
                    if (l and l not in cset):
                        cset.append(l)
                if (self.validateOption(k,cset)): data[k]=cset
        self.updateFonts(update=True)
        self.setOptions()
        self.reset()
    def reject(self):
        self.hide()
    def updateFonts(self,update=False):
        data=self._options
        scss=""
        for kfont in Q7Option.labdict:
          if (update):
            fm=self.getopt('%s_Family'%kfont).currentFont().family()
            it=self.getopt('%s_Italic'%kfont).isChecked()
            bd=self.getopt('%s_Bold'%kfont).isChecked()
            sz=int(self.getopt('%s_Size'%kfont).text())
            data['%s_Family'%kfont]=fm
            data['%s_Size'%kfont]=sz
            data['%s_Bold'%kfont]=bd
            data['%s_Italic'%kfont]=it
          fm=getattr(OCTXT,'%s_Family'%kfont)
          sz=getattr(OCTXT,'%s_Size'%kfont)
          bd=getattr(OCTXT,'%s_Bold'%kfont)
          it=getattr(OCTXT,'%s_Italic'%kfont)
          if (bd): wg=QFont.Bold
          else: wg=QFont.Normal
          qf=QFont(fm,italic=it,weight=wg,pointSize=sz)
          setattr(OCTXT,'_%s_Font'%kfont,qf)
          self.getopt('%s_Family'%kfont).setCurrentFont(qf)
          for wtype in Q7Option.labdict[kfont]:
              bf=''
              tf=''
              if (bd): bf='bold'
              if (it): tf='italic'
              scss+="""%s { font:  %s %s %dpx "%s" }\n"""%(wtype,bf,tf,sz,fm)
        self._control._application.setStyleSheet(scss)

# -----------------------------------------------------------------
