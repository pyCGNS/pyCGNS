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
from CGNS.NAV.Q7DiagWindow import Ui_Q7DiagWindow
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.moption import Q7OptionContext  as OCTXT
import CGNS.VAL.simplecheck as CGV
import CGNS.VAL.parse.messages as CGM

# -----------------------------------------------------------------
class Q7CheckList(Q7Window,Ui_Q7DiagWindow):
    def __init__(self,parent,data,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_DIAG,parent,None,fgprint)
        self.bClose.clicked.connect(self.reject)
        self.bExpandAll.clicked.connect(self.expand)
        self.bCollapseAll.clicked.connect(self.collapse)
        self.bPrevious.clicked.connect(self.previousfiltered)
        self.bNext.clicked.connect(self.nextfiltered)
        self.bClear.clicked.connect(self.clearfiltered)
        self.cWarnings.clicked.connect(self.warnings)
        self._data=data
    def warnings(self):
        if (not self.cWarnings.isChecked()): self.reset(False)
        else: self.reset(True)
    def previousfiltered(self):
        print 'PREVIOUS'
    def nextfiltered(self):
        print 'NEXT'
    def clearfiltered(self):
        print 'CLEAR'
    def expand(self):
        self.diagTable.expandAll()
    def collapse(self):
        self.diagTable.collapseAll()
    def show(self):
        self.reset()
        super(Q7CheckList, self).show()
    def reset(self,warnings=True):
        v=self.diagTable
        v.clear()
        v.setHeaderHidden(True)
        for path in self._data:
          state=self._data.getWorstDiag(path)
          if ((state==CGM.CHECK_WARN) and not warnings):
            pass
          else:
            it=QTreeWidgetItem(None,(path,))
            it.setFont(0,OCTXT.FixedFontTable)
            if (state==CGM.CHECK_FAIL): it.setIcon(0,self.I_C_SFL)
            if (state==CGM.CHECK_WARN): it.setIcon(0,self.I_C_SWR)
            v.insertTopLevelItem(0, it)
            for (diag,pth) in self._data.diagnosticsByPath(path):
              if ((diag[0]==CGM.CHECK_WARN) and not warnings):
                pass
              else:
                dit=QTreeWidgetItem(it,(self._data.message(diag),))
                dit.setFont(0,OCTXT.FixedFontTable)
                if (diag[0]==CGM.CHECK_FAIL): dit.setIcon(0,self.I_C_SFL)
                if (diag[0]==CGM.CHECK_WARN): dit.setIcon(0,self.I_C_SWR)
        self.cFilter.addItems(self._data.allMessageKeys())
    def reject(self):
        self.close()
         
# -----------------------------------------------------------------
