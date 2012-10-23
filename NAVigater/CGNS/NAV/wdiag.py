#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
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
        self.cWarnings.clicked.connect(self.warnings)
        self.bSave.clicked.connect(self.diagnosticssave)
        QObject.connect(self.cFilter,
                        SIGNAL("currentIndexChanged(int)"),
                        self.filterChange)
        self._data=data
        self.filterChange()
    def diagnosticssave(self):
        n='data=%s\n'%self._data
        filename=QFileDialog.getSaveFileName(self,
                                             "Save diagnostics",".","*.py")
        if (filename[0]==""): return
        f=open(str(filename[0]),'w+')
        f.write(n)
        f.close()
    def warnings(self):
        if (not self.cWarnings.isChecked()): self.reset(False)
        else: self.reset(True)
    def previousfiltered(self):
      iold=self._filterItems[self.cFilter.currentText()][self._currentItem]
      if (self._currentItem!=0): self._currentItem-=1
      inew=self._filterItems[self.cFilter.currentText()][self._currentItem]
      iold.setSelected(False)
      inew.setSelected(True)
      self.diagTable.scrollToItem(inew)
      self.eCount.setText(str(self._currentItem+1))
    def nextfiltered(self):
      ilist=self._filterItems[self.cFilter.currentText()]
      if (self._currentItem>=len(ilist)-1):
        return
      iold=ilist[self._currentItem]
      self._currentItem+=1
      inew=ilist[self._currentItem]
      iold.setSelected(False)
      inew.setSelected(True)
      self.diagTable.scrollToItem(inew)
      self.eCount.setText(str(self._currentItem+1))
    def filterChange(self):
      self._currentItem=0
      self.eCount.setText("")
      self.diagTable.clearSelection()
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
        plist=self._data.keys()
        plist.sort()
        plist.reverse()
        keyset=set()
        self.cFilter.clear()
        self._filterItems={}
        for path in plist:
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
              if ((diag.level==CGM.CHECK_WARN) and not warnings):
                pass
              else:
                dit=QTreeWidgetItem(it,(self._data.message(diag),))
                dit.setFont(0,OCTXT.FixedFontTable)
                keyset.add(diag.key)
                if (diag.level==CGM.CHECK_FAIL): dit.setIcon(0,self.I_C_SFL)
                if (diag.level==CGM.CHECK_WARN): dit.setIcon(0,self.I_C_SWR)
                if (diag.key not in self._filterItems):
                  self._filterItems[diag.key]=[dit]
                else:
                  self._filterItems[diag.key].insert(0,dit)
        for k in keyset:
            self.cFilter.addItem(k)
    def reject(self):
        self.close()
         
# -----------------------------------------------------------------
