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
from CGNS.NAV.Q7LinkWindow import Ui_Q7LinkWindow
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.moption import Q7OptionContext  as OCTXT
import CGNS.MAP as CGM

# -----------------------------------------------------------------
class Q7LinkList(Q7Window,Ui_Q7LinkWindow):
    def __init__(self,parent,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_LINK,parent,None,fgprint)
        self.bClose.clicked.connect(self.reject)
        self._fgprint=fgprint
        self._links=fgprint.links
        self.setLabel(self.eDirSource,fgprint.filedir)
        self.setLabel(self.eFileSource,fgprint.filename)
    def show(self):
        self.reset()
        super(Q7LinkList, self).show()
    def statusIcon(self,status):
        if (status ==CGM.S2P_LKOK):
            it=QTableWidgetItem(self.I_L_OKL,'')
            it.setToolTip('Link ok')
            return it
        if (status & CGM.S2P_LKNOFILE):
            it=QTableWidgetItem(self.I_L_NFL,'')
            it.setToolTip('File not found in search path')
            return it
        if (status & CGM.S2P_LKFILENOREAD):
            it=QTableWidgetItem(self.I_L_NRL,'')
            it.setToolTip('File found, not readable')
            return it
        if (status & CGM.S2P_LKNONODE):
            it=QTableWidgetItem(self.I_L_NNL,'')
            it.setToolTip('File ok, node path not found')
            return it
        it=QTableWidgetItem(self.I_L_ERL,'')
        it.setToolTip('Unknown error')
        return it
    def reset(self):
        v=self.linkTable
        for r in range(v.rowCount()):
            v.removeRow(r+1)
        lh=v.horizontalHeader()
        lv=v.verticalHeader()
        h=['S','Source Node','Linked-to file','Linked-to Node','Found in dir']
        for i in range(len(h)):
            hi=QTableWidgetItem(h[i])
            v.setHorizontalHeaderItem(i,hi)
            lh.setResizeMode(i,QHeaderView.ResizeToContents)
        lh.setResizeMode(len(h)-1,QHeaderView.Stretch)
        for lk in self._links:
          (ld,lf,ln,sn,st)=lk
          v.setRowCount(v.rowCount()+1)
          r=v.rowCount()-1
          t1item=self.statusIcon(st)
          t2item=QTableWidgetItem(sn)
          t2item.setFont(QFont("Courier"))
          t3item=QTableWidgetItem(lf)
          t3item.setFont(QFont("Courier"))
          t4item=QTableWidgetItem(ln)
          t4item.setFont(QFont("Courier"))
          t5item=QTableWidgetItem(ld)
          t5item.setFont(QFont("Courier"))
          v.setItem(r,0,t1item)
          v.setItem(r,1,t2item)
          v.setItem(r,2,t3item)
          v.setItem(r,3,t4item)
          v.setItem(r,4,t5item)
        for i in (2,3):
          v.resizeColumnToContents(i)
        for i in range(v.rowCount()):
          v.resizeRowToContents(i)
    def reject(self):
        self.close()
         
# -----------------------------------------------------------------
