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

# -----------------------------------------------------------------
class Q7CheckList(Q7Window,Ui_Q7DiagWindow):
    def __init__(self,parent,data,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_DIAG,parent,None,fgprint)
        self.bClose.clicked.connect(self.reject)
        self._data=data
    def show(self):
        self.reset()
        super(Q7CheckList, self).show()
    def reset(self):
        v=self.diagTable
        lh=v.horizontalHeader()
        lv=v.verticalHeader()
        h=['S','N','Node','Diagnostic']
        for i in range(len(h)):
            hi=QTableWidgetItem(h[i])
            v.setHorizontalHeaderItem(i,hi)
            lh.setResizeMode(i,QHeaderView.ResizeToContents)
        lh.setResizeMode(len(h)-1,QHeaderView.Stretch)
        for path in self._data:
            for diag in self._data[path][1]:
                v.setRowCount(v.rowCount()+1)
                r=v.rowCount()-1
                stat=diag[0]
                tpitem=QTableWidgetItem(self.I_D_INF,'')
                if (stat==CGV.CHECK_FAIL):
                    tpitem=QTableWidgetItem(self.I_D_ERR,'')
                tpitem.setTextAlignment(Qt.AlignCenter)
                v.setItem(r,0,tpitem)
                v.setItem(r,1,QTableWidgetItem('%.4d'%(r+1)))
                v.setItem(r,2,QTableWidgetItem('%s '%path))
                v.setItem(r,3,QTableWidgetItem('%s '%diag[1]))
        for i in (2,3):
            v.resizeColumnToContents(i)
        for i in range(v.rowCount()):
            v.resizeRowToContents(i)
    def reject(self):
        self.close()
         
# -----------------------------------------------------------------
