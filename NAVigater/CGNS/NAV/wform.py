#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
from PySide.QtCore  import *
from PySide.QtGui   import *
from CGNS.NAV.Q7FormWindow import Ui_Q7FormWindow
from CGNS.NAV.mtable import Q7TableModel
from CGNS.NAV.wstylesheets import Q7TABLEVIEWSTYLESHEET
from CGNS.NAV.wfingerprint import Q7Window

import CGNS.PAT.cgnsutils as CGU

# -----------------------------------------------------------------
class Q7TableItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        t=index.data(Qt.DisplayRole)
        r=option.rect.adjusted(2, 2, -2, -2);
        painter.drawText(r,Qt.AlignVCenter|Qt.AlignLeft,t, r)
        #QStyledItemDelegate.paint(self, painter, option, index)

# -----------------------------------------------------------------
class Q7Form(Q7Window,Ui_Q7FormWindow):
    showas=['Array','Text','Python']
    def __init__(self,control,node,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_FORM,control,
                          node.sidsPath(),fgprint)
        self._node=node
        for t in self._node.sidsTypeList():
            self.eType.addItem(t)
        for t in Q7Form.showas:
            self.cShowAs.addItem(t)
        self.bApply.clicked.connect(self.accept)
        self.bClose.clicked.connect(self.reject)
        for dt in node.sidsDataType(all=True):
            self.cDataType.addItem(dt)
    def accept(self):
        pass
    def reject(self):
        self.close()
    def show(self):
        self.reset()
        super(Q7Form, self).show()
    def reset(self):
        self.eName.setText(self._node.sidsName())
        self.ePath.setText(self._node.sidsPath())
        self.setCurrentType(self._node.sidsType())
        dims=self._node.sidsDims()
        its=reduce(lambda x,y: x*y, dims)
        ndz=its*self._node.sidsDataTypeSize()
        self.eDims.setText(str(dims))
        self.setDataType(self._node)
        self.eItems.setText(str(its))
        self.ls=dims[0]
        self.cs=1
        if (len(dims)>1): self.cs=reduce(lambda x,y: x*y, dims[1:])
        showparams={'cs':self.cs,'ls':self.ls,'mode':'IJ'}
        self.model=Q7TableModel(self._node,showparams)
        self.tableView.setModel(self.model)
        self.tableView.setStyleSheet(self._stylesheet)
        QObject.connect(self.cMinimize,
                        SIGNAL("stateChanged(int)"),
                        self.minimizeCells)
        #self.tableView.setItemDelegate(Q7TableItemDelegate(self))
    def minimizeCells(self,*args):
        if (self.cMinimize.isChecked()):
            self.tableView.resizeColumnsToContents()
            self.tableView.resizeRowsToContents()
    def setCurrentType(self,ntype):
        idx=self.eType.findText(ntype)
        self.eType.setCurrentIndex(idx)
    def setDataType(self,node):
        dt=node.sidsDataType()
        ix=self.cDataType.findText(dt)
        if (ix==-1): return
        self.cDataType.setCurrentIndex(ix)
    def addControlLine(self):
        pass
# -----------------------------------------------------------------
