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
import CGNS.PAT.cgnskeywords as CGK

# -----------------------------------------------------------------
def divpairs(n):
    d=n
    l=[]
    while (d != 0):
        m=n//d
        r=n%d
        if (r==0): l+=[(d,m)]
        d-=1
    return l

# -----------------------------------------------------------------
class Q7TableItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        t=index.data(Qt.DisplayRole)
        r=option.rect.adjusted(2, 2, -2, -2);
        painter.drawText(r,Qt.AlignVCenter|Qt.AlignLeft,t, r)
        #QStyledItemDelegate.paint(self, painter, option, index)

# -----------------------------------------------------------------
class Q7Form(Q7Window,Ui_Q7FormWindow):
    def __init__(self,control,node,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_FORM,control,
                          node.sidsPath(),fgprint)
        self._node=node
        for t in self._node.sidsTypeList():
            self.eType.addItem(t)
        self.bApply.clicked.connect(self.accept)
        self.bClose.clicked.connect(self.reject)
        for dt in node.sidsDataType(all=True):
            self.cDataType.addItem(dt)
    def updateTreeStatus(self):
        print 'form up'
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
        self.ePath.setReadOnly(True)
        self.setCurrentType(self._node.sidsType())
        dims=self._node.sidsDims()
        its=reduce(lambda x,y: x*y, dims)
        ndz=its*self._node.sidsDataTypeSize()
        self.eDims.setText(str(dims))
        self.setDataType(self._node)
        self.eItems.setText(str(its))
        self.ls=1
        self.cs=dims[0]
        if (len(dims)>1): self.ls=reduce(lambda x,y: x*y, dims[1:])
        self.showparams={'cs':self.cs,'ls':self.ls,'mode':'IJ'}
        for dp in divpairs(self.cs*self.ls):
           self.cRowColSize.addItem("%d x %d"%dp)
        ix=self.cRowColSize.findText("%d x %d"%(self.cs,self.ls))
        self.cRowColSize.setCurrentIndex(ix)
        self.model=Q7TableModel(self._node,self.showparams)
        self.tableView.setModel(self.model)
        self.tableView.setStyleSheet(self._stylesheet)
        self.bMinimize.clicked.connect(self.minimizeCells)
        QObject.connect(self.tableView,
                        SIGNAL("clicked(QModelIndex)"),
                        self.clickedNode)
        QObject.connect(self.cRowColSize,
                        SIGNAL("currentIndexChanged(int)"),
                        self.resizeTable)
    def resizeTable(self):
        s=self.cRowColSize.currentText()
        (r,c)=s.split('x')
        self.cs=int(r)
        self.ls=int(c)
        self.showparams={'cs':self.cs,'ls':self.ls,'mode':'IJ'}
        self.model=Q7TableModel(self._node,self.showparams)
        self.tableView.setModel(self.model)
        self.tableView.setStyleSheet(self._stylesheet)
    def clickedNode(self,index):
        tp=self.model.getEnumeratedValueIfPossible(index)
        if (tp): self.setEnumerateValue(*tp)
    def minimizeCells(self,*args):
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
    def setEnumerateValue(self,etype,evalue,etypeidx,evalueidx):
        self.cEnumType.clear()
        self.cEnumValue.clear()
        for et in etype:
            self.cEnumType.addItem(et)
        for ev in evalue:
            self.cEnumValue.addItem(ev)
        self.cEnumType.setCurrentIndex(etypeidx)
        self.cEnumValue.setCurrentIndex(evalueidx)
            
# -----------------------------------------------------------------
