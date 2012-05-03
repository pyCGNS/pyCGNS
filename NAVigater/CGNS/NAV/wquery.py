#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
from PySide.QtCore  import *
from PySide.QtGui   import *
from CGNS.NAV.Q7QueryWindow import Ui_Q7QueryWindow
from CGNS.NAV.Q7SelectionWindow import Ui_Q7SelectionWindow
from CGNS.NAV.mquery import Q7QueryTableModel,Q7ComboBoxDelegate
from CGNS.NAV.wfingerprint import Q7Window

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

# -----------------------------------------------------------------
class Q7SelectionList(Q7Window,Ui_Q7SelectionWindow):
    def __init__(self,control,data,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_SELECT,control,None,fgprint)
        self.bClose.clicked.connect(self.reject)
        self._data=data
        self._fgprint=fgprint
    def show(self):
        self.reset()
        super(Q7SelectionList, self).show()
    def reset(self):
        v=self.diagTable
        v.setColumnCount(self._fgprint.depth)
        lh=v.horizontalHeader()
        lv=v.verticalHeader()
        h=['-']*self._fgprint.depth
        for i in range(len(h)):
            hi=QTableWidgetItem(h[i])
            v.setHorizontalHeaderItem(i,hi)
            lh.setResizeMode(i,QHeaderView.ResizeToContents)
        lh.setResizeMode(len(h)-1,QHeaderView.Stretch)
        for path in self._data:
            v.setRowCount(v.rowCount()+1)
            r=v.rowCount()-1
            plist=path.split('/')[1:]
            for i in range(len(plist)):
                v.setItem(r,i,QTableWidgetItem('%s '%plist[i]))
        for i in range(len(plist)):
            v.resizeColumnToContents(i)
        for i in range(v.rowCount()):
            v.resizeRowToContents(i)
    def reject(self):
        self.close()

# -----------------------------------------------------------------
class Q7QueryTableItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        t=index.data(Qt.DisplayRole)
        r=option.rect.adjusted(2, 2, -2, -2);
        painter.drawText(r,Qt.AlignVCenter|Qt.AlignLeft,t, r)

# -----------------------------------------------------------------
class Q7Query(Q7Window,Ui_Q7QueryWindow):
    def __init__(self,control,fgprint,treeview):
        Q7Window.__init__(self,Q7Window.VIEW_QUERY,control,'/',fgprint)
        self.querytablemodel=treeview.querymodel
        self.querytableview.setModel(self.querytablemodel)
        self.querytableview.setEditTriggers(QAbstractItemView.CurrentChanged)
        self.querytableview.viewport().installEventFilter(self)
        self.querytablemodel.setDelegates(self.querytableview,self.editFrame)
        self.bClose.clicked.connect(self.reject)
        self.bSave.clicked.connect(self.queriessave)
        QObject.connect(self.cQueryName,
                        SIGNAL("currentIndexChanged(int)"),
                        self.changeCurrentQuery)
        self.resizeAll()
        self.showQuery(self.querytablemodel.getCurrentQuery())
    def updateTreeStatus(self):
        print 'query up'
    def queriessave(self):
        self.querytablemodel.saveUserQueries()
    def changeCurrentQuery(self,*args):
        qtm=self.querytablemodel
        qtm.setCurrentQuery(self.cQueryName.currentText())
        qtm.refreshRows(self.querytableview)
    def resizeAll(self):
        for c in range(self.querytablemodel._cols):
            self.querytableview.resizeColumnToContents(c)
    def reject(self):
        self.close()
    def reset(self):
        for qn in self.querytablemodel.queriesNamesList():
            self.cQueryName.addItem(qn)
    def queries(self):
        return self.querytablemodel.queriesNamesList()
    def showQuery(self,name):
        if (name in self.querytablemodel.queriesNamesList()):
            pass
    def show(self):
        self.reset()
        super(Q7Query, self).show()

# -----------------------------------------------------------------
