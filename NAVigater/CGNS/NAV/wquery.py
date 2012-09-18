#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
import numpy

from PySide.QtCore  import *
from PySide.QtGui   import QFileDialog
from PySide.QtGui   import *
from CGNS.NAV.Q7QueryWindow import Ui_Q7QueryWindow
from CGNS.NAV.Q7SelectionWindow import Ui_Q7SelectionWindow
from CGNS.NAV.mquery import Q7QueryTableModel,Q7ComboBoxDelegate
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.mtree import COLUMN_VALUE,COLUMN_DATATYPE,COLUMN_SIDS,COLUMN_NAME
from CGNS.NAV.mtree import HIDEVALUE

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

# -----------------------------------------------------------------
class Q7SelectionList(Q7Window,Ui_Q7SelectionWindow):
    def __init__(self,control,model,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_SELECT,control,None,fgprint)
        self.bClose.clicked.connect(self.reject)
        self._model=model
        self._data=model._selected
        self._fgprint=fgprint
        self.bSave.clicked.connect(self.selectionsave)
        self.bDelete.clicked.connect(self.deletelocal)
    def show(self):
        self.reset()
        super(Q7SelectionList, self).show()
    def selectionsave(self):
        n='data=[\n'
        for path in self._data:
            node=self._model.nodeFromPath(path)
            t=node.data(COLUMN_SIDS)
            d=node.data(COLUMN_DATATYPE)
            v=node.sidsValue()
            if (type(v)==numpy.ndarray):
                if (v.dtype.char in ['S','c']):
                    s='"'+v.tostring()+'"'
                else:
                    s=str(v.tolist())
            else:
                s=str(v)
            n+='("%s","%s","%s",%s),\n'%(path,t,d,s)
        n+=']\n'
        filename=QFileDialog.getSaveFileName(self,"Save list",".","*.py")
        f=open(str(filename[0]),'w+')
        f.write(n)
        f.close()
    def deletelocal(self):
        pass
    def reset(self):
        tlvcols=3
        tlvcolsnames=['SIDS type','Data Type','Value']
        v=self.diagTable
        v.setColumnCount(self._fgprint.depth+tlvcols)
        lh=v.horizontalHeader()
        lv=v.verticalHeader()
        h=['/level%d'%d for d in range(self._fgprint.depth)]+tlvcolsnames
        n=len(h)
        for i in range(n):
            hi=QTableWidgetItem(h[i])
            v.setHorizontalHeaderItem(i,hi)
        plist=[]
        for path in self._data:
            v.setRowCount(v.rowCount()+1)
            r=v.rowCount()-1
            plist=path.split('/')[1:]
            for i in range(len(plist)):
                it=QTableWidgetItem('%s '%plist[i])
                it.setFont(QFont("Courier"))
                v.setItem(r,i,it)
            if tlvcols:
                node=self._model.nodeFromPath(path)
                if (node):
                  it1=QTableWidgetItem(node.data(COLUMN_SIDS))
                  it2=QTableWidgetItem(node.data(COLUMN_DATATYPE))
                  val=node.data(COLUMN_VALUE)
                  if (val==HIDEVALUE):
                      val=QIcon(QPixmap(":/images/icons/data-array-large.gif"))
                      it3=QTableWidgetItem(val,'')
                  else:
                      it3=QTableWidgetItem(val)
                      it3.setFont(QFont("Courier"))
                  it1.setFont(QFont("Courier"))
                  it3.setFont(QFont("Courier"))
                  v.setItem(r,n-3,it1)
                  v.setItem(r,n-2,it2)
                  v.setItem(r,n-1,it3)
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
