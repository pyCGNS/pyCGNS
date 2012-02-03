#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

from PySide.QtCore import Slot as pyqtSlot
from PySide.QtCore import *
from PySide.QtGui  import *
from CGNS.NAV.Q7TreeWindow import Ui_Q7TreeWindow
from CGNS.NAV.wform import Q7Form
from CGNS.NAV.wvtk import Q7VTK
from CGNS.NAV.wquery import Q7Query
from CGNS.NAV.mquery import Q7QueryTableModel
from CGNS.NAV.mtree import Q7TreeModel
from CGNS.NAV.mtree import Q7TreeItem
import CGNS.NAV.wconstants as Q7WC
from CGNS.NAV.wfingerprint import Q7Window

# -----------------------------------------------------------------
class Q7TreeItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if ((index.column()==0) and
            (index.internalPointer().sidsName() not in Q7WC.reservedNames)):
            option.font.setWeight(QFont.Bold)
            QStyledItemDelegate.paint(self, painter, option, index)
            option.font.setWeight(QFont.Light)
        elif (index.column()==8):
            option.font.setFamily(Q7WC.FixedFontTable)
            QStyledItemDelegate.paint(self, painter, option, index)
        elif (index.column() in [2,4,5,6,7]):
            option.decorationPosition=QStyleOptionViewItem.Top
            QStyledItemDelegate.paint(self, painter, option, index)
            option.decorationPosition=QStyleOptionViewItem.Left
        else:
            QStyledItemDelegate.paint(self, painter, option, index)

# -----------------------------------------------------------------
class Q7Tree(Q7Window,Ui_Q7TreeWindow):
    def __init__(self,control,path,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_TREE,control,path,fgprint)
        self._depthExpanded=0
        self._lastEntered=None
        QObject.connect(self.treeview,
                        SIGNAL("expanded(QModelIndex)"),
                        self.expandNode)
        QObject.connect(self.treeview,
                        SIGNAL("collapsed()"),
                        self.collapseNode)
        QObject.connect(self.treeview,
                        SIGNAL("clicked(QModelIndex)"),
                        self.clickedNode)
        QObject.connect(self.treeview,
                        SIGNAL("customContextMenuRequested(QPoint)"),
                        self.clickedNode)
        self.querymodel=Q7QueryTableModel(self)
        self._currentQuery=self.querymodel.currentQuery
        for q in self.querymodel.defaultQueriesList: self.cQuery.addItem(q)
        ix=self.cQuery.findText(self._currentQuery)
        if (ix!=-1): self.cQuery.setCurrentIndex(ix)
        self.bApply.clicked.connect(self.forceapply)
        self.bClose.clicked.connect(self.leave)
        self.bZoomIn.clicked.connect(self.expandLevel)
        self.bZoomOut.clicked.connect(self.collapseLevel)
        self.bZoomAll.clicked.connect(self.expandMinMax)
        self.bForm.clicked.connect(self.formview)
        self.bMarkAll.clicked.connect(self.markall)
        self.bUnmarkAll.clicked.connect(self.unmarkall)
        self.bPreviousMark.clicked.connect(self.previousmark)
        self.bNextMark.clicked.connect(self.nextmark)
        self.bSwapMarks.clicked.connect(self.swapmarks)
        self.bApply.clicked.connect(self.applyquery)
        self.bVTK.clicked.connect(self.vtkview)
        self.bOpenOperateView.clicked.connect(self.queryview)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.popupmenu = QMenu()
        self.popupmenu.addAction(QAction("Open form",self))
        self.treeview.setModel(self._fgprint.model)
        self.treeview.setItemDelegate(Q7TreeItemDelegate(self))
        self.treeview.setControlWindow(self,self._fgprint.model)
    def expandMinMax(self):
        if (self._depthExpanded==self._fgprint.depth-2):
            self._depthExpanded=-1
            self.treeview.collapseAll()
        else:
            self._depthExpanded=self._fgprint.depth-2
            self.treeview.expandAll()
        self.resizeAll()
    def expandLevel(self):
        if (self._depthExpanded<self._fgprint.depth-2):
            self._depthExpanded+=1
        self.treeview.expandToDepth(self._depthExpanded)
        self.resizeAll()
    def collapseLevel(self):
        if (self._depthExpanded!=-1): self._depthExpanded-=1
        if (self._depthExpanded==-1):
            self.treeview.collapseAll()
        else:
            self.treeview.expandToDepth(self._depthExpanded)
        self.resizeAll()
    def updateStatus(self,node):
        self.lineEdit.clear()
        self.lineEdit.insert(node.sidsPath())
    def updateMenu(self,node):
        print node.sidsPath()
        self.popupmenu.addAction(QAction("About %s"%node.sidsType(),self))
    def setLastEntered(self):
        self._lastEntered=self.treeview.currentIndex()
    def getLastEntered(self):
        return self._lastEntered
    def clickedNode(self,index):
        self.setLastEntered()
        if (self.treeview.lastButton==Qt.RightButton):
            self.updateMenu(self.treeview.currentIndex().internalPointer())
            self.popupmenu.popup(self.treeview.lastPos)
    def expandNode(self,*args):
        print 'EXPAND ',args
        self.resizeAll()
    def collapseNode(self,*args):
        print 'COLLAPSE ',args
    def resizeAll(self):
        for n in range(9):
            self.treeview.resizeColumnToContents(n)
    def show(self):
        super(Q7Tree, self).show()
    def applyquery(self):
        q=self.cQuery.currentText()
        if (q==''):
            self.unmarkall()
            return
        qry=self.querymodel
        if (q in qry.queries()):
            sl=qry.queries()[q].run(self._fgprint.tree)
            self.treeview.model().markExtendToList(sl)
            self.treeview.model().updateSelected()
    def previousmark(self):
        self.treeview.changeSelectedMark(-1)
    def nextmark(self):
        self.treeview.changeSelectedMark(+1)
    def markall(self):
        self.treeview.model().markAll()
        self.treeview.model().updateSelected()
    def unmarkall(self):
        self.treeview.model().unmarkAll()
        self.treeview.model().updateSelected()
    def swapmarks(self):
        self.treeview.model().swapMarks()
        self.treeview.model().updateSelected()
    def formview(self):
        node=self.treeview.currentIndex().internalPointer()
        if (node.sidsType()=='CGNSTree_t'): return
        form=Q7Form(self._control,node,self._fgprint)
        form.show()
    def vtkview(self):
        node=self.treeview.currentIndex().internalPointer()
        vtk=Q7VTK(self._control,node,self._fgprint)
        vtk.show()
    def queryview(self):
        qry=Q7Query(self._control,self._fgprint,self)
        qry.show()
    def closeAlone(self):
        pass
    def forceapply(self):
        pass
    def leave(self):
        self.close()
        
# -----------------------------------------------------------------
