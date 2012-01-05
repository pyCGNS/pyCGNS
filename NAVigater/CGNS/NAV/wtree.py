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
from CGNS.NAV.mtree import Q7TreeModel
import CGNS.NAV.wconstants as Q7WC
from CGNS.NAV.wfingerprint import Q7Window

# -----------------------------------------------------------------
class Q7ItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if ((index.column()==0) and
            (index.internalPointer().sidsName() not in Q7WC.reservedNames)):
            option.font.setWeight(QFont.Bold)
            QStyledItemDelegate.paint(self, painter, option, index)
        elif (index.column()==8):
            option.font.setFamily(Q7WC.FixedFontTable)
            QStyledItemDelegate.paint(self, painter, option, index)
        else:
            QStyledItemDelegate.paint(self, painter, option, index)

# -----------------------------------------------------------------
class Q7Tree(Q7Window,Ui_Q7TreeWindow):
    def __init__(self,control,path,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_TREE,control,path,fgprint)
        self._depthExpanded=0
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
        queries=['','BCWall','ORPHAN','Trigger']
        for q in queries:
          self.Q7QueryComboWidget.addItem(q)
        self.bApply.clicked.connect(self.forceapply)
        self.bClose.clicked.connect(self.leave)
        self.bZoomIn.clicked.connect(self.expandLevel)
        self.bZoomOut.clicked.connect(self.collapseLevel)
        self.bZoomAll.clicked.connect(self.expandMinMax)
        self.bForm.clicked.connect(self.formview)
        self.bVTK.clicked.connect(self.vtkview)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.popupmenu = QMenu()
        self.popupmenu.addAction(QAction("Open form",self))
        self.treeview.setModel(self._fgprint.model)
        self.treeview.setItemDelegate(Q7ItemDelegate(self))
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
    def clickedNode(self,index):
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
    def formview(self):
        node=self.treeview.currentIndex().internalPointer()
        form=Q7Form(self._control,node,self._fgprint)
        form.show()
    def vtkview(self):
        node=self.treeview.currentIndex().internalPointer()
        vtk=Q7VTK(self._control,node,self._fgprint)
        vtk.show()
    def closeEvent(self, event):
        event.accept()
    def closeAlone(self):
        pass
    def forceapply(self):
        pass
    def leave(self):
        self.close()
        
# -----------------------------------------------------------------
