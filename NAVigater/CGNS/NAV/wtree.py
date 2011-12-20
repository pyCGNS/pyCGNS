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
from CGNS.NAV.wstylesheets import Q7TREEVIEWSTYLESHEET
import CGNS.NAV.wconstants as Q7WC

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
class Q7Tree(QWidget,Ui_Q7TreeWindow):
    def __init__(self,parent):
        QWidget.__init__(self,None)
        self._stylesheet=Q7TREEVIEWSTYLESHEET
        self._parent=parent
        self._fingerprint=None
        self._depthExpanded=0
        self._closealone=False
        self.form=None
        self.setupUi(self)
        self.setWindowTitle("Tree")
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
    def expandMinMax(self):
        if (self._depthExpanded==self._fingerprint.depth-2):
            self._depthExpanded=-1
            self.treeview.collapseAll()
        else:
            self._depthExpanded=self._fingerprint.depth-2
            self.treeview.expandAll()
        self.resizeAll()
    def expandLevel(self):
        if (self._depthExpanded<self._fingerprint.depth-2):
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
        self.form=Q7Form(self,node)
        self.form.show()
    def vtkview(self):
        self.vtk=Q7VTK(self)
        self.vtk.show()
    def bindTree(self,fgprint):
        self._fingerprint=fgprint
        self.treeview._parent=self
        self.treeview.setModel(self._fingerprint.model)
        self.treeview.setStyleSheet(self._stylesheet)
        self.treeview.setItemDelegate(Q7ItemDelegate(self))
    def closeEvent(self, event):
        if (not self._closealone):
            if (self._parent.close()):
                if (self.form): self.form.close()
                event.accept()
            else: event.ignore()
        else:
          if (self.form): self.form.close()
          event.accept()
    def closeAlone(self):
        if (self.form): self.form.close()
    def forceapply(self):
        pass
    def leave(self):
        self._closealone=True
        self.close()
        
# -----------------------------------------------------------------
