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
from CGNS.NAV.wquery import Q7Query, Q7SelectionList
from CGNS.NAV.wdiag import Q7CheckList
from CGNS.NAV.mquery import Q7QueryTableModel
from CGNS.NAV.mtree import Q7TreeModel, Q7TreeItem
import CGNS.NAV.mtree as NMT
from CGNS.NAV.wfingerprint import Q7Window,Q7fingerPrint
from CGNS.NAV.moption import Q7OptionContext as OCTXT
import CGNS.PAT.cgnskeywords as CGK

(CELLCOMBO,CELLTEXT)=range(2)
CELLEDITMODE=(CELLCOMBO,CELLTEXT)

# -----------------------------------------------------------------
class Q7TreeItemDelegate(QStyledItemDelegate):
    def __init__(self, owner):
        QStyledItemDelegate.__init__(self, owner)
        self._parent=owner
        self._mode=CELLTEXT
    def createEditor(self, parent, option, index):
        ws=option.rect.width()
        hs=option.rect.height()+4
        xs=option.rect.x()
        ys=option.rect.y()-2
        if (index.column() in [NMT.COLUMN_NAME,NMT.COLUMN_VALUE]):
          self._mode=CELLTEXT
          editor=QLineEdit(parent)
          editor.transgeometry=(xs,ys,ws,hs)
          editor.installEventFilter(self)
          self.setEditorData(editor,index)
          return editor
        if (index.column()==NMT.COLUMN_SIDS):
          self._mode=CELLCOMBO
          editor=QComboBox(parent)
          editor.transgeometry=(xs,ys,ws,hs)
          itemslist=index.internalPointer().sidsTypeList()
          editor.addItems(itemslist)
          editor.setCurrentIndex(0)
          editor.installEventFilter(self)
          self.setEditorData(editor,index)
          return editor
        if (index.column()==NMT.COLUMN_DATATYPE):
          self._mode=CELLCOMBO
          editor=QComboBox(parent)
          editor.transgeometry=(xs,ys,ws,hs)
          itemslist=index.internalPointer().sidsDataTypeList()
          editor.addItems(itemslist)
          editor.setCurrentIndex(0)
          editor.installEventFilter(self)
          self.setEditorData(editor,index)
          return editor
        return None
    def setEditorData(self, editor, index):
        if (self._mode==CELLTEXT):
            value=index.data()
            editor.clear()
            editor.insert(value)
        elif (self._mode==CELLCOMBO):
            value = index.data()
            ix=editor.findText(value)
            if (ix!=-1): editor.setCurrentIndex(ix)
        else:
            pass
    def setModelData(self,editor,model,index):
        value=None
        if (self._mode==CELLCOMBO):
            value=editor.currentText()
        if (self._mode==CELLTEXT):
            value=editor.text()
        model.setData(index,value,role=Qt.EditRole)
    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(*editor.transgeometry)
    def paint(self, painter, option, index):
        if (index.column()==NMT.COLUMN_NAME):
          if (index.internalPointer().sidsName() not in OCTXT._ReservedNames):
            option.font.setWeight(QFont.Bold)
          if (index.internalPointer().userState()in NMT.STUSR_1):
            pass
          QStyledItemDelegate.paint(self, painter, option, index)
          option.font.setWeight(QFont.Light)
        elif (index.column() in [NMT.COLUMN_VALUE,NMT.COLUMN_DATATYPE]):
          option.font.setFamily(OCTXT.FixedFontTable)
          if (index.column() == NMT.COLUMN_DATATYPE):
              option.font.setPointSize(8)
          QStyledItemDelegate.paint(self, painter, option, index)
        elif (index.column() in NMT.COLUMN_FLAGS):
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
        qlist=self.querymodel.queriesNamesList()
        qlist.sort()
        for q in qlist: self.cQuery.addItem(q)
        ix=self.cQuery.findText(self.querymodel.getCurrentQuery())
        if (ix!=-1): self.cQuery.setCurrentIndex(ix)
        self.bSave.clicked.connect(self.savetree)
        self.bApply.clicked.connect(self.forceapply)
        self.bClose.clicked.connect(self.leave)
        self.bZoomIn.clicked.connect(self.expandLevel)
        self.bZoomOut.clicked.connect(self.collapseLevel)
        self.bZoomAll.clicked.connect(self.expandMinMax)
        self.bFormView.clicked.connect(self.formview)
        self.bMarkAll.clicked.connect(self.markall)
        self.bUnmarkAll.clicked.connect(self.unmarkall)
        self.bPreviousMark.clicked.connect(self.previousmark)
        self.bNextMark.clicked.connect(self.nextmark)
        self.bSwapMarks.clicked.connect(self.swapmarks)
        self.bMarksAsList.clicked.connect(self.selectionlist)
        self.bApply.clicked.connect(self.applyquery)
        self.bVTKView.clicked.connect(self.vtkview)
        self.bQueryView.clicked.connect(self.queryview)
        self.bScreenShot.clicked.connect(self.screenshot)
        self.bCheck.clicked.connect(self.check)
        self.bCheckList.clicked.connect(self.checklist)
        self.bClearChecks.clicked.connect(self.clearchecks)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.popupmenu = QMenu()
        self.treeview.setModel(self._fgprint.model)
        self.treeview.setItemDelegate(Q7TreeItemDelegate(self.treeview))
        self.treeview.setControlWindow(self,self._fgprint.model)
        if (self._control.transientRecurse): self.expandMinMax()
        if (self._control.transientVTK):     self.vtkview()
        self.clearchecks()
    def savetree(self):
        if ((self._fgprint.converted) or
            not (self._fgprint.isModified())): return
        self._fgprint.modifiedTreeStatus(Q7fingerPrint.STATUS_UNCHANGED)
        self.updateTreeStatus()
    def screenshot(self):
        sshot=QPixmap.grabWindow(self.treeview.winId())
        sshot.save('/tmp/foo.png','png')
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
    def popform(self):
        self.formview()
    def openSubTree(self):
        self.busyCursor()
        node=self.lastNodeMenu.internalPointer().sidsPath()
        child=Q7Tree(self._control,node,self._fgprint)
        self.readyCursor()
        child.show()
    def pop0(self):
        pass
    def newnodebrother(self):
        if (self.lastNodeMenu.isValid()):
            nodeitem=self.lastNodeMenu.internalPointer()
            self._fgprint.model.newNodeBrother(nodeitem)
    def newnodechild(self):
        if (self.lastNodeMenu.isValid()):
            nodeitem=self.lastNodeMenu.internalPointer()
            self._fgprint.model.newNodeChild(nodeitem)
    def marknode(self):
        if (self.lastNodeMenu.isValid()):
             nodeitem=self.lastNodeMenu.internalPointer()
             self.treeview.markNode(nodeitem)
    def mcopy(self):
        if (self.lastNodeMenu.isValid()):
            nodeitem=self.lastNodeMenu.internalPointer()
            self._fgprint.model.copyNode(nodeitem)
    def mcutselected(self):
        self._fgprint.model.cutAllSelectedNodes()
    def mcut(self):
        if (self.lastNodeMenu.isValid()):
            nodeitem=self.lastNodeMenu.internalPointer()
            self.lastNodeMenu=self.lastNodeMenu.parent()
            self._fgprint.model.cutNode(nodeitem)
    def mpasteasbrotherselected(self):
        self._fgprint.model.pasteAsBrotherAllSelectedNodes()
    def mpasteasbrother(self):
        if (self.lastNodeMenu.isValid()):
            nodeitem=self.lastNodeMenu.internalPointer()
            self._fgprint.model.pasteAsBrother(nodeitem)
            self._fgprint.model.dataChanged.emit(self.lastNodeMenu,
                                                 self.lastNodeMenu)
    def mpasteaschildselected(self):
        self._fgprint.model.pasteAsChildAllSelectedNodes()
    def mpasteaschild(self):
        if (self.lastNodeMenu.isValid()):
            nodeitem=self.lastNodeMenu.internalPointer()
            self._fgprint.model.pasteAsChild(nodeitem)
    def updateMenu(self,nodeidx):
        self.lastNodeMenu=nodeidx
        if (nodeidx != -1):
          node=nodeidx.internalPointer()
          actlist=(("About %s"%node.sidsType(),self.pop0,None),
                   None,
                   ("Mark/unmark node",self.marknode,'Space'),
                   ("Add new child node",self.newnodechild,'Ctrl+A'),
                   ("Add new brother node",self.newnodebrother,'Ctrl+Z'),
                   None,
                   ("Open form",self.popform,'Ctrl+F'),
                   ("Open view",self.openSubTree,'Ctrl+W'),
                   None,
                   ("Copy",self.mcopy,'Ctrl+C'),
                   ("Cut",self.mcut,'Ctrl+X'),
                   ("Paste as brother",self.mpasteasbrother,'Ctrl+V'),
                   ("Paste as child",self.mpasteaschild,'Ctrl+Y'),
                   None,
                   ("Cut all selected",self.mcutselected,'Ctrl+O'),
                   ("Paste as brother for each selected",
                    self.mpasteasbrotherselected,'Ctrl+K'),
                   ("Paste as child for each selected",
                    self.mpasteaschildselected,'Ctrl+I'),
                   )
          self.popupmenu.clear()
          self.popupmenu.setTitle('Node menu')
          for aparam in actlist:
              if (aparam is None): self.popupmenu.addSeparator()
              else:
                  a=QAction(aparam[0],self,triggered=aparam[1])
                  if (aparam[2] is not None): a.setShortcut(aparam[2])
                  self.popupmenu.addAction(a)
    def setLastEntered(self):
        nix=self.treeview.currentIndex()
        if (nix.isValid()):
            self._lastEntered=nix.internalPointer()
    def getLastEntered(self):
        return self._lastEntered
    def clickedNode(self,index):
        self.setLastEntered()
        if (self.treeview.lastButton==Qt.RightButton):
            self.updateMenu(self.treeview.currentIndex())
            self.popupmenu.popup(self.treeview.lastPos)
    def expandNode(self,*args):
        self.resizeAll()
    def collapseNode(self,*args):
        pass
    def resizeAll(self):
        for n in range(NMT.COLUMN_LAST+1):
            self.treeview.resizeColumnToContents(n)
    def show(self):
        super(Q7Tree, self).show()
    def applyquery(self):
        q=self.cQuery.currentText()
        v=self.eUserVariable.text()
        if (q in ['',' ']):
            self.unmarkall()
            return
        qry=self.querymodel
        if (q in qry.queriesNamesList()):
            sl=qry.getQuery(q).run(self._fgprint.tree,v)
            self.treeview.model().markExtendToList(sl)
            self.treeview.model().updateSelected()
        self.treeview.refreshView()
    def check(self):
        self.busyCursor()
        self.lastdiag=self.treeview.model().checkSelected()
        self.readyCursor()
        self.treeview.refreshView()
    def checklist(self):
        if (self.lastdiag is None): return
        self.diagview=Q7CheckList(self._control,self.lastdiag,self._fgprint)
        self.diagview.show()
    def clearchecks(self):
        self.treeview.model().checkClear()
        self.treeview.refreshView()
        self.lastdiag=None
    def selectionlist(self):
        slist=Q7SelectionList(self._control,self.treeview.model()._selected,
                              self._fgprint)
        slist.show()
    def previousmark(self):
        self.treeview.changeSelectedMark(-1)
    def nextmark(self):
        self.treeview.changeSelectedMark(+1)
    def markall(self):
        self.treeview.model().markAll()
        self.treeview.model().updateSelected()
        self.treeview.refreshView()
    def unmarkall(self):
        self.treeview.model().unmarkAll()
        self.treeview.model().updateSelected()
        self.treeview.refreshView()
    def swapmarks(self):
        self.treeview.model().swapMarks()
        self.treeview.model().updateSelected()
        self.treeview.refreshView()
    def formview(self):
        node=self.treeview.currentIndex().internalPointer()
        if (node is None): return
        if (node.sidsType()==CGK.CGNSTree_ts): return
        form=Q7Form(self._control,node,self._fgprint)
        form.show()
    def vtkview(self):
        zlist=self.treeview.model().getSelectedZones()
        node=self.treeview.currentIndex().internalPointer()
        self.busyCursor()
        vtk=Q7VTK(self._control,node,self._fgprint,self.treeview.model(),zlist)
        self.readyCursor()
        vtk.show()
    def queryview(self):
        q=self.querymodel.getCurrentQuery()
        self.querymodel.setCurrentQuery(' ')
        qry=Q7Query(self._control,self._fgprint,self)
        qry.show()
        self.querymodel.setCurrentQuery(q)
    def closeAlone(self):
        pass
    def forceapply(self):
        pass
    def leave(self):
        self.close()
    def updateTreeStatus(self):
        if (self._fgprint.converted): return
        if (self._fgprint._status==Q7fingerPrint.STATUS_MODIFIED):
            self.bSave.setIcon(self.I_MODIFIED)
        else:
            self.bSave.setIcon(self.I_UNCHANGED)
        
# -----------------------------------------------------------------
