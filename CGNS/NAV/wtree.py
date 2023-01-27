#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

import sys
import gc
import functools

from ..PAT import cgnskeywords as CGK
from ..PAT import cgnsutils as CGU
from ..PAT import SIDS as CGS

from qtpy.QtCore import Qt, QModelIndex
from qtpy.QtWidgets import (
    QStyledItemDelegate,
    QLineEdit,
    QComboBox,
    QSizePolicy,
    QStyleOptionViewItem,
    QMenu,
    QAction,
)
from qtpy.QtGui import QFont, QPalette, QScreen, QKeySequence

from CGNS.pyCGNSconfig import HAS_VTK

from .Q7TreeWindow import Ui_Q7TreeWindow
from .wform import Q7Form
from .wpattern import Q7PatternList
from .wquery import Q7Query, Q7SelectionList
from .wdiag import Q7CheckList
from .wlink import Q7LinkList
from .mtree import Q7TreeModel, Q7TreeItem
from .wfingerprint import Q7Window, Q7FingerPrint
from .moption import Q7OptionContext as OCTXT

from . import mtree as NMT
from . import wmessages as MSG

(CELLCOMBO, CELLTEXT) = range(2)
CELLEDITMODE = (CELLCOMBO, CELLTEXT)


# -----------------------------------------------------------------
class Q7TreeItemDelegate(QStyledItemDelegate):
    def __init__(self, wtree, model):
        QStyledItemDelegate.__init__(self, wtree)
        self._parent = wtree
        self._mode = CELLTEXT
        self._model = model

    def createEditor(self, parent, option, index):
        if self._parent.isLocked():
            return None
        if index.internalPointer().sidsIsCGNSTree():
            return None
        if index.internalPointer().sidsIsLink():
            return None
        if index.internalPointer().sidsIsLinkChild():
            return None
        ws = option.rect.width()
        hs = option.rect.height() + 4
        xs = option.rect.x()
        ys = option.rect.y() - 2
        if not index.internalPointer().hasEditCheck() or (
            index.column() in [NMT.COLUMN_NAME]
        ):
            self._mode = CELLTEXT
            editor = QLineEdit(parent)
            editor.transgeometry = (xs, ys, ws, hs)
            editor.installEventFilter(self)
            self.setEditorData(editor, index)
            return editor
        if index.column() in [NMT.COLUMN_VALUE]:
            node = self._parent.modelData(index)
            if node.hasValueView():
                pt = node.sidsPath().split("/")[1:]
                lt = node.sidsTypePath()
                fc = self._parent._control._control.userFunctionFromPath(pt, lt)
                if fc is not None:
                    en = fc.getEnumerate(pt, lt)
                else:
                    en = node.sidsValueEnum()
                if en is None:
                    self._mode = CELLTEXT
                    editor = QLineEdit(parent)
                    editor.transgeometry = (xs, ys, ws, hs)
                else:
                    self._mode = CELLCOMBO
                    editor = QComboBox(parent)
                    editor.transgeometry = (xs, ys, ws, hs)
                    editor.addItems(en)
                    try:
                        tix = en.index(node.sidsValue().tostring().decode("ascii"))
                    except ValueError:
                        editor.insertItem(
                            0, node.sidsValue().tostring().decode("ascii")
                        )
                        tix = 0
                    editor.setCurrentIndex(tix)
                editor.installEventFilter(self)
                self.setEditorData(editor, index)
                return editor
        if index.column() == NMT.COLUMN_SIDS:
            self._mode = CELLCOMBO
            editor = QComboBox(parent)
            editor.transgeometry = (xs, ys, ws, hs)
            tnode = self._parent.modelData(index)
            itemslist = tnode.sidsTypeList()
            editor.addItems(itemslist)
            try:
                tix = itemslist.index(tnode.sidsType())
            except ValueError:
                editor.insertItem(0, tnode.sidsType())
                tix = 0
            editor.setCurrentIndex(tix)
            editor.installEventFilter(self)
            self.setEditorData(editor, index)
            return editor
        if index.column() == NMT.COLUMN_DATATYPE:
            self._mode = CELLCOMBO
            editor = QComboBox(parent)
            editor.transgeometry = (xs, ys, ws, hs)
            editor.setProperty("Q7SIDSDataTypeComboBox", "True")
            itemslist = self._parent.modelData(index).sidsDataType(all=True)
            editor.addItems(itemslist)
            editor.installEventFilter(self)
            sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            editor.setSizePolicy(sizePolicy)
            self.setEditorData(editor, index)
            return editor
        return None

    def setEditorData(self, editor, index):
        if self._mode == CELLTEXT:
            value = index.data()
            editor.clear()
            editor.insert(value)
        elif self._mode == CELLCOMBO:
            value = index.data()
            ix = editor.findText(value)
            print("VALUE", value, ix)
            if ix != -1:
                editor.setCurrentIndex(ix)
        else:
            pass

    def setModelData(self, editor, model, index):
        value = None
        if self._mode == CELLCOMBO:
            value = editor.currentText()
        if self._mode == CELLTEXT:
            value = editor.text()
        pth = self._parent.modelData(index).sidsPath()
        model.setData(index, value, role=Qt.EditRole)
        node = index.internalPointer().lastEdited()
        if node is not None:
            nindex = self._parent.model().indexByPath(node.sidsPath())
            self._parent.setLastEntered(nindex)
        if self._parent._control._selectwindow is not None:
            self._parent._control._selectwindow.reset()

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(*editor.transgeometry)

    def paint(self, painter, option, index):
        if self._parent.modelIndex(index).column() == NMT.COLUMN_NAME:
            if self._parent.modelData(index).sidsName() not in OCTXT._ReservedNames:
                option.font.setWeight(QFont.Bold)
            uf = self._parent.modelData(index).userState()
            if uf in NMT.USERSTATES:
                if self._model.hasUserColor(uf):
                    cl = self._model.getUserColor(uf)
                    option.palette.brush(QPalette.Text).setColor(cl)
            QStyledItemDelegate.paint(self, painter, option, index)
            option.font.setWeight(QFont.Light)
        elif index.column() in [NMT.COLUMN_VALUE, NMT.COLUMN_DATATYPE]:
            option.font.setFamily(OCTXT.Table_Family)
            if index.column() == NMT.COLUMN_DATATYPE:
                # option.font.setPointSize(8)
                pass
            QStyledItemDelegate.paint(self, painter, option, index)
        elif index.column() in NMT.COLUMN_FLAGS:
            option.decorationPosition = QStyleOptionViewItem.Top
            QStyledItemDelegate.paint(self, painter, option, index)
            option.decorationPosition = QStyleOptionViewItem.Left
        else:
            QStyledItemDelegate.paint(self, painter, option, index)

    def doRelease(self):
        self._model = None


# -----------------------------------------------------------------
class Q7Tree(Q7Window, Ui_Q7TreeWindow):
    def __init__(self, control, path, fgprintindex):
        Q7Window.__init__(self, Q7Window.VIEW_TREE, control, path, fgprintindex)
        self._depthExpanded = 0
        self._lastEntered = None
        self.lastdiag = None
        self._linkwindow = None
        self._querywindow = None
        self._vtkwindow = None
        self._selectwindow = None
        self._column = {
            NMT.COLUMN_SIDS: OCTXT.ShowSIDSColumn,
            NMT.COLUMN_FLAG_LINK: OCTXT.ShowLinkColumn,
            NMT.COLUMN_FLAG_SELECT: OCTXT.ShowSelectColumn,
            NMT.COLUMN_FLAG_CHECK: OCTXT.ShowCheckColumn,
            NMT.COLUMN_FLAG_USER: OCTXT.ShowUserColumn,
            NMT.COLUMN_SHAPE: OCTXT.ShowShapeColumn,
            NMT.COLUMN_DATATYPE: OCTXT.ShowDataTypeColumn,
        }
        self.selectForLinkSrc = None  # one link source per tree view allowed

        # self.treeview.expanded[QModelIndex].connect(self.expandNode)
        self.treeview.collapsed.connect(self.collapseNode)
        self.treeview.pressed[QModelIndex].connect(self.clickedPressedNode)
        self.treeview.customContextMenuRequested.connect(self.clickedNode)

        # QObject.connect(self.treeview,
        #                SIGNAL("expanded(QModelIndex)"),
        #                self.expandNode)
        # QObject.connect(self.treeview,
        #                SIGNAL("collapsed()"),
        #                self.collapseNode)
        # QObject.connect(self.treeview,
        #                SIGNAL("pressed(QModelIndex)"),
        #                self.clickedPressedNode)
        # QObject.connect(self.treeview,
        #                SIGNAL("customContextMenuRequested(QPoint)"),
        #                self.clickedNode)

        self.bSave.clicked.connect(self.savetree)
        self.lockable(self.bSave)
        self.bQueryView.clicked.connect(self.queryview)
        self.lockable(self.bQueryView)
        self.bSaveAs.clicked.connect(self.savetreeas)
        self.lockable(self.bSaveAs)
        self.bInfo.clicked.connect(self.infoTreeView)
        self.bZoomIn.clicked.connect(self.expandLevel)
        self.bZoomOut.clicked.connect(self.collapseLevel)
        self.bZoomAll.clicked.connect(self.expandMinMax)
        self.bFormView.clicked.connect(self.formview)
        self.bMarkAll.clicked.connect(self.markall)
        self.bUnmarkAll_1.clicked.connect(self.unmarkall)
        self.bUnmarkAll_2.clicked.connect(self.unmarkall)
        self.bPreviousMark.clicked.connect(self.previousmark)
        self.bNextMark.clicked.connect(self.nextmark)
        self.bSwapMarks.clicked.connect(self.swapmarks)
        self.bMarksAsList.clicked.connect(self.selectionlist)
        self.bVTKView.clicked.connect(self.vtkview)
        self.lockable(self.bVTKView)
        self.bScreenShot.clicked.connect(self.screenshot)
        self.bCheck.clicked.connect(self.check)
        self.bCheckList.clicked.connect(self.checklist)
        self.bClearChecks.clicked.connect(self.clearchecks)
        self.bLinkView.clicked.connect(self.linklist)
        self.bPatternView.clicked.connect(self.patternlist)
        self.bToolsView.clicked.connect(self.tools)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.popupmenu = QMenu()
        self.diagview = None
        lmodel = self.FG.model
        self.treeview.setModel(lmodel)
        self.treeview.setItemDelegate(Q7TreeItemDelegate(self.treeview, lmodel))
        self.treeview.setControlWindow(self, self.FG.index)
        if self._control.transientRecurse or OCTXT.RecursiveTreeDisplay:
            self.expandMinMax()
        if self._control.transientVTK:
            self.vtkview()
        self._control.transientRecurse = False
        self._control.transientVTK = False
        self.clearchecks()
        #
        self.bCheckList.setDisabled(True)
        if not OCTXT._HasProPackage:
            self.bToolsView.setDisabled(True)
        self.bCheckView.setDisabled(True)
        self.bPatternDB.setDisabled(True)
        self.bAddLink.clicked.connect(self.linkadd)
        self.bSelectLinkSrc.clicked.connect(self.linkselectsrc)
        self.bSelectLinkDst.clicked.connect(self.linkselectdst)
        self.bAddLink.setDisabled(True)
        self.lineEdit.returnPressed.connect(self.jumpToNode)
        # QObject.connect(self.lineEdit,
        #                SIGNAL("returnPressed()"),
        #                self.jumpToNode)
        tvh = self.treeview.header()
        tvh.setContextMenuPolicy(Qt.CustomContextMenu)
        tvh.customContextMenuRequested.connect(self.headerMenu)
        self._hmenu = QMenu()
        self._hmenu._idx = {}
        self._tlist = (
            ("SIDS type", NMT.COLUMN_SIDS),
            ("Link flag", NMT.COLUMN_FLAG_LINK),
            ("Mark flag", NMT.COLUMN_FLAG_SELECT),
            ("Check flag", NMT.COLUMN_FLAG_CHECK),
            ("User flag", NMT.COLUMN_FLAG_USER),
            ("Shape", NMT.COLUMN_SHAPE),
            ("Data type", NMT.COLUMN_DATATYPE),
        )
        for (tag, idx) in self._tlist:
            a = QAction(tag, self._hmenu, checkable=True)
            self._hmenu._idx[idx] = a
            if self._column[idx]:
                a.setChecked(True)
            else:
                a.setChecked(False)
            self._hmenu.addAction(a)
            self.treeview.setColumnHidden(idx, not self._column[idx])
        self._recursiveAddNewNode = False
        self.updateTreeStatus()

    def headerMenu(self, pos):
        for (tag, idx) in self._tlist:
            self._hmenu._idx[idx].setChecked(self._column[idx])
        self._hmenu.exec_(self.treeview.mapToGlobal(pos))
        for (tag, idx) in self._tlist:
            if self._hmenu._idx[idx].isChecked():
                self._column[idx] = True
            else:
                self._column[idx] = False
            self.treeview.setColumnHidden(idx, not self._column[idx])

    def model(self):
        return self.FG.model

    def modelIndex(self, idx):
        if not idx.isValid():
            return -1
        midx = idx
        if idx.model() != self.treeview.M():
            midx = self.treeview.model().mapToSource(idx)
        return midx

    def modelData(self, idx):
        if not idx.isValid():
            return None
        return self.modelIndex(idx).internalPointer()

    def savetree(self):
        if not (self.FG.isSaveable() and self.FG.isModified()):
            return
        self._control.savedirect(self.FG)
        self.updateTreeStatus()

    def tools(self):
        from .wtools import Q7ToolsView

        if self._control._toolswindow is None:
            self._control._toolswindow = Q7ToolsView(self._control, self.FG, self)
            self._control._toolswindow.show()
        else:
            self._control._toolswindow.raise_()

    def savetreeas(self):
        self._control.save(self.FG)
        self.updateTreeStatus()

    def infoTreeView(self):
        self._control.helpWindow("Tree")

    def screenshot(self):
        self.treeview.model().sort(0)
        sshot = QScreen.grabWindow(self.treeview.winId())
        sshot.save("/tmp/foo.png", "png")

    def expandMinMax(self):
        if self._depthExpanded == self.FG.depth - 2:
            self._depthExpanded = -1
            self.treeview.collapseAll()
        else:
            self._depthExpanded = self.FG.depth - 2
            self.treeview.expandAll()
        self.resizeAll()

    def resetOptions(self):
        if OCTXT.AutoExpand:
            self.treeview.setAutoExpandDelay(1000)
        else:
            self.treeview.setAutoExpandDelay(-1)

    def expandLevel(self):
        if self._depthExpanded < self.FG.depth - 2:
            self._depthExpanded += 1
        self.treeview.expandToDepth(self._depthExpanded)
        self.resizeAll()

    def collapseLevel(self):
        if self._depthExpanded != -1:
            self._depthExpanded -= 1
        if self._depthExpanded == -1:
            self.treeview.collapseAll()
        else:
            self.treeview.expandToDepth(self._depthExpanded)
        self.resizeAll()

    def updateStatus(self, node):
        if not self.lineEditLock.isChecked():
            self.lineEdit.clear()
            self.lineEdit.insert(node.sidsPath())

    def jumpToNode(self):
        path = self.lineEdit.text()
        self.treeview.selectByPath(path)

    def popform(self):
        self.formview()

    def openLkTree(self):
        self.busyCursor()
        filename = self.getLastEntered().sidsLinkFilename()
        if filename is not None:
            self._control.loadfile(filename)
        self.readyCursor()

    def openSubTree(self):
        self.busyCursor()
        node = self.getLastEntered().sidsPath()
        child = Q7Tree(self._control, node, self.FG)
        self.readyCursor()
        child.show()

    def pop0(self):
        pass

    def newnodebrother(self):
        if self.getLastEntered() is not None:
            self.model().newNodeBrother(self.getLastEntered())

    def newnodechild(self):
        if self.getLastEntered() is not None:
            self.model().newNodeChild(self.getLastEntered())

    def marknode(self):
        if self.getLastEntered() is not None:
            self.treeview.markNode(self.getLastEntered())

    def mcopy(self):
        if self.getLastEntered() is not None:
            self.model().copyNode(self.getLastEntered())
            self.clearOtherSelections()

    def mcutselected(self):
        self.model().cutAllSelectedNodes()
        self.clearLastEntered()
        self.clearOtherSelections()

    def mcut(self):
        if self.getLastEntered() is not None:
            self.model().cutNode(self.getLastEntered())
            self.clearLastEntered()
            self.clearOtherSelections()

    def mdel(self):
        if self.getLastEntered() is not None:
            self.model().deleteNode(self.getLastEntered())
            self.clearLastEntered()
            self.clearOtherSelections()

    def mpasteasbrotherselected(self):
        self.model().pasteAsBrotherAllSelectedNodes()

    def mpasteasbrother(self):
        if self.getLastEntered() is not None:
            self.model().pasteAsBrother(self.getLastEntered())

    def mpasteaschildselected(self):
        self.model().pasteAsChildAllSelectedNodes()

    def mpasteaschild(self):
        if self.getLastEntered() is not None:
            self.model().pasteAsChild(self.getLastEntered())

    def updateMenu(self, nodeidxs):
        nodeidx = self.modelIndex(nodeidxs)
        if not nodeidx.isValid:
            return False
        if nodeidx.internalPointer() is None:
            return False
        if nodeidx.internalPointer().sidsPath() == "/CGNSTree":
            return False
        self.setLastEntered(nodeidxs)
        if nodeidx != -1:
            node = nodeidx.internalPointer()
            lknode = not node.sidsIsLink()
            lznode = node.hasLazyLoad()
            actlist = (
                ("%s goodies" % node.sidsType(),),
                None,
                ("Expand sub-tree from this node", self.expand_sb, "Ctrl++", False),
                (
                    "Collapses sub-tree from this node",
                    self.collapse_sb,
                    "Ctrl+-",
                    False,
                ),
                None,
                [
                    "Mark nodes...",
                    [
                        ("Mark/unmark node", self.marknode, "Space", False),
                        None,
                        (
                            "Mark all nodes same SIDS type",
                            self.marknode_t,
                            "Ctrl+1",
                            False,
                        ),
                        ("Mark all nodes same name", self.marknode_n, "Ctrl+2", False),
                        ("Mark all nodes same value", self.marknode_v, "Ctrl+3", False),
                        None,
                        ("Mark parent path", self.marknode_p, "Ctrl+4", False),
                    ],
                ],
                ("Add new child node", self.newnodechild, "Ctrl+A", False),
                ("Add new brother node", self.newnodebrother, "Ctrl+Z", False),
                #            None,
                #            ("Open form",self.popform,'Ctrl+F',False),
                #            ("Open view",self.openSubTree,'Ctrl+W',False),
                #            ("Open view on linked-to file",self.openLkTree,'Ctrl+O',lknode),
                None,
                ("Load node data in memory", self.dataLoad, "Ctrl+L", not lznode),
                ("Release memory node data", self.dataRelease, "Ctrl+R", lznode),
                None,
                ("Copy current", self.mcopy, "Ctrl+C", False),
                ("Cut current", self.mcut, "Ctrl+X", False),
                ("Paste as brother", self.mpasteasbrother, "Ctrl+V", False),
                ("Paste as child", self.mpasteaschild, "Ctrl+Y", False),
                ("Delete current", self.mdel, QKeySequence(Qt.Key_Delete), False),
                None,
                [
                    "On selected nodes...",
                    [
                        (
                            "Expand sub-tree from all selected nodes",
                            self.sexpand_sb,
                            "Ctrl+Shift++",
                            False,
                        ),
                        (
                            "Collapses sub-tree from all selected nodes",
                            self.scollapse_sb,
                            "Ctrl+Shift+-",
                            False,
                        ),
                        None,
                        ("Cut all selected", self.mcutselected, "Ctrl+Shift+X", False),
                        (
                            "Paste as brother for each selected",
                            self.mpasteasbrotherselected,
                            "Ctrl+Shift+V",
                            False,
                        ),
                        (
                            "Paste as child for each selected",
                            self.mpasteaschildselected,
                            "Ctrl+Shift+Y",
                            False,
                        ),
                        None,
                        (
                            "Load nodes data in memory for each selected",
                            self.dataLoadSelected,
                            "Ctrl+Shift+L",
                            False,
                        ),
                        (
                            "Release memory node data for each selected",
                            self.dataReleaseSelected,
                            "Ctrl+Shift+R",
                            False,
                        ),
                    ],
                ],
            )
            self.popupmenu.clear()
            self.popupmenu.setTitle("Node menu")
            for aparam in actlist:
                if aparam is None:
                    self.popupmenu.addSeparator()
                elif len(aparam) == 1:
                    stp = node.sidsType()
                    tag = "_GM_{}".format(stp)
                    subm = self.popupmenu.addMenu("{}...".format(stp))
                    a = QAction(
                        "About %s" % node.sidsType(), self, triggered=self.aboutSIDS
                    )
                    subm.addAction(a)
                    patmenu = subm.addMenu("Insert pattern")
                    self.patternMenu(patmenu, node.sidsNode())
                    subm.addSeparator()
                    if hasattr(self, tag):
                        getattr(self, tag)(subm, node)
                else:
                    if isinstance(aparam, list):
                        subm = self.popupmenu.addMenu(aparam[0])
                        for aaparam in aparam[1]:
                            if aaparam is None:
                                subm.addSeparator()
                            else:
                                a = QAction(aaparam[0], self, triggered=aaparam[1])
                                if aaparam[2] is not None:
                                    a.setShortcut(aaparam[2])
                                subm.addAction(a)
                                a.setDisabled(aaparam[3])
                    else:
                        a = QAction(aparam[0], self, triggered=aparam[1])
                        if aparam[2] is not None:
                            a.setShortcut(aparam[2])
                        self.popupmenu.addAction(a)
                        a.setDisabled(aparam[3])
            return True

    def _runAndSelect(self, qname, value):
        q = Q7Query.getQuery(qname)
        sl = q.run(self.FG.tree, self.FG.links, list(self.FG.lazy), False, value)
        self.model().markExtendToList(sl)
        self.model().updateSelected()
        self.treeview.refreshView()

    def patternMenu(self, menu, node):
        a = QAction("Recursive sub-pattern add", self, checkable=True)
        menu.addAction(a)
        a.setChecked(self._recursiveAddNewNode)
        menu.addSeparator()
        for t in [n[0] for n in CGU.getAuthChildren(node)]:

            def genCopyPattern(arg):
                def copyPattern():
                    pattern = CGS.profile.get(arg, None)
                    if pattern is None:
                        str_msg = (
                            "Pattern for {} was not found in current profile".format(
                                arg
                            )
                        )
                        MSG.wError(
                            self, 404, "Cannot create recursive sub-pattern", str_msg
                        )
                        return
                    self.model().copyNodeRaw(pattern[0])
                    if self.getLastEntered() is not None:
                        self.model().pasteAsChild(self.getLastEntered())

                return copyPattern

            a = QAction("{}".format(t), self, triggered=genCopyPattern(t))
            menu.addAction(a)

    def _gm_family_1(self, node):
        self._runAndSelect("013. FamilyName reference", "'%s'" % node.sidsName())

    def _gm_family_2(self, node):
        self._runAndSelect("003. Node type", "'Family_t'")

    def _GM_Family_t(self, m, node):
        a = QAction("Select references to myself", self)
        a.triggered.connect(functools.partial(self._gm_family_1, node))
        m.addAction(a)
        a = QAction("Select all families", self)
        a.triggered.connect(functools.partial(self._gm_family_2, node))
        m.addAction(a)
        m.addSeparator()
        return True

    def _GM_IndexRange_t(self, m, node):
        if node.sidsName() != CGK.ElementRange_s:
            v = 0
            a = QAction("Range size: %d" % (v), self)
            m.addAction(a)
        else:
            v = node.sidsValue()[1] - node.sidsValue()[0]
            etp = CGU.getEnumAsString(node.sidsParent())
            a = QAction("Number of elements of type [%s]: %d" % (etp, v), self)
            m.addAction(a)
        return True

    def _GM_Elements_t(self, m, node):
        etp = CGU.getEnumAsString(node.sidsNode())
        npe = CGK.ElementTypeNPE[etp]
        a = QAction("Element type [%s] npe [%d]" % (etp, npe), self)
        m.addAction(a)
        return True

    def marknode_t(self):
        node = self.getLastEntered()
        self._runAndSelect("003. Node type", "'%s'" % node.sidsType())

    def marknode_n(self):
        node = self.getLastEntered()
        self._runAndSelect("001. Node name", "'%s'" % node.sidsName())

    def marknode_v(self):
        node = self.getLastEntered()
        value = node.sidsValue()
        self._runAndSelect("005. Node value", value)

    def marknode_p(self):
        node = self.getLastEntered()
        node.switchMarked()
        path = node.sidsPath()
        while path is not None:
            path = CGU.getPathAncestor(path)
            if path not in ["/", None]:
                node = self.model().nodeFromPath("/CGNSTree" + path)
                node.switchMarked()
        self.model().updateSelected()

    def setLastEntered(self, nix=None):
        if (nix is None) or (not nix.isValid()):
            nix = self.treeview.modelCurrentIndex()
        self._lastEntered = None
        if nix.isValid():
            self.treeview.exclusiveSelectRow(nix, False)
            self._lastEntered = self.modelData(nix)

    def getLastEntered(self):
        return self._lastEntered

    def clearLastEntered(self):
        self._lastEntered = None
        self.treeview.selectionModel().clearSelection()
        return None

    def clearOtherSelections(self):
        if self._control._patternwindow is not None:
            self._control._patternwindow.clearSelection()

    def clickedPressedNode(self, index):
        self.clickedNode(index)

    def clickedNode(self, index):
        self.treeview.exclusiveSelectRow(index)
        if self.treeview.lastButton == Qt.RightButton:
            if self.updateMenu(index):
                self.popupmenu.popup(self.treeview.lastPos)

    def expandNode(self, *args):
        self.resizeAll()

    def collapseNode(self, *args):
        pass

    def expand_sb(self):
        self.treeview.expand_sb()

    def collapse_sb(self):
        self.treeview.collapse_sb()

    def sexpand_sb(self):
        self.treeview.sexpand_sb()

    def scollapse_sb(self):
        self.treeview.scollapse_sb()

    def resizeAll(self):
        for n in range(NMT.COLUMN_LAST + 1):
            self.treeview.resizeColumnToContents(n)

    def show(self):
        super(Q7Tree, self).show()

    def linkselectsrc(self):
        if self.bSelectLinkSrc.isChecked():
            if self.getLastEntered() is None:
                return
            self.bAddLink.setDisabled(False)
            node = self.getLastEntered()
            self.selectForLinkSrc = (node, node.sidsPath())
        else:
            self.bAddLink.setDisabled(True)
            self.selectForLinkSrc = None

    def linkselectdst(self):
        if self.getLastEntered() is None:
            return
        node = self.getLastEntered()
        if node is None:
            return
        if node.sidsIsLink():
            return
        if node.sidsType() == CGK.CGNSTree_ts:
            return
        if self._control.selectForLinkDst is not None:
            bt = self._control.selectForLinkDst[-1].bSelectLinkDst
            bt.setChecked(Qt.Unchecked)
            if self._control.selectForLinkDst[-1] == self:
                self._control.selectForLinkDst = None
                return
        self._control.selectForLinkDst = (
            node,
            node.sidsPath(),
            self.FG.filedir,
            self.FG.filename,
            self,
        )
        self.bSelectLinkDst.setChecked(Qt.Checked)
        if self._linkwindow is not None:
            n = node.sidsPath()
            d = self.FG.filedir
            f = self.FG.filename
            self._linkwindow.updateSelected(d, f, n)

    def linkadd(self):
        if self._control.selectForLinkDst is None:
            return
        dst = self._control.selectForLinkDst
        str_dst = "%s:%s" % (dst[3], dst[1])
        tpath = "relative"
        newname = CGU.getPathLeaf(dst[1])
        if CGU.checkDuplicatedName(
            self.selectForLinkSrc[0].sidsNode(), newname, dienow=False
        ):
            str_cnm = "New child node name is <b>%s</b>" % newname
        else:
            count = 0
            while not CGU.checkDuplicatedName(
                self.selectForLinkSrc[0].sidsNode(), newname, dienow=False
            ):
                count += 1
                newname = "{%s#%.3d}" % (dst[0].sidsType(), count)
            str_cnm = (
                """As a child with this name already exists, the name <b>%s</b> is used (generated name)"""
                % newname
            )
        str_src = "%s:%s/%s" % (self.FG.filename, self.selectForLinkSrc[1], newname)
        str_msg = (
            "you want to create a link from <b>%s</b> to <b>%s</b><br>%s<br>Your current user options do force "
            "the link to use <b>%s</b> destination file path."
            "" % (str_src, str_dst, str_cnm, tpath)
        )
        reply = MSG.wQuestion(self, 231, "Create link as a new node", str_msg)

    def linklist(self):
        if self._linkwindow is None:
            self._linkwindow = Q7LinkList(self._control, self.FG.index, self)
            self._linkwindow.show()
        else:
            self._linkwindow.raise_()

    def patternlist(self):
        if self._control._patternwindow is None:
            self._control._patternwindow = Q7PatternList(self._control, self.FG)
            self._control._patternwindow.show()
        self._control._patternwindow.raise_()

    def check(self):
        self.busyCursor()
        if self.diagview is not None:
            self.diagview.close()
            self.diagview = None
        self.lastdiag = self.model().checkSelected()
        self.readyCursor()
        self.treeview.refreshView()
        self.bCheckList.setDisabled(False)

    def checklist(self):
        if self.lastdiag is None:
            return
        self.diagview = Q7CheckList(self, self.lastdiag, self.FG.index)
        self.diagview.show()

    def clearchecks(self):
        self.model().checkClear()
        self.treeview.refreshView()
        self.lastdiag = None
        self.bCheckList.setDisabled(True)

    def selectionlist(self):
        if self._selectwindow is not None:
            self._selectwindow.close()
            self._selectwindow = None
        self._selectwindow = Q7SelectionList(self, self.model(), self.FG.index)
        self._selectwindow.show()
        self._selectwindow.raise_()

    def previousmark(self):
        self.treeview.changeSelectedMark(-1)

    def nextmark(self):
        self.treeview.changeSelectedMark(+1)

    def markall(self):
        self.model().markAll()
        self.model().updateSelected()
        self.treeview.refreshView()

    def unmarkall(self):
        self.model().unmarkAll()
        self.model().updateSelected()
        self.treeview.refreshView()

    def swapmarks(self):
        self.model().swapMarks()
        self.model().updateSelected()
        self.treeview.refreshView()

    def formview(self):
        ix = self.treeview.modelCurrentIndex()
        node = self.modelData(ix)
        if node is None:
            MSG.wInfo(
                self,
                254,
                "Form view:",
                """You have to select a node to open its form view""",
                again=False,
            )
            return
        if node.sidsType() == CGK.CGNSTree_ts:
            return
        form = Q7Form(self._control, node, self.FG.index)
        form.show()

    def vtkview(self):
        if not HAS_VTK:
            return
        from .wvtk import Q7VTK

        if self._vtkwindow is None:
            self.busyCursor()
            ix = self.treeview.modelCurrentIndex()
            zlist = self.model().getSelectedZones()
            node = self.modelData(ix)
            self._vtkwindow = Q7VTK(
                self._control, self, node, self.FG.index, self.model(), zlist
            )
            if self._vtkwindow._vtkstatus:
                self._vtkwindow.show()
            else:
                self._vtkwindow.close()
                self._vtkwindow = None
            self.readyCursor()
        else:
            self._vtkwindow.raise_()

    def plotview(self):
        return

    def queryview(self):
        if self._querywindow is None:
            self._querywindow = Q7Query(self._control, self.FG.index, self)
            self._querywindow.show()
        else:
            self._querywindow.raise_()

    def aboutSIDS(self):
        path = self.getLastEntered().sidsPath()

    def dataLoadSelected(self):
        self.model().dataLoadSelected()

    def dataReleaseSelected(self):
        self.model().dataReleaseSelected()

    def dataLoad(self):
        node = self.getLastEntered()
        self.model().dataLoadSelected(single=node.sidsPath())

    def dataRelease(self):
        node = self.getLastEntered()
        self.model().dataReleaseSelected(single=node.sidsPath())

    def forceapply(self):
        pass

    def updateTreeStatus(self):
        if (Q7FingerPrint.STATUS_MODIFIED in self.FG._status) and (
            Q7FingerPrint.STATUS_SAVEABLE in self.FG._status
        ):
            self.bSave.setEnabled(True)
        else:
            self.bSave.setEnabled(False)

    def doRelease(self):
        # break cyclic refs to allow garbage
        self.treeview.itemDelegate().doRelease()
        self.treeview.setItemDelegate(None)
        self.treeview.doRelease()
        self.treeview = None


# -----------------------------------------------------------------
