#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
from .moption import Q7OptionContext as OCTXT

from ..PAT import cgnsutils as CGU

from qtpy.QtCore import Qt, QModelIndex, QPoint
from qtpy.QtWidgets import QStyledItemDelegate, QMenu
from qtpy.QtGui import QColor, QFont, QIcon, QPixmap

from .Q7MergeWindow import Ui_Q7MergeWindow
from .wfingerprint import Q7Window

from . import mtree as NMT

(DIFF_NX, DIFF_NA, DIFF_ND, DIFF_CQ, DIFF_CT, DIFF_CS, DIFF_CV) = range(7)
(MERGE_NX, MERGE_NA, MERGE_NB) = range(3)


# -----------------------------------------------------------------
class Q7MergeItemDelegate(QStyledItemDelegate):
    def __init__(self, owner, model, diag, merge):
        QStyledItemDelegate.__init__(self, owner)
        self._parent = owner
        self._model = model
        self._diag = diag
        self._merge = merge
        self.orange = QColor("#FFA500")
        self.red = QColor("#FF0000")
        self.green = QColor("#008000")
        self.gray = QColor("#C0C0C0")

    def paint(self, painter, option, index):
        idx = self._parent.modelIndex(index)
        col = idx.column()
        nnm = self._parent.modelData(index).sidsName()
        pth = CGU.getPathNoRoot(self._parent.modelData(index).sidsPath())
        if (self._merge is not None) and (pth in self._merge):
            if self._merge[pth] == MERGE_NA:
                self._parent.modelData(index).setUserStatePrivate("A")
            if self._merge[pth] == MERGE_NB:
                self._parent.modelData(index).setUserStatePrivate("B")
        if col == NMT.COLUMN_NAME:
            if nnm not in OCTXT._ReservedNames:
                option.font.setWeight(QFont.Bold)
            QStyledItemDelegate.paint(self, painter, option, index)
            option.font.setWeight(QFont.Light)
            if (self._diag is not None) and (pth in self._diag):
                color = self.gray
                if self._diag[pth] == DIFF_NX:
                    color = self.gray
                if self._diag[pth] == DIFF_NA:
                    color = self.green
                if self._diag[pth] == DIFF_ND:
                    color = self.red
                painter.save()
                painter.setBrush(Qt.NoBrush)
                painter.setPen(color)
                painter.drawRect(option.rect)
                painter.restore()
        elif col == NMT.COLUMN_VALUE:
            option.font.setFamily(OCTXT.Table_Family)
            QStyledItemDelegate.paint(self, painter, option, index)
            if (self._diag is not None) and (pth in self._diag):
                if self._diag[pth] == DIFF_CV:
                    painter.save()
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(self.orange)
                    painter.drawRect(option.rect)
                    painter.restore()
        elif col == NMT.COLUMN_DATATYPE:
            option.font.setFamily(OCTXT.Table_Family)
            option.font.setPointSize(8)
            QStyledItemDelegate.paint(self, painter, option, index)
            if (self._diag is not None) and (pth in self._diag):
                if self._diag[pth] == DIFF_CQ:
                    painter.save()
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(self.orange)
                    painter.drawRect(option.rect)
                    painter.restore()
        elif col == NMT.COLUMN_SHAPE:
            QStyledItemDelegate.paint(self, painter, option, index)
            if (self._diag is not None) and (pth in self._diag):
                if self._diag[pth] == DIFF_CS:
                    painter.save()
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(self.orange)
                    painter.drawRect(option.rect)
                    painter.restore()
        elif col == NMT.COLUMN_SIDS:
            QStyledItemDelegate.paint(self, painter, option, index)
            if (self._diag is not None) and (pth in self._diag):
                if self._diag[pth] == DIFF_CT:
                    painter.save()
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(self.orange)
                    painter.drawRect(option.rect)
                    painter.restore()
        else:
            QStyledItemDelegate.paint(self, painter, option, index)


# -----------------------------------------------------------------
class Q7Merge(Q7Window, Ui_Q7MergeWindow):
    def __init__(self, control, fgprint, diag):
        Q7Window.__init__(self, Q7Window.VIEW_DIFF, control, None, fgprint)
        self._depthExpanded = 0
        self._lastEntered = None
        self._fgprint = fgprint
        (ldiag, lmerge) = self.diagAnalysis(diag)
        self.treeview.expanded[QModelIndex].connect(self.expandNode)
        self.treeview.collapsed.connect(self.collapseNode)
        self.treeview.clicked[QModelIndex].connect(self.clickedNode)
        self.treeview.customContextMenuRequested[QPoint].connect(self.clickedNode)
        # QObject.connect(self.treeview,
        #                 SIGNAL("expanded(QModelIndex)"),
        #                 self.expandNode)
        # QObject.connect(self.treeview,
        #                 SIGNAL("collapsed()"),
        #                 self.collapseNode)
        # QObject.connect(self.treeview,
        #                 SIGNAL("clicked(QModelIndex)"),
        #                 self.clickedNode)
        # QObject.connect(self.treeview,
        #                 SIGNAL("customContextMenuRequested(QPoint)"),
        #                 self.clickedNode)
        self.bClose.clicked.connect(self.leave)
        self.bInfo.clicked.connect(self.infoTreeView)
        self.bZoomIn.clicked.connect(self.expandLevel)
        self.bZoomOut.clicked.connect(self.collapseLevel)
        self.bZoomAll.clicked.connect(self.expandMinMax)
        self.bSaveDiff.clicked.connect(self.savediff)
        self.bSelectA.clicked.connect(self.showSelected)
        self.bSelectB.clicked.connect(self.showSelected)
        self.bSelectOrderSwap.clicked.connect(self.swapSelected)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.popupmenu = QMenu()
        self.proxyA = self._fgprint.model
        self.treeview.setModel(self.proxyA)
        self.treeview.setItemDelegate(
            Q7MergeItemDelegate(self.treeview, self._fgprint.model, ldiag, lmerge)
        )
        self.treeview.setControlWindow(self, self._fgprint.model)
        self.treeview.hideColumn(NMT.COLUMN_FLAG_LINK)
        self.treeview.hideColumn(NMT.COLUMN_FLAG_CHECK)
        self.treeview.hideColumn(NMT.COLUMN_FLAG_SELECT)
        self._fgprint.model.addA = True
        self._fgprint.model.addB = True
        self._A = QIcon(QPixmap(":/images/icons/user-A.png"))
        self._B = QIcon(QPixmap(":/images/icons/user-B.png"))
        self._order = 0  # A first

    def swapSelected(self):
        if not self._order:
            self.bSelectA.setIcon(self._B)
            self.bSelectB.setIcon(self._A)
            self._order = 1
        else:
            self.bSelectA.setIcon(self._A)
            self.bSelectB.setIcon(self._B)
            self._order = 0

    def showSelected(self):
        self._fgprint.model.addA = False
        self._fgprint.model.addB = False
        if self.bSelectA.isChecked():
            self._fgprint.model.addA = True
        if self.bSelectB.isChecked():
            self._fgprint.model.addB = True

    def diagAnalysis(self, diag):
        ldiag = {}
        lmerge = {}
        for k in diag:
            ldiag[k] = DIFF_NX
            for d in diag[k]:
                if d[0] == "NA":
                    ldiag[d[1]] = DIFF_NA
                    lmerge[d[1]] = MERGE_NB
                if d[0] == "ND":
                    ldiag[d[1]] = DIFF_ND
                    lmerge[d[1]] = MERGE_NA
                if d[0] in ["CT"]:
                    ldiag[k] = DIFF_CT
                    lmerge[k] = MERGE_NA
                if d[0] in ["C3", "C1", "C2"]:
                    ldiag[k] = DIFF_CQ
                    lmerge[k] = MERGE_NA
                if d[0] in ["C6", "C7"]:
                    ldiag[k] = DIFF_CV
                    lmerge[k] = MERGE_NA
                if d[0] in ["C4", "C5"]:
                    ldiag[k] = DIFF_CS
                    lmerge[k] = MERGE_NA
        return (ldiag, lmerge)

    def model(self):
        return self._fgprint.model

    def modelIndex(self, idx):
        if not idx.isValid():
            return -1
        midx = idx
        # if (idx.model() != self.treeview.M()):
        #    midx=self.treeview.model().mapToSource(idx)
        return midx

    def modelData(self, idx):
        if not idx.isValid():
            return None
        return self.modelIndex(idx).internalPointer()

    def infoTreeView(self):
        self._control.helpWindow("Tree")

    def savediff(self):
        pass

    def expandMinMax(self):
        if self._depthExpanded == self._fgprint.depth - 2:
            self._depthExpanded = -1
            self.treeview.collapseAll()
        else:
            self._depthExpanded = self._fgprint.depth - 2
            self.treeview.expandAll()
        self.resizeAll()

    def expandLevel(self):
        if self._depthExpanded < self._fgprint.depth - 2:
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
        return
        self.lineEdit.clear()
        self.lineEdit.insert(node.sidsPath())

    def updateMenu(self, nodeidxs):
        return
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
                ("About %s" % node.sidsType(), self.aboutSIDS, None, False),
                None,
                ("Mark/unmark node", self.marknode, "Space", False),
                ("Add new child node", self.newnodechild, "Ctrl+A", False),
                ("Add new brother node", self.newnodebrother, "Ctrl+Z", False),
                None,
                ("Open form", self.popform, "Ctrl+F", False),
                ("Open view", self.openSubTree, "Ctrl+W", False),
                ("Open view on linked-to file", self.openLkTree, "Ctrl+O", lknode),
                None,
                ("Load node data in memory", self.dataLoad, "Ctrl+L", not lznode),
                ("Release memory node data", self.dataRelease, "Ctrl+R", lznode),
                None,
                ("Copy", self.mcopy, "Ctrl+C", False),
                ("Cut", self.mcut, "Ctrl+X", False),
                ("Paste as brother", self.mpasteasbrother, "Ctrl+V", False),
                ("Paste as child", self.mpasteaschild, "Ctrl+Y", False),
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
            )
            self.popupmenu.clear()
            self.popupmenu.setTitle("Node menu")
            for aparam in actlist:
                if aparam is None:
                    self.popupmenu.addSeparator()
                else:
                    a = QAction(aparam[0], self)
                    a.triggered.connect(aparam[1])
                    if aparam[2] is not None:
                        a.setShortcut(aparam[2])
                    self.popupmenu.addAction(a)
                    a.setDisabled(aparam[3])
            return True

    def setLastEntered(self, nix=None):
        self._lastEntered = None

    def getLastEntered(self):
        return self._lastEntered

    def clearLastEntered(self):
        self._lastEntered = None
        self.treeview.selectionModel().clearSelection()
        return None

    def clickedNode(self, index):
        pass

    def expandNode(self, *args):
        self.resizeAll()

    def collapseNode(self, *args):
        pass

    def resizeAll(self):
        for n in range(NMT.COLUMN_LAST + 1):
            self.treeview.resizeColumnToContents(n)

    def show(self):
        super(Q7Merge, self).show()

    def closeAlone(self):
        pass

    def leave(self):
        self.close()


# -----------------------------------------------------------------
