#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

from CGNS.NAV.moption import Q7OptionContext as OCTXT

import CGNS.PAT.cgnsutils as CGU

from qtpy.QtCore import Qt, QModelIndex, QPoint
from qtpy.QtWidgets import QStyledItemDelegate, QMenu
from qtpy.QtGui import QColor, QFont

from CGNS.NAV.Q7DiffWindow import Ui_Q7DiffWindow
from CGNS.NAV.wfingerprint import Q7Window, Q7FingerPrint
from CGNS.NAV.mdifftreeview import Q7DiffTreeModel

import CGNS.NAV.mtree as NMT

(DIFF_NX, DIFF_NA, DIFF_ND, DIFF_CQ, DIFF_CT, DIFF_CS, DIFF_CV) = range(7)


# -----------------------------------------------------------------
class Q7DiffItemDelegate(QStyledItemDelegate):
    def __init__(self, owner, model, diag):
        QStyledItemDelegate.__init__(self, owner)
        self._parent = owner
        self._model = model
        self._diag = diag
        self.orange = QColor("#FFA500")
        self.red = QColor("#FF0000")
        self.green = QColor("#008000")
        self.gray = QColor("#C0C0C0")

    def paint(self, painter, option, index):
        idx = self._parent.modelIndex(index)
        col = idx.column()
        nnm = self._parent.modelData(index).sidsName()
        pth = CGU.getPathNoRoot(self._parent.modelData(index).sidsPath())
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
class Q7Diff(Q7Window, Ui_Q7DiffWindow):
    def __init__(self, control, fgprintindexA, fgprintindexB, diag):
        Q7Window.__init__(self, Q7Window.VIEW_DIFF, control, None, fgprintindexA)
        self._depthExpanded = 0
        self._lastEntered = None
        self._fgidxA = fgprintindexA
        self._fgidxB = fgprintindexB
        ldiag = self.diagAnalysis(diag)
        self.treeviewA.expanded[QModelIndex].connect(self.expandNode)
        self.treeviewA.collapsed.connect(self.collapseNode)
        self.treeviewA.clicked[QModelIndex].connect(self.clickedNode)
        self.treeviewA.customContextMenuRequested[QPoint].connect(self.clickedNode)
        self.treeviewB.expanded[QModelIndex].connect(self.expandNode)
        self.treeviewB.collapsed.connect(self.collapseNode)
        self.treeviewB.clicked[QModelIndex].connect(self.clickedNode)
        self.treeviewB.customContextMenuRequested[QPoint].connect(self.clickedNode)
        # QObject.connect(self.treeviewA,
        #                 SIGNAL("expanded(QModelIndex)"),
        #                 self.expandNode)
        # QObject.connect(self.treeviewA,
        #                 SIGNAL("collapsed()"),
        #                 self.collapseNode)
        # QObject.connect(self.treeviewA,
        #                 SIGNAL("clicked(QModelIndex)"),
        #                 self.clickedNode)
        # QObject.connect(self.treeviewA,
        #                 SIGNAL("customContextMenuRequested(QPoint)"),
        #                 self.clickedNode)
        # QObject.connect(self.treeviewB,
        #                 SIGNAL("expanded(QModelIndex)"),
        #                 self.expandNode)
        # QObject.connect(self.treeviewB,
        #                 SIGNAL("collapsed()"),
        #                 self.collapseNode)
        # QObject.connect(self.treeviewB,
        #                 SIGNAL("clicked(QModelIndex)"),
        #                 self.clickedNode)
        # QObject.connect(self.treeviewB,
        #                 SIGNAL("customContextMenuRequested(QPoint)"),
        #                 self.clickedNode)
        # QObject.connect(self.bLockScroll,
        #                 SIGNAL("clicked()"),
        #                 self.syncScrolls)
        self.bLockScroll.clicked.connect(self.syncScrolls)
        self.bClose.clicked.connect(self.leave)
        self.bInfo.clicked.connect(self.infoTreeView)
        self.bZoomIn.clicked.connect(self.expandLevel)
        self.bZoomOut.clicked.connect(self.collapseLevel)
        self.bZoomAll.clicked.connect(self.expandMinMax)
        self.bSaveDiff.clicked.connect(self.savediff)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.popupmenu = QMenu()
        self.proxyA = Q7DiffTreeModel(self._fgidxA)
        self.proxyB = Q7DiffTreeModel(self._fgidxB)
        self.proxyA.setDiag(ldiag)
        self.proxyB.setDiag(ldiag)
        self.treeviewA.setModel(self.proxyA)
        self.treeviewB.setModel(self.proxyB)
        fga = Q7FingerPrint.getByIndex(self._fgidxA)
        fgb = Q7FingerPrint.getByIndex(self._fgidxB)
        model_a = fga.model
        model_b = fgb.model
        self.treeviewA.setItemDelegate(
            Q7DiffItemDelegate(self.treeviewA, model_a, ldiag)
        )
        self.treeviewB.setItemDelegate(
            Q7DiffItemDelegate(self.treeviewB, model_b, ldiag)
        )
        self.treeviewA.setControlWindow(self, model_a)
        self.treeviewB.setControlWindow(self, model_b)
        self.treeviewA.hideColumn(NMT.COLUMN_FLAG_LINK)
        self.treeviewA.hideColumn(NMT.COLUMN_FLAG_CHECK)
        self.treeviewA.hideColumn(NMT.COLUMN_FLAG_SELECT)
        self.treeviewA.hideColumn(NMT.COLUMN_FLAG_USER)
        self.treeviewB.hideColumn(NMT.COLUMN_FLAG_LINK)
        self.treeviewB.hideColumn(NMT.COLUMN_FLAG_CHECK)
        self.treeviewB.hideColumn(NMT.COLUMN_FLAG_SELECT)
        self.treeviewB.hideColumn(NMT.COLUMN_FLAG_USER)
        self.wvsa = self.treeviewA.verticalScrollBar()
        self.wvsb = self.treeviewB.verticalScrollBar()
        self.uvsa = self.verticalScrollBarA
        self.uvsb = self.verticalScrollBarB
        self.uvsa.setToolTip("%s/%s" % (fga.filedir, fga.filename))
        self.uvsb.setToolTip("%s/%s" % (fgb.filedir, fgb.filename))
        self.syncScrolls(True)

    def diagAnalysis(self, diag):
        ldiag = {}
        for k in diag:
            ldiag[k] = DIFF_NX
            for d in diag[k]:
                if d[0] == "NA":
                    ldiag[d[1]] = DIFF_NA
                if d[0] == "ND":
                    ldiag[d[1]] = DIFF_ND
                if d[0] in ["CT"]:
                    ldiag[k] = DIFF_CT
                if d[0] in ["C3", "C1", "C2"]:
                    ldiag[k] = DIFF_CQ
                if d[0] in ["C6", "C7"]:
                    ldiag[k] = DIFF_CV
                if d[0] in ["C4", "C5"]:
                    ldiag[k] = DIFF_CS
        return ldiag

    def syncScrolls(self, force=False):
        self.uvsa.valueChanged[int].connect(self.wvsa.setValue[int])
        self.uvsb.valueChanged[int].connect(self.wvsb.setValue[int])
        self.wvsa.valueChanged[int].connect(self.uvsa.setValue[int])
        self.wvsb.valueChanged[int].connect(self.uvsb.setValue[int])
        #
        # QObject.connect(self.uvsa, SIGNAL("valueChanged(int)"),
        #                 self.wvsa, SLOT("setValue(int)"))
        # QObject.connect(self.uvsb, SIGNAL("valueChanged(int)"),
        #                 self.wvsb, SLOT("setValue(int)"))
        # QObject.connect(self.wvsa, SIGNAL("valueChanged(int)"),
        #                 self.uvsa, SLOT("setValue(int)"))
        # QObject.connect(self.wvsb, SIGNAL("valueChanged(int)"),
        #                 self.uvsb, SLOT("setValue(int)"))
        if force or self.bLockScroll.isChecked():
            self.wvsa.valueChanged[int].connect(self.wvsb.setValue[int])
            self.wvsb.valueChanged[int].connect(self.wvsa.setValue[int])
            # QObject.connect(self.wvsa, SIGNAL("valueChanged(int)"),
            #                 self.wvsb, SLOT("setValue(int)"))
            # QObject.connect(self.wvsb, SIGNAL("valueChanged(int)"),
            #                 self.wvsa, SLOT("setValue(int)"))
        else:
            self.wvsa.valueChanged[int].disconnect(self.wvsb.setValue[int])
            self.wvsb.valueChanged[int].disconnect(self.wvsa.setValue[int])
            # QObject.disconnect(self.wvsa, SIGNAL("valueChanged(int)"),
            #                   self.wvsb, SLOT("setValue(int)"))
            # QObject.disconnect(self.wvsb, SIGNAL("valueChanged(int)"),
            #                   self.wvsa, SLOT("setValue(int)"))

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
        if self._depthExpanded == Q7FingerPrint.getByIndex(self._fgidxA).depth - 2:
            self._depthExpanded = -1
            self.treeviewA.collapseAll()
            self.treeviewB.collapseAll()
        else:
            self._depthExpanded = Q7FingerPrint.getByIndex(self._fgidxA).depth - 2
            self.treeviewA.expandAll()
            self.treeviewB.expandAll()
        self.resizeAll()

    def expandLevel(self):
        if self._depthExpanded < Q7FingerPrint.getByIndex(self._fgidxA).depth - 2:
            self._depthExpanded += 1
        self.treeviewA.expandToDepth(self._depthExpanded)
        self.treeviewB.expandToDepth(self._depthExpanded)
        self.resizeAll()

    def collapseLevel(self):
        if self._depthExpanded != -1:
            self._depthExpanded -= 1
        if self._depthExpanded == -1:
            self.treeviewA.collapseAll()
            self.treeviewB.collapseAll()
        else:
            self.treeviewA.expandToDepth(self._depthExpanded)
            self.treeviewB.expandToDepth(self._depthExpanded)
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
            self.treeviewA.resizeColumnToContents(n)
            self.treeviewB.resizeColumnToContents(n)

    def show(self):
        super(Q7Diff, self).show()

    def closeAlone(self):
        pass

    def leave(self):
        self.close()

    def doRelease(self):
        pass


# -----------------------------------------------------------------
