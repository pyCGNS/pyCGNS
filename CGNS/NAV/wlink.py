#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
from .moption import Q7OptionContext as OCTXT

import sys
import string
import os.path

from .. import MAP as CGM
from ..PAT import cgnsutils as CGU
from ..PAT import cgnskeywords as CGK

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QTableWidgetItem, QStyledItemDelegate, QLineEdit, QHeaderView

from .Q7LinkWindow import Ui_Q7LinkWindow
from .wfingerprint import Q7Window as QW

from . import wmessages as MSG


# -----------------------------------------------------------------
class Q7LinkItemDelegate(QStyledItemDelegate):
    def __init__(self, owner, model):
        self._table = owner.linkTable
        QStyledItemDelegate.__init__(self, self._table)
        self._parent = owner
        self._model = model
        self._filename = ""
        self._dirname = ""

    def createEditor(self, parent, option, index):
        ws = option.rect.width()
        hs = option.rect.height() + 4
        xs = option.rect.x()
        ys = option.rect.y() - 2
        if index.column() in [2]:
            filt = "CGNS Files (*.hdf *.cgns)"
            filename = str(
                QFileDialog.getOpenFileName(self._table, "Select file", filter=filt)[0]
            )
            (dname, fname) = os.path.split(filename)
            if not dname or not fname:
                return None
            itf = QTableWidgetItem(fname)
            itd = QTableWidgetItem(dname)
            self._table.setItem(index.row(), 2, itf)
            self._table.setItem(index.row(), 4, itd)
            return None
        if index.column() in [1, 3]:
            editor = QLineEdit(parent)
            editor.transgeometry = (xs, ys, ws, hs)
            editor.installEventFilter(self)
            self.setEditorData(editor, index)
            return editor
        return None

    def setEditorData(self, editor, index):
        value = index.data()
        editor.clear()
        editor.insert(value)

    def setModelData(self, editor, model, index):
        col = index.column()
        row = index.row()
        v = str(editor.text())
        if col in [1, 3]:
            v = CGU.getPathNoRoot(v)
        self._parent._links[row][self._parent._col2lk[col]] = v
        self._parent.reset()

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(*editor.transgeometry)

    def paint(self, painter, option, index):
        QStyledItemDelegate.paint(self, painter, option, index)


# -----------------------------------------------------------------
class Q7LinkList(QW, Ui_Q7LinkWindow):
    def __init__(self, parent, fgindex, master):
        QW.__init__(self, QW.VIEW_LINK, parent, None, fgindex)
        self.bClose.clicked.connect(self.reject)
        self._lk2col = [4, 2, 3, 1, 0]
        self._col2lk = [4, 3, 1, 2, 0]
        self._master = master
        self._links = self.FG.links
        self.linkTable.setItemDelegate(Q7LinkItemDelegate(self, self.FG.model))
        self.setLabel(self.eDirSource, self.FG.filedir)
        self.setLabel(self.eFileSource, self.FG.filename)
        self.bInfo.clicked.connect(self.infoLinkView)
        self.bAddLink.clicked.connect(self.newLink)
        self.bDeleteLink.clicked.connect(self.removeLink)
        self.bDuplicateLink.clicked.connect(self.duplicateLink)
        self.bSave.clicked.connect(self.infoLinkView)
        self.bCheckLink.clicked.connect(self.checkLinks)
        self.bLoadTree.clicked.connect(self.loadLinkFile)

    def doRelease(self):
        self.reject()

    def loadLinkFile(self):
        i = self.linkTable.currentItem()
        if i is None:
            return
        r = i.row()
        d = self.linkTable.item(r, 4).text()
        f = self.linkTable.item(r, 2).text()
        filename = "%s/%s" % (d, f)
        self.busyCursor()
        if filename is not None:
            self._control.loadfile(filename)
        self.readyCursor()

    def infoLinkView(self):
        self._control.helpWindow("Link")

    def duplicateLink(self):
        if self._links:
            i = self.linkTable.currentItem()
            if i is None:
                return
            r = i.row()
            self._links.append(self._links[r])
            self.reset()
        return

    def removeLink(self):
        if not self._links:
            return
        i = self.linkTable.currentItem()
        if i is None:
            return
        reply = MSG.wQuestion(
            self,
            311,
            "Remove link entry",
            """Do you want to remove the selected link entry?
                                 Existing sub-tree would be <b>merged</b> in the
                                 top file during the next save.""",
        )
        if reply:
            r = i.row()
            self._links.pop(r)
            self.reset()

    def checkLinks(self):
        pass

    def newLink(self):
        self._links.append(["", "", "", "", CGM.S2P_LKIGNORED])
        self.reset()

    def show(self):
        self.reset()
        super(Q7LinkList, self).show()

    def statusIcon(self, status):
        if status == CGM.S2P_LKOK:
            it = QTableWidgetItem(self.IC(QW.I_L_OKL), "")
            it.setToolTip("Link ok")
            return it
        if status & CGM.S2P_LKNOFILE:
            it = QTableWidgetItem(self.IC(QW.I_L_NFL), "")
            it.setToolTip("File not found in search path")
            return it
        if status & CGM.S2P_LKIGNORED:
            it = QTableWidgetItem(self.IC(QW.I_L_IGN), "")
            it.setToolTip("Link was ignored during load")
            return it
        if status & CGM.S2P_LKFILENOREAD:
            it = QTableWidgetItem(self.IC(QW.I_L_NRL), "")
            it.setToolTip("File found, not readable")
            return it
        if status & CGM.S2P_LKNONODE:
            it = QTableWidgetItem(self.IC(QW.I_L_NNL), "")
            it.setToolTip("File ok, node path not found")
            return it
        it = QTableWidgetItem(self.IC(QW.I_L_ERL), "")
        it.setToolTip("Unknown error")
        return it

    def reset(self):
        v = self.linkTable
        v.clear()
        v.setRowCount(0)
        lh = v.horizontalHeader()
        lv = v.verticalHeader()
        h = ["S", "Source Node", "Linked-to file", "Linked-to Node", "Found in dir"]
        for i, hstr in enumerate(h):
            hi = QTableWidgetItem(hstr)
            hi.setFont(OCTXT._Label_Font)
            v.setHorizontalHeaderItem(i, hi)
            lh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        lh.setSectionResizeMode(len(h) - 1, QHeaderView.Stretch)
        v.setRowCount(len(self._links))
        r = 0
        for lk in self._links:
            (ld, lf, ln, sn, st) = lk
            t1item = self.statusIcon(st)
            t2item = QTableWidgetItem(sn)
            t2item.setFont(OCTXT._Table_Font)
            t3item = QTableWidgetItem(lf)
            t3item.setFont(OCTXT._Table_Font)
            t4item = QTableWidgetItem(ln)
            t4item.setFont(OCTXT._Table_Font)
            t5item = QTableWidgetItem(ld)
            t5item.setFont(OCTXT._Table_Font)
            v.setItem(r, 0, t1item)
            v.setItem(r, 1, t2item)
            v.setItem(r, 2, t3item)
            v.setItem(r, 3, t4item)
            v.setItem(r, 4, t5item)
            r += 1
        for i in (2, 3):
            v.resizeColumnToContents(i)
        for i in range(v.rowCount()):
            v.resizeRowToContents(i)

    def reject(self):
        if self._master._linkwindow is not None:
            self._master._linkwindow = None
        self.close()


# -----------------------------------------------------------------
