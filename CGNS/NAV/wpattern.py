#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
from CGNS.NAV.moption import Q7OptionContext as OCTXT

import sys
import numpy
import os
import importlib
import glob

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.SIDS

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QTableWidgetItem, QStyledItemDelegate
from qtpy.QtGui import QFont

from CGNS.NAV.Q7PatternWindow import Ui_Q7PatternWindow
from CGNS.NAV.wfingerprint import Q7Window as QW

# -----------------------------------------------------------------
class Q7PatternList(QW, Ui_Q7PatternWindow):
    def __init__(self, control, fgprint):
        QW.__init__(self, QW.VIEW_PATTERN, control, None, None)
        self.bClose.clicked.connect(self.reject)
        self.bSave.clicked.connect(self.patternsave)
        self.bInfo.clicked.connect(self.infoPatternView)
        self.bCopy.clicked.connect(self.copyPatternInBuffer)
        self.bDelete.clicked.connect(self.deletelocal)
        self.bSave.setDisabled(True)
        self.bDelete.setDisabled(True)
        self.bAdd.setDisabled(True)
        self._profiles = {}
        self._profiles["SIDS"] = CGNS.PAT.SIDS.profile
        self.loadUserProfiles()
        self._modified = False
        self._initialized = False
        self._selected = None
        self._master = control
        self.patternTable._panel = self

    def infoPatternView(self):
        self._control.helpWindow("Pattern")

    def clearSelection(self):
        if self._selected is not None:
            rold = self.findRow(*self._selected)
            if rold != -1:
                it1 = QTableWidgetItem(self.IC(QW.I_EMPTY), "")
                self.patternTable.setItem(rold, 0, it1)

    def findRow(self, pit, nit):
        maxrow = self.patternTable.rowCount()
        for row in range(maxrow):
            rnit = self.patternTable.item(row, 1).text()
            rpit = self.patternTable.item(row, 2).text()
            if (rpit, rnit) == (pit, nit):
                return row
        return -1

    def copyPatternInBuffer(self):
        ix = self.patternTable.currentIndex()
        if not ix.isValid():
            return
        row = ix.row()
        nit = self.patternTable.item(ix.row(), 1).text()
        pit = self.patternTable.item(ix.row(), 2).text()
        if (pit, nit) == self._selected:
            self.clearSelection()
            self._selected = None
            return
        self.clearSelection()
        self._selected = (pit, nit)
        self._control.copyPasteBuffer = self._profiles[pit][nit][0]
        it1 = QTableWidgetItem(self.IC(QW.I_MARK), "")
        self.patternTable.setItem(row, 0, it1)

    def show(self):
        if not self._initialized:
            self.reset()
        super(Q7PatternList, self).show()
        self.raise_()

    def patternsave(self):
        filename = QFileDialog.getSaveFileName(self, "Save pattern", ".", "*.py")
        if filename[0] == "":
            return
        # f=open(str(filename[0]),'w+')
        # f.write(n)
        # f.close()

    def deletelocal(self):
        pass

    def reset(self):
        tlvcols = 4
        tlvcolsnames = ["S", "Pattern", "P", "Comment"]
        v = self.patternTable
        v.setColumnCount(tlvcols)
        lh = v.horizontalHeader()
        lv = v.verticalHeader()
        h = tlvcolsnames
        n = len(h)
        for i in range(n):
            hi = QTableWidgetItem(h[i])
            v.setHorizontalHeaderItem(i, hi)
        for profkey in self._profiles:
            prof = self._profiles[profkey]
            for k in prof:
                pentry = prof[k]
                v.setRowCount(v.rowCount() + 1)
                r = v.rowCount() - 1
                it1 = QTableWidgetItem(self.IC(QW.I_EMPTY), "")
                it2 = QTableWidgetItem(k)
                it2.setFont(QFont("Courier"))
                it3 = QTableWidgetItem(profkey)
                it4 = QTableWidgetItem(pentry[2])
                v.setItem(r, 0, it1)
                v.setItem(r, 1, it2)
                v.setItem(r, 2, it3)
                v.setItem(r, 3, it4)
        self.patternTable.resizeColumnsToContents()
        self.patternTable.resizeRowsToContents()
        plist = []
        for i in range(len(plist)):
            v.resizeColumnToContents(i)
        for i in range(v.rowCount()):
            v.resizeRowToContents(i)
        self._initialized = True

    def reject(self):
        if self._master._patternwindow is not None:
            self._master._patternwindow = None
        self.close()

    def loadUserProfiles(self):
        pthlist = OCTXT.ProfileSearchPathList
        if not pthlist:
            return
        pthlistok = []
        for pth in pthlist:
            if os.path.isdir(pth):
                pthlistok += [pth]
        if not pthlistok:
            return
        for pth in pthlistok:
            sys.path.append(pth)
            modlist = [
                os.path.splitext(os.path.basename(mod))[0]
                for mod in glob.glob("%s/*" % pth)
            ]
            for m in modlist:
                pym = importlib.import_module(m)
                self._profiles[m] = pym.profile


# -----------------------------------------------------------------
class Q7PatternTableItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        t = index.data(Qt.DisplayRole)
        r = option.rect.adjusted(2, 2, -2, -2)
        painter.drawText(r, Qt.AlignVCenter | Qt.AlignLeft, t, r)


# -----------------------------------------------------------------
