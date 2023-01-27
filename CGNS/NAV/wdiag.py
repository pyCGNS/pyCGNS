#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

import os

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QTreeView, QFileDialog, QTreeWidgetItem, QWidget


from . import wmessages as MSG
from ..VAL.parse import messages as CGM
from .Q7DiagWindow import Ui_Q7DiagWindow
from .moption import Q7OptionContext as OCTXT
from .wfingerprint import Q7Window as QW
from ..VAL.parse.findgrammar import locateGrammars


# -----------------------------------------------------------------
class Q7CheckList(QW, Ui_Q7DiagWindow):
    def __init__(self, parent, data, fgindex):
        QW.__init__(self, QW.VIEW_DIAG, parent._control, None, fgindex)
        self._currentItem = 0
        self._filterItems = {}
        self.bClose.clicked.connect(self.reject)
        self.bExpandAll.clicked.connect(self.expand)
        self.bInfo.clicked.connect(self.infoDiagView)
        self.bCollapseAll.clicked.connect(self.collapse)
        self.bPrevious.clicked.connect(self.previousfiltered)
        self.bNext.clicked.connect(self.nextfiltered)
        self.cWarnings.clicked.connect(self.reset)
        self.cDiagFirst.clicked.connect(self.reset)
        self.bSave.clicked.connect(self.diagnosticssave)
        self.bWhich.clicked.connect(self.grammars)
        self.cFilter.currentIndexChanged[int].connect(self.filterChange)
        # QObject.connect(self.cFilter,
        #                 SIGNAL("currentIndexChanged(int)"),
        #                 self.filterChange)
        self._data = data
        self._parent = parent
        self.diagTable.keyPressEvent = self.diagTableKeyPressEvent
        self.filterChange()

    def doRelease(self):
        pass

    def grammars(self):
        self.FG.pushGrammarPaths()
        gl = ""
        for g in locateGrammars():
            gl += "%s : %s<br>" % (g[0], os.path.split(g[1])[0])
        self.FG.popGrammarPaths()
        MSG.wInfo(
            self,
            400,
            "Diag view:",
            """Checks performed using the following grammars:<br>%s""" % gl,
            again=False,
        )

    def diagTableKeyPressEvent(self, event):
        kmod = event.modifiers()
        kval = event.key()
        if kval == Qt.Key_Space:
            itlist = self.diagTable.selectedItems()
            it = itlist[0]
            itxt = it.text(0)
            if itxt[0] != "/":
                itxt = it.parent().text(0)
            self._parent.treeview.selectByPath(itxt)
        else:
            QTreeView.keyPressEvent(self.diagTable, event)

    def infoDiagView(self):
        self._control.helpWindow("Diagnosis")

    def diagnosticssave(self):
        n = "data=%s\n" % self._data

        filename = QFileDialog.getSaveFileName(self, "Save diagnosis", ".", "*.py")
        if filename[0] == "":
            return
        with open(str(filename[0]), "w+") as f:
            f.write(n)

    def previousfiltered(self):
        iold = self._filterItems[self.cFilter.currentText()][self._currentItem]
        if self._currentItem != 0:
            self._currentItem -= 1
        inew = self._filterItems[self.cFilter.currentText()][self._currentItem]
        iold.setSelected(False)
        inew.setSelected(True)
        self.diagTable.scrollToItem(inew)
        self.eCount.setText(str(self._currentItem + 1))

    def nextfiltered(self):
        ilist = self._filterItems[self.cFilter.currentText()]
        if self._currentItem >= len(ilist) - 1:
            return
        iold = ilist[self._currentItem]
        self._currentItem += 1
        inew = ilist[self._currentItem]
        iold.setSelected(False)
        inew.setSelected(True)
        self.diagTable.scrollToItem(inew)
        self.eCount.setText(str(self._currentItem + 1))

    def filterChange(self):
        self._currentItem = 0
        self.eCount.setText("")
        self.diagTable.clearSelection()

    def expand(self):
        self.diagTable.expandAll()

    def collapse(self):
        self.diagTable.collapseAll()

    def show(self):
        self.reset()
        super(Q7CheckList, self).show()

    def reset(self):
        v = self.diagTable
        v.clear()
        v.setHeaderHidden(True)
        plist = list(self._data)
        plist.sort()
        plist.reverse()
        keyset = set()
        self.cFilter.clear()
        self._filterItems = {}
        diagfirst = self.cDiagFirst.isChecked()
        warnings = self.cWarnings.isChecked()
        diagstack = {}
        for path in plist:
            path_item = None
            state = self._data.getWorstDiag(path)
            if state in [CGM.CHECK_NONE, CGM.CHECK_PASS]:
                pass
            elif (state == CGM.CHECK_WARN) and not warnings:
                pass
            else:
                for (diag, pth) in self._data.diagnosticsByPath(path):
                    if (diag.level == CGM.CHECK_WARN) and not warnings:
                        pass
                    elif diagfirst:
                        if diag.key not in keyset:
                            keyset.add(diag.key)
                            diag_item = self.addDiagEntry(None, diag, top=True)
                            v.insertTopLevelItem(0, diag_item)
                            diagstack[diag.key] = diag_item
                        path_item = self.addPathEntry(
                            diagstack[diag.key], path, state, top=False
                        )
                    else:
                        if diag.key not in keyset:
                            keyset.add(diag.key)
                        if path_item is None:
                            path_item = self.addPathEntry(None, path, state, top=True)
                            v.insertTopLevelItem(0, path_item)
                        diag_item = self.addDiagEntry(path_item, diag, top=False)
                    keylist = list(keyset)
        keylist.sort()
        for k in keylist:
            self.cFilter.addItem(k)

    def addPathEntry(self, parent, path, state, top=False):
        it = QTreeWidgetItem(parent, (path,))
        ft = QFont(OCTXT._Table_Font)
        if top:
            ft.setBold(True)
        it.setFont(0, ft)
        if state == CGM.CHECK_FAIL:
            it.setIcon(0, self.IC(QW.I_C_SFL))
        if state == CGM.CHECK_WARN:
            it.setIcon(0, self.IC(QW.I_C_SWR))
        return it

    def addDiagEntry(self, parent, diag, top=False):
        dit = QTreeWidgetItem(parent, (self._data.message(diag),))
        ft = QFont(OCTXT._Table_Font)
        if top:
            ft.setBold(True)
        dit.setFont(0, ft)
        if diag.level == CGM.CHECK_FAIL:
            dit.setIcon(0, self.IC(QW.I_C_SFL))
        if diag.level == CGM.CHECK_WARN:
            dit.setIcon(0, self.IC(QW.I_C_SWR))
        if diag.key not in self._filterItems:
            self._filterItems[diag.key] = [dit]
        else:
            self._filterItems[diag.key].insert(0, dit)
        return dit

    def reject(self):
        self.close()

    def close(self):
        self._parent.diagview = None
        QWidget.close(self)


# -----------------------------------------------------------------
