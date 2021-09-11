#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
from CGNS.NAV.moption import Q7OptionContext as OCTXT

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

from qtpy.QtCore import Qt, QModelIndex
from qtpy.QtWidgets import QStyledItemDelegate

from CGNS.NAV.Q7FormWindow import Ui_Q7FormWindow
from CGNS.NAV.mtable import Q7TableModel
from CGNS.NAV.wstylesheets import Q7TABLEVIEWSTYLESHEET
from CGNS.NAV.wfingerprint import Q7Window


# -----------------------------------------------------------------
def divpairs(n):
    d = n
    l = []
    while d != 0:
        m = n // d
        r = n % d
        if r == 0:
            l += [(d, m)]
        d -= 1
    return l


# -----------------------------------------------------------------
class Q7TableItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        t = index.data(Qt.DisplayRole)
        r = option.rect.adjusted(2, 2, -2, -2)
        painter.drawText(r, Qt.AlignVCenter | Qt.AlignLeft, t, r)
        # QStyledItemDelegate.paint(self, painter, option, index)


# -----------------------------------------------------------------
class Q7Form(Q7Window, Ui_Q7FormWindow):
    def __init__(self, control, node, fgprintindex):
        Q7Window.__init__(
            self, Q7Window.VIEW_FORM, control, node.sidsPath(), fgprintindex
        )
        self._node = node
        self._operatorlist = [""]
        for t in self._node.sidsTypeList():
            self.eType.addItem(t)
        self.bClose.clicked.connect(self.reject)
        self.bInfo.clicked.connect(self.infoFormView)
        self.cFortranOrderOff.setChecked(False)
        self.cFortranOrderOff.setDisabled(True)
        for dt in node.sidsDataType(all=True):
            self.cDataType.addItem(dt)

    def infoFormView(self):
        self._control.helpWindow("Form")

    def updateTreeStatus(self):
        pass

    def accept(self):
        pass

    def reject(self):
        self.close()

    def show(self):
        self.reset()
        super(Q7Form, self).show()

    def reset(self):
        self.eName.setText(self._node.sidsName())
        self.ePath.setText(self._node.sidsPath())
        self.ePath.setReadOnly(True)
        self.sMinH.setRange(-999999999, 999999999)
        self.sMinV.setRange(-999999999, 999999999)
        self.sMinH.setValue(1)
        self.sMinV.setValue(1)
        self.setCurrentType(self._node.sidsType())
        dims = self._node.sidsDims()
        its = 1
        for x in dims:
            its *= x
        ndz = its * self._node.sidsDataTypeSize()
        self.eDims.setText(str(dims))
        self.setDataType(self._node)
        self.eItems.setText(str(its))
        self.ls = 1
        self.cs = dims[0]
        if len(dims) > 1:
            self.ls = 1
            for x in dims[1:]:
                self.ls *= x
        self.showparams = {"cs": self.cs, "ls": self.ls, "mode": "IJ"}
        for dp in divpairs(self.cs * self.ls):
            self.cRowColSize.addItem("%d x %d" % dp)
        ix = self.cRowColSize.findText("%d x %d" % (self.cs, self.ls))
        self.cRowColSize.setCurrentIndex(ix)
        self.model = Q7TableModel(self._node, self.showparams)
        self.tableView.setModel(self.model)
        self.tableView.setStyleSheet(self._stylesheet)
        self.bMinimize.clicked.connect(self.minimizeCells)
        #
        self.tableView.clicked[QModelIndex].connect(self.clickedNode)
        self.cRowColSize.currentIndexChanged[int].connect(self.resizeTable)
        self.sMinH.valueChanged[int].connect(self.resetIndex)
        self.sMinV.valueChanged[int].connect(self.resetIndex)
        # QObject.connect(self.tableView,
        #                SIGNAL("clicked(QModelIndex)"),
        #                self.clickedNode)
        # QObject.connect(self.cRowColSize,
        #                SIGNAL("currentIndexChanged(int)"),
        #                self.resizeTable)
        # QObject.connect(self.sMinH,
        #                SIGNAL("valueChanged(int)"),
        #                self.resetIndex)
        # QObject.connect(self.sMinV,
        #                SIGNAL("valueChanged(int)"),
        #                self.resetIndex)
        lk = self.FG.isLink(self._node.sidsPath())
        if lk:
            self.setLabel(self.eDestDir, lk[0])
            self.setLabel(self.eDestFile, lk[1])
            self.setLabel(self.eDestPath, lk[2])
        else:
            self.tFiles.setDisabled(True)
        self.tPython.setDisabled(True)
        if self._node.sidsDataType() == CGK.C1:
            if self._node.sidsValue().ndim > 1:
                txt = self._node.sidsValue().T.tostring().decode("ascii")
                self.cFortranOrderOff.setChecked(True)
                self.cFortranOrderOff.setDisabled(True)
            else:
                txt = self._node.sidsValue().tostring().decode("ascii")
            self.eText.initText(txt)

    def resizeTable(self):
        s = self.cRowColSize.currentText()
        (r, c) = s.split("x")
        self.cs = int(r)
        self.ls = int(c)
        self.showparams = {"cs": self.cs, "ls": self.ls, "mode": "IJ"}
        self.model = Q7TableModel(self._node, self.showparams)
        self.tableView.setModel(self.model)
        self.tableView.setStyleSheet(self._stylesheet)

    def clickedNode(self, index):
        pass

    def minimizeCells(self, *args):
        self.tableView.resizeColumnsToContents()
        self.tableView.resizeRowsToContents()

    def setCurrentType(self, ntype):
        idx = self.eType.findText(ntype)
        self.eType.setCurrentIndex(idx)

    def setDataType(self, node):
        dt = node.sidsDataType()
        ix = self.cDataType.findText(dt)
        if ix == -1:
            return
        self.cDataType.setCurrentIndex(ix)

    def resetIndex(self):
        hm = self.sMinH.value()
        vm = self.sMinV.value()
        self.model.setRange(hm, vm)
        self.model.headerDataChanged.emit(Qt.Vertical, 1, 1)
        self.model.headerDataChanged.emit(Qt.Horizontal, 1, 1)

    def addControlLine(self):
        pass

    def doRelease(self):
        pass


# -----------------------------------------------------------------
