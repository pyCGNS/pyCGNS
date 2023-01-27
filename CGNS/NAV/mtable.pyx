#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

import sys

from qtpy.QtCore import (Qt, QAbstractTableModel)
from qtpy.QtGui import (QFont, QFontMetrics)
from qtpy.QtWidgets import QTableView

from ..PAT import cgnskeywords as CGK
from ..PAT import cgnsutils as CGU
from ..NAV.moption import Q7OptionContext as OCTXT


# -----------------------------------------------------------------
class Q7TableView(QTableView):
    def __init__(self, parent):
        super(Q7TableView, self).__init__(None)
        self._parent = None
        self.lastPos = None
        self.lastButton = None

    def mousePressEvent(self, event):
        self.lastPos = event.globalPos()
        self.lastButton = event.button()
        QTableView.mousePressEvent(self, event)


# -----------------------------------------------------------------
class Q7TableModel(QAbstractTableModel):
    def __init__(self, node, showparams, parent=None):
        super(Q7TableModel, self).__init__(parent)
        self.node = node
        self.showmode = showparams['mode']
        self.cs = showparams['cs']
        self.ls = showparams['ls']
        self.fmt = "%-s"
        tx = self.fmt % 'w' * 16
        if self.node.sidsDataType() in [CGK.R4, CGK.R8]:
            self.fmt = "% 0.8e"
            tx = self.fmt % 9.9999999999999
        if self.node.sidsDataType() in [CGK.I4, CGK.I8]:
            self.fmt = "%8d"
            tx = self.fmt % 999999999
        if self.node.sidsDataType() in [CGK.C1]:
            self.fmt = "%1s"
            tx = self.fmt % 'w'
        if self.node.sidsDataType() not in [CGK.MT, CGK.LK]:
            if self.node.sidsValueFortranOrder():
                self.flatindex = self.flatindex_F
            else:
                self.flatindex = self.flatindex_C
            self.flatarray = self.node.sidsValue().flat
        else:
            self.flatarray = None
        self.hmin = 1
        self.vmin = 1
        self.font = QFont("Courier new", 10)
        fm = QFontMetrics(self.font)
        self.colwidth = fm.width(tx)

    def setRange(self, minh, minv):
        self.hmin = minh
        self.vmin = minv

    def headerData(self, section, orientation, role):
        if (orientation == Qt.Horizontal) and (role == Qt.DisplayRole):
            hix = section + self.hmin
            return hix
        if (orientation == Qt.Vertical) and (role == Qt.DisplayRole):
            vix = section + self.vmin
            return vix
        return None

    def flatindex_C(self, index):
        return index.row() * self.cs + index.column()

    def flatindex_F(self, index):
        return index.row() + index.column() * self.ls

    def columnCount(self, parent):
        return self.cs

    def rowCount(self, parent):
        return self.ls

    def index(self, row, column, parent):
        return self.createIndex(row, column, 0)

    def data(self, index, role=Qt.DisplayRole):
        if role not in [Qt.DisplayRole, Qt.FontRole]:
            return None
        if self.flatarray is None:
            return None
        if role == Qt.FontRole:
            return self.font
        return self.fmt % self.flatarray[self.flatindex(index)]

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def getValue(self, index):
        return self.flatarray[self.flatindex(index)]

# -----------------------------------------------------------------
