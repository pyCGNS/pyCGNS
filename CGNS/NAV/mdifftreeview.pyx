#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

import numpy
from ..PAT import cgnsutils as CGU

from ..NAV.moption import Q7OptionContext as OCTXT
from ..NAV import mtree as NMT
from ..NAV.mtree import (Q7TreeItem, Q7TreeModel, Q7TreeView)

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

(DIFF_NX, DIFF_NA, DIFF_ND, DIFF_CQ, DIFF_CT, DIFF_CS, DIFF_CV) = range(7)


# -----------------------------------------------------------------
class Q7DiffTreeModel(Q7TreeModel):
    def __init__(self, fgprintindex, parent=None):
        Q7TreeModel.__init__(self, fgprintindex, parent)
        self._diag = None
        self.orange = QColor(255, 165, 0, 255)
        self.orangeLight = QColor(255, 165, 0, 120)
        self.red = QColor(255, 0, 0, 255)
        self.redLight = QColor(255, 0, 0, 90)
        self.green = QColor(0, 128, 0, 255)
        self.greenLight = QColor(0, 128, 0, 90)
        self.gray = QColor(192, 192, 192, 255)
        self.grayLight = QColor(192, 192, 192, 192)

    def setDiag(self, diag):
        self._diag = diag

    def getDiag(self):
        return self._diag

    def data(self, index, role):
        idx = index
        col = idx.column()
        #nnm = index.internalPointer().sidsName()
        pth = CGU.getPathNoRoot(index.internalPointer().sidsPath())
        if ((role not in [Qt.BackgroundRole]) or
                (self._diag is None) or
                (pth not in self._diag)):
            return Q7TreeModel.data(self, index, role)
        if col == NMT.COLUMN_NAME:
            if self._diag[pth] == DIFF_NX:
                return self.grayLight
            if self._diag[pth] == DIFF_NA:
                return self.greenLight
            if self._diag[pth] == DIFF_ND:
                return self.redLight
            return self.grayLight
        elif col == NMT.COLUMN_DATATYPE:
            if self._diag[pth] == DIFF_CQ:
                return self.orangeLight
        elif col == NMT.COLUMN_SHAPE:
            if self._diag[pth] == DIFF_CS:
                return self.orangeLight
        elif col == NMT.COLUMN_SIDS:
            if self._diag[pth] == DIFF_CT:
                return self.orangeLight
        return Q7TreeModel.data(self, index, role)


# -----------------------------------------------------------------
class Q7DiffTreeView(Q7TreeView):
    def __init__(self, parent):
        Q7TreeView.__init__(self, parent)


# -----------------------------------------------------------------
class Q7DiffItem(Q7TreeItem):
    def __init__(self, fgprint, data, model, tag="", parent=None):
        super(Q7DiffItem, self).__init__(self, fgprint, data, model, tag, parent)

    def columnCount(self):
        return NMT.COLUMN_LAST + 1

    def data(self, column):
        if self._itemnode is None:
            return self._title[column]
        if column == NMT.COLUMN_NAME:
            return self.sidsName()
        if column == NMT.COLUMN_SIDS:
            return self.sidsType()
        if column == NMT.COLUMN_FLAG_LINK:
            return self.sidsLinkStatus()
        if column == NMT.COLUMN_SHAPE:
            if self.sidsValue() is None:
                return None
            if not isinstance(self.sidsValue(), numpy.ndarray):
                return None
            return str(self.sidsValue().shape)
        if column == NMT.COLUMN_FLAG_SELECT:
            return self._states['mark']
        if column == NMT.COLUMN_DATATYPE:
            return self.sidsDataType()
        if column == NMT.COLUMN_FLAG_CHECK:
            return self._states['check']
        if column == NMT.COLUMN_FLAG_USER:
            return self._states['user']
        if column == NMT.COLUMN_VALUE:
            if self._lazy:
                return NMT.LAZYVALUE
            if self.sidsValue() is None:
                return None
            if isinstance(self.sidsValue(), numpy.ndarray):
                vsize = 1
                for x in self.sidsValue().shape:
                    vsize *= x
                if ((vsize > OCTXT.MaxDisplayDataSize) and
                        (OCTXT.MaxDisplayDataSize > 0)):
                    return NMT.HIDEVALUE
                if self.sidsValue().dtype.char in ['S', 'c']:
                    if len(self.sidsValue().shape) == 1:
                        return self.sidsValue().tostring().decode('ascii')
                    if len(self.sidsValue().shape) > 2:
                        return NMT.HIDEVALUE
                    return NMT.HIDEVALUE
                if (self.sidsValue().shape == (1,)) and OCTXT.Show1DAsPlain:
                    return str(self.sidsValue()[0])
            return str(self.sidsValue().tolist())[:100]
        return None

# -----------------------------------------------------------------
