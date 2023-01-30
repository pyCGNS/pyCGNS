#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

import numpy

from . import mtree as NMT
from .mtree import Q7TreeView, Q7TreeItem, Q7TreeModel
from .moption import Q7OptionContext as OCTXT


## -----------------------------------------------------------------
# class Q7MergeTreeView(Q7MergeView):
#    def  __init__(self,parent):
#        Q7TreeView.__init__(self,parent)

# -----------------------------------------------------------------
class Q7TreeMergeModel(Q7TreeModel):
    def __init__(self, fgprint, parent=None):
        super(Q7TreeMergeModel, self).__init__(fgprint, parent)
        self.addA = True
        self.addB = True

    def itemNew(self, fg, nt, md, tg, pi):
        return Q7MergeItem(fg, nt, md, tg, pi)


TAG_FRONT = '???'
TAG_BACK = '&'  # single char required
TAG_LENGTH = 5


# -----------------------------------------------------------------
class Q7MergeItem(Q7TreeItem):
    def __init__(self, fgprint, data, model, tag="", parent=None):
        super(Q7MergeItem, self).__init__(fgprint, data, model, tag, parent)
        self._model = model

    def columnCount(self):
        return NMT.COLUMN_LAST + 1

    def notagName(self):
        name = self.sidsName()
        if ((len(name) > TAG_LENGTH) and
                (name[-1] == TAG_BACK) and
                (name[-TAG_LENGTH:-len(TAG_BACK) - 1] == TAG_FRONT)):
            return name[:-TAG_LENGTH]
        return name

    def data(self, column):
        if not self._model.addA and self.userState() == NMT.STUSR_A:
            print('NO A')
            return None
        if not self._model.addB and self.userState() == NMT.STUSR_B:
            print('NO B')
        if self._itemnode is None:
            return self._title[column]
        if column == NMT.COLUMN_NAME:
            return self.notagName()
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
