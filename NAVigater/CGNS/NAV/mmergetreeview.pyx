#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import numpy

from PySide.QtCore    import *
from PySide.QtGui     import *
from CGNS.NAV.moption import Q7OptionContext as OCTXT
from CGNS.NAV.mtree import Q7TreeView,Q7TreeItem,Q7TreeModel
import CGNS.NAV.mtree as NMT

## -----------------------------------------------------------------
#class Q7MergeTreeView(Q7MergeView):
#    def  __init__(self,parent):
#        Q7TreeView.__init__(self,parent)
       
# -----------------------------------------------------------------
class Q7TreeMergeModel(Q7TreeModel):
    def __init__(self,fgprint,parent=None):
        Q7TreeModel.__init__(self,fgprint,parent)  
    def itemNew(self,fg,nt,md,tg,pi):
        return Q7MergeItem(fg,nt,md,tg,pi)

TAG_FRONT='???'
TAG_BACK='&' # single char required
TAG_LENGTH=5

# -----------------------------------------------------------------
class Q7MergeItem(Q7TreeItem):
    def __init__(self,fgprint,data,model,tag="",parent=None):
        Q7TreeItem.__init__(self,fgprint,data,model,tag,parent)
    def columnCount(self):
        return NMT.COLUMN_LAST+1
    def notagName(self):
        name=self.sidsName()
        if ((len(name)>TAG_LENGTH)
            and (name[-1]==TAG_BACK)
            and (name[-TAG_LENGTH:-len(TAG_BACK)-1]==TAG_FRONT)):
            return name[:-TAG_LENGTH]
        return name
    def data(self,column):
        if (self._itemnode==None):     return self._title[column]
        if (column==NMT.COLUMN_NAME):  return self.notagName()
        if (column==NMT.COLUMN_SIDS):      return self.sidsType()
        if (column==NMT.COLUMN_FLAG_LINK): return self.sidsLinkStatus()
        if (column==NMT.COLUMN_SHAPE):
            if (self.sidsValue() is None): return None
            if (type(self.sidsValue())!=numpy.ndarray): return None
            return str(self.sidsValue().shape)
        if (column==NMT.COLUMN_FLAG_SELECT): return self._states['mark']
        if (column==NMT.COLUMN_DATATYPE):    return self.sidsDataType()
        if (column==NMT.COLUMN_FLAG_CHECK):  return self._states['check']
        if (column==NMT.COLUMN_FLAG_USER):   return self._states['user']
        if (column==NMT.COLUMN_VALUE):
            if (self._lazy): return LAZYVALUE
            if (self.sidsValue() is None): return None
            if (type(self.sidsValue())==numpy.ndarray):
                vsize=reduce(lambda x,y: x*y, self.sidsValue().shape)
                if ((vsize>OCTXT.MaxDisplayDataSize) and
                    (OCTXT.MaxDisplayDataSize>0)):
                    return NMT.HIDEVALUE
                if (self.sidsValue().dtype.char in ['S','c']):
                    if (len(self.sidsValue().shape)==1):
                        return self.sidsValue().tostring()
                    if (len(self.sidsValue().shape)>2):
                        return NMT.HIDEVALUE
                    return NMT.HIDEVALUE
                if ((self.sidsValue().shape==(1,)) and OCTXT.Show1DAsPlain):
                    return str(self.sidsValue()[0])
            return str(self.sidsValue().tolist())[:100]
        return None

# -----------------------------------------------------------------
