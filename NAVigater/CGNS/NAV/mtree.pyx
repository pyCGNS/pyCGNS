#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
import numpy

from PySide.QtCore    import *
from PySide.QtGui     import *
from CGNS.NAV.Q7TreeWindow import Ui_Q7TreeWindow

HIDEVALUE='@@HIDE@@'
COLUMNICO=[2,4,5,6,7,8]
DATACOLUMN=8

STLKTOPOK='@@LTOPOK@@' # top link entry ok
STLKCHDOK='@@LCHDOK@@' # child link entry ok
STLKTOPBK='@@LTOPBK@@' # broken top link entry
STLKTOPNF='@@LTOPNF@@' # top link entry ok not followed
STLKNOLNK='@@LNOLNK@@' # no link

import CGNS.PAT.cgnsutils as CT

# -----------------------------------------------------------------
class Q7TreeView(QTreeView):
    def  __init__(self,parent):
        QTreeView.__init__(self,None)
        self._parent=parent
    def selectionChanged(self,old,new):
        QTreeView.selectionChanged(self,old,new)
        if (old.count()):
            self._parent.updateStatus(old[0].topLeft().internalPointer())
    def mousePressEvent(self,event):
       self.lastPos=event.globalPos()
       self.lastButton=event.button()
       QTreeView.mousePressEvent(self,event)
       
# -----------------------------------------------------------------
class Q7TreeItem(object):
    dtype=['MT','I4','I8','R4','R8','C1','LK']
    stype={'MT':0,'I4':4,'I8':8,'R4':4,'R8':8,'C1':1,'LK':0}
    def __init__(self,control,data,parent=None):  
        self._parentitem=parent  
        self._itemnode=data  
        self._childrenitems=[]
        self._title=['Name','SIDS type','L','Shape','M','S','C','F','Value']
        if (parent is not None): self._path=parent.sidsPath()+'/'+data[0]
        else:                    self._path=''
        self._depth=self._path.count('/')
        self._size=None
        self._control=control
    def sidsParent(self):
        return self._parentitem._itemnode
    def sidsPath(self):
        return self._path
    def sidsName(self):
        return self._itemnode[0]
    def sidsValue(self):
        return self._itemnode[1]
    def sidsChildren(self):
        return self._itemnode[2]
    def sidsType(self):
        return self._itemnode[3]
    def sidsDataType(self,all=False):
        if (all): return Q7TreeItem.dtype
        return CT.getValueDataType(self._itemnode)
    def sidsDataTypeSize(self):
        return Q7TreeItem.stype[CT.getValueDataType(self._itemnode)]
    def sidsTypeList(self):
        tlist=CT.getNodeAllowedChildrenTypes(self._parentitem._itemnode,
                                             self._itemnode)
        return tlist
    def sidsDims(self):
        if (type(self.sidsValue())==numpy.ndarray):
            return self.sidsValue().shape
        return (0,)
    def sidsLinkStatus(self):
        return STLKNOLNK
    def addChild(self,item):  
        self._childrenitems.append(item)  
    def child(self,row):  
        return self._childrenitems[row]  
    def childCount(self):  
        return len(self._childrenitems)  
    def columnCount(self):
        return 9
    def data(self,column):
        if (self._itemnode==None): return self._title[column]
        if (column==0):
            return self.sidsName()
        if (column==1): return self.sidsType()
        if (column==2):
            return self.sidsLinkStatus()
        if (column==3):
            if (self.sidsValue() is None): return None
            return str(self.sidsValue().shape)
        if (column==8):
            if (self.sidsValue() is None): return None
            if (type(self.sidsValue())==numpy.ndarray):
                vsize=reduce(lambda x,y: x*y, self.sidsValue().shape)
                if (vsize>self._control.getOptionValue('maxlengthdatadisplay')):
                    return HIDEVALUE
                if (self.sidsValue().dtype.char in ['S','c']):
                    return self.sidsValue().tostring()
            return str(self.sidsValue().tolist())
        return None
    def parent(self):  
        return self._parentitem  
    def row(self):  
        if self._parentitem:  
            return self._parentitem._childrenitems.index(self)  
        return 0  
  
# -----------------------------------------------------------------
class Q7TreeModel(QAbstractItemModel):  
    def __init__(self,fgprint,parent=None):  
        super(Q7TreeModel, self).__init__(parent)  
        self._rootitem = Q7TreeItem(fgprint.control,(None))  
        self._fingerprint=fgprint
        self.parseAndUpdate(self._rootitem, self._fingerprint.tree)
        fgprint.model=self
    def columnCount(self, parent):  
        if (parent.isValid()): return parent.internalPointer().columnCount()  
        else:                  return self._rootitem.columnCount()  
    def data(self, index, role):
        if (not index.isValid()): return None  
        if (role not in [Qt.DisplayRole,Qt.DecorationRole]): return None
        if ((role == Qt.DecorationRole)
            and (index.column() not in COLUMNICO)): return None
        item = index.internalPointer()
        disp = item.data(index.column())
        if ((index.column()==DATACOLUMN) and (role == Qt.DecorationRole)):
            if (disp == HIDEVALUE):
                disp=QIcon(QPixmap(":/images/icons/data-array-large.gif"))
            else:
                return None
        if ((index.column()==DATACOLUMN) and (role == Qt.DisplayRole)):
            if (disp == HIDEVALUE):
                return None
        if (disp == STLKNOLNK):
            disp=QIcon(QPixmap(":/images/icons/link-node.gif"))
        return disp
    def flags(self, index):  
        if (not index.isValid()):  return Qt.NoItemFlags  
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable  
    def headerData(self, section, orientation, role):  
        if ((orientation == Qt.Horizontal) and (role == Qt.DisplayRole)):  
            return self._rootitem.data(section)  
        return None  
    def index(self, row, column, parent):  
        if (not self.hasIndex(row, column, parent)): return QModelIndex()  
        if (not parent.isValid()): parentitem = self._rootitem  
        else:                      parentitem = parent.internalPointer()  
        childitem = parentitem.child(row)  
        if (childitem): return self.createIndex(row, column, childitem)  
        return QModelIndex()  
    def parent(self, index):  
        if (not index.isValid()): return QModelIndex()  
        childitem = index.internalPointer()  
        parentitem = childitem.parent()  
        if (parentitem == self._rootitem): return QModelIndex()  
        return self.createIndex(parentitem.row(), 0, parentitem)  
    def rowCount(self, parent):  
        if (parent.column() > 0): return 0  
        if (not parent.isValid()): parentitem = self._rootitem  
        else:                      parentitem = parent.internalPointer()  
        return parentitem.childCount()  
    def parseAndUpdate(self,parent,node):
        newnode=Q7TreeItem(self._fingerprint.control,(node),parent)
        parent.addChild(newnode)
        for childnode in node[2]:
            c=self.parseAndUpdate(newnode,childnode)
            self._fingerprint.depth=max(c._depth,self._fingerprint.depth)
        return newnode

# -----------------------------------------------------------------
