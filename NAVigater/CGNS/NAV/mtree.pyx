#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
import numpy
import copy

from PySide.QtCore    import *
from PySide.QtGui     import *
from CGNS.NAV.Q7TreeWindow import Ui_Q7TreeWindow
from CGNS.NAV.moption import Q7OptionContext as OCTXT

HIDEVALUE='@@HIDE@@'
COLUMNICO=[2,4,5,6,7,8]
DATACOLUMN=8

STLKTOPOK='@@LTOPOK@@' # top link entry ok
STLKCHDOK='@@LCHDOK@@' # child link entry ok
STLKTOPBK='@@LTOPBK@@' # broken top link entry
STLKTOPNF='@@LTOPNF@@' # top link entry ok not followed
STLKNOLNK='@@LNOLNK@@' # no link

STCHKUNKN='@@CKUNKN@@' # unknown
STCHKGOOD='@@CKGOOD@@' # good, ok
STCHKWARN='@@CKWARN@@' # warning
STCHKFAIL='@@CKFAIL@@' # fail, bad

STFORUNKN='@@FORUNK@@'

STMARK_ON='@@MARK_ON@@'
STMARKOFF='@@MARKOFF@@'

STSHRUNKN='@@SHRUNKN@@'

EDITNODE='@@NODEDIT@@'
MARKNODE='@@NODEMARK@@'
DOWNNODE='@@NODEDOWN@@'
UPNODE  ='@@NODEUP@@'
COPY='@@NODECOPY@@'
CUT='@@NODECUT@@'
PASTEBROTHER='@@NODEPASTEB@@'
PASTECHILD='@@NODEPASTEC@@'

ICONMAPPING={
 STLKNOLNK:":/images/icons/link-node.gif",

 STCHKUNKN:":/images/icons/empty.gif",
 STCHKGOOD:":/images/icons/subtree-sids-ok.gif",
 STCHKFAIL:":/images/icons/subtree-sids-failed.gif",
 STCHKWARN:":/images/icons/subtree-sids-warning.gif",

 STFORUNKN:":/images/icons/empty.gif",   
 STSHRUNKN:":/images/icons/empty.gif",   
 STMARKOFF:":/images/icons/empty.gif",
 STMARK_ON:":/images/icons/mark-node.gif",   
}

KEYMAPPING={
 MARKNODE:   Qt.Key_Space,
 EDITNODE:   Qt.Key_Insert,
 UPNODE  :   Qt.Key_Up,
 DOWNNODE:   Qt.Key_Down,
 COPY    :   Qt.Key_C,
 CUT     :   Qt.Key_X,
 PASTECHILD   :   Qt.Key_Y,
 PASTEBROTHER :   Qt.Key_V,
}

EDITKEYMAPPINGS=[
    KEYMAPPING[COPY],
    KEYMAPPING[CUT],
    KEYMAPPING[PASTECHILD],
    KEYMAPPING[PASTEBROTHER],
    ]

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

# -----------------------------------------------------------------
class Q7TreeView(QTreeView):
    def  __init__(self,parent):
        QTreeView.__init__(self,None)
        self._parent=parent
        self._control=None
        self._model=None
    def setControlWindow(self,control,model):
        self._control=control
        self._model=model
    def getLastEntered(self):
        if (self._control is not None): return self._control.getLastEntered()
        return None
    def setLastEntered(self):
        if (self._control is not None): self._control.setLastEntered()
    def selectionChanged(self,old,new):
        QTreeView.selectionChanged(self,old,new)
        if (old.count()):
            self._parent.updateStatus(old[0].topLeft().internalPointer())
    def mousePressEvent(self,event):
        self.lastPos=event.globalPos()
        self.lastButton=event.button()
        QTreeView.mousePressEvent(self,event)
    def wink(self,index):
        if (self.isExpanded(index)):
            self.setExpanded(index,False)
            self.setExpanded(index,True)
        else:
            self.setExpanded(index,True)
            self.setExpanded(index,False)
    def keyPressEvent(self,event):
        last=self.getLastEntered()
        if (last is not None):
          kmod=event.modifiers()
          kval=event.key()
          if (kval in EDITKEYMAPPINGS):
              if (kmod==Qt.ControlModifier):
                if (kval==KEYMAPPING[COPY]):
                  self._model.copyNode(last)
                if (kval==KEYMAPPING[CUT]):
                  self._model.cutNode(last)
                if (kval==KEYMAPPING[PASTECHILD]):
                  self._model.pasteAsChild(last)
                if (kval==KEYMAPPING[PASTEBROTHER]):
                  self._model.pasteAsBrother(last)
          if (kval==KEYMAPPING[MARKNODE]):
              last.internalPointer().switchMarked()
              last.internalPointer()._model.updateSelected()
              self.changeRow(last.row())
          if (kval==KEYMAPPING[UPNODE]):
              if   (kmod==Qt.ControlModifier): self.upRowLevel(last)
              elif (kmod==Qt.ShiftModifier):   self.upRowMarked()
              else: QTreeView.keyPressEvent(self,event)
          if (kval==KEYMAPPING[DOWNNODE]):
              if (kmod==Qt.ControlModifier): self.downRowLevel(last)
              elif (kmod==Qt.ShiftModifier): self.downRowMarked()
              else: QTreeView.keyPressEvent(self,event)
        self.setLastEntered()
        self.scrollTo(self.getLastEntered())
    def refreshView(self):
        ixc=self.currentIndex()
        self._model.refreshModel(ixc)
    def upRowLevel(self,index):
        self.relativeMoveToRow(-1,index)
    def downRowLevel(self,index):
        self.relativeMoveToRow(+1,index)
    def upRowMarked(self):
        self.changeSelectedMark(-1)
    def downRowMarked(self):
        self.changeSelectedMark(+1)
    def relativeMoveToRow(self,shift,index):
        row=index.row()
        col=index.column()
        if (not index.sibling(row+shift,col).isValid()): return
        parent=index.parent()
        nix=self.model().index(row+shift,col,parent)
        self.exclusiveSelectRow(nix)
    def changeRow(self,row):
        ix1=self._model.createIndex(row,0)
        ix2=self._model.createIndex(row,DATACOLUMN-1)
        self._model.dataChanged.emit(ix1,ix2)
    def exclusiveSelectRow(self,index):
        row=index.row()
        col=index.column()
        parent=index.parent()
        nix=self.model().index(row,col,parent)
        mod=QItemSelectionModel.SelectCurrent|QItemSelectionModel.Rows
        self.selectionModel().setCurrentIndex(nix,mod)
        self.setLastEntered()
        self.scrollTo(index)
    def changeSelectedMark(self,delta):
        if (self.model()._selected==[]): return
        sidx=self.model()._selectedIndex
        if (self.model()._selectedIndex==-1): self.model()._selectedIndex=0
        elif ((delta==-1) and (sidx==0)):
            self.model()._selectedIndex=len(self.model()._selected)-1
        elif (delta==-1):
            self.model()._selectedIndex-=1
        elif ((delta==+1) and (sidx==len(self.model()._selected)-1)):
            self.model()._selectedIndex=0
        elif (delta==+1):
            self.model()._selectedIndex+=1
        if ((self.model()._selectedIndex!=-1)
             and (self.model()._selectedIndex<len(self.model()._selected))):
            path=self.model()._selected[self.model()._selectedIndex]
            idx=self.model().match(self.model().index(0,0,QModelIndex()),
                                   Qt.UserRole,
                                   path,
                                   flags=Qt.MatchExactly|Qt.MatchRecursive)
            if (idx[0].isValid()): self.exclusiveSelectRow(idx[0])
       
# -----------------------------------------------------------------
class Q7TreeItem(object):
    dtype=['MT','I4','I8','R4','R8','C1','LK']
    stype={'MT':0,'I4':4,'I8':8,'R4':4,'R8':8,'C1':1,'LK':0}
    def __init__(self,fgprint,data,model,tag="",parent=None):  
        self._parentitem=parent  
        self._itemnode=data  
        self._childrenitems=[]
        self._title=['Name','SIDS type','L','Shape','M','S','C','F','Value']
        if (parent is not None): self._path=parent.sidsPath()+'/'+data[0]
        else:                    self._path=''
        self._depth=self._path.count('/')
        self._size=None
        self._fingerprint=fgprint
        self._control=self._fingerprint.control
        self._states={'mark':STMARKOFF,'check':STCHKUNKN,
                      'fortran':STFORUNKN,'shared':STSHRUNKN}
        self._model=model
        self._tag=tag
        if ((parent is not None) and (model is not None)):
            self._model._extension[self._path]=self
    def orderTag(self):
        return self._tag+"0"*(self._fingerprint.depth*4-len(self._tag))
    def sidsParent(self):
        return self._parentitem._itemnode
    def sidsPath(self):
        return self._path
    def sidsName(self):
        return self._itemnode[0]
    def sidsNameSet(self,name):
        self._itemnode[0]=name
    def sidsValue(self):
        return self._itemnode[1]
    def sidsChildren(self):
        return self._itemnode[2]
    def sidsType(self):
        return self._itemnode[3]
    def sidsDataType(self,all=False):
        if (all): return Q7TreeItem.dtype
        return CGU.getValueDataType(self._itemnode)
    def sidsDataTypeSize(self):
        return Q7TreeItem.stype[CGU.getValueDataType(self._itemnode)]
    def sidsTypeList(self):
        tlist=CGU.getNodeAllowedChildrenTypes(self._parentitem._itemnode,
                                             self._itemnode)
        return tlist
    def sidsDims(self):
        if (type(self.sidsValue())==numpy.ndarray):
            return self.sidsValue().shape
        return (0,)
    def sidsLinkStatus(self):
        return STLKNOLNK
    def sidsAddChild(self,node):
        newtree=copy.deepcopy(node._itemnode)
        name=newtree[0]
        ntype=newtree[3]
        parent=self._itemnode
        count=0
        while (not CGU.checkDuplicatedName(parent,name,dienow=False)):
            count+=1
            name='{%s#%.3d}'%(ntype,count)
        newtree[0]=name
        self._itemnode[2].append(newtree)
        newpath=self.sidsPath()+'/%s'%name
        return (newtree,newpath)
    def addChild(self,item):  
        self._childrenitems.append(item)  
    def child(self,row):  
        return self._childrenitems[row]
    def hasChildren(self):
        if (self.childCount()>0): return True
        return False
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
        if (column==4): return self._states['mark']
        if (column==5): return self._states['shared']
        if (column==6): return self._states['check']
        if (column==7): return self._states['fortran']
        if (column==8):
            if (self.sidsValue() is None): return None
            if (type(self.sidsValue())==numpy.ndarray):
                vsize=reduce(lambda x,y: x*y, self.sidsValue().shape)
                if (vsize>OCTXT.MaxLengthDataDisplay):
                    return HIDEVALUE
                if (self.sidsValue().dtype.char in ['S','c']):
                    return self.sidsValue().tostring()
            return str(self.sidsValue().tolist())
        return None
    def parentItem(self):  
        return self._parentitem  
    def row(self):  
        if self._parentitem:  
            return self._parentitem._childrenitems.index(self)  
        return 0
    def switchMarked(self):
        if (self._states['mark']==STMARK_ON): self._states['mark']=STMARKOFF
        else:                                 self._states['mark']=STMARK_ON

SORTTAG="%.4x"

# -----------------------------------------------------------------
class Q7TreeModel(QAbstractItemModel):
    _icons={}
    def __init__(self,fgprint,parent=None):  
        super(Q7TreeModel, self).__init__(parent)  
        self._extension={}
        self._rootitem=Q7TreeItem(fgprint,(None),None)  
        self._fingerprint=fgprint
        self._slist=OCTXT._SortedTypeList
        self._count=0
        self._maxrow=0
        self._index={}
        self.parseAndUpdate(self._rootitem, self._fingerprint.tree)
        fgprint.model=self
        for ik in ICONMAPPING:
            Q7TreeModel._icons[ik]=QIcon(QPixmap(ICONMAPPING[ik]))
        self._selected=[]
        self._selectedIndex=-1
        self.copyPasteBuffer=None
        self._control=self._fingerprint.control
    def setIndex(self,path,row,col,parent):
        self._index[path]=(row,col,parent)
    def getIndex(self,path):
        if (self._index.has_key(path)):
            return self.createIndex(*self._index[path])
        return QModelIndex()
    def nodeFromPath(self,path):
        if (path in self._extension.keys()): return self._extension[path]
        return None
    def sortNamesAndTypes(self,paths):
        t=[]
        if (paths is None): return []
        for p in paths:
            n=self.nodeFromPath(p)
            if (n is not None):
                t+=[(n.orderTag(),p)]
        t.sort()
        return [e[1] for e in t]
    def getSelectedZones(self):
        zlist=[]
        for pth in self._selected:
            spth=pth.split('/')
            if (len(spth)>3):
                zpth='/'.join(['']+spth[2:4])
                node=CGU.getNodeByPath(self._fingerprint.tree,zpth)
                if (    (node is not None)
                    and (node[3]==CGK.Zone_ts)
                    and (zpth not in zlist)):
                    zlist+=[zpth]
            if (len(spth)==3):
                bpth='/'.join(['']+spth[2:])
                for node in CGU.getChildrenByPath(self._fingerprint.tree,bpth):
                    if (node[3] == CGK.Zone_ts):
                        zpth=bpth+'/'+node[0]
                        if (zpth not in zlist):
                            zlist+=[zpth]
        return zlist
    def getSelectedShortCut(self):
        slist=[]
        for pth in self._selected:
            if (CGU.hasFirstPathItem(pth)): pth=CGU.removeFirstPathItem(pth)
            slist+=[pth]
        return slist
    def updateSelected(self):
        self._selected=[]
        self._selectedIndex=-1
        for k in self._extension:
           if (self._extension[k]._states['mark']==STMARK_ON):
               self._selected+=[self._extension[k].sidsPath()]
        self._selected=self.sortNamesAndTypes(self._selected)
    def markExtendToList(self,mlist):
        for k in self._extension:
            if (k in mlist):
                self._extension[k]._states['mark']=STMARK_ON
    def markAll(self):
        for k in self._extension:
           self._extension[k]._states['mark']=STMARK_ON
    def unmarkAll(self):
        for k in self._extension:
           self._extension[k]._states['mark']=STMARKOFF
    def swapMarks(self):
        for k in self._extension:
           self._extension[k].switchMarked()
    def columnCount(self, parent):  
        if (parent.isValid()): return parent.internalPointer().columnCount()  
        else:                  return self._rootitem.columnCount()  
    def data(self, index, role):
        if (not index.isValid()): return None
        if (role == Qt.UserRole): return index.internalPointer().sidsPath()
        if (role not in [Qt.EditRole,Qt.DisplayRole,Qt.DecorationRole]):
            return None
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
        if (disp in ICONMAPPING.keys()):
            disp=Q7TreeModel._icons[disp]
        return disp
    def flags(self, index):  
        if (not index.isValid()):  return Qt.NoItemFlags  
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable  
    def headerData(self, section, orientation, role):  
        if ((orientation == Qt.Horizontal) and (role == Qt.DisplayRole)):  
            return self._rootitem.data(section)  
        return None  
    def index(self, row, column, parent):
        self._maxrow=max(row,self._maxrow)
        if (not self.hasIndex(row, column, parent)):
            return QModelIndex()  
        if (not parent.isValid()): parentitem = self._rootitem  
        else:                      parentitem = parent.internalPointer()  
        childitem = parentitem.child(row)  
        if (childitem):
            return self.createIndex(row, column, childitem)  
        return QModelIndex()  
    def parent(self, index):  
        if (not index.isValid()): return QModelIndex()
        childitem = index.internalPointer()
        if (type(childitem)==type(QModelIndex())): return QModelIndex()
        if (childitem is None): return QModelIndex()  
        parentitem = childitem.parentItem()  
        if (parentitem == self._rootitem): return QModelIndex()  
        return self.createIndex(parentitem.row(), 0, parentitem)  
    def rowCount(self, parent):  
        if (parent.column() > 0): return 0  
        if (not parent.isValid()): parentitem = self._rootitem  
        else:                      parentitem = parent.internalPointer()  
        return parentitem.childCount()  
    def parseAndUpdate(self,parentItem,node,parenttag=""):
        self._count+=1
        tag=parenttag+SORTTAG%self._count
        newItem=Q7TreeItem(self._fingerprint,(node),self,tag,parentItem)
        row=parentItem.childCount()
        self.beginInsertRows(self.getIndex(parentItem.sidsPath()),row,row)
        parentItem.addChild(newItem)
        self.endInsertRows()
        self.setIndex(newItem.sidsPath(),row,0,parentItem)
        for childnode in CGU.getNextChildSortByType(node,criteria=self._slist):
            c=self.parseAndUpdate(newItem,childnode,tag)
            self._fingerprint.depth=max(c._depth,self._fingerprint.depth)
        return newItem
    def refreshModel(self,nodeidx):
        row=nodeidx.row()
        dlt=5
        parentidx=nodeidx.parent()
        row1=min(0,abs(row-dlt))
        row2=min(row+dlt,self._maxrow)
        ix1=self.createIndex(row1,0,parentidx)
        ix2=self.createIndex(row2,DATACOLUMN-1,parentidx)
        self.dataChanged.emit(ix1,ix2)
    def copyNode(self,nodeidx):
        nodeitem=nodeidx.internalPointer()
        self.copyPasteBuffer=nodeitem
    def cutNode(self,nodeidx):
        nodeitem=nodeidx.internalPointer()
        self.copyPasteBuffer=nodeitem
    def pasteAsChild(self,nodeidx):
        if (self.copyPasteBuffer is None): return
        nodeitem=nodeidx.internalPointer()
        row=nodeitem.row()
        (ntree,npath)=nodeitem.sidsAddChild(self.copyPasteBuffer)
        self.parseAndUpdate(nodeitem,ntree,nodeitem._tag)
        child=self.getIndex(npath)
        self.refreshModel(nodeidx.parent())
        self.refreshModel(nodeidx)
        self.refreshModel(child)
    def pasteAsBrother(self,nodeidx):
        self.pasteAsChild(nodeidx.parent())

# -----------------------------------------------------------------
