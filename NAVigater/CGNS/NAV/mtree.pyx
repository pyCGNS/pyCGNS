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
from CGNS.NAV.wfingerprint import Q7fingerPrint
import CGNS.VAL.simplecheck as CGV

HIDEVALUE='@@HIDE@@'

(COLUMN_NAME,\
 COLUMN_SIDS,\
 COLUMN_FLAG_LINK,\
 COLUMN_FLAG_SELECT,\
 COLUMN_FLAG_CHECK,\
 COLUMN_FLAG_USER,\
 COLUMN_SHAPE,\
 COLUMN_DATATYPE,\
 COLUMN_VALUE)=range(9)

COLUMN_LAST=COLUMN_VALUE

COLUMN_ICO  =[COLUMN_FLAG_LINK,COLUMN_FLAG_SELECT,COLUMN_FLAG_CHECK,
              COLUMN_FLAG_USER,COLUMN_VALUE]
COLUMN_FLAGS=[COLUMN_FLAG_LINK,COLUMN_FLAG_SELECT,COLUMN_FLAG_CHECK,
              COLUMN_FLAG_USER]

COLUMN_EDIT=[COLUMN_NAME,COLUMN_SIDS,COLUMN_DATATYPE,COLUMN_VALUE]

COLUMN_TITLE=['-']*(COLUMN_LAST+1)

COLUMN_TITLE[COLUMN_NAME]='Name'
COLUMN_TITLE[COLUMN_SIDS]='SIDS type'
COLUMN_TITLE[COLUMN_FLAG_LINK]='L'
COLUMN_TITLE[COLUMN_SHAPE]='Shape'
COLUMN_TITLE[COLUMN_FLAG_SELECT]='M'
COLUMN_TITLE[COLUMN_FLAG_CHECK]='C'
COLUMN_TITLE[COLUMN_FLAG_USER]='U'
COLUMN_TITLE[COLUMN_DATATYPE]='D'
COLUMN_TITLE[COLUMN_VALUE]='Value'

STLKTOPOK='@@LTOPOK@@' # top link entry ok
STLKCHDOK='@@LCHDOK@@' # child link entry ok
STLKTOPBK='@@LTOPBK@@' # broken top link entry
STLKTOPNF='@@LTOPNF@@' # top link entry ok not followed
STLKNOLNK='@@LNOLNK@@' # no link

STCHKUNKN='@@CKUNKN@@' # unknown
STCHKGOOD='@@CKGOOD@@' # good, ok
STCHKWARN='@@CKWARN@@' # warning
STCHKFAIL='@@CKFAIL@@' # fail, bad
STCHKUSER='@@CKUSER@@' # user condition

STCHKLIST=(STCHKUNKN,STCHKGOOD,STCHKWARN,STCHKFAIL,STCHKUSER)

STUSR_P='@@USR_%d@@'
STUSR_X='@@USR_X@@'
STUSR_0='@@USR_0@@'
STUSR_1='@@USR_1@@'
STUSR_2='@@USR_2@@'
STUSR_3='@@USR_3@@'
STUSR_4='@@USR_4@@'
STUSR_5='@@USR_5@@'
STUSR_6='@@USR_6@@'
STUSR_7='@@USR_7@@'
STUSR_8='@@USR_8@@'
STUSR_9='@@USR_9@@'

USERSTATES=[STUSR_0,STUSR_1,STUSR_2,STUSR_3,STUSR_4,
            STUSR_5,STUSR_6,STUSR_7,STUSR_8,STUSR_9]

STMARK_ON='@@MARK_ON@@'
STMARKOFF='@@MARKOFF@@'

STSHRUNKN='@@SHRUNKN@@'

EDITNODE='@@NODEDIT@@'
NEWCHILDNODE='@@NODENEWC@@'
NEWBROTHERNODE='@@NODENEWB@@'
MARKNODE='@@NODEMARK@@'
DOWNNODE='@@NODEDOWN@@'
UPNODE  ='@@NODEUP@@'
COPY='@@NODECOPY@@'
CUT='@@NODECUT@@'
PASTEBROTHER='@@NODEPASTEB@@'
PASTECHILD='@@NODEPASTEC@@'
CUTSELECTED='@@NODECUTS@@'
PASTEBROTHERSELECTED='@@NODEPASTEBS@@'
PASTECHILDSELECTED='@@NODEPASTECS@@'
OPENFORM='@@OPENFORM@@'
OPENVIEW='@@OPENVIEW@@'

USERFLAG_0='@@USER0@@'
USERFLAG_1='@@USER1@@'
USERFLAG_2='@@USER2@@'
USERFLAG_3='@@USER3@@'
USERFLAG_4='@@USER4@@'
USERFLAG_5='@@USER5@@'
USERFLAG_6='@@USER6@@'
USERFLAG_7='@@USER7@@'
USERFLAG_8='@@USER8@@'
USERFLAG_9='@@USER9@@'

USERFLAGS=[USERFLAG_0,USERFLAG_1,USERFLAG_2,USERFLAG_3,USERFLAG_4,
           USERFLAG_5,USERFLAG_6,USERFLAG_7,USERFLAG_8,USERFLAG_9]

ICONMAPPING={
 STLKNOLNK:":/images/icons/empty.gif",
 STLKTOPOK:":/images/icons/link.gif",
 STLKCHDOK:":/images/icons/link.gif",
 STLKTOPBK:":/images/icons/link-break.gif",
 STLKTOPNF:":/images/icons/link-error.gif",

 STCHKUNKN:":/images/icons/empty.gif",
 STCHKGOOD:":/images/icons/check-ok.gif",
 STCHKFAIL:":/images/icons/check-fail.gif",
 STCHKWARN:":/images/icons/check-warn.gif",

 STUSR_X:  ":/images/icons/empty.gif",   
 STUSR_0:  ":/images/icons/empty.gif",   
 STSHRUNKN:":/images/icons/empty.gif",   
 STMARKOFF:":/images/icons/empty.gif",
 STMARK_ON:":/images/icons/mark-node.gif",   
}

KEYMAPPING={
 MARKNODE:              Qt.Key_Space,
 UPNODE  :              Qt.Key_Up,
 DOWNNODE:              Qt.Key_Down,
 OPENFORM     :         Qt.Key_F,
 OPENVIEW     :         Qt.Key_W,

 NEWCHILDNODE:          Qt.Key_A,
 NEWBROTHERNODE:        Qt.Key_Z,
 EDITNODE:              Qt.Key_Insert,
 COPY    :              Qt.Key_C,
 CUT     :              Qt.Key_X,
 PASTECHILD   :         Qt.Key_Y,
 PASTEBROTHER :         Qt.Key_V,
 CUTSELECTED  :         Qt.Key_O,
 PASTECHILDSELECTED   : Qt.Key_I,
 PASTEBROTHERSELECTED : Qt.Key_K,

 USERFLAG_0:            Qt.Key_0,
 USERFLAG_1:            Qt.Key_1,
 USERFLAG_2:            Qt.Key_2,
 USERFLAG_3:            Qt.Key_3,
 USERFLAG_4:            Qt.Key_4,
 USERFLAG_5:            Qt.Key_5,
 USERFLAG_6:            Qt.Key_6,
 USERFLAG_7:            Qt.Key_7,
 USERFLAG_8:            Qt.Key_8,
 USERFLAG_9:            Qt.Key_9,  
    
}

EDITKEYMAPPINGS=[
    KEYMAPPING[NEWCHILDNODE],
    KEYMAPPING[NEWBROTHERNODE],
    KEYMAPPING[COPY],
    KEYMAPPING[CUT],
    KEYMAPPING[PASTECHILD],
    KEYMAPPING[PASTEBROTHER],
    KEYMAPPING[CUTSELECTED],
    KEYMAPPING[PASTECHILDSELECTED],
    KEYMAPPING[PASTEBROTHERSELECTED],
    ]

USERKEYMAPPINGS=[ KEYMAPPING[USERFLAG_0],KEYMAPPING[USERFLAG_1],
                  KEYMAPPING[USERFLAG_2],KEYMAPPING[USERFLAG_3],
                  KEYMAPPING[USERFLAG_4],KEYMAPPING[USERFLAG_5],
                  KEYMAPPING[USERFLAG_6],KEYMAPPING[USERFLAG_7],
                  KEYMAPPING[USERFLAG_8],KEYMAPPING[USERFLAG_9] ]

ALLKEYMAPPINGS=[KEYMAPPING[v] for v in KEYMAPPING]

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
    def clearLastEntered(self):
        if (self._control is not None): return self._control.clearLastEntered()
        return None
    def getLastEntered(self):
        if (self._control is not None): return self._control.getLastEntered()
        return None
    def setLastEntered(self,ix=None):
        if (self._control is not None): self._control.setLastEntered(ix)
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
        kmod=event.modifiers()
        kval=event.key()
        if (kval not in ALLKEYMAPPINGS): return
        last=self.getLastEntered()
        if (last is not None):
          nix=self._model.indexByPath(last.sidsPath())
          if (not nix.isValid()):
              if (last.sidsType()==CGK.CGNSTree_ts):
                 if (kval==KEYMAPPING[PASTECHILD]):
                     self._model.pasteAsChild(last)
                 if (kval==KEYMAPPING[NEWCHILDNODE]):
                     self._model.newNodeChild(last)
              return
          if (kval in EDITKEYMAPPINGS):
              if (kmod==Qt.ControlModifier):
                if (kval==KEYMAPPING[NEWCHILDNODE]):
                  self._model.newNodeChild(last)
                if (kval==KEYMAPPING[NEWBROTHERNODE]):
                  self._model.newNodeBrother(last)
                if (kval==KEYMAPPING[COPY]):
                  self._model.copyNode(last)
                if (kval==KEYMAPPING[CUT]):
                  self._model.cutNode(last)
                if (kval==KEYMAPPING[PASTECHILD]):
                  self._model.pasteAsChild(last)
                if (kval==KEYMAPPING[PASTEBROTHER]):
                  self._model.pasteAsBrother(last)
                if (kval==KEYMAPPING[CUTSELECTED]):
                  self._model.cutAllSelectedNodes()
                if (kval==KEYMAPPING[PASTECHILDSELECTED]):
                  self._model.pasteAsChildAllSelectedNodes()
                if (kval==KEYMAPPING[PASTEBROTHERSELECTED]):
                  self._model.pasteAsBrotherAllSelectedNodes()
          elif (kval in USERKEYMAPPINGS):
              last.setUserState(kval-48)
              self._model.refreshModel(nix)
          elif (kval==KEYMAPPING[EDITNODE]):
              if (kmod==Qt.ControlModifier):
                  eix=self._model.createIndex(nix.row(),COLUMN_SIDS,
                                              nix.internalPointer())
                  self.edit(eix)
              elif (kmod==Qt.ShiftModifier):
                  eix=self._model.createIndex(nix.row(),COLUMN_VALUE,
                                              nix.internalPointer())
                  self.edit(eix)
              else:
                  self.edit(nix)
          elif (kval==KEYMAPPING[MARKNODE]):
              self.markNode(last)
          elif (kval==KEYMAPPING[UPNODE]):
              if   (kmod==Qt.ControlModifier): self.upRowLevel(nix)
              elif (kmod==Qt.ShiftModifier):   self.upRowMarked()
              else: QTreeView.keyPressEvent(self,event)
          elif (kval==KEYMAPPING[DOWNNODE]):
              if (kmod==Qt.ControlModifier): self.downRowLevel(nix)
              elif (kmod==Qt.ShiftModifier): self.downRowMarked()
              else: QTreeView.keyPressEvent(self,event)
          elif (kval==KEYMAPPING[OPENFORM]):
              self._parent.formview()
          elif (kval==KEYMAPPING[OPENVIEW]):
              self._parent.openSubTree()
          if (nix.isValid()): self.scrollTo(nix)
    def refreshView(self):
        ixc=self.currentIndex()
        self._model.refreshModel(ixc)
    def markNode(self,node):
        node.switchMarked()
        node._model.updateSelected()
        self.changeRow(node)
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
    def changeRow(self,nodeitem):
        pix=self._model.indexByPath(nodeitem.sidsPath()).parent()
        row=pix.row()
        ix1=self._model.createIndex(row,0,nodeitem)
        ix2=self._model.createIndex(row,COLUMN_LAST,nodeitem)
        self._model.dataChanged.emit(ix1,ix2)
    def exclusiveSelectRow(self,index,setlast=True):
        row=index.row()
        col=index.column()
        parent=index.parent()
        nix=self.model().index(row,col,parent)
        mod=QItemSelectionModel.SelectCurrent|QItemSelectionModel.Rows
        self.selectionModel().setCurrentIndex(nix,mod)
        if (setlast):
            self.clearLastEntered()
            self.setLastEntered(index)
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
def __sortItems(i1,i2):
    c1=i1._model.getSortedChildRow(i1.sidsPath())
    c2=i2._model.getSortedChildRow(i2.sidsPath())
    return c1 - c2

# -----------------------------------------------------------------
class Q7TreeItem(object):
    dtype=['MT','I4','I8','R4','R8','C1','LK']
    stype={'MT':0,'I4':4,'I8':8,'R4':4,'R8':8,'C1':1,'LK':0}
    atype={'I4':numpy.int32,'I8':numpy.int64,
           'R4':numpy.float32,'R8':numpy.float64}
    def __init__(self,fgprint,data,model,tag="",parent=None):
        self._parentitem=parent  
        self._itemnode=data  
        self._childrenitems=[]
        self._title=COLUMN_TITLE
        if (parent is not None): self._path=parent.sidsPath()+'/'+data[0]
        else:                    self._path=''
        self._depth=self._path.count('/')
        self._size=None
        self._fingerprint=fgprint
        self._control=self._fingerprint.control
        self._states={'mark':STMARKOFF,'check':STCHKUNKN,
                      'user':STUSR_X,'shared':STSHRUNKN}
        self._model=model
        self._tag=tag
        if ((parent is not None) and (model is not None)):
            self._model._extension[self._path]=self
            self._nodes=len(self._model._extension)
        else:
            self._nodes=0
    def orderTag(self):
        return self._tag+"0"*(self._fingerprint.depth*4-len(self._tag))
    def sidsParent(self):
        return self._parentitem._itemnode
    def sidsPathSet(self,path):
        del self._model._extension[self._path]
        self._path=path
        self._model._extension[self._path]=self
    def sidsPath(self):
        return self._path
    def sidsName(self):
        return self._itemnode[0]
    def sidsNameSet(self,name):
        if (type(name) not in [str,unicode]): return False
        name=str(name)
        if (name==''): return False
        if (not CGU.checkNodeName(name)): return False
        if (not CGU.checkDuplicatedName(self.sidsParent(),name,dienow=False)):
            return False
        if (name==self._itemnode[0]): return False
        self._itemnode[0]=name
        return True
    def sidsValue(self):
        return self._itemnode[1]
    def sidsValueArray(self):
        if (type(self._itemnode[1])==numpy.ndarray): return True
        return False
    def sidsValueFortranOrder(self):
        if (self.sidsValueArray()):
            return numpy.isfortran(self.sidsValue())
        return False
    def sidsValueSet(self,value):
        try:
            aval=numpy.array([float(value)])
            self._itemnode[1]=aval
            return True
        except ValueError:
            pass
        try:
            aval=numpy.array([int(value)])
            self._itemnode[1]=aval
            return True
        except ValueError:
            pass
        try:
            aval=numpy.array(eval(value))
            self._itemnode[1]=aval
            return True
        except (ValueError, SyntaxError, NameError ):
            pass
        aval=CGU.setStringAsArray(value)
        self._itemnode[1]=aval
        return True
    def sidsChildren(self):
        return self._itemnode[2]
    def sidsType(self):
        return self._itemnode[3]
    def sidsTypeSet(self,value):
        if (type(value) not in [str,unicode]): return False
        value=str(value)
        if (value not in self.sidsTypeList()): return False
        if (value == self._itemnode[3]): return False
        self._itemnode[3]=value
        return True
    def sidsDataType(self,all=False):
        if (all): return Q7TreeItem.dtype
        return CGU.getValueDataType(self._itemnode)
    def sidsDataTypeSet(self,value):
        if (value not in Q7TreeItem.atype): return False
        ndt=Q7TreeItem.atype[value]
        oval=self._itemnode[1]
        nval=numpy.array(oval,dtype=ndt)
        self._itemnode[1]=nval
        return True
    def sidsDataTypeList(self,all=False):
        return Q7TreeItem.dtype
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
        pth=CGU.getPathNoRoot(self.sidsPath())
        if (pth in [lk[-1] for lk in self._fingerprint.links]):
            return STLKTOPOK
        return STLKNOLNK
    def sidsRemoveChild(self,node):
        children=self.sidsChildren()
        idx=0
        while (idx<len(children)):
            childnode=children[idx]
            if ((childnode[0]==node[0]) and (childnode[3]==node[3])): break
            idx+=1
        if (idx<len(children)): children.pop(idx)
    def sidsAddChild(self,node):
        if (node is None):
          ntype=CGK.UserDefinedData_ts
          name='{%s#%.3d}'%(ntype,0)
          newtree=CGU.newNode(name,None,[],ntype)
        else:
          newtree=copy.deepcopy(node)
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
    def sortChildrenInPlace(self):
        self._childrenitems.sort(__sortItems)
    def addChild(self,item,idx):
        self._childrenitems.insert(idx,item)
    def delChild(self,item):
        idx=0
        while (idx<self.childrenCount()):
            if (item==self._childrenitems[idx]): break
            idx+=1
        if (idx<self.childrenCount()): self._childrenitems.pop(idx)
    def children(self):  
        return self._childrenitems
    def child(self,row):  
        return self._childrenitems[row]
    def childRow(self):
        pth=self.sidsPath()
        parentitem=self.parentItem()
        row=0
        for child in parentitem.children():
            if (child.sidsPath()==pth): return row
            row+=1
        return -1
    def hasChildren(self):
        if (self.childrenCount()>0): return True
        return False
    def childrenCount(self):  
        return len(self._childrenitems)  
    def columnCount(self):
        return COLUMN_LAST+1
    def data(self,column):
        if (self._itemnode==None):     return self._title[column]
        if (column==COLUMN_NAME):      return self.sidsName()
        if (column==COLUMN_SIDS):      return self.sidsType()
        if (column==COLUMN_FLAG_LINK): return self.sidsLinkStatus()
        if (column==COLUMN_SHAPE):
            if (self.sidsValue() is None): return None
            return str(self.sidsValue().shape)
        if (column==COLUMN_FLAG_SELECT): return self._states['mark']
        if (column==COLUMN_DATATYPE):    return self.sidsDataType()
        if (column==COLUMN_FLAG_CHECK):  return self._states['check']
        if (column==COLUMN_FLAG_USER):   return self._states['user']
        if (column==COLUMN_VALUE):
            if (self.sidsValue() is None): return None
            if (type(self.sidsValue())==numpy.ndarray):
                vsize=reduce(lambda x,y: x*y, self.sidsValue().shape)
                if (vsize>OCTXT.MaxLengthDataDisplay):
                    return HIDEVALUE
                if (self.sidsValue().dtype.char in ['S','c']):
                    if (len(self.sidsValue().shape)==1):
                        return self.sidsValue().tostring()
                    if (len(self.sidsValue().shape)>2):
                        return HIDEVALUE
                    # TODO: use qtextedit for multiple lines
                    #v=self.sidsValue().T
                    #v=numpy.append(v,numpy.array([['\n']]*v.shape[0]),1)
                    #v=v.tostring()
                    #return v
                    return HIDEVALUE
                if ((self.sidsValue().shape==(1,)) and OCTXT.Show1DAsPlain):
                    return str(self.sidsValue()[0])
            return str(self.sidsValue().tolist())[:100]
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
    def setCheck(self,check):
        if (check in STCHKLIST): self._states['check']=check
    def userState(self):
        return self._states['user']
    def setUserState(self,s):
        try:
            if (int(s) not in range(10)): return
        except ValueError: return
        state=STUSR_P%int(s)
        if (self._states['user'] == state): self._states['user']=STUSR_X
        else: self._states['user']=state

SORTTAG="%.4x"

# -----------------------------------------------------------------
class Q7TreeModel(QAbstractItemModel):
    _icons={}
    def __init__(self,fgprint,parent=None):
        QAbstractItemModel.__init__(self,parent)  
        self._extension={}
        self._rootitem=Q7TreeItem(fgprint,(None),None)  
        self._fingerprint=fgprint
        self._slist=OCTXT._SortedTypeList
        self._count=0
        self._movedPaths={}
        self._fingerprint.version=CGU.getVersion(self._fingerprint.tree)
        self.parseAndUpdate(self._rootitem,
                            self._fingerprint.tree,
                            QModelIndex(),0)
        fgprint.model=self
        for ik in ICONMAPPING:
            Q7TreeModel._icons[ik]=QIcon(QPixmap(ICONMAPPING[ik]))
        self._selected=[]
        self._selectedIndex=-1
        self._control=self._fingerprint.control
    def nodeFromPath(self,path):
        if (path in self._extension.keys()): return self._extension[path]
        return None
    def modifiedPaths(self,node,parentpath,newname,oldname):
        self._movedPaths={}
        newpath=parentpath+'/'+newname
        oldpath=parentpath+'/'+oldname
        oldplen=len(oldpath)
        node.sidsPathSet(newpath)
        plist=list(self._selected)
        for p in plist:
            if (CGU.hasSameRootPath(oldpath,p)):
                self._selected.remove(p)
                self._selected+=[str(newpath+p[oldplen:])]
        for p in self._extension:
            if (CGU.hasSameRootPath(oldpath,p)):
                pn=str(newpath+p[oldplen:])
                nd=self._extension[p]
                nd.sidsPathSet(pn)
                self._movedPaths[p]=pn
        node.parentItem().sortChildrenInPlace()
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
    def checkSelected(self):
        return self.checkTree(self._fingerprint.tree,self._selected)
    def checkClear(self):
        for k in self._extension:
           self._extension[k]._states['check']=STCHKUNKN
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
            and (index.column() not in COLUMN_ICO)): return None
        item = index.internalPointer()
        disp = item.data(index.column())
        if ((index.column()==COLUMN_VALUE) and (role == Qt.DecorationRole)):
            if (disp == HIDEVALUE):
                disp=QIcon(QPixmap(":/images/icons/data-array-large.gif"))
            else:
                return None
        if ((index.column()==COLUMN_VALUE) and (role == Qt.DisplayRole)):
            if (disp == HIDEVALUE):
                return None
        if ((index.column()==COLUMN_FLAG_USER) and (role == Qt.DisplayRole)):
             return None
        if (disp in ICONMAPPING.keys()):
            disp=Q7TreeModel._icons[disp]
        return disp
    def flags(self, index):  
        if (not index.isValid()):  return Qt.NoItemFlags  
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
    def headerData(self, section, orientation, role):  
        if ((orientation == Qt.Horizontal) and (role == Qt.DisplayRole)):  
            return self._rootitem.data(section)  
        return None
    def indexByPath(self, path):
        if (path in self._movedPaths):
            npath=self._movedPaths[path]
            path=npath
        row=self.getSortedChildRow(path)
        col=COLUMN_NAME
        ix=self.createIndex(row, col, self.nodeFromPath(path))
        if (not ix.isValid()):
            return QModelIndex()
        return ix
    def index(self, row, column, parent):
        if (not self.hasIndex(row, column, parent)):
            return QModelIndex()  
        if (not parent.isValid()):
            parentitem = self._rootitem
        else:
            parentitem = parent.internalPointer()
        childitem = parentitem.child(row)  
        if (childitem):
            return self.createIndex(row, column, childitem)
        return QModelIndex()  
    def parent(self, index):  
        if (not index.isValid()): return QModelIndex()
        childitem = index.internalPointer()
        if (childitem is None): return QModelIndex()
        parentitem = childitem.parentItem()
        if (parentitem is None): return QModelIndex()
        return self.createIndex(parentitem.row(), 0, parentitem)  
    def rowCount(self, parent):  
        if (parent.column() > 0): return 0  
        if (not parent.isValid()): parentitem = self._rootitem  
        else:                      parentitem = parent.internalPointer()  
        if (type(parentitem)==type(QModelIndex())): return 0
        return parentitem.childrenCount()  
    def getSortedChildRow(self,path):
        npath=CGU.getPathNoRoot(path)
        if (npath=='/'): return -1
        targetname=CGU.getPathLeaf(path)
        parentpath=CGU.getPathAncestor(path)
        node=CGU.getNodeByPath(self._fingerprint.tree,parentpath)
        if (node is None):
            node=self._fingerprint.tree
        row=0
        for childnode in CGU.getNextChildSortByType(node,criteria=self._slist):
            if (childnode[0]==targetname):
                return row
            row+=1
        return -1
    def setData(self,index,value,role):
        if (   (value is None)
            or (role!=Qt.EditRole)
            or (not index.isValid())
            or (index.column() not in COLUMN_EDIT)):
            return
        node=index.internalPointer()
        st=False
        if (index.column()==COLUMN_NAME):
            oldname=node.sidsName()
            newname=value
            parentpath=node.parentItem().sidsPath()
            st=node.sidsNameSet(value)
            if (st):
                self.modifiedPaths(node,parentpath,newname,oldname)
                nindex=self.indexByPath(parentpath+'/'+newname)
                self.rowsMoved.emit(index.parent(),index.row(),index.row(),
                                    nindex,nindex.row())
                self.refreshModel(nindex.parent())
                self.refreshModel(nindex)
        if (index.column()==COLUMN_SIDS):     st=node.sidsTypeSet(value)
        if (index.column()==COLUMN_VALUE):    st=node.sidsValueSet(value)
        if (index.column()==COLUMN_DATATYPE): st=node.sidsDataTypeSet(value)
        if (st):
            self._fingerprint.modifiedTreeStatus(Q7fingerPrint.STATUS_MODIFIED)
    def removeItem(self,parentitem,targetitem,row):
        parentindex=self.indexByPath(parentitem.sidsPath())
        self.beginRemoveRows(parentindex,row,row)
        path=targetitem.sidsPath()
        parentitem.delChild(targetitem)
        del self._extension[path]
        self.endRemoveRows()
    def removeItemTree(self,nodeitem):
        self.parseAndRemove(nodeitem)
        row=nodeitem.childRow()
        self.removeItem(nodeitem.parentItem(),nodeitem,row)
    def parseAndRemove(self,parentitem):
        if (not parentitem.hasChildren()): return
        while (parentitem.childrenCount()!=0):
            r=parentitem.childrenCount()
            child=parentitem.child(r-1)
            self.parseAndRemove(child)
            self.removeItem(parentitem,child,r-1)
    def parseAndUpdate(self,parentItem,node,parentIndex,row,parenttag=""):
        self._count+=1
        tag=parenttag+SORTTAG%self._count
        newItem=Q7TreeItem(self._fingerprint,(node),self,tag,parentItem)
        self._fingerprint.depth=max(newItem._depth,self._fingerprint.depth)
        self._fingerprint.nodes=max(newItem._nodes,self._fingerprint.nodes)
        self.beginInsertRows(parentIndex,row,row)
        parentItem.addChild(newItem,row)
        self.endInsertRows()
        parentItem.sortChildrenInPlace()
        newIndex=self.createIndex(row,0,newItem)
        crow=0
        for childnode in CGU.getNextChildSortByType(node,criteria=self._slist):
            c=self.parseAndUpdate(newItem,childnode,newIndex,crow,tag)
            self._fingerprint.depth=max(c._depth,self._fingerprint.depth)
            self._fingerprint.nodes=max(c._nodes,self._fingerprint.nodes)
            crow+=1
        return newItem
    def refreshModel(self,nodeidx):
        if (not nodeidx.isValid()): return
        row=nodeidx.row()
        dlt=2
        parentidx=nodeidx.parent()
        maxrow=self.rowCount(parentidx)
        row1=min(0,abs(row-dlt))
        row2=min(row+dlt,maxrow)
        ix1=self.createIndex(row1,0,parentidx.internalPointer())
        ix2=self.createIndex(row2,COLUMN_LAST,parentidx.internalPointer())
        if (ix1.isValid() and ix2.isValid()):
            self.dataChanged.emit(ix1,ix2)
    def newNodeChild(self,nodeitem):
        if (nodeitem is None): return
        row=nodeitem.row()
        nix=self.indexByPath(nodeitem.sidsPath())
        (ntree,npath)=nodeitem.sidsAddChild(None)
        self.parseAndUpdate(nodeitem,ntree,nix,0,nodeitem._tag)
        nix=self.indexByPath(nodeitem.sidsPath())
        pix=self.indexByPath(CGU.getPathAncestor(npath))
        cix=self.indexByPath(npath)
        self.refreshModel(pix)
        self.refreshModel(nix)
        self.refreshModel(cix)
        self._fingerprint.modifiedTreeStatus(Q7fingerPrint.STATUS_MODIFIED)
    def newNodeBrother(self,nodeitem):
        if (nodeitem is None): return
        nix=self.indexByPath(nodeitem.sidsPath())
        self.newNodeChild(nix.parent().internalPointer())
        self._fingerprint.modifiedTreeStatus(Q7fingerPrint.STATUS_MODIFIED)
    def copyNode(self,nodeitem):
        if (nodeitem is None): return
        self._control.copyPasteBuffer=copy.deepcopy(nodeitem._itemnode)
    def cutAllSelectedNodes(self):
        for pth in self._selected:
            nodeitem=self.nodeFromPath(pth)
            self.cutNode(nodeitem)
    def cutNode(self,nodeitem):
        if (nodeitem is None): return
        self._control.copyPasteBuffer=copy.deepcopy(nodeitem._itemnode)
        parentitem=nodeitem.parentItem()
        path=CGU.getPathAncestor(nodeitem.sidsPath())
        self.removeItemTree(nodeitem)
        pix=self.indexByPath(path)
        parentitem.sidsRemoveChild(self._control.copyPasteBuffer)
        self.refreshModel(pix)
        self._fingerprint.modifiedTreeStatus(Q7fingerPrint.STATUS_MODIFIED)
    def pasteAsChildAllSelectedNodes(self):
        for pth in self._selected:
            nodeitem=self.nodeFromPath(pth)
            self.pasteAsChild(nodeitem)
    def pasteAsChild(self,nodeitem):
        if (nodeitem is None): return
        if (self._control.copyPasteBuffer is None): return
        row=nodeitem.row()
        nix=self.indexByPath(nodeitem.sidsPath())
        (ntree,npath)=nodeitem.sidsAddChild(self._control.copyPasteBuffer)
        self.parseAndUpdate(nodeitem,ntree,nix,0,nodeitem._tag)
        nix=self.indexByPath(nodeitem.sidsPath())
        pix=self.indexByPath(CGU.getPathAncestor(npath))
        cix=self.indexByPath(npath)
        self.refreshModel(pix)
        self.refreshModel(nix)
        self.refreshModel(cix)
        self._fingerprint.modifiedTreeStatus(Q7fingerPrint.STATUS_MODIFIED)
    def pasteAsBrotherAllSelectedNodes(self):
        for pth in self._selected:
            nodeitem=self.nodeFromPath(pth)
            self.pasteAsBrother(nodeitem)
    def pasteAsBrother(self,nodeitem):
        if (nodeitem is None): return
        nix=self.indexByPath(nodeitem.sidsPath())
        self.pasteAsChild(nix.parent().internalPointer())
        self._fingerprint.modifiedTreeStatus(Q7fingerPrint.STATUS_MODIFIED)
    def checkTree(self,T,pathlist):
        checkdiag=CGV.checkTree(T)
        for path in pathlist:
            pth=CGU.getPathNoRoot(path)
            if (checkdiag.has_key(pth)):
                item=self._extension[path]
                stat=checkdiag[pth][0]
                if (stat==CGV.CHECK_NONE): item.setCheck(STCHKUNKN)
                if (stat==CGV.CHECK_GOOD): item.setCheck(STCHKGOOD)
                if (stat==CGV.CHECK_FAIL): item.setCheck(STCHKFAIL)
                if (stat==CGV.CHECK_WARN): item.setCheck(STCHKWARN)
                if (stat==CGV.CHECK_USER): item.setCheck(STCHKUSER)
        return checkdiag
    def hasUserColor(self,k):
        cl=OCTXT.UserColors
        c=k[-3]
        if (cl[int(c)] is None): return False
        return True
    def getUserColor(self,k):
        cb=Qt.black
        cl=OCTXT.UserColors
        c=int(k[-3])
        return QColor(cl[c])

# -----------------------------------------------------------------
