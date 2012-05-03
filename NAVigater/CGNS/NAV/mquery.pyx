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

from CGNS.NAV.moption import Q7OptionContext as OCTXT
import CGNS.NAV.moption as OCST

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

BLANKQUERYKEY=' '
BLANKQUERY=[[ BLANKQUERYKEY, # name
             [   # clause
                 (' '*8,' '*16,' '*16,' '),
                 (' ',' ',' ',' '),
                 (' ',' ',' ',' '),
                 (' ',' ',' ',' '),
                 (' ',' ',' ',' '),
                 (' ',' ',' ',' '),
                 (' ',' ',' ',' '),
                 (' ',' ',' ',' '),
                 (' ',' ',' ',' '),
                 (' ',' ',' ',' '),
                 ]]]
# -----------------------------------------------------------------
def sameVal(n,v):
    if (n is None):
        if (v is None): return True
        return False
    if (n.dtype.char in ['S','c']): return (n.tostring() == v)
    if (n.dtype.char in ['d','f','i','l']): return (n.flat[0] == v)
    if (n==v): return True
    return False
# -----------------------------------------------------------------
def sameValType(n,v):
    if (type(n)!=numpy.ndarray): return False
    if (n.dtype.char==v): return True
    return False
# -----------------------------------------------------------------
def evalScript(node,parent,tree,path,val,args):
    l=locals()
    l[OCST.Q_VAR_RESULT_LIST]=[False]
    l[OCST.Q_VAR_NAME]=node[0]
    l[OCST.Q_VAR_VALUE]=node[1]
    l[OCST.Q_VAR_CGNSTYPE]=node[3]
    l[OCST.Q_VAR_CHILDREN]=node[2]
    l[OCST.Q_VAR_TREE]=tree
    l[OCST.Q_VAR_PATH]=path
    l[OCST.Q_VAR_USER]=args
    pre=OCST.Q_SCRIPT_PRE+val+OCST.Q_SCRIPT_POST
    try:
      eval(compile(pre,'<string>','exec'),globals(),l)
    except IndexError:
      RESULT=False
    RESULT=l[OCST.Q_VAR_RESULT_LIST][0]
    return RESULT
# -----------------------------------------------------------------
def parseAndSelect(tree,node,parent,path,clause,args):
    Q=False
    path=path+'/'+node[0]
    for (opr,tgt,att,val) in clause:
      P=False
      
      N=(tgt==OCST.Q_PARENT)      
      P|=(N and(att==OCST.Q_CGNSTYPE)  and (parent[3]==val))
      P|=(N and(att==OCST.Q_NAME)      and (parent[0]==val))
      P|=(N and(att==OCST.Q_VALUE)     and (sameVal(parent[1],val)))
      P|=(N and(att==OCST.Q_VALUETYPE) and (sameValType(parent[1],val)))
      P|=(N and(att==OCST.Q_SCRIPT)    and (evalScript(node,parent,tree,
                                                       path,val,args)))

      N=(tgt==OCST.Q_NODE)      
      P|=(N and(att==OCST.Q_CGNSTYPE)  and (node[3]==val))
      P|=(N and(att==OCST.Q_NAME)      and (node[0]==val))
      P|=(N and(att==OCST.Q_VALUE)     and (sameVal(node[1],val)))
      P|=(N and(att==OCST.Q_VALUETYPE) and (sameValType(node[1],val)))
      P|=(N and(att==OCST.Q_SCRIPT)    and (evalScript(node,parent,tree,
                                                       path,val,args)))

      if (opr==OCST.Q_OR):     Q|= P
      if (opr==OCST.Q_AND):    Q&= P
      if (opr==OCST.Q_ORNOT):  Q|=~P
      if (opr==OCST.Q_ANDNOT): Q&=~P
      
    R=[]
    if (Q): R=[path]
    for C in node[2]:
        R+=parseAndSelect(tree,C,node,path,clause,args)
    return R

# -----------------------------------------------------------------
class Q7QueryEntry(object):
    def __init__(self,name):
        self.name=name
        self.clause=[]
        self.clausecount=0
    def addClause(self,operator,target,attribute,value):
        self.clause.append((operator,target,attribute,value))
        self.clausecount+=1
    def clauseChange(self,r,c,v):
        if (r>self.clausecount): return 0
        p=self.clause[r]
        if   (c==0): p=(v,p[1],p[2],p[3])
        elif (c==1): p=(p[0],v,p[2],p[3])
        elif (c==2): p=(p[0],p[1],v,p[3])
        elif (c==3): p=(p[0],p[1],p[2],v)
        self.clause[r]=p
        return 1
    def __str__(self):
        s="['%s',["%self.name
        for c in self.clause:
            if (c[0]==OCST.Q_OR):       c0="CGO.Q_OR"
            if (c[0]==OCST.Q_AND):      c0="CGO.Q_AND"
            if (c[0]==OCST.Q_ORNOT):    c0="CGO.Q_ORNOT"
            if (c[0]==OCST.Q_ANDNOT):   c0="CGO.Q_ANDNOT"
            if (c[1]==OCST.Q_PARENT):   c1="CGO.Q_PARENT"
            if (c[1]==OCST.Q_NODE):     c1="CGO.Q_NODE"
            if (c[1]==OCST.Q_CHILDREN): c1="CGO.Q_CHILDREN"
            if (c[2]==OCST.Q_CGNSTYPE): c2="CGO.Q_CGNSTYPE"
            if (c[2]==OCST.Q_VALUETYPE):c2="CGO.Q_VALUETYPE"
            if (c[2]==OCST.Q_NAME):     c2="CGO.Q_NAME"
            if (c[2]==OCST.Q_VALUE):    c2="CGO.Q_VALUE"
            if (c[2]==OCST.Q_SCRIPT):   c2="CGO.Q_SCRIPT"
            s+='(%s,%s,%s,"""%s"""),'%(c0,c1,c2,c[3])
        s+="]]"
        return s
    def run(self,tree,*args):
        self.args=args
        result=parseAndSelect(tree,tree,[None,None,[],None],'',
                              self.clause,self.args)
        return result
    
# -----------------------------------------------------------------
class Q7QueryTableView(QTableView):
    def  __init__(self,parent):
        QTableView.__init__(self,parent)
        self._parent=None
       
# -----------------------------------------------------------------
class Q7EditBoxDelegate(QItemDelegate):
     def __init__(self, owner, editwidget):
         QItemDelegate.__init__(self, owner)
         self.edit=editwidget
         self.parent=owner
     def createEditor(self, parent, option, index):
         editor = QPlainTextEdit(self.edit)
         fs=self.edit.frameSize()
         editor.transgeometry=(0,0,fs.width(),fs.height())
         editor.installEventFilter(self)
         self.setEditorData(editor,index)
         return editor
     def setEditorData(self, editor, index):
         value = index.data()
         editor.clear()
         editor.insertPlainText(value)
     def setModelData(self,editor,model,index):
         value = editor.toPlainText()
         model.setData(index,value,role=Qt.EditRole)
     def updateEditorGeometry(self, editor, option, index):
         editor.setGeometry(*editor.transgeometry)

# -----------------------------------------------------------------
class Q7ComboBoxDelegate(QItemDelegate):
     def __init__(self, owner, itemslist):
         QItemDelegate.__init__(self, owner)
         self.itemslist = itemslist
         self.parent=owner
     def createEditor(self, parent, option, index):
         editor = QComboBox(parent)
         editor.addItems(self.itemslist)
         editor.setCurrentIndex(0)
         editor.installEventFilter(self)
         self.setEditorData(editor,index)
         return editor
     def setEditorData(self, editor, index):
         value = index.data()
         ix=editor.findText(value)
         if (ix!=-1): editor.setCurrentIndex(ix)
     def setModelData(self,editor,model,index):
         value = editor.currentText()
         model.setData(index,value,role=Qt.EditRole)

# -----------------------------------------------------------------
class Q7QueryTableModel(QAbstractTableModel):  
    def __init__(self,parent):
        QAbstractTableModel.__init__(self,parent)
        self._allQueries={}
        self._cols=4
        self._parent=parent
        self._previousRows=0
        self._currentQuery=None
        self.loadUserQueries()
        self.fillQueries()
        self.setCurrentQuery(self.queriesNamesList()[0]) 
    def getCurrentQuery(self):
        return self._currentQuery
    def setCurrentQuery(self,name):
        self._previousRows=self.rowCount()
        self._currentQuery=name
        self.removeRows(0,self._previousRows)
        self.insertRows(0,self.rowCount())
        sig=SIGNAL("dataChanged(const QModelIndex&, const QModelIndex &)")
        imin=self.minIndex()
        imax=self.maxIndex()
        QObject.emit(self,sig,imin,imax)
    def refreshRows(self,view):
        pass
    def columnCount(self, parent):
        return self._cols
    def rowCount(self, parent=None):
        if ((self._currentQuery is not None) and (self._allQueries!={})):
            return self._allQueries[self._currentQuery].clausecount
        return 0
    def minIndex(self):
        return QModelIndex(self.createIndex(0,0))
    def maxIndex(self):
        maxl=self._allQueries[self._currentQuery].clausecount
        return QModelIndex(self.createIndex(maxl,self._cols))
    def insertRows(self,row,count):
        q=self._allQueries[self._currentQuery]
        for r in range(count):
            for c in range(self._cols):
                self.setData(self.createIndex(r,c),q.clause[r][c])
    def removeRows(self,row,count):
        for r in range(count):
            for c in range(self._cols):
                self.setData(self.createIndex(r,c),'')
    def data(self, index, role=Qt.DisplayRole):
        l=index.row()
        c=index.column()
        if (role not in (Qt.EditRole,Qt.DisplayRole)) :return
        q=self._allQueries[self._currentQuery]
        if ((c>self._cols) or (l>q.clausecount)): return
        return q.clause[l][c]
    def flags(self, index):
        if (index.column() in range(self._cols)):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        return QAbstractTableModel.flags(self, index)
    def setData(self, index, value, role = Qt.DisplayRole):
        r=index.row()
        c=index.column()
        if (role==Qt.EditRole):
            self._allQueries[self._currentQuery].clauseChange(r,c,value)
        return value
    def fillQueries(self):
        allqueriestext=self._userQueriesText+OCTXT._UsualQueriesText+BLANKQUERY
        self._defaultQueriesNames=[n[0] for n in OCTXT._UsualQueriesText]
        self._defaultQueriesNames+=[BLANKQUERYKEY]
        for qe in allqueriestext:
            q=Q7QueryEntry(qe[0])
            for c in qe[1]:
                q.addClause(*c)
            self._allQueries[qe[0]]=q
    def getQuery(self,name):
        if (name in self.queriesNamesList()): return self._allQueries[name]
        return None
    def loadUserQueries(self):
        self._userQueriesText=OCTXT._readQueries(self)
        if (self._userQueriesText is None): self._userQueriesText=[]
        return self._userQueriesText
    def saveUserQueries(self):
        ql=[]
        for q in self._allQueries:
            if (q not in self._defaultQueriesNames):
                ql+=[str(self._allQueries[q])]
        OCTXT._writeQueries(self,ql)
    def queriesNamesList(self):
        k=self._allQueries.keys()
        k.sort()
        return k
    def setDelegates(self,tableview,editframe):
        opd=Q7ComboBoxDelegate(self,OCST.Q_OPERATOR)
        tgd=Q7ComboBoxDelegate(self,OCST.Q_TARGET)
        atd=Q7ComboBoxDelegate(self,OCST.Q_ATTRIBUTE)
        scd=Q7EditBoxDelegate(self,editframe)
        tableview.setItemDelegateForColumn(0,opd)
        tableview.setItemDelegateForColumn(1,tgd)
        tableview.setItemDelegateForColumn(2,atd)
        tableview.setItemDelegateForColumn(3,scd)

# -----------------------------------------------------------------
