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

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

Q_OR    ='or'
Q_AND   ='and'
Q_ORNOT ='or not'
Q_ANDNOT='and not'

Q_OPERATOR=(Q_OR,Q_AND,Q_ORNOT,Q_ANDNOT)

Q_PARENT  ='Parent'
Q_NODE    ='Node'
Q_CHILDREN='Children'

Q_TARGET=(Q_PARENT,Q_NODE,Q_CHILDREN)

Q_CGNSTYPE='CGNS type'
Q_VALUETYPE='Value type'
Q_NAME='Name'
Q_VALUE='Value'
Q_SCRIPT='Python script'

Q_ATTRIBUTE=(Q_CGNSTYPE,Q_VALUETYPE,Q_NAME,Q_VALUE,Q_SCRIPT)

Q_VAR_NAME='NAME'
Q_VAR_VALUE='VALUE'
Q_VAR_CGNSTYPE='CGNSTYPE'
Q_VAR_CHILDREN='CHILDREN'
Q_VAR_TREE='TREE'
Q_VAR_PATH='PATH'
Q_VAR_RESULT='RESULT'
Q_VAR_RESULT_LIST='__Q7_QUERY_RESULT__'

Q_SCRIPT_PRE="""
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils as CGU
"""

Q_SCRIPT_POST="""
%s[0]=%s
"""%(Q_VAR_RESULT_LIST,Q_VAR_RESULT)

DEFAULTQUERIES=[
    {'name':'Families','clauses':[
        (Q_OR,  Q_NODE, Q_CGNSTYPE, CGK.Family_ts)
        ]},
    {'name':'Family names','clauses':[
        (Q_OR,  Q_NODE, Q_CGNSTYPE, CGK.FamilyName_ts)
        ]},
    {'name':'BCs','clauses':[
        (Q_OR,  Q_NODE, Q_CGNSTYPE, CGK.BC_ts)
        ]},
    {'name':'QUADs','clauses':[
        (Q_OR,  Q_NODE, Q_CGNSTYPE,  CGK.Elements_ts),
        (Q_AND, Q_NODE, Q_SCRIPT, 'RESULT=VALUE[0] in (CGK.QUAD_4, CGK.QUAD_8, CGK.QUAD_9)')
        ]},
    {'name':'TRIs','clauses':[
        (Q_OR,  Q_NODE, Q_CGNSTYPE,  CGK.Elements_ts),
        (Q_AND, Q_NODE, Q_SCRIPT, """RESULT=VALUE[0] in (CGK.TRI_3, CGK.TRI_6)""")
        ]},
    {'name':' ','clauses':[
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
        ]},
]
# -----------------------------------------------------------------
def sameVal(n,v):
    if (n==v): return True
    return False
# -----------------------------------------------------------------
def sameValType(n,v):
    if (type(n)!=numpy.ndarray): return False
    if (n.dtype.char==v): return True
    return False
# -----------------------------------------------------------------
def evalScript(node,parent,tree,path,val):
    l=locals()
    l[Q_VAR_RESULT_LIST]=[False]
    l[Q_VAR_NAME]=node[0]
    l[Q_VAR_VALUE]=node[1]
    l[Q_VAR_CGNSTYPE]=node[3]
    l[Q_VAR_CHILDREN]=node[2]
    l[Q_VAR_TREE]=tree
    l[Q_VAR_PATH]=path
    pre=Q_SCRIPT_PRE+val+Q_SCRIPT_POST
    try:
      eval(compile(pre,'<string>','exec'),globals(),l)
    except:
      RESULT=False
    RESULT=l[Q_VAR_RESULT_LIST][0]
    return RESULT
# -----------------------------------------------------------------
def parseAndSelect(tree,node,parent,path,clause):
    Q=False
    path=path+'/'+node[0]
    for (opr,tgt,att,val) in clause:
      P=False
      
      N=(tgt==Q_PARENT)      
      P|=(N and(att==Q_CGNSTYPE)  and (parent[3]==val))
      P|=(N and(att==Q_NAME)      and (parent[0]==val))
      P|=(N and(att==Q_VALUE)     and (sameVal(parent[1],val)))
      P|=(N and(att==Q_VALUETYPE) and (sameValType(parent[1],val)))
      P|=(N and(att==Q_SCRIPT)    and (evalScript(node,parent,tree,path,val)))

      N=(tgt==Q_NODE)      
      P|=(N and(att==Q_CGNSTYPE)  and (node[3]==val))
      P|=(N and(att==Q_NAME)      and (node[0]==val))
      P|=(N and(att==Q_VALUE)     and (sameVal(node[1],val)))
      P|=(N and(att==Q_VALUETYPE) and (sameValType(node[1],val)))
      P|=(N and(att==Q_SCRIPT)    and (evalScript(node,parent,tree,path,val)))

      if (opr==Q_OR):     Q|= P
      if (opr==Q_AND):    Q&= P
      if (opr==Q_ORNOT):  Q|=~P
      if (opr==Q_ANDNOT): Q&=~P
      
    R=[]
    if (Q): R=[path]
    for C in node[2]:
        R+=parseAndSelect(tree,C,node,path,clause)
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
        s="{'name':'%s','clauses':["%self.name
        for c in self.clause:
            c0=Q_ENUMSTRINGS[c[0]]
            c1=Q_ENUMSTRINGS[c[1]]
            c2=Q_ENUMSTRINGS[c[2]]
            s+='(%s,%s,%s,"""%s"""),'%(c0,c1,c2,c[3])
        s+="]},"
        return s
    def run(self,tree):
        result=parseAndSelect(tree,tree,[None,None,[],None],'',self.clause)
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
        self._queries={}
        self._cols=4
        self._parent=parent
        self._previousrows=0
        self.currentQuery=None
        self.fillQueries()
        self.defaultQueriesList=self._queries.keys()
        self.setDefaultQuery()
    def setDefaultQuery(self):
        self.setCurrentQuery(self.defaultQueriesList[0])
    def getCurrentQuery(self):
        return self.currentQuery
    def setCurrentQuery(self,name):
        self._previousrows=self.rowCount()
        self.currentQuery=name
        self.removeRows(0,self._previousrows)
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
        if ((self.currentQuery is not None) and (self._queries!={})):
            return self._queries[self.currentQuery].clausecount
        return 0
    def minIndex(self):
        return QModelIndex(self.createIndex(0,0))
    def maxIndex(self):
        maxl=self._queries[self.currentQuery].clausecount
        return QModelIndex(self.createIndex(maxl,self._cols))
    def insertRows(self,row,count):
        q=self._queries[self.currentQuery]
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
        q=self._queries[self.currentQuery]
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
            self._queries[self.currentQuery].clauseChange(r,c,value)
        return value
    def fillQueries(self):
        for qe in DEFAULTQUERIES:
            q=Q7QueryEntry(qe['name'])
            for c in qe['clauses']:
                q.addClause(*c)
            self._queries[qe['name']]=q
    def queries(self):
        return self._queries
    def setDelegates(self,tableview,editframe):
        opd=Q7ComboBoxDelegate(self,Q_OPERATOR)
        tgd=Q7ComboBoxDelegate(self,Q_TARGET)
        atd=Q7ComboBoxDelegate(self,Q_ATTRIBUTE)
        scd=Q7EditBoxDelegate(self,editframe)
        tableview.setItemDelegateForColumn(0,opd)
        tableview.setItemDelegateForColumn(1,tgd)
        tableview.setItemDelegateForColumn(2,atd)
        tableview.setItemDelegateForColumn(3,scd)

# -----------------------------------------------------------------
