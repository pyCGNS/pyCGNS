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

DEFAULTQUERIES=[
    {'name':'BC type','clauses':[
        (Q_OR,  Q_NODE, Q_CGNSTYPE, CGK.BC_ts)
        ]},
    {'name':'QUADs','clauses':[
        (Q_OR,  Q_NODE, Q_CGNSTYPE,  CGK.Elements_ts),
        (Q_AND, Q_NODE, Q_SCRIPT, 'VALUE[0] in (CGK.QUAD_4, CGK.QUAD_8, CGK.QUAD_9)')
        ]},
]

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
        result=[]
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
         model.setData(index,value)

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
         model.setData(index,value)

# -----------------------------------------------------------------
class Q7QueryTableModel(QAbstractTableModel):  
    def __init__(self,parent):  
        QAbstractTableModel.__init__(self,parent)
        self._tableview=parent.querytableview
        self._edit=parent.editFrame
        self._queries={}
        self._cols=4
        self._current='QUADs'
        self.fillQueries()
        self.resizeAll()
    def columnCount(self, parent):
        return self._cols
    def rowCount(self, parent):
        if (self._queries!={}):
            return self._queries[self._current].clausecount
        return 0
    def index(self, row, column, parent):
        return self.createIndex(row, column, 0)  
    def data(self, index, role=Qt.DisplayRole):
        l=index.row()
        c=index.column()
        if (role not in (Qt.EditRole,Qt.DisplayRole)) :return
        q=self._queries[self._current]
        if ((c>self._cols) or (l>q.clausecount)): return
        return q.clause[l][c]
    def flags(self, index):
        if (index.column() in range(self._cols)):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        return QAbstractTableModel.flags(self, index)
    def setData(self, index, value, role = Qt.DisplayRole):
        r=index.row()
        c=index.column()
        self._queries[self._current].clauseChange(r,c,value)
        self.resizeAll()
        return value
    def resizeAll(self):
        for c in range(self._cols):
            self._tableview.resizeColumnToContents(c)
    def fillQueries(self):
        for qe in DEFAULTQUERIES:
            q=Q7QueryEntry(qe['name'])
            for c in qe['clauses']:
                q.addClause(*c)
            self._queries[qe['name']]=q
    def setDelegates(self,tableview):
        opd=Q7ComboBoxDelegate(self,Q_OPERATOR)
        tgd=Q7ComboBoxDelegate(self,Q_TARGET)
        atd=Q7ComboBoxDelegate(self,Q_ATTRIBUTE)
        scd=Q7EditBoxDelegate(self,self._edit)
        tableview.setItemDelegateForColumn(0,opd)
        tableview.setItemDelegateForColumn(1,tgd)
        tableview.setItemDelegateForColumn(2,atd)
        tableview.setItemDelegateForColumn(3,scd)

# -----------------------------------------------------------------
