#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import sys
import numpy
import os

from PySide.QtCore  import *
from PySide.QtGui   import QFileDialog
from PySide.QtGui   import *
from CGNS.NAV.Q7QueryWindow import Ui_Q7QueryWindow
from CGNS.NAV.Q7SelectionWindow import Ui_Q7SelectionWindow
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.mtree import COLUMN_VALUE,COLUMN_DATATYPE,COLUMN_SIDS,COLUMN_NAME
from CGNS.NAV.mtree import HIDEVALUE
from CGNS.NAV.moption import Q7OptionContext as OCTXT
from CGNS.NAV.mquery import Q7QueryEntry

import CGNS.NAV.wmessages as MSG

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

(CELLCOMBO,CELLTEXT)=range(2)
CELLEDITMODE=(CELLCOMBO,CELLTEXT)

# -----------------------------------------------------------------
class Q7SelectionItemDelegate(QStyledItemDelegate):
    def __init__(self,owner,model):
        QStyledItemDelegate.__init__(self,owner.selectionTable)
        self._parent=owner
        self._mode=CELLTEXT
        self._model=model
        self._lastCol=None
    def createEditor(self, parent, option, index):
        tnode=self._model.nodeFromPath(self._parent._data[index.row()])
        if (tnode.sidsIsCGNSTree()): return None
        if (tnode.sidsIsLink()): return None
        if (tnode.sidsIsLinkChild()): return None
        ws=option.rect.width()
        hs=option.rect.height()+4
        xs=option.rect.x()
        ys=option.rect.y()-2
        if (index.column()==2):
          self._lastCol=COLUMN_SIDS
          self._mode=CELLCOMBO
          editor=QComboBox(parent)
          editor.transgeometry=(xs,ys,ws,hs)
          itemslist=tnode.sidsTypeList()
          editor.addItems(itemslist)
          try:
              tix=itemslist.index(tnode.sidsType())
          except ValueError:
              editor.insertItem(0,tnode.sidsType())
              tix=0
          editor.setCurrentIndex(tix)
          editor.installEventFilter(self)
          self.setEditorData(editor,index)
          return editor
        if (index.column()==3):
          self._lastCol=COLUMN_DATATYPE
          self._mode=CELLCOMBO
          editor=QComboBox(parent)
          editor.transgeometry=(xs,ys,ws,hs)
          itemslist=tnode.sidsDataType(all=True)
          editor.addItems(itemslist)
          editor.setCurrentIndex(0)
          editor.installEventFilter(self)
          self.setEditorData(editor,index)
          return editor
        if (index.column()==4):
          self._lastCol=COLUMN_VALUE
          if (tnode.hasValueView()):
            pt=tnode.sidsPath().split('/')[1:]
            lt=tnode.sidsTypePath()
            fc=self._parent._control.userFunctionFromPath(pt,lt)
            if (fc is not None): en=fc.getEnumerate(pt,lt)
            else:                en=tnode.sidsValueEnum()
            if (en is None):
              self._mode=CELLTEXT
              editor=QLineEdit(parent)
              editor.transgeometry=(xs,ys,ws,hs)
            else:
              self._mode=CELLCOMBO
              editor=QComboBox(parent)
              editor.transgeometry=(xs,ys,ws,hs)
              editor.addItems(en)
              try:
                tix=en.index(tnode.sidsValue().tostring())
              except ValueError:
                editor.insertItem(0,tnode.sidsValue().tostring())
                tix=0
              editor.setCurrentIndex(tix)
            editor.installEventFilter(self)
            self.setEditorData(editor,index)
            return editor
        return None
    def setEditorData(self, editor, index):
        if (self._mode==CELLTEXT):
            value=index.data()
            editor.clear()
            editor.insert(value)
        elif (self._mode==CELLCOMBO):
            value = index.data()
            ix=editor.findText(value)
            if (ix!=-1): editor.setCurrentIndex(ix)
        else:
            pass
    def setModelData(self,editor,model,index):
        value=None
        if (self._mode==CELLCOMBO):
            value=editor.currentText()
        if (self._mode==CELLTEXT):
            value=editor.text()
        tnode=self._model.nodeFromPath(self._parent._data[index.row()])
        col=self._lastCol
        if (col is None): return
        nix=self._model.createIndex(index.row(),col,tnode)
        self._model.setData(nix,value,role=Qt.EditRole)
        self._parent.updateRowData(index.row(),self._parent._data[index.row()])
    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(*editor.transgeometry)

# -----------------------------------------------------------------
class Q7SelectionList(Q7Window,Ui_Q7SelectionWindow):
    def __init__(self,parent,model,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_SELECT,
                          parent._control,None,fgprint)
        self.bClose.clicked.connect(self.reject)
        self._parent=parent
        self._model=model
        self._data=model.getSelected()
        self._fgprint=fgprint
        self.bSave.clicked.connect(self.selectionsave)
        self.bInfo.clicked.connect(self.infoSelectView)
        QObject.connect(self.cShowPath,
                        SIGNAL("stateChanged(int)"),self.colCheck)
        QObject.connect(self.cShowSIDS,
                        SIGNAL("stateChanged(int)"),self.colCheck)
        self.bFirst.clicked.connect(self.sClear)
        self.selectionTable.setItemDelegate(Q7SelectionItemDelegate(self,self._fgprint.model))
        self.bRemoveToSelect.clicked.connect(self.sRemove)
        self.bReverse.clicked.connect(self.sReverse)
        self.bSelectAll.clicked.connect(self.sAll)
        self.bUnselectAll.clicked.connect(self.sClear)
    def colCheck(self):
        if (self.cShowPath.checkState()==Qt.Checked):
            self.selectionTable.showColumn(0)
        else:
            self.selectionTable.hideColumn(0)
        if (self.cShowSIDS.checkState()==Qt.Checked):
            self.selectionTable.showColumn(2)
        else:
            self.selectionTable.hideColumn(2)
    def sRemove(self):
        dsel=[]
        for r in self.selectionTable.selectionModel().selectedRows():
            dsel.append(self._data[r.row()])
        for p in dsel:
            self._data.remove(p)
        self.reset()
    def sReverse(self):
        rall=set(range(self.selectionTable.rowCount()))
        rsel=set()
        for r in self.selectionTable.selectionModel().selectedRows():
            rsel.add(r.row())
        runs=rall.difference(rsel)
        self.selectionTable.clearSelection()
        for r in runs:
            self.selectionTable.selectRow(r)
    def sAll(self):
        self.selectionTable.selectAll()
    def sClear(self):
        self.selectionTable.clearSelection()
    def infoSelectView(self):
        self._control.helpWindow('Selection')
    def show(self):
        self.reset()
        super(Q7SelectionList, self).show()
        self.raise_()
    def selectionsave(self):
        n='data=[\n'
        for path in self._data:
            node=self._model.nodeFromPath(path)
            t=node.data(COLUMN_SIDS)
            d=node.data(COLUMN_DATATYPE)
            v=node.sidsValue()
            if (type(v)==numpy.ndarray):
                if (v.dtype.char in ['S','c']):
                    s='"'+v.tostring()+'"'
                else:
                    s=str(v.tolist())
            else:
                s=str(v)
            n+='("%s","%s","%s",%s),\n'%(path,t,d,s)
        n+=']\n'
        filename=QFileDialog.getSaveFileName(self,
                                             "Save selection list",".","*.py")
        if (filename[0]==""): return
        f=open(str(filename[0]),'w+')
        f.write(n)
        f.close()
    def deletelocal(self):
        pass
    def reset(self):
        tlvcolsnames=['Path','Name','SIDS type','DT','Value']
        v=self.selectionTable
        v.clear()
        v.setColumnCount(len(tlvcolsnames))
        v.setRowCount(0)
        lh=v.horizontalHeader()
        lv=v.verticalHeader()
        for i in xrange(len(tlvcolsnames)):
            hi=QTableWidgetItem(tlvcolsnames[i])
            v.setHorizontalHeaderItem(i,hi)
        for path in self._data:
            v.setRowCount(v.rowCount()+1)
            r=v.rowCount()-1
            self.updateRowData(r,path)
        self.colCheck()
        for i in xrange(v.rowCount()):
            v.resizeRowToContents(i)
        for i in xrange(len(tlvcolsnames)):
            v.resizeColumnToContents(i)
        self.setFixedSize(v.horizontalHeader().length()+70,
                          v.verticalHeader().length()+180)
        self._parent.treeview.refreshView()
    def updateRowData(self,r,path):
        v=self.selectionTable
        it0=QTableWidgetItem('%s '%'/'.join(path.split('/')[:-1]))
        it0.setFont(QFont("Courier"))
        v.setItem(r,0,it0)
        it1=QTableWidgetItem('%s '%path.split('/')[-1])
        ft=QFont("Courier")
        ft.setWeight(QFont.Bold)
        it1.setFont(ft)
        v.setItem(r,1,it1)
        node=self._model.nodeFromPath(path)
        if (node):
          it1=QTableWidgetItem(node.data(COLUMN_SIDS))
          it2=QTableWidgetItem(node.data(COLUMN_DATATYPE))
          val=node.data(COLUMN_VALUE)
          if (val==HIDEVALUE):
              val=QIcon(QPixmap(":/images/icons/data-array-large.png"))
              it3=QTableWidgetItem(val,'')
          else:
              it3=QTableWidgetItem(val)
              it3.setFont(QFont("Courier"))
          it1.setFont(QFont("Courier"))
          it2.setFont(QFont("Courier"))
          it3.setFont(QFont("Courier"))
          v.setItem(r,2,it1)
          v.setItem(r,3,it2)
          v.setItem(r,4,it3)
    def reject(self):
        self.close()

# -----------------------------------------------------------------
class Q7Query(Q7Window,Ui_Q7QueryWindow):
    _allQueries={}
    def __init__(self,control,fgprint,treeview):
        Q7Window.__init__(self,Q7Window.VIEW_QUERY,control,'/',fgprint)
        self.bClose.clicked.connect(self.reject)
        self.bRun.clicked.connect(self.runCurrent)
        self.bAdd.clicked.connect(self.queryAdd)
        self.bDel.clicked.connect(self.queryDel)
        self.bRevert.clicked.connect(self.queryRevert)
        self.bCommitDoc.clicked.connect(self.queryCommit)
        self.bRevertDoc.clicked.connect(self.queryRevert)
        self.bSave.clicked.connect(self.queriesSave)
        self.bSaveAsScript.clicked.connect(self.queryScriptSave)
        self._master=treeview
        QObject.connect(self.cQueryName,
                        SIGNAL("currentIndexChanged(int)"),
                        self.changeCurrentQuery)
        QObject.connect(self.cQueryGroup,
                        SIGNAL("currentIndexChanged(int)"),
                        self.changeCurrentGroup)
        self.cQueryName.editTextChanged.connect(self.checkNewQueryName)
        QObject.connect(self.cQueryName,
                        SIGNAL("editTextChanged()"),
                        self.checkNewQueryName)
        QObject.connect(self.cQueryName,
                        SIGNAL("editTextChanged(str)"),
                        self.checkNewQueryName)
        QObject.connect(self.eText,
                        SIGNAL("textChanged()"),
                        self.changeText)
        QObject.connect(self.eQueryDoc,
                        SIGNAL("textChanged()"),
                        self.changeText)
        self.bInfo.clicked.connect(self.infoQueryView)
        self.bAdd.setEnabled(False)
        self.setCurrentQuery()
        self._modified=False
        self.showQuery()
    def infoQueryView(self):
        self._control.helpWindow('Query')
    def updateTreeStatus(self):
        print 'query up'
    def queriesSave(self):
        self.queryCommit()
        Q7Query.saveUserQueries()
        self._modified=False
    def queryScriptSave(self):
        n=self.cQueryName.currentText()
        q=self.getQuery(n)
        c=self.eText.toPlainText()
        v=self.eUserVariable.text()
        f="%s/%s"%(self._fgprint.filedir,self._fgprint.filename)
        s=q.getFullScript(f,c,v)
        filename=QFileDialog.getSaveFileName(self,
                                             "Save query script",".","*.py")
        if (filename[0]==""): return
        f=open(str(filename[0]),'w+')
        f.write(s)
        f.close()
        os.chmod(filename[0],0o0750)
        self._modified=False
    def queryDel(self):
        self._modified=True
        q=self.cQueryName.currentText()
        i=self.cQueryName.currentIndex()
        reply = MSG.message('Delete query',
                            """Do you really want to delete query [%s]?"""%q,
                            MSG.YESNO)
        if (reply == QMessageBox.Yes):
            del Q7Query._allQueries[q]
            self.cQueryName.removeItem(i)
    def checkNewQueryName(self,*args):
        qname=self.cQueryName.currentText().strip()
        if (qname not in Q7Query._allQueries):
            self.bAdd.setEnabled(True)
    def queryAdd(self):
        self.bAdd.setEnabled(False)
        qname=self.cQueryName.currentText().strip()
        if (qname in Q7Query._allQueries): return
        q=Q7QueryEntry(qname)
        Q7Query._allQueries[qname]=q
        self.queryCommit()
        self.cQueryName.addItem(qname)
    def queryCommit(self):
        self._modified=True
        q=self.cQueryName.currentText()
        if (q not in Q7Query._allQueries): self.queryAdd()
        com=self.eText.toPlainText()
        doc=self.eQueryDoc.toPlainText()
        if (self.cRequireUpdate.checkState()==Qt.Checked):
            self.setRequireTreeUpdate(True)
        else:
            self.setRequireTreeUpdate(False)
        Q7Query._allQueries[q].setScript(com)
        Q7Query._allQueries[q].setDoc(doc)
        self.bRevert.setEnabled(False)
        self.bCommitDoc.setEnabled(False)
        self.bRevertDoc.setEnabled(False)
    def queryRevert(self):
        self.showQuery()
    def reject(self):
        reply=True
        if (self._modified):
            reply = MSG.wQuestion(self._control,'Leave query panel',
                                """There are unsaved modified queries,
leave this panel without save?""",again=False)
        if (not reply):
            return
        else:
            self._modified=False
        if (self._master._querywindow is not None):
            self._master._querywindow=None
        self.close()
    def reset(self):
        gq=set()
        for qn in self.queriesNamesList():
            self.cQueryName.addItem(qn)
            gq.add(Q7Query._allQueries[qn].group)
        self.cQueryGroup.addItem('*')    
        for gqn in gq:
            self.cQueryGroup.addItem(gqn)
    def showQuery(self,name=None):
        pstate=self._modified
        if (name is None):
            name=self.getCurrentQuery().name
        if (name in Q7Query.queriesNamesList()):
          txt=self.getCurrentQuery().script
          self.eText.initText(txt)
          doc=self.getCurrentQuery().doc
          self.eQueryDoc.initText(doc)
        if (self.getCurrentQuery().requireTreeUpdate()):
            self.cRequireUpdate.setCheckState(Qt.Checked)
        else:
            self.cRequireUpdate.setCheckState(Qt.Unchecked)            
        self.bRevert.setEnabled(False)
        self.bCommitDoc.setEnabled(False)
        self.bRevertDoc.setEnabled(False)
        self._modified=pstate
    def show(self):
        self.reset()
        super(Q7Query, self).show()
    def runCurrent(self):
        com=self.eText.toPlainText()
        v=self.eUserVariable.text()
        q=Q7QueryEntry('__tmp__query__')
        q.setScript(com)
        skp=self._fgprint.lazy.keys()
        r=q.run(self._fgprint.tree,self._fgprint.links,skp,True,v,
                self._fgprint.model.getSelected())
        self.eResult.initText(str(r))
        if (q.requireTreeUpdate()):
            self._fgprint.model.modelReset()
    @classmethod
    def fillQueries(self):
        allqueriestext=Q7Query._userQueriesText+OCTXT._UsualQueries
        Q7Query._defaultQueriesNames=[n[0] for n in OCTXT._UsualQueries]
        for qe in allqueriestext:
            try:
                if (len(qe)<5):
                    q=Q7QueryEntry(qe[0],qe[1],qe[2],qe[3])
                else:
                    q=Q7QueryEntry(qe[0],qe[1],qe[2],qe[3],qe[4])
                Q7Query._allQueries[qe[0]]=q
            except IndexError: pass
    @classmethod
    def getQuery(self,name):
        if (name in Q7Query.queriesNamesList()):
            return Q7Query._allQueries[name]
        return None
    @classmethod
    def loadUserFunctions(self):
        try:
          Q7Query._userFunction=OCTXT._readFunctions(self)()
        except:
          Q7Query._userFunction=None
    @classmethod
    def loadUserQueries(self):
        Q7Query._userQueriesText=OCTXT._readQueries(self)
        if (Q7Query._userQueriesText is None): Q7Query._userQueriesText=[]
        return Q7Query._userQueriesText
    @classmethod
    def saveUserQueries(self):
        ql=[]
        for q in Q7Query._allQueries:
            if (q not in Q7Query._defaultQueriesNames):
                ql+=[str(Q7Query._allQueries[q])]
        OCTXT._writeQueries(self,ql)
    @classmethod
    def queriesNamesList(self):
        k=Q7Query._allQueries.keys()
        k.sort()
        return k
    @classmethod
    def queriesGroupsList(self):
      gl=set()
      for q in Q7Query._allQueries:
        g=Q7Query._allQueries[q].group
        gl.add(g)
      return ['*']+list(gl)
    def getCurrentQuery(self):
        return self._currentQuery
    def changeCurrentGroup(self):
        group=self.cQueryGroup.currentText()
        self.cQueryName.clear()
        for qn in self.queriesNamesList():
            gp=Q7Query._allQueries[qn].group
            if ((group=='*') or (gp==group)):
                self.cQueryName.addItem(qn)
        self.showQuery()
    def changeCurrentQuery(self,*args):
        name=self.cQueryName.currentText()
        if (name not in self.queriesNamesList()): return
        self.setCurrentQuery(name)
        self.showQuery()
    def setCurrentQuery(self,name=None):
        if (name is None): name=self.queriesNamesList()[0]
        self._currentQuery=Q7Query._allQueries[name]
    def changeText(self):
        self.bRevert.setEnabled(True)
        self.bCommitDoc.setEnabled(True)
        self.bRevertDoc.setEnabled(True)
        self._modified=True
    @classmethod
    def queries(self):
        return Q7Query.queriesNamesList()

# -----------------------------------------------------------------
