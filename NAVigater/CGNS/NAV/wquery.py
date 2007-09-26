#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
import numpy

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

# -----------------------------------------------------------------
class Q7SelectionList(Q7Window,Ui_Q7SelectionWindow):
    def __init__(self,control,model,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_SELECT,control,None,fgprint)
        self.bClose.clicked.connect(self.reject)
        self._model=model
        self._data=model._selected
        self._fgprint=fgprint
        self.bSave.clicked.connect(self.selectionsave)
        self.bInfo.clicked.connect(self.infoSelectView)
        self.bDelete.clicked.connect(self.deletelocal)
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
        tlvcols=3
        tlvcolsnames=['SIDS type','Data Type','Value']
        v=self.diagTable
        v.setColumnCount(self._fgprint.depth+tlvcols)
        lh=v.horizontalHeader()
        lv=v.verticalHeader()
        h=['/level%d'%d for d in range(self._fgprint.depth)]+tlvcolsnames
        n=len(h)
        for i in range(n):
            hi=QTableWidgetItem(h[i])
            v.setHorizontalHeaderItem(i,hi)
        plist=[]
        for path in self._data:
            v.setRowCount(v.rowCount()+1)
            r=v.rowCount()-1
            plist=path.split('/')[1:]
            for i in range(len(plist)):
                it=QTableWidgetItem('%s '%plist[i])
                it.setFont(QFont("Courier"))
                v.setItem(r,i,it)
            if tlvcols:
                node=self._model.nodeFromPath(path)
                if (node):
                  it1=QTableWidgetItem(node.data(COLUMN_SIDS))
                  it2=QTableWidgetItem(node.data(COLUMN_DATATYPE))
                  val=node.data(COLUMN_VALUE)
                  if (val==HIDEVALUE):
                      val=QIcon(QPixmap(":/images/icons/data-array-large.gif"))
                      it3=QTableWidgetItem(val,'')
                  else:
                      it3=QTableWidgetItem(val)
                      it3.setFont(QFont("Courier"))
                  it1.setFont(QFont("Courier"))
                  it3.setFont(QFont("Courier"))
                  v.setItem(r,n-3,it1)
                  v.setItem(r,n-2,it2)
                  v.setItem(r,n-1,it3)
        for i in range(len(plist)):
            v.resizeColumnToContents(i)
        for i in range(v.rowCount()):
            v.resizeRowToContents(i)
    def reject(self):
        self.close()

# -----------------------------------------------------------------
class Q7QueryTableItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        t=index.data(Qt.DisplayRole)
        r=option.rect.adjusted(2, 2, -2, -2);
        painter.drawText(r,Qt.AlignVCenter|Qt.AlignLeft,t, r)

# -----------------------------------------------------------------
class Q7Query(Q7Window,Ui_Q7QueryWindow):
    _allQueries={}
    def __init__(self,control,fgprint,treeview):
        Q7Window.__init__(self,Q7Window.VIEW_QUERY,control,'/',fgprint)
        self.bClose.clicked.connect(self.reject)
        self.bRun.clicked.connect(self.runCurrent)
        self.bAdd.clicked.connect(self.queryAdd)
        self.bDel.clicked.connect(self.queryDel)
        self.bCommit.clicked.connect(self.queryCommit)
        self.bRevert.clicked.connect(self.queryRevert)
        self.bCommitDoc.clicked.connect(self.queryCommit)
        self.bRevertDoc.clicked.connect(self.queryRevert)
        self.bSave.clicked.connect(self.queriesSave)
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
        self.showQuery()
    def infoQueryView(self):
        self._control.helpWindow('Query')
    def updateTreeStatus(self):
        print 'query up'
    def queriesSave(self):
        Q7Query.saveUserQueries()
    def queryDel(self):
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
        q=self.cQueryName.currentText()
        com=self.eText.toPlainText()
        doc=self.eQueryDoc.toPlainText()
        Q7Query._allQueries[q].setScript(com)
        Q7Query._allQueries[q].setDoc(doc)
        self.bCommit.setEnabled(False)
        self.bRevert.setEnabled(False)
        self.bCommitDoc.setEnabled(False)
        self.bRevertDoc.setEnabled(False)
    def queryRevert(self):
        self.showQuery()
    def reject(self):
        self._master.qryview=None
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
        if (name is None):
            name=self.getCurrentQuery().name
        if (name in Q7Query.queriesNamesList()):
          txt=self.getCurrentQuery().script
          self.eText.initText(txt)
          doc=self.getCurrentQuery().doc
          self.eQueryDoc.initText(doc)
        self.bCommit.setEnabled(False)
        self.bRevert.setEnabled(False)
        self.bCommitDoc.setEnabled(False)
        self.bRevertDoc.setEnabled(False)
    def show(self):
        self.reset()
        super(Q7Query, self).show()
    def runCurrent(self):
        com=self.eText.toPlainText()
        v=self.eUserVariable.text()
        q=Q7QueryEntry('__tmp__query__')
        q.setScript(com)
        r=q.run(self._fgprint.tree,True,v)
        self.eResult.initText(str(r))
    @classmethod
    def fillQueries(self):
        allqueriestext=Q7Query._userQueriesText+OCTXT._UsualQueries
        Q7Query._defaultQueriesNames=[n[0] for n in OCTXT._UsualQueries]
        for qe in allqueriestext:
            try:
                q=Q7QueryEntry(qe[0],qe[1],qe[2],qe[3])
                Q7Query._allQueries[qe[0]]=q
            except IndexError: pass
    @classmethod
    def getQuery(self,name):
        if (name in Q7Query.queriesNamesList()):
            return Q7Query._allQueries[name]
        return None
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
        self.bCommit.setEnabled(True)
        self.bRevert.setEnabled(True)
        self.bCommitDoc.setEnabled(True)
        self.bRevertDoc.setEnabled(True)
    @classmethod
    def queries(self):
        return Q7Query.queriesNamesList()

# -----------------------------------------------------------------
