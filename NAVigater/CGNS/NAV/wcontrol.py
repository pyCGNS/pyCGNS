#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys

from PySide.QtCore import *
from PySide.QtGui import *

from CGNS.NAV.Q7ControlWindow import Ui_Q7ControlWindow
from CGNS.NAV.wfile import Q7File
from CGNS.NAV.woption import Q7Option
from CGNS.NAV.moption import Q7OptionContext as OCTXT
from CGNS.NAV.wtree import Q7Tree
from CGNS.NAV.mtree import Q7TreeModel
from CGNS.NAV.wfingerprint import Q7fingerPrint, Q7Window

import CGNS.NAV.wmessages as MSG

# -----------------------------------------------------------------
class Q7SignalPool(QObject):
    loadFile=Signal()
    saveFile=Signal()
    buffer=None

# -----------------------------------------------------------------
class Q7ControlItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if (index.column() in [0,1]):
            option.decorationPosition=QStyleOptionViewItem.Top
            QStyledItemDelegate.paint(self, painter, option, index)
        else:
            QStyledItemDelegate.paint(self, painter, option, index)

# -----------------------------------------------------------------
class Q7Main(Q7Window, Ui_Q7ControlWindow):
    def __init__(self, parent=None):
        Q7Window.__init__(self,Q7Window.VIEW_CONTROL,self,None,None)
        self.bAbout.clicked.connect(self.about)
        self.bOptionView.clicked.connect(self.option)
        self.bTreeLoadLast.clicked.connect(self.loadlast)
        self.bTreeLoad.clicked.connect(self.load)
        self.bEditTree.clicked.connect(self.edit)
        #self.bResetScrollBars.clicked.connect(self.resetScrolls)
        self.bClose.clicked.connect(self.close)
        QObject.connect(self.controlTable,
                        SIGNAL("cellClicked(int,int)"),
                        self.clickedLine)
        self.wOption=None
        self.initControlTable()
        self.I_UNCHANGED=QIcon(QPixmap(":/images/icons/save-done.gif"))
        self.I_MODIFIED=QIcon(QPixmap(":/images/icons/save.gif"))
        self.I_TREE=QIcon(QPixmap(":/images/icons/tree-load.gif"))
        self.I_VTK=QIcon(QPixmap(":/images/icons/vtk.gif"))
        self.I_QUERY=QIcon(QPixmap(":/images/icons/operate-execute.gif"))
        self.I_FORM=QIcon(QPixmap(":/images/icons/form-open.gif"))
        self.controlTable.setItemDelegate(Q7ControlItemDelegate(self))
        self.signals=Q7SignalPool()
        self.signals.loadFile.connect(self.loading)     
        self.signals.saveFile.connect(self.saving)
        self.getHistory()
        self.getOptions()
    def clickedLine(self,*args):
        Q7fingerPrint.raiseView(self.getIdxFromLine(args[0]))
    def option(self):
        if (self.wOption==None):
            self.wOption=Q7Option(self)
        self.wOption.show()
    def about(self):
        MSG.message("About %s"%OCTXT._ToolName,
               """<b>%s %s</b><p>%s<p>http://www.onera.fr"""\
               %(OCTXT._ToolName,OCTXT._ToolVersion,OCTXT._CopyrightNotice),
               MSG.INFO)
    def closeApplication(self):
        reply = MSG.message('Double check...',
                            """Do you want to quit %s,<b>close all views<b><br>
                            and forget unsaved modifications?""" \
                            %OCTXT._ToolName,
                            MSG.YESNO)
        if (reply == QMessageBox.Yes):
            Q7fingerPrint.closeAllTrees()
            return True
        else:
            return False
    def closeEvent(self, event):
        if (self.closeApplication()):
            event.accept()
            return True
        else:
            event.ignore()
            return False
    def resetScrolls(self):
        self.controlTable.verticalScrollBar().setSliderPosition(0)
        self.controlTable.horizontalScrollBar().setSliderPosition(0)
    def initControlTable(self):
        ctw=self.controlTable
        cth=ctw.horizontalHeader()
        ctw.verticalHeader().hide()
        h=['S','T','View','Dir','File','Node']
        for i in range(len(h)):
            hi=QTableWidgetItem(h[i])
            ctw.setHorizontalHeaderItem(i,hi)
            cth.setResizeMode(i,QHeaderView.ResizeToContents)
        cth.setResizeMode(len(h)-1,QHeaderView.Stretch)
    def addLine(self,l):
        ctw=self.controlTable
        ctw.setRowCount(ctw.rowCount()+1)
        r=ctw.rowCount()-1
        if (l[0]==Q7Window.STATUS_UNCHANGED):
            stitem=QTableWidgetItem(self.I_UNCHANGED,'')
        if (l[0]==Q7Window.STATUS_MODIFIED):
            stitem=QTableWidgetItem(self.I_MODIFIED,'')
        if (l[1]==Q7Window.VIEW_TREE):
            tpitem=QTableWidgetItem(self.I_TREE,'')
        if (l[1]==Q7Window.VIEW_FORM):
            tpitem=QTableWidgetItem(self.I_FORM,'')
        if (l[1]==Q7Window.VIEW_VTK):
            tpitem=QTableWidgetItem(self.I_VTK,'')
        if (l[1]==Q7Window.VIEW_QUERY):
            tpitem=QTableWidgetItem(self.I_QUERY,'')
        ctw.setItem(r,0,stitem)
        ctw.setItem(r,1,tpitem)
        for i in range(len(l)-2):
            it=QTableWidgetItem('%s'%(l[i+2]))
            if (i in [0,1]):   it.setTextAlignment(Qt.AlignCenter)
            if (i in [2,3,4]): it.setFont(QFont("Courier"))
            ctw.setItem(r,i+2,it)
        for i in (0,1,2):
            ctw.resizeColumnToContents(i)
        for i in range(self.controlTable.rowCount()):
            ctw.resizeRowToContents(i)
    def delLine(self,idx):
        i=int(self.getLineFromIdx(idx))
        if (i!=-1):
            self.controlTable.removeRow(i)
    def getIdxFromLine(self,l):
        self.controlTable.setCurrentCell(l,2)
        it=self.controlTable.currentItem()
        return it.text()
    def getLineFromIdx(self,idx):
        found=-1
        for n in range(self.controlTable.rowCount()):
            if (int(idx)==int(self.controlTable.item(n,2).text())):found=n
        return found
    def loading(self,*args):
        self.busyCursor()
        fgprint=Q7fingerPrint.treeLoad(self,self.signals.buffer)
        if (fgprint is None): return
        Q7TreeModel(fgprint)
        child=Q7Tree(self,'/',fgprint)
        self.readyCursor()
        child.show()
        self.setHistory(fgprint.filedir,fgprint.filename)
    def saving(self,*args):
        print 'SAVING ...', self.signals.buffer
    def load(self):
        self.fdialog=Q7File(self)
        self.fdialog.show()
    def loadlast(self):
        if (self.getLastFile() is None): return
        self.signals.buffer=self.getLastFile()[0]+'/'+self.getLastFile()[1]
        if (self.signals.buffer is None): self.load()
        else: self.signals.loadFile.emit()
    def save(self):
        self.fdialog=Q7File(self,1)
        self.fdialog.show()
    def edit(self):
        pass

# -----------------------------------------------------------------
