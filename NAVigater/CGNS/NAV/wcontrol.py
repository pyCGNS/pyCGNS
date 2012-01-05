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
from CGNS.NAV.wtree import Q7Tree
from CGNS.NAV.mtree import Q7TreeModel
from CGNS.NAV.wfingerprint import Q7fingerPrint, Q7Window

import CGNS.NAV.wmessages as MSG
from CGNS.NAV.defaults import G__TOOLNAME, G__TOOLVERSION, G__COPYRIGHTNOTICE

# -----------------------------------------------------------------
class Q7SignalPool(QObject):
    loadFile=Signal()
    saveFile=Signal()
    buffer=None

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
        self.wOption=None
        self.initControlTable()
        self.signals=Q7SignalPool()
        self.signals.loadFile.connect(self.loading)     
        self.signals.saveFile.connect(self.saving)
        self.getHistory()
        self.getOptions()
    def option(self):
        if (self.wOption==None):
            self.wOption=Q7Option(self)
        self.wOption.show()
    def about(self):
        MSG.message("About CGNS.NAV",
                    """<b>%s %s</b><p>%s<p>http://www.onera.fr"""\
                    %(G__TOOLNAME,G__TOOLVERSION,G__COPYRIGHTNOTICE),MSG.INFO)
    def closeApplication(self):
        reply = MSG.message('Double check...',
            """Do you want to quit %s,<p>close all views<P>
               and <b>loose</b> any modification?"""%G__TOOLNAME,MSG.YESNO)
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
        self.Q7ControlTable.verticalScrollBar().setSliderPosition(0)
        self.Q7ControlTable.horizontalScrollBar().setSliderPosition(0)
    def initControlTable(self):
        ctw=self.Q7ControlTable
        cth=ctw.horizontalHeader()
        ctw.verticalHeader().hide()
        h=['S','T','View','Dir','File','Node']
        for i in range(len(h)):
            hi=QTableWidgetItem(h[i])
            ctw.setHorizontalHeaderItem(i,hi)
            cth.setResizeMode(i,QHeaderView.ResizeToContents)
        cth.setResizeMode(len(h)-1,QHeaderView.Stretch)
    def addLine(self,l):
        ctw=self.Q7ControlTable
        ctw.setRowCount(ctw.rowCount()+1)
        r=ctw.rowCount()-1
        stitem=QTableWidgetItem()
        if (l[0]==Q7Window.STATUS_UNCHANGED):
            stitem.setIcon(QIcon(QPixmap(":/images/icons/save-done.gif")))
        if (l[0]==Q7Window.STATUS_MODIFIED):
            stitem.setIcon(QIcon(QPixmap(":/images/icons/save.gif")))
        ctw.setItem(r,0,stitem)
        for i in range(len(l)-1):
            it=QTableWidgetItem('%s'%(l[i+1]))
            if (i in [0,1]):   it.setTextAlignment(Qt.AlignCenter)
            if (i in [2,3,4]): it.setFont(QFont("Courier"))
            ctw.setItem(r,i+1,it)
    def loading(self,*args):
        fgprint=Q7fingerPrint.treeLoad(self,self.signals.buffer)
        if (fgprint is None): return
        Q7TreeModel(fgprint)
        child=Q7Tree(self,'/',fgprint)
        child.show()
        self.setHistory(fgprint.filedir,fgprint.filename)
    def saving(self,*args):
        print 'SAVING ...', self.signals.buffer
    def load(self):
        self.fdialog=Q7File(self)
        self.fdialog.show()
    def loadlast(self):
        self.signals.buffer=self.getLastFile()[0]+'/'+self.getLastFile()[1]
        if (self.signals.buffer is None): self.load()
        else: self.signals.loadFile.emit()
    def save(self):
        self.fdialog=Q7File(self,1)
        self.fdialog.show()
    def edit(self):
        pass

# -----------------------------------------------------------------
