#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys

from PySide.QtCore import *
from PySide.QtGui import *

from CGNS.NAV.Q7MainWindow import Ui_Q7MainWindow
from CGNS.NAV.wfile import Q7File
from CGNS.NAV.woption import Q7Option
from CGNS.NAV.wtree import Q7Tree
from CGNS.NAV.mtree import Q7TreeModel
from CGNS.NAV.wfingerprint import Q7fingerPrint
from CGNS.NAV.defaults import G__options, G__toolname, G__toolversion

import CGNS.NAV.wmessages as msg
import CGNS.NAV.moption   as mop

COPYRIGHTNOTICE="""
Copyright (c) 2010-2012 Marc Poinot - Onera - The French Aerospace Labs
All rights reserved in accordance with GPL v2 
NO WARRANTY :
Check GPL v2 sections 15 and 16 about loss of data or corrupted data
"""

(STATUS_UNCHANGED, STATUS_MODIFIED)=('U','M')
(VIEW_TREE,VIEW_TABLE,VIEW_FORM)=('T','A','F')

# -----------------------------------------------------------------
class Q7SignalPool(QObject):
    loadFile=Signal()
    saveFile=Signal()
    buffer=None

# -----------------------------------------------------------------
class Q7Main(QMainWindow, Ui_Q7MainWindow):
    def __init__(self, parent=None):
        super(Q7Main, self).__init__(parent)
        self.setupUi(self)
        self.options=G__options
        self.bAbout.clicked.connect(self.about)
        self.bOptionView.clicked.connect(self.option)
        self.bTreeLoad.clicked.connect(self.load)
        self.bEditTree.clicked.connect(self.edit)
        self.bResetScrollBars.clicked.connect(self.resetScrolls)
        self.bClose.clicked.connect(self.close)
        self.wOption=None
        self.setWindowTitle("%s %s"%(G__toolname,G__toolversion))
        self.initControlTable()
        self.signals=Q7SignalPool()
        self.signals.loadFile.connect(self.loading)     
        self.signals.saveFile.connect(self.saving)
        self.getHistory()
        self.getOptions()
    def getHistory(self):
        self.history=mop.readHistory(self)
        if (self.history is None): self.history={}
        return self.history
    def setHistory(self,filedir,filename):
        for d in self.history.keys():
            if (d==filedir):
                if (filename not in self.history[filedir]):
                    self.history[filedir].append(filename)
            else:
                self.history[filedir]=[filename]
        if (self.history=={}): self.history[filedir]=[filename]
        mop.writeHistory(self)
        return self.history
    def addChildWindow(self,child,fingerprint):
        index=fingerprint.addChild(VIEW_TREE,child)
        path='/'
        l=[STATUS_UNCHANGED,VIEW_TREE,'%.3d'%index]
        l+=[fingerprint.filedir,fingerprint.filename,path]
        self.addLine(l)
        return child
    def validateOption(self,name,value):
        return True
    def getOptions(self):
        self.options=mop.readOptions(self)
        if (self.options is None): self.options=G__options
        return self.options
    def getOptionValue(self,name):
        return self.options[name]
    def setOptionValue(self,name,value):
        if (self.validateOption(name,value)):
            self.options[name]=value
            return value
        return self.options[name]
    def option(self):
        if (self.wOption==None):
            self.wOption=Q7Option(self)
        self.wOption.show()
    def about(self):
        msg.message("About CGNS.NAV",
                    """<b>%s %s</b><p>%s<p>http://www.onera.fr"""\
                    %(G__toolname,G__toolversion,COPYRIGHTNOTICE),msg.INFO)
    def closeApplication(self):
        reply = msg.message('Double check...',
            "Do you want to quit %s, close all views and loose any modification?"%G__toolname,
            msg.YESNO)
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
        if (l[0]==STATUS_UNCHANGED):
            stitem.setIcon(QIcon(QPixmap(":/images/icons/save-done.gif")))
        if (l[0]==STATUS_MODIFIED):
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
        child=self.addChildWindow(Q7Tree(self),fgprint)
        child.bindTree(fgprint)
        self.setHistory(fgprint.filedir,fgprint.filename)
        child.show()
    def saving(self,*args):
        print 'SAVING ...', self.signals.buffer
    def load(self):
        self.fdialog=Q7File(self)
        self.fdialog.show()
    def save(self):
        self.fdialog=Q7File(self,1)
        self.fdialog.show()
    def edit(self):
        pass

# -----------------------------------------------------------------
