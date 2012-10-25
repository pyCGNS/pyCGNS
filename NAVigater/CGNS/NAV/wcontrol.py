#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import sys

from PySide.QtCore import *
from PySide.QtGui import *

from CGNS.NAV.Q7ControlWindow import Ui_Q7ControlWindow
from CGNS.NAV.wfile import Q7File
from CGNS.NAV.winfo import Q7Info
from CGNS.NAV.woption import Q7Option
from CGNS.NAV.moption import Q7OptionContext as OCTXT
from CGNS.NAV.wtree import Q7Tree
from CGNS.NAV.mtree import Q7TreeModel
from CGNS.NAV.wfingerprint import Q7fingerPrint, Q7Window
from CGNS.NAV.wquery import Q7Query
from CGNS.NAV.whelp import Q7Help

import CGNS.NAV.wmessages as MSG

import CGNS.PAT.cgnslib   as CGL
import CGNS.PAT.cgnsutils as CGU

# -----------------------------------------------------------------
class Q7SignalPool(QObject):
    loadFile=Signal()
    saveFile=Signal()
    buffer=None
    fgprint=None

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
        self.getHistory()
        self.bAbout.clicked.connect(self.about)
        self.bOptionView.clicked.connect(self.option)
        self.bTreeLoadLast.clicked.connect(self.loadlast)
        self.bTreeLoad.clicked.connect(self.load)
        self.bEditTree.clicked.connect(self.edit)
        self.bInfo.clicked.connect(self.infoControl)
        self.bPatternView.setDisabled(True)
        #self.bResetScrollBars.clicked.connect(self.resetScrolls)
        self.bClose.clicked.connect(self.close)
        QObject.connect(self.controlTable,
                        SIGNAL("cellClicked(int,int)"),
                        self.clickedLine)
        self.initControlTable()
        self.controlTable.setItemDelegate(Q7ControlItemDelegate(self))
        self.signals=Q7SignalPool()
        self.signals.loadFile.connect(self.loading)     
        self.signals.saveFile.connect(self.saving)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.popupmenu = QMenu()
        self.transientRecurse=False
        self.transientVTK=False
        self.copyPasteBuffer=None
        self.wOption=None
        self.selectForLink=None
        self.__newtreecount=1
        self.help=None
        Q7Query.loadUserQueries()
        Q7Query.fillQueries()
    def clickedLine(self,*args):
        if (self.controlTable.lastButton==Qt.LeftButton):
            #Q7fingerPrint.raiseView(self.getIdxFromLine(args[0]))
            pass
        if (self.controlTable.lastButton==Qt.RightButton):
            self.updateMenu(self.controlTable.currentIndex())
            self.popupmenu.popup(self.controlTable.lastPos)
    def closeView(self):
        self.updateLastView()
        if (self.lastView): Q7fingerPrint.getView(self.lastView).close()
    def raiseView(self):
        self.updateLastView()
        if (self.lastView): Q7fingerPrint.raiseView(self.lastView)
    def infoControl(self):
        self.helpWindow('Control')
    def helpWindow(self,key):
        if (self.help is not None): self.help.close()
        self.help=Q7Help(self,key)
        self.help.show()
    def info(self):
        self.updateLastView()
        (f,v,d)=Q7fingerPrint.infoView(self.lastView)
        if (not f.isFile()): return
        self.w=Q7Info(self,d,f)
        self.w.show()
    def closeTree(self):
        self.updateLastView()
        (f,v,d)=Q7fingerPrint.infoView(self.lastView)
        reply = MSG.message('Double check...',
                            """Do you want to close the tree and all its views,<br>
                            and <b>forget unsaved</b> modifications?""",
                            MSG.YESNO)
        if (reply == QMessageBox.Yes):
            f.closeAllViews()
    def closeAllViews(self):
        reply = MSG.message('Double check...',
                            """Do you want to close all the views,<br>
                            and <b>forget unsaved</b> modifications?""",
                            MSG.YESNO)
        if (reply == QMessageBox.Yes):
            Q7fingerPrint.closeAllTrees()
    def pop6(self):
        pass
    def pop7(self):
        pass
    def updateLastView(self):
        r=self.controlTable.currentItem().row()
        self.lastView=self.getIdxFromLine(r)
        return self.lastView
    def updateMenu(self,idx):
        lv=self.getIdxFromLine(idx.row())
        if (lv is not None):
          self.lastView=lv
          actlist=(("View information (Enter)",self.info),
                   ("Raise view (Space)",self.raiseView),
#                   ("Update tree",self.pop6),
                   None,
                   ("Close tree",self.closeTree),
                   ("Close all views",self.closeAllViews),
                   ("Close view (Del)",self.closeView))
          self.popupmenu.clear()
          self.popupmenu.setTitle('Control view menu')
          for aparam in actlist:
              if (aparam is None): self.popupmenu.addSeparator()
              else:
                  a=QAction(aparam[0],self,triggered=aparam[1])
                  self.popupmenu.addAction(a)
    def loadOptions(self):
        if (self.wOption==None): self.wOption=Q7Option(self)
        self.wOption.reset()
    def option(self):
        if (self.wOption==None): self.wOption=Q7Option(self)
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
            if (self.help is not None): self.help.close()
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
        ctw.control=self
        cth=ctw.horizontalHeader()
        ctw.verticalHeader().hide()
        h=['S','T','View','Dir','File','Node']
        for i in range(len(h)):
            hi=QTableWidgetItem(h[i])
            ctw.setHorizontalHeaderItem(i,hi)
            cth.setResizeMode(i,QHeaderView.ResizeToContents)
        cth.setResizeMode(len(h)-1,QHeaderView.Stretch)
    def updateViews(self):
        for i in self.getAllIdx():
            f=Q7fingerPrint.getFingerPrint(i)
            v=Q7fingerPrint.getView(i)
            l=self.getLineFromIdx(i)
            self.modifiedLine(l,f._status,f)
            try:
                v.updateTreeStatus()
            except AttributeError:
                pass
    def modifiedLine(self,n,stat,fg):
        if (     (Q7fingerPrint.STATUS_MODIFIED in stat)
             and (Q7fingerPrint.STATUS_SAVEABLE in stat)):
            stitem=QTableWidgetItem(self.I_MOD_SAV,'')
            stitem.setToolTip('Tree modified and saveable')
        if (     (Q7fingerPrint.STATUS_MODIFIED in stat)
             and (Q7fingerPrint.STATUS_SAVEABLE not in stat)):
            stitem=QTableWidgetItem(self.I_MOD_USAV,'')
            stitem.setToolTip('Tree modified but NOT saveable')
        if (     (Q7fingerPrint.STATUS_MODIFIED not in stat)
             and (Q7fingerPrint.STATUS_SAVEABLE not in stat)):
            stitem=QTableWidgetItem(self.I_UMOD_USAV,'')
            stitem.setToolTip('Tree unmodified and NOT saveable')
        if (     (Q7fingerPrint.STATUS_MODIFIED not in stat)
             and (Q7fingerPrint.STATUS_SAVEABLE in stat)):
            stitem=QTableWidgetItem(self.I_UMOD_SAV,'')
            stitem.setToolTip('Tree unmodified and saveable')
        stitem.setTextAlignment(Qt.AlignCenter)
        self.controlTable.setItem(n,0,stitem)
        self.controlTable.item(n,3).setText(str(fg.filedir))
        self.controlTable.item(n,4).setText(str(fg.filename))
    def addLine(self,l,fg):
        ctw=self.controlTable
        ctw.setRowCount(ctw.rowCount()+1)
        r=ctw.rowCount()-1
        if (l[1]==Q7Window.VIEW_TREE):
            tpitem=QTableWidgetItem(self.I_TREE,'')
        if (l[1]==Q7Window.VIEW_FORM):
            tpitem=QTableWidgetItem(self.I_FORM,'')
        if (l[1]==Q7Window.VIEW_VTK):
            tpitem=QTableWidgetItem(self.I_VTK,'')
        if (l[1]==Q7Window.VIEW_QUERY):
            tpitem=QTableWidgetItem(self.I_QUERY,'')
        if (l[1]==Q7Window.VIEW_SELECT):
            tpitem=QTableWidgetItem(self.I_SELECT,'')
        if (l[1]==Q7Window.VIEW_DIAG):
            tpitem=QTableWidgetItem(self.I_DIAG,'')
        if (l[1]==Q7Window.VIEW_LINK):
            tpitem=QTableWidgetItem(self.I_LINK,'')
        tpitem.setTextAlignment(Qt.AlignCenter)
        ctw.setItem(r,1,tpitem)
        for i in range(len(l)-2):
            it=QTableWidgetItem('%s '%(l[i+2]))
            if (i in [0]): it.setTextAlignment(Qt.AlignCenter)
            else: it.setTextAlignment(Qt.AlignLeft)
            it.setFont(QFont("Courier"))
            ctw.setItem(r,i+2,it)
        self.modifiedLine(r,l[0],fg)
        for i in (0,1,2):
            ctw.resizeColumnToContents(i)
        for i in range(self.controlTable.rowCount()):
            ctw.resizeRowToContents(i)
    def selectLine(self,idx):
        i=int(self.getLineFromIdx(idx))
        if (i!=-1):
            self.controlTable.setCurrentCell(i,2)
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
    def getAllIdx(self):
        all=[]
        for n in range(self.controlTable.rowCount()):
            all.append(self.controlTable.item(n,2).text())
        return all
    def loading(self,*args):
        self._T('loading: [%s]'%self.signals.buffer)
        self.busyCursor()
        fgprint=Q7fingerPrint.treeLoad(self,self.signals.buffer)
        if (fgprint is None): return
        Q7TreeModel(fgprint)
        child=Q7Tree(self,'/',fgprint)
        child.show()
        self.setHistory(fgprint.filedir,fgprint.filename)
        self.updateViews()
        self.readyCursor()
    def saving(self,*args):
        self._T('saving as: [%s]'%self.signals.buffer)
        self.busyCursor()
        Q7fingerPrint.treeSave(self,self.signals.fgprint,self.signals.buffer)
        self.setHistory(self.signals.fgprint.filedir,
                        self.signals.fgprint.filename)
        self.updateViews()
        self.readyCursor()
    def load(self):
        self.fdialog=Q7File(self)
        self.fdialog.show()
    def loadlast(self):
        if (self.getLastFile() is None): return
        self.signals.buffer=self.getLastFile()[0]+'/'+self.getLastFile()[1]
        if (self.signals.buffer is None): self.load()
        else: self.signals.loadFile.emit()
    def loadfile(self,name):
        self.signals.buffer=name
        self.signals.loadFile.emit()
    def save(self,fgprint):
        self.signals.fgprint=fgprint
        self.fdialog=Q7File(self,1)
        self.fdialog.show()
    def savedirect(self,fgprint):
        return
        self.signals.fgprint=fgprint
        self.signals.buffer=fgprint.filedir+'/'+fgprint.filename
        self.signals.saveFile.emit()
    def edit(self):
        self._T('edit new')
        tree=CGL.newCGNSTree()
        tc=self.__newtreecount
        self.busyCursor()
        fgprint=Q7fingerPrint(self,'.','new#%.3d.hdf'%tc,tree,[],[])
        Q7TreeModel(fgprint)
        child=Q7Tree(self,'/',fgprint)
        fgprint._status=[Q7fingerPrint.STATUS_MODIFIED]
        self.readyCursor()
        self.__newtreecount+=1
        child.show()

   

# -----------------------------------------------------------------
