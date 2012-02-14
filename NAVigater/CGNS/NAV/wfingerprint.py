#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
from PySide.QtCore       import *
from PySide.QtGui        import *
import os.path
import CGNS.MAP

import CGNS.NAV.wmessages as MSG
from CGNS.NAV.moption import Q7OptionContext as OCTXT
from CGNS.NAV.wstylesheets import Q7TREEVIEWSTYLESHEET, Q7TABLEVIEWSTYLESHEET
from CGNS.NAV.wstylesheets import Q7CONTROLVIEWSTYLESHEET

# -----------------------------------------------------------------
class Q7Window(QWidget,object):
    VIEW_CONTROL='C'
    VIEW_OPTION='O'
    VIEW_TREE='T'
    VIEW_VTK='G'
    VIEW_FORM='F'
    VIEW_QUERY='Q'
    STATUS_UNCHANGED='U'
    STATUS_MODIFIED='M'
    HISTORYLASTKEY='///LAST///'
    def __init__(self,vtype,control,path,fgprint):
        QWidget.__init__(self,None)
        self._stylesheet=None
        if (vtype==Q7Window.VIEW_TREE): self._stylesheet=Q7TREEVIEWSTYLESHEET
        if (vtype==Q7Window.VIEW_FORM): self._stylesheet=Q7TABLEVIEWSTYLESHEET
        if (vtype==Q7Window.VIEW_CONTROL):
            self._stylesheet=Q7CONTROLVIEWSTYLESHEET
        self.setupUi(self)
        if (self._stylesheet is not None): self.setStyleSheet(self._stylesheet)
        self.getOptions()
        self._timercount=0
        self._vtype=vtype
        self._path=path
        self._control=control
        self._fgprint=fgprint
        self._index=self.addChildWindow()
        if (self._index==0):
            tit="%s:%s%.3d"%(OCTXT._ToolName,self._vtype,self._index)
        else:
            tit="%s:Control"%(OCTXT._ToolName)            
        self.setWindowTitle(tit)
        try:
            self.bBackControl.clicked.connect(self.backcontrol)
        except AttributeError:
            pass
    def validateOption(self,name,value):
        return True
    def getOptions(self):
        self._options=OCTXT._readOptions(self)
        if (self._options is None): self._options=OCTXT()
        return self._options
    def setOptions(self):
        OCTXT._writeOptions(self)
    def getOptionValue(self,name):
        return self._options[name]
    def setOptionValue(self,name,value):
        if (self.validateOption(name,value)):
            self._options[name]=value
            return value
        return self._options[name]
    def getHistoryLastKey(self):
        return Q7Window.HISTORYLASTKEY
    def getHistory(self):
        self._history=OCTXT._readHistory(self)
        if (self._history is None): self._history={}
        return self._history
    def getHistoryFile(self,file):
        if (self._history is None): return None
        for d in self._history:
            if (file in self._history[d]): return (d,file)
        return None
    def setHistory(self,filedir,filename):
        for d in self._history.keys():
            if (d==filedir):
                if (filename not in self._history[filedir]):
                    self._history[filedir].append(filename)
            else:
                self._history[filedir]=[filename]
        if (self._history=={}): self._history[filedir]=[filename]
        self._history[Q7Window.HISTORYLASTKEY]=(filedir,filename)
        OCTXT._writeHistory(self)
        return self._history
    def getLastFile(self):
        if ((self._history=={})
            or not self._history.has_key(Q7Window.HISTORYLASTKEY)):
            return None
        return self._history[Q7Window.HISTORYLASTKEY]
    def addChildWindow(self):
        if (self._fgprint is None): return 0
        self._index=self._fgprint.addChild(self._vtype,self)
        l=[Q7Window.STATUS_UNCHANGED,self._vtype,'%.3d'%self._index]
        l+=[self._fgprint.filedir,self._fgprint.filename,self._path]
        self._control.addLine(l)
        return self._index
    def closeEvent(self, event):
        self._control.delLine('%.3d'%self._index)
        event.accept()
    def backcontrol(self):
        self._fgprint.raiseControlView()
    def busyCursor(self):
        QApplication.setOverrideCursor(QCursor(QPixmap(":/images/icons/cgSpy.gif")))
    def readyCursor(self):
        QApplication.restoreOverrideCursor()

# -----------------------------------------------------------------
class Q7fingerPrint:
    __viewscounter=0
    __extension=[]
    @classmethod
    def treeLoad(cls,control,selectedfile):
        f=selectedfile
        (filedir,filename)=(os.path.normpath(os.path.dirname(f)),
                            os.path.basename(f))
        slp=OCTXT.LinkSearchPathList
        slp+=[filedir]
        flags=CGNS.MAP.S2P_DEFAULT
        if (OCTXT.CHLoneTrace): flags|=CGNS.MAP.S2P_TRACE
        try:
            (tree,links)=CGNS.MAP.load(f,flags,lksearch=slp)
        except CGNS.MAP.error,e:
            control.readyCursor()
            txt="""The current operation has been aborted, while trying to load a file, the following error occurs:"""
            MSG.wError(e[0],txt,e[1])
            return None
        return Q7fingerPrint(control,filedir,filename,tree,links)
    @classmethod
    def closeAllTrees(cls):
        for x in cls.__extension: x.closeAllViews()
    @classmethod
    def raiseView(cls,idx):
        for x in cls.__extension:
            for vtype in x.views:
                for (v,i) in x.views[vtype]:
                    if (i==int(idx)): v.raise_()
    def __init__(self,control,filedir,filename,tree,links,**kw):
        self.filename=filename
        self.tree=tree
        self.filedir=filedir
        self.links=links
        self.model=None
        self.depth=0
        self.views={}
        self.control=control
        Q7fingerPrint.__extension.append(self)
    def raiseControlView(self):
        self.control.raise_()
    def addChild(self,viewtype,view):
        Q7fingerPrint.__viewscounter+=1
        idx=Q7fingerPrint.__viewscounter
        if not self.views.has_key(viewtype): self.views[viewtype]=[]
        self.views[viewtype].append((view,idx))
        return Q7fingerPrint.__viewscounter
    def closeAllViews(self):
        for vtype in self.views:
            for (v,i) in self.views[vtype]: v.close()
        
# -----------------------------------------------------------------
