#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
from PySide.QtCore       import *
from PySide.QtGui        import *
import os
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
    VIEW_SELECT='S'
    HISTORYLASTKEY='///LAST///'
    def __init__(self,vtype,control,path,fgprint):
        QWidget.__init__(self,None)
        self._stylesheet=None
        self.I_UNCHANGED=QIcon(QPixmap(":/images/icons/save-done.gif"))
        self.I_MODIFIED=QIcon(QPixmap(":/images/icons/save.gif"))
        self.I_CONVERTED=QIcon(QPixmap(":/images/icons/save-converted.gif"))
        self.I_TREE=QIcon(QPixmap(":/images/icons/tree-load.gif"))
        self.I_VTK=QIcon(QPixmap(":/images/icons/vtk.gif"))
        self.I_QUERY=QIcon(QPixmap(":/images/icons/operate-execute.gif"))
        self.I_FORM=QIcon(QPixmap(":/images/icons/form.gif"))
        self.I_SELECT=QIcon(QPixmap(":/images/icons/operate-list.gif"))
        if (vtype==Q7Window.VIEW_TREE): self._stylesheet=Q7TREEVIEWSTYLESHEET
        if (vtype==Q7Window.VIEW_FORM): self._stylesheet=Q7TABLEVIEWSTYLESHEET
        if (vtype==Q7Window.VIEW_CONTROL):
            self._stylesheet=Q7CONTROLVIEWSTYLESHEET
        self.setupUi(self)
        if (self._stylesheet is not None): self.setStyleSheet(self._stylesheet)
        self._busyx=QCursor(QPixmap(":/images/icons/cgSpy.gif"))
        self.getOptions()
        self._timercount=0
        self._vtype=vtype
        self._path=path
        self._control=control
        self._fgprint=fgprint
        self._index=self.addChildWindow()
        if (self._index!=0):
            tit="%s:%s%.3d"%(OCTXT._ToolName,self._vtype,self._index)
        else:
            tit="%s:Control"%(OCTXT._ToolName)            
        self.setWindowTitle(tit)
        try:
            self.bBackControl.clicked.connect(self.backcontrol)
        except AttributeError:
            pass
        self._readonly=False
    def validateOption(self,name,value):
        #if (name[0]=='_'): return False
        return True
    def getOptions(self):
        try:
            if (self._options is None): self._options={}
        except AttributeError:
            self._options={}
        user_options=OCTXT._readOptions(self)
        for k in dir(OCTXT):   self.setOptionValue(k,getattr(OCTXT,k))
        for k in user_options: self.setOptionValue(k,user_options[k])
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
    def getQueries(self):
        self._queries=OCTXT._readQueries(self)
        if (self._queries is None): self._queries=[]
        return self._queries
    def setQueries(self):
        OCTXT._writeQueries(self)
        return self._queries
    def getLastFile(self):
        if ((self._history=={})
            or not self._history.has_key(Q7Window.HISTORYLASTKEY)):
            return None
        return self._history[Q7Window.HISTORYLASTKEY]
    def addChildWindow(self):
        if (self._fgprint is None): return 0
        self._index=self._fgprint.addChild(self._vtype,self)
        l=[self._fgprint._status,self._vtype,'%.3d'%self._index]
        l+=[self._fgprint.filedir,self._fgprint.filename,self._path]
        self._control.addLine(l)
        return self._index
    def closeEvent(self, event):
        self._control.delLine('%.3d'%self._index)
        event.accept()
    def backcontrol(self):
        self._fgprint.raiseControlView()
    def busyCursor(self):
        QApplication.setOverrideCursor(self._busyx)
    def readyCursor(self):
        QApplication.restoreOverrideCursor()
    def _T(self,msg):
        if (self.getOptionValue('NAVTrace')):
            print '### CGNS.NAV:', msg
# -----------------------------------------------------------------
class Q7fingerPrint:
    __viewscounter=0
    __extension=[]
    STATUS_UNCHANGED='U'
    STATUS_MODIFIED='M'
    STATUS_CONVERTED='C'
    STATUS_LIST=(STATUS_UNCHANGED,STATUS_MODIFIED,STATUS_CONVERTED)
    @classmethod
    def fileconversion(cls,fdir,filein,control):
        fileout=control.getOptionValue('TemporaryDirectory')+'/'+filein+'.hdf'
        count=1
        while (os.path.exists(fileout)):
            fileout=control.getOptionValue('TemporaryDirectory')+\
                     '/'+filein+'.%.3d.hdf'%count
            count+=1
        com='(cd %s; %s -h %s %s)'%(fdir,
                                    control.getOptionValue('ADFConversionCom'),
                                    filein,fileout)
        os.system(com)
        return fileout
    @classmethod
    def treeLoad(cls,control,selectedfile):
        kw={}
        f=selectedfile
        (filedir,filename)=(os.path.normpath(os.path.dirname(f)),
                            os.path.basename(f))
        slp=control.getOptionValue('LinkSearchPathList')
        slp+=[filedir]
        if (   (os.path.splitext(filename)[1]=='.cgns')
            and control.getOptionValue('ConvertADFFiles')):
            f=cls.fileconversion(filedir,filename,control)
            kw['converted']=True
        flags=CGNS.MAP.S2P_DEFAULT
        if (control.getOptionValue('CHLoneTrace')): flags|=CGNS.MAP.S2P_TRACE
        try:
            (tree,links)=CGNS.MAP.load(f,flags,lksearch=slp)
        except CGNS.MAP.error,e:
            control.readyCursor()
            txt="""The current operation has been aborted, while trying to load a file, the following error occurs:"""
            MSG.wError(e[0],txt,e[1])
            return None
        return Q7fingerPrint(control,filedir,filename,tree,links,**kw)
    @classmethod
    def closeAllTrees(cls):
        for x in cls.__extension: x.closeAllViews()
    @classmethod
    def raiseView(cls,idx):
        for x in cls.__extension:
            for vtype in x.views:
                for (v,i) in x.views[vtype]:
                    if (i==int(idx)): v.raise_()
    @classmethod
    def getView(cls,idx):
        for x in cls.__extension:
            for vtype in x.views:
                for (v,i) in x.views[vtype]:
                    if (i==int(idx)): return v
        return None
    @classmethod
    def getFingerPrint(cls,idx):
        for x in cls.__extension:
            for vtype in x.views:
                for (v,i) in x.views[vtype]:
                    if (i==int(idx)): return x
        return None
    def __init__(self,control,filedir,filename,tree,links,**kw):
        self.filename=filename
        self.tree=tree
        self.filedir=filedir
        self.links=links
        self.model=None
        self.depth=0
        self.views={}
        self.control=control
        self.converted=False
        self._status=Q7fingerPrint.STATUS_UNCHANGED
        if (kw.has_key('converted')):
            self.converted=kw['converted']
            if (self.converted):
                self._status=Q7fingerPrint.STATUS_CONVERTED
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
    def isModified(self):
        return (self._status==Q7fingerPrint.STATUS_MODIFIED)
    def modifiedTreeStatus(self,status=None):
        if (status not in Q7fingerPrint.STATUS_LIST): return
        if (status is not None): self._status=status
        elif (self._status==Q7fingerPrint.STATUS_UNCHANGED):
            self._status=Q7fingerPrint.STATUS_MODIFIED
        elif (self._status==Q7fingerPrint.STATUS_MODIFIED):
            self._status=Q7fingerPrint.STATUS_UNCHANGED
        elif (self._status==Q7fingerPrint.STATUS_CONVERTED):
            self._status=Q7fingerPrint.STATUS_MODIFIED
        else:
            pass
        self.control.updateViews()
# -----------------------------------------------------------------
