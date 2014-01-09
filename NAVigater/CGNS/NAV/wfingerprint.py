#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from PySide.QtCore import QCoreApplication
from PySide.QtCore import *
from PySide.QtGui  import *
import os
import os.path
import re

try:
  from pwd import getpwuid
except ImportError:
  def getpwuid(a):
    return (0,)
try:
    from grp import getgrgid
except ImportError:
  def getgrgid(a):
    return (0,)
import time
import stat
import sys
import CGNS.MAP
import CGNS.PAT.cgnsutils as CGU

import CGNS.NAV.wmessages as MSG
from CGNS.NAV.moption import Q7OptionContext as OCTXT
from CGNS.NAV.wstylesheets import Q7TREEVIEWSTYLESHEET, Q7TABLEVIEWSTYLESHEET
from CGNS.NAV.wstylesheets import Q7CONTROLVIEWSTYLESHEET
from CGNS.NAV.wfile import checkFilePermission


# -----------------------------------------------------------------
class Q7CHLoneProxy(object):
    def __init__(self,hasthreads=False):
        self._data=None
        self._hasthreads=hasthreads
        self._thread=None
    def kill(self):
        if (self._thread is not None): self._thread.quit()
    def load(self,control,selectedfile):
        self._control=control
        self._thread=Q7CHLoneThread(control,selectedfile)
        self._thread.datacompleted.connect(self.proxyCompleted, Qt.QueuedConnection)
        self._data=None
        if (self._hasthreads): self._thread.start()
        else:                  self._thread.run()
    @property
    def data(self):
        return self._data
    @Slot(str)
    def proxyCompleted(self, data):
        if (data[0] is None):  self._data=data
        else : self._data=data[0]
        self._control.signals.fgprint=self._data
        self._control.signals.loadCompleted.emit()

# -----------------------------------------------------------------
class Q7CHLoneThread(QThread):
    datacompleted=Signal(tuple)
    def __init__(self, control, selectedfile):
        super(Q7CHLoneThread, self).__init__(None)
        self._control=control
        self._selectedfile=selectedfile
        self._data=None
    def run(self):
        control=self._control
        selectedfile=self._selectedfile
        Q7FingerPrint.Lock()
        kw={}
        (filedir,filename)=(os.path.abspath(os.path.dirname(selectedfile)),
                            os.path.basename(selectedfile))
        if ("%s/%s"%(filedir,filename) in Q7FingerPrint.getExpandedFilenameList()):
            Q7FingerPrint.Unlock()
            txt="""The current file is already open:"""
            self._data=(None,(0,"%s/%s"%(filedir,filename),txt))
            self.datacompleted.emit(self._data)
            control.readyCursor()
            return 
        slp=OCTXT.LinkSearchPathList
        slp+=[filedir]
        loadfilename=selectedfile
        flags=CGNS.MAP.S2P_NONE&~CGNS.MAP.S2P_REVERSEDIMS
        maxdataload=-1
        if (OCTXT.CHLoneTrace):
            flags|=CGNS.MAP.S2P_TRACE
        if (OCTXT.DoNotLoadLargeArrays):
            flags|=CGNS.MAP.S2P_NODATA
            maxdataload=OCTXT.MaxLoadDataSize
        if (OCTXT.FollowLinksAtLoad):
            flags|=CGNS.MAP.S2P_FOLLOWLINKS
        try:
            if (not CGNS.MAP.probe(loadfilename)):
              if ((os.path.splitext(filename)[1] in OCTXT.CGNSFileExtension)
                  and OCTXT._ConvertADFFiles):
                loadfilename=Q7FingerPrint.fileconversion(filedir,
                                                          filename,control)
                kw['converted']=True
                kw['convertedAs']=loadfilename
            if (maxdataload):
                (tree,links,paths)=CGNS.MAP.load(loadfilename,flags=flags,
                                                 lksearch=slp,
                                                 maxdata=maxdataload)
            else:
                (tree,links,paths)=CGNS.MAP.load(loadfilename,flags=flags,
                                                 lksearch=slp)
        except (CGNS.MAP.error,),chlex:
            Q7FingerPrint.Unlock()
            txt="""The current load operation has been aborted:"""
            self._data=(None,(chlex[0],txt,chlex[1]))
            self.datacompleted.emit(self._data)
            control.readyCursor()
            return 
        except Exception, e:
            Q7FingerPrint.Unlock()
            txt="""The current operation has been aborted: %s"""%e
            self._data=(None,(0,txt,''))
            self.datacompleted.emit(self._data)
            control.readyCursor()
            return 
        kw['isfile']=True
        Q7FingerPrint.Unlock()
        self._data=(Q7FingerPrint(control,filedir,filename,
                                  tree,links,paths,**kw),)
        self.datacompleted.emit(self._data)

# -----------------------------------------------------------------
class Q7Window(QWidget,object):
    VIEW_CONTROL='C'
    VIEW_DIAG='D'
    VIEW_TOOLS='X'
    VIEW_TREE='T'
    VIEW_OPTION='O'
    VIEW_VTK='G'
    VIEW_FORM='F'
    VIEW_QUERY='Q'
    VIEW_SELECT='S'
    VIEW_PATTERN='P'
    VIEW_INFO='I'
    VIEW_LINK='L'
    VIEW_MESSAGE='M'
    HISTORYLASTKEY='///LAST///'
    control_log=None
    def __init__(self,vtype,control,path,fgprint):
        QWidget.__init__(self,None)
        self._stylesheet=None
        self.I_MOD_SAV=QIcon(QPixmap(":/images/icons/save-S-M.png"))
        self.I_UMOD_SAV=QIcon(QPixmap(":/images/icons/save-S-UM.png"))
        self.I_MOD_USAV=QIcon(QPixmap(":/images/icons/save-US-M.png"))
        self.I_UMOD_USAV=QIcon(QPixmap(":/images/icons/save-US-UM.png"))
        self.I_TREE=QIcon(QPixmap(":/images/icons/tree-load.png"))
        self.I_VTK=QIcon(QPixmap(":/images/icons/vtkview.png"))
        self.I_QUERY=QIcon(QPixmap(":/images/icons/operate-execute.png"))
        self.I_FORM=QIcon(QPixmap(":/images/icons/form.png"))
        self.I_SELECT=QIcon(QPixmap(":/images/icons/mark-node.png"))
        self.I_PATTERN=QIcon(QPixmap(":/images/icons/pattern.png"))
        self.I_DIAG=QIcon(QPixmap(":/images/icons/check-all.png"))
        self.I_TOOLS=QIcon(QPixmap(":/images/icons/toolsview.png"))
        self.I_LINK=QIcon(QPixmap(":/images/icons/link-node.png"))
        self.I_D_INF=QIcon(QPixmap(":/images/icons/subtree-sids-warning.png"))
        self.I_D_ERR=QIcon(QPixmap(":/images/icons/subtree-sids-failed.png"))
        self.I_L_OKL=QIcon(QPixmap(":/images/icons/link.png"))
        self.I_L_NFL=QIcon(QPixmap(":/images/icons/link-nofile.png"))
        self.I_L_NRL=QIcon(QPixmap(":/images/icons/link-noreadable.png"))
        self.I_L_NNL=QIcon(QPixmap(":/images/icons/link-nonode.png"))
        self.I_L_IGN=QIcon(QPixmap(":/images/icons/link-error.png"))
        self.I_L_ERL=QIcon(QPixmap(":/images/icons/link-ignore.png"))
        self.I_C_SFL=QIcon(QPixmap(":/images/icons/check-fail.png"))
        self.I_C_SWR=QIcon(QPixmap(":/images/icons/check-warn.png"))
        self.I_EMPTY=QIcon(QPixmap(":/images/icons/empty.png"))
        self.I_MARK=QIcon(QPixmap(":/images/icons/mark-node.png"))
        if (vtype==Q7Window.VIEW_TREE):
            self._stylesheet=Q7TREEVIEWSTYLESHEET
        if (vtype==Q7Window.VIEW_CONTROL):
            self._stylesheet=Q7CONTROLVIEWSTYLESHEET
        self.setupUi(self)
        if (self._stylesheet is not None): self.setStyleSheet(self._stylesheet)
        self._busyx=QCursor(QPixmap(":/images/icons/cgSpy.png"))
        self.getOptions()
        self._timercount=0
        self._vtype=vtype
        self._path=path
        self._control=control
        self._fgprint=fgprint
        self._index=self.addChildWindow()
        self._lockableWidgets=[]
        self._lockedWidgets=[]
        fn=''
        if (fgprint is not None): fn=fgprint.filename
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
        if (not control.verbose):
          sys.stdout=self.control_log
          sys.stderr=self.control_log
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
        for k in OCTXT._Default_Fonts:
            self.setOptionValue(k,OCTXT._Default_Fonts[k])
        if (user_options is not None):
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
    def matchFileExtensions(self,filename):
        xl=self.getFileExtensions()
        (name,ext)=os.path.splitext(os.path.basename(filename))
        return ((xl==[]) or (ext in xl))
    def getFileExtensions(self):
        xlist=[]
        if (OCTXT.FilterCGNSFiles): xlist+=OCTXT.CGNSFileExtension
        if (OCTXT.FilterHDFFiles):  xlist+=OCTXT.HDFFileExtension
        if (OCTXT.FilterOwnFiles):  xlist+=OCTXT.OwnFileExtension
        return xlist
    def getHistoryLastKey(self):
        return Q7Window.HISTORYLASTKEY
    def getHistory(self,fromfile=True):
        if (fromfile):
          self._history=OCTXT._readHistory(self)
          if (self._history is None): self._history={}
        return self._history
    def getHistoryFile(self,file):
        if (self._history is None): return None
        for d in self._history:
            if (file in self._history[d]): return (d,file)
        return None
    def destroyHistory(self):
        self._history={}
        OCTXT._writeHistory(self)
        return self._history
    def getDirNotFoundFromHistory(self):
        nf=[]
        for d in self._history.keys():
          if (not os.path.exists(d) and not (d==Q7Window.HISTORYLASTKEY)):
            nf.append(d)
        return nf
    def getDirNoHDFFromHistory(self):
        nf=[]
        for d in self._history.keys():
          if (not os.path.exists(d) or (d==Q7Window.HISTORYLASTKEY)):
            pass
          else:
            fl=[f for f in os.listdir(d) if self.matchFileExtensions(f)]
            if (not fl): nf.append(d)
        return nf
    def removeDirFromHistory(self,filedir):
        if (self._history.has_key(filedir)):
          del self._history[filedir]
        if ((self.getLastFile() is not None)
            and (self._history[Q7Window.HISTORYLASTKEY][0]==filedir)):
            del self._history[Q7Window.HISTORYLASTKEY]
        OCTXT._writeHistory(self)
    def removeFileFromHistory(self,filedir,filename):
        if (self._history.has_key(filedir)
            and (filename in self._history[filedir])):
            self._history[filedir].remove(filename)
        if ((self.getLastFile() is not None)
            and (self._history[Q7Window.HISTORYLASTKEY][1]==filename)):
            del self._history[Q7Window.HISTORYLASTKEY]
        OCTXT._writeHistory(self)
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
        self._control.addLine(l,self._fgprint)
        return self._index
    def closeEvent(self, event):
        self._control.delLine('%.3d'%self._index)
        if (self._fgprint is not None):
            self._fgprint.closeView(self._index)
        event.accept()
    def backcontrol(self):
        if (self._fgprint is not None):
            self._fgprint.raiseControlView()
        self._control.selectLine('%.3d'%self._index)
    def busyCursor(self):
        self._control.lockView(True)
        Q7FingerPrint.disableAllViewsButtons(True)
    def readyCursor(self):
        self._control.lockView(False)
        Q7FingerPrint.disableAllViewsButtons(False)
    def lockView(self,lock):
        for wid in self._lockableWidgets:
            if (not lock and wid in self._lockedWidgets):
                wid.setDisabled(lock)
                self._lockedWidgets.remove(wid)
            if (lock and wid.isEnabled()):
                self._lockedWidgets.append(wid)
                wid.setDisabled(lock)
        if (self._fgprint is not None): self._fgprint._locked=lock
    def isLocked(self):
        if (self._fgprint is None): return self._fgprint._locked
        return False
    def lockable(self,widget):
        self._lockableWidgets.append(widget)
    def refreshScreen(self):
        QCoreApplication.processEvents()
    def setLabel(self,it,text):
        it.setText(text)
        it.setFont(QFont("Courier"))
        it.setReadOnly(True)
    def _T(self,msg):
        if (self.getOptionValue('NAVTrace')):
            print '### CGNS.NAV:', msg
    def reject(self):
        pass
# -----------------------------------------------------------------
class Q7FingerPrint:
    __viewscounter=0
    __extension=[]
    STATUS_UNCHANGED='U'
    STATUS_MODIFIED='M'
    STATUS_SAVEABLE='S'
    STATUS_CONVERTED='C'
    STATUS_LIST=(STATUS_UNCHANGED,STATUS_MODIFIED,STATUS_CONVERTED,
                 STATUS_SAVEABLE)
    __mutex=QMutex()
    __chloneproxy=None
    @classmethod
    def proxy(cls):
        cls.Lock()
        if (cls.__chloneproxy is None):
            thrd=OCTXT.ActivateMultiThreading
            # BLOCK ACTUAL MULTI-THREADING HERE
            cls.__chloneproxy=Q7CHLoneProxy(thrd) 
        cls.Unlock()
        return cls.__chloneproxy
    @classmethod
    def killProxy(cls):
        cls.Lock()
        if (cls.__chloneproxy is not None): cls.__chloneproxy.kill()
        cls.Unlock()
    @classmethod
    def Lock(cls):
        cls.__mutex.lock()
    @classmethod
    def Unlock(cls):
        cls.__mutex.unlock()
    @classmethod
    def fileconversion(cls,fdir,filein,control):
        fileout=OCTXT.TemporaryDirectory+'/'+filein+'.hdf'
        count=1
        while (os.path.exists(fileout)):
            fileout=OCTXT.TemporaryDirectory+'/'+filein+'.%.3d.hdf'%count
            count+=1
        com='(cd %s; %s -f -h %s %s)'%(fdir,
                                       OCTXT.ADFConversionCom,
                                       filein,fileout)
        os.system(com)
        return fileout
    @classmethod
    def treeLoad(cls,control,selectedfile):
        control.loadOptions()
        proxy=cls.proxy()
        proxy.load(control,selectedfile)
        cls.refreshScreen()
        return
    @classmethod
    def treeSave(cls,control,fgprint,f,saveas):
        flags=CGNS.MAP.S2P_DEFAULT
        if (OCTXT.CHLoneTrace): 
          flags|=CGNS.MAP.S2P_TRACE
        if (not saveas):        flags|=CGNS.MAP.S2P_UPDATE
        tree=fgprint.tree
        lazylist=[ CGU.getPathNoRoot(path) for path in fgprint.lazy.keys() ]
        try:
            CGNS.MAP.save(f,tree,links=fgprint.links,flags=flags,skip=lazylist)
        except (CGNS.MAP.error,),chlex:
            txt="""The current save operation has been aborted (CHLone):"""
            control.readyCursor()
            MSG.wError(control,chlex[0],txt,chlex[1])
            return None
        except (Exception,), e:
            txt="""The current save operation has been aborted: %s"""%e
            control.readyCursor()
            MSG.wError(control,0,txt,'')
            return None
        fgprint.updateFileStats(f,saveas=True)
    @classmethod
    def closeAllTrees(cls):
        while cls.__extension:
            x=cls.__extension[0]
            x.closeAllViews()
    @classmethod
    def refreshScreen(cls):
        QCoreApplication.processEvents()
    @classmethod
    def disableAllViewsButtons(cls,lock):
        cls.Lock()
        for x in cls.__extension:
          for vtype in x.views:
            for (v,i) in x.views[vtype]:
              for wid in v._lockableWidgets:
                if (not lock and wid in v._lockedWidgets):
                    wid.setDisabled(lock)
                    v._lockedWidgets.remove(wid)
                if (lock and wid.isEnabled()):
                    v._lockedWidgets.append(wid)
                    wid.setDisabled(lock)
              if (v._fgprint is not None): v._fgprint._locked=lock
        cls.Unlock()
    @classmethod
    def raiseView(cls,idx):
        cls.Lock()
        for x in cls.__extension:
            for vtype in x.views:
                for (v,i) in x.views[vtype]:
                    if (i==int(idx)): v.raise_()
        cls.Unlock()
    @classmethod
    def infoView(cls,idx):
        f=cls.getFingerPrint(idx)
        v=cls.getView(idx)
        if (f is None): return (None,None,None)
        if (not f.isfile): return (f,None,None)
        return (f,v,f.getInfo())
    @classmethod
    def getView(cls,idx):
        for x in cls.__extension:
            for vtype in x.views:
                for (v,i) in x.views[vtype]:
                    if (i==int(idx)): return v
        return None
    @classmethod
    def getViewType(cls,idx):
        vw=cls.getView(idx)
        for x in cls.__extension:
            for vtype in x.views:
                for (v,i) in x.views[vtype]:
                    if ((v==vw) and (i==int(idx))): return vtype
        return None
    @classmethod
    def getFingerPrint(cls,idx):
        for x in cls.__extension:
            for vtype in x.views:
                for (v,i) in x.views[vtype]:
                    if (i==int(idx)): return x
        return None
    @classmethod
    def getUniqueTreeViewIdList(cls):
        r=set()
        for x in cls.__extension:
            for vtype in x.views:
                if (vtype==Q7Window.VIEW_TREE):
                  for (v,i) in x.views[vtype]:
                      r.add(i)
        return list(r)
    @classmethod
    def getExpandedFilenameList(cls):
        l=[]
        for x in cls.__extension:
            l.append("%s/%s"%(x.filedir,x.filename))
        return l
    # -------------------------------------------------------------
    def __init__(self,control,filedir,filename,tree,links,paths,**kw):
        if (control is None): return # __root instance, empty
        self.filename=filename
        self.tree=tree
        self.filedir=filedir
        self.links=links
        self.model=None
        self.depth=0
        self.nodes=0
        self.views={}
        self.control=control
        self.converted=False
        self.isfile=False
        self.infoData={}
        self.tmpfile=''
        self._kw=kw
        self._status=[]
        self._locked=False
        if (checkFilePermission(filedir+'/'+filename,write=True)):
            self._status=[Q7FingerPrint.STATUS_SAVEABLE]
        if (kw.has_key('isfile')):
            self.isfile=True
        if (kw.has_key('converted')):
            self.converted=kw['converted']
            self.tmpfile=kw['convertedAs']
            if (self.converted):
                self._status=[]
        self.lazy={}
        for p in paths: self.lazy['/CGNSTree'+p[0]]=p[1]
        Q7FingerPrint.__extension.append(self)
        self.updateFileStats(filedir+'/'+filename)
    def __len__(self):
        return 0
    def isLocked(self):
        return self._locked
    def updateNodeData(self,pathdict):
        tfile="%s/%s"%(self.filedir,self.filename)
        slp=OCTXT.LinkSearchPathList
        slp+=[self.filedir]
        minpath=CGU.getPathListCommonAncestor(pathdict.keys())
        flags=CGNS.MAP.S2P_NONE&~CGNS.MAP.S2P_REVERSEDIMS
        if (OCTXT.CHLoneTrace):
            flags|=CGNS.MAP.S2P_TRACE
        if (OCTXT.FollowLinksAtLoad):
            flags|=CGNS.MAP.S2P_FOLLOWLINKS
        (t,l,p)=CGNS.MAP.load(tfile,flags=flags,path=minpath,lksearch=slp,
                              update=pathdict)
        return (t,l,p)
    def updateFileStats(self,fname,saveas=False):
        (filedir,filename)=(os.path.normpath(os.path.dirname(fname)),
                            os.path.basename(fname))
        self.filename=filename
        self.filedir=filedir
        self.version=CGU.getVersion(self.tree)
        self.removeTreeStatus(Q7FingerPrint.STATUS_MODIFIED)
        self.addTreeStatus(Q7FingerPrint.STATUS_UNCHANGED)
        if (saveas):
            self.converted=False
            self.isfile=True
            self.tmpfile=''
            self.removeTreeStatus(Q7FingerPrint.STATUS_CONVERTED)
            self.addTreeStatus(Q7FingerPrint.STATUS_SAVEABLE)
    def isFile(self):
        return self.isfile
    def isLink(self,path):
        pth=CGU.getPathNoRoot(path)
        for lk in self.links:
            if (lk[3]==pth): return lk
        return False
    def fileHasChanged(self):
        dnow={}
        self.readInfoFromOS(dnow)
        for k in dnow:
            if (dnow[k]!=self.infoData[k]):
                print k,dnow[k],self.infoData[k]
                return True
        return False
    def readInfoFromOS(self,d):
        f='%s/%s'%(self.filedir,self.filename)
        d['eFilename']=f
        d['eDirSource']=self.filedir
        d['eFileSource']=self.filename
        d['eTmpFile']=self.tmpfile
        d['eDepth']=self.depth
        d['eNodes']=self.nodes
        d['eVersion']=str(self.version)
        d['eVersionHDF5']='???'
        st=os.stat(f)
        d['eFileSize']="%.3f Mb"%(1.0*st[6]/(1024*1024))
        d['eMergeSize']="%.3f Mb"%(1.0*st[6]/(1024*1024))
        dfmt="%Y-%m-%d %H:%M"
        d['eLastDate']=time.strftime(dfmt,time.localtime(int(st[7])))
        d['eModifDate']=time.strftime(dfmt,time.localtime(int(st[8])))
        e=getpwuid(st[4])
        g=getgrgid(st[5])
        d['eOwner']=e[0]
        d['eGroup']=g[0]
        d['cNoFollow']=False
        d['cHasLinks']=(len(self.links)!=0)
        d['cSameFS']=False
        d['cBadLinks']=False
        d['cModeProp']=False
        m=""
        if (st[0] & stat.S_IRUSR):m+="r"
        else: m+="-"
        if (st[0] & stat.S_IWUSR):m+="w"
        else: m+="-"
        if (st[0] & stat.S_IXUSR):m+="x"
        else: m+="-"
        if (st[0] & stat.S_IRGRP):m+="r"
        else: m+="-"
        if (st[0] & stat.S_IWGRP):m+="w"
        else: m+="-"
        if (st[0] & stat.S_IXGRP):m+="x"
        else: m+="-"
        if (st[0] & stat.S_IROTH):m+="r"
        else: m+="-"
        if (st[0] & stat.S_IWOTH):m+="w"
        else: m+="-"
        if (st[0] & stat.S_IXOTH):m+="x"
        else: m+="-"
        d['eRights']=m
        d['cConverted']=self.converted
        d['cADF']=self.converted
        d['cHDF5']=not self.converted
        d['cREAD']=not self.isSaveable()
        d['cMODIFY']=self.isSaveable()
        d['cNODATA']=False
        d['cHasInt64']=False
        return d
    def getInfo(self,force=False):
        if (force or not self.infoData): 
            self.readInfoFromOS(self.infoData)
        return self.infoData
    def raiseControlView(self):
        self.control.raise_()
    def addChild(self,viewtype,view):
        Q7FingerPrint.__viewscounter+=1
        idx=Q7FingerPrint.__viewscounter
        if not self.views.has_key(viewtype): self.views[viewtype]=[]
        self.views[viewtype].append((view,idx))
        return Q7FingerPrint.__viewscounter
    def closeView(self,i):
        idx=int(i)
        fg=self.getFingerPrint(idx)
        vw=self.getView(idx)
        vt=self.getViewType(idx)
        if ((vw is not None) and self.views.has_key(vt)):
          self.views[vt].remove((vw,idx))
          if (self.views[vt]==[]): del self.views[vt]
        if (vw is not None): vw.reject()
        if ((self.views=={}) and (fg in self.__extension)):
            self.__extension.remove(fg)
    def unlockAllViews(self):
        self.lockAllViews(lock=False)
    def lockAllViews(self,lock=True):
        vtlist=self.views.keys()
        for vtype in vtlist:
            for (v,i) in self.views[vtype]: v.lockView(lock)
    def closeAllViews(self):
        vtlist=self.views.keys()
        for vtype in vtlist:
            for (v,i) in self.views[vtype]: self.closeView(i)
    def isModified(self):
        return (Q7FingerPrint.STATUS_MODIFIED in self._status)
    def isSaveable(self):
        return (Q7FingerPrint.STATUS_SAVEABLE in self._status)
    def removeTreeStatus(self,status):
        if (status not in Q7FingerPrint.STATUS_LIST): return
        if (status in self._status): self._status.remove(status)
        self.control.updateViews()
    def addTreeStatus(self,status):
        if (status not in Q7FingerPrint.STATUS_LIST): return
        if (status not in self._status): self._status.append(status)
        self.control.updateViews()
# -----------------------------------------------------------------
