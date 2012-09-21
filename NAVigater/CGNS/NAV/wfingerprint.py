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
import pwd
import grp
import time
import stat
import CGNS.MAP
import CGNS.PAT.cgnsutils as CGU

import CGNS.NAV.wmessages as MSG
from CGNS.NAV.moption import Q7OptionContext as OCTXT
from CGNS.NAV.wstylesheets import Q7TREEVIEWSTYLESHEET, Q7TABLEVIEWSTYLESHEET
from CGNS.NAV.wstylesheets import Q7CONTROLVIEWSTYLESHEET

# -----------------------------------------------------------------
class Q7Window(QWidget,object):
    VIEW_CONTROL='C'
    VIEW_DIAG='D'
    VIEW_TREE='T'
    VIEW_OPTION='O'
    VIEW_VTK='G'
    VIEW_FORM='F'
    VIEW_QUERY='Q'
    VIEW_SELECT='S'
    VIEW_INFO='I'
    VIEW_LINK='L'
    HISTORYLASTKEY='///LAST///'
    def __init__(self,vtype,control,path,fgprint):
        QWidget.__init__(self,None)
        self._stylesheet=None
        self.I_UNCHANGED=QIcon(QPixmap(":/images/icons/save-inactive.gif"))
        self.I_MODIFIED=QIcon(QPixmap(":/images/icons/save.gif"))
        self.I_CONVERTED=QIcon(QPixmap(":/images/icons/save-converted.gif"))
        self.I_TREE=QIcon(QPixmap(":/images/icons/tree-load.gif"))
        self.I_VTK=QIcon(QPixmap(":/images/icons/vtkview.gif"))
        self.I_QUERY=QIcon(QPixmap(":/images/icons/operate-execute.gif"))
        self.I_FORM=QIcon(QPixmap(":/images/icons/form.gif"))
        self.I_SELECT=QIcon(QPixmap(":/images/icons/operate-list.gif"))
        self.I_DIAG=QIcon(QPixmap(":/images/icons/check-all.gif"))
        self.I_LINK=QIcon(QPixmap(":/images/icons/link.gif"))
        self.I_D_INF=QIcon(QPixmap(":/images/icons/subtree-sids-warning.gif"))
        self.I_D_ERR=QIcon(QPixmap(":/images/icons/subtree-sids-failed.gif"))
        self.I_L_OKL=QIcon(QPixmap(":/images/icons/link.gif"))
        self.I_L_NFL=QIcon(QPixmap(":/images/icons/link-nofile.gif"))
        self.I_L_NRL=QIcon(QPixmap(":/images/icons/link-noreadable.gif"))
        self.I_L_NNL=QIcon(QPixmap(":/images/icons/link-nonode.gif"))
        self.I_L_IGN=QIcon(QPixmap(":/images/icons/link-error.gif"))
        self.I_L_ERL=QIcon(QPixmap(":/images/icons/link-ignore.gif"))
        if (vtype==Q7Window.VIEW_TREE):
            self._stylesheet=Q7TREEVIEWSTYLESHEET
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
        fn=''
        if (fgprint is not None): fn=fgprint.filename
        if (self._index!=0):
            tit="%s:[%s]:%s%.3d"%(OCTXT._ToolName,fn,self._vtype,self._index)
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
        self._control.selectLine('%.3d'%self._index)
    def busyCursor(self):
        QApplication.setOverrideCursor(self._busyx)
    def readyCursor(self):
        QApplication.restoreOverrideCursor()
    def setLabel(self,it,text):
        it.setText(text)
        it.setFont(QFont("Courier"))
        it.setReadOnly(True)
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
        control.loadOptions()
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
        kw={}
        f=selectedfile
        (filedir,filename)=(os.path.normpath(os.path.dirname(f)),
                            os.path.basename(f))
        slp=OCTXT.LinkSearchPathList
        slp+=[filedir]
        if (   (os.path.splitext(filename)[1]=='.cgns')
            and OCTXT._ConvertADFFiles):
            f=cls.fileconversion(filedir,filename,control)
            kw['converted']=True
            kw['convertedAs']=f
        flags=CGNS.MAP.S2P_NONE&~CGNS.MAP.S2P_REVERSEDIMS
        maxdataload=None
        if (OCTXT.CHLoneTrace):
            flags|=CGNS.MAP.S2P_TRACE
        if (OCTXT.DoNotLoadLargeArrays):
            flags|=CGNS.MAP.S2P_NODATA
            maxdataload=OCTXT.MaxLoadDataSize
        if (OCTXT.FollowLinksAtLoad):
            flags|=CGNS.MAP.S2P_FOLLOWLINKS
        try:
            if (maxdataload):
                (tree,links,paths,diag)=CGNS.MAP.load2(f,flags,lksearch=slp,
                                                       maxdata=maxdataload)
            else:
                (tree,links,paths,diag)=CGNS.MAP.load2(f,flags,lksearch=slp)
            if (diag is not None):
                txt="""Loading process returns:"""
                control.readyCursor()
                MSG.wError(diag[0],txt,diag[1])
        except Exception, e:
            txt="""The current operation has been aborted: %s"""%e
            control.readyCursor()
            MSG.wError(0,txt,'')
            return None
        kw['isfile']=True
        return Q7fingerPrint(control,filedir,filename,tree,links,paths,**kw)
    @classmethod
    def treeSave(cls,control,fgprint,f):
        flags=CGNS.MAP.S2P_DEFAULT
        if (OCTXT.CHLoneTrace): flags|=CGNS.MAP.S2P_TRACE
        tree=fgprint.tree
        #for p in CGU.getAllPaths(tree): print p
        lk=[]
        CGNS.MAP.save(f,tree,lk,flags)
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
    def infoView(cls,idx):
        f=cls.getFingerPrint(idx)
        v=cls.getView(idx)
        if (f is None): return (None,None,None)
        if (not f.isfile): return (f,None,None)
        return (f,v,f.getInfo(v))
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
    # -------------------------------------------------------------
    def __init__(self,control,filedir,filename,tree,links,paths,**kw):
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
        self.tmpfile=''
        self._status=Q7fingerPrint.STATUS_UNCHANGED
        if (kw.has_key('isfile')):
            self.isfile=True
        if (kw.has_key('converted')):
            self.converted=kw['converted']
            self.tmpfile=kw['convertedAs']
            if (self.converted):
                self._status=Q7fingerPrint.STATUS_CONVERTED
        self.lazy={}
        for p in paths: self.lazy['/CGNSTree'+p[0]]=p[1]
        Q7fingerPrint.__extension.append(self)
    def isFile(self):
        return self.isfile
    def isLink(self,path):
        pth=CGU.getPathNoRoot(path)
        for lk in self.links:
            if (lk[3]==pth): return lk
        return False
    def getInfo(self,view):
        d={}
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
        e=pwd.getpwuid(st[4])
        g=grp.getgrgid(st[5])
        d['eOwner']=e[0]
        d['eGroup']=g[0]
        d['cNoFollow']=False
        d['cHasLinks']=False
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
        d['cREAD']=self.converted
        d['cMODIFY']=False
        d['cNODATA']=False
        d['cHasInt64']=False
        return d
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
