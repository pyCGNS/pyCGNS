#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from PySide.QtCore    import *
from PySide.QtGui     import *
from CGNS.NAV.Q7FileWindow import Ui_Q7FileWindow

from CGNS.NAV.moption import Q7OptionContext  as OCTXT
import CGNS.NAV.wmessages as MSG

import os.path
import stat
import string
import time

LOADBUTTON='Load'
SAVEBUTTON='Save'
(LOADMODE,SAVEMODE)=(0,1)

def checkFilePermission(path,write=False):
    return True
    if (not os.path.exists(path)): return False
    r=False
    w=False
    st=os.stat(path)
    m=st.st_mode
    if ((st.st_uid == os.getuid()) and (m & stat.S_IRUSR)): r=True
    if ((st.st_gid == os.getgid()) and (m & stat.S_IRGRP)): r=True
    if (m & stat.S_IROTH): r=True
    if (not r): return False
    if ((st.st_uid == os.getuid()) and (m & stat.S_IWUSR)): w=True
    if ((st.st_gid == os.getgid()) and (m & stat.S_IWGRP)): w=True
    if (m & stat.S_IWOTH): w=True
    if (write and not w): return False
    return True

# -----------------------------------------------------------------
class Q7FileFilterProxy(QSortFilterProxyModel):
    def __init__(self,parent):
        QSortFilterProxyModel.__init__(self,parent)
        self.model=parent.model
        self.treeview=parent.treeview
        self.control=parent.parent
        self.wparent=parent
        self.setDynamicSortFilter(True)
        import locale
        locale.setlocale(locale.LC_ALL, 'C')
    def filterAcceptsRow(self,row,parentindex):
        idx=self.model.index(row,1,parentindex)
        p=self.model.filePath(idx)
        if not self.checkPermission(p): return False
        if (os.path.isdir(p)):
            if (self.wparent.cShowDirs.checkState()!=Qt.Checked):
                if (len(p)>len(self.wparent.selecteddir)): return False
            return True
        (fname,ext)=os.path.splitext(os.path.basename(p))
        xlist=[]
        self.wparent.getBoxes()
        if (OCTXT.FilterCGNSFiles): xlist+=OCTXT.CGNSFileExtension
        if (OCTXT.FilterHDFFiles):  xlist+=OCTXT.HDFFileExtension
        if (self.wparent.cShowAll.checkState()==Qt.Checked): xlist=[]
        if ((xlist == []) or (ext in xlist)): return True
        return False
    def checkPermission(self,path,write=False):
        return checkFilePermission(path,write)
    def lessThan(self,left,right):
        c=self.sortColumn()
        a=self.model.data(left)
        b=self.model.data(right)
        if (c in (0,2)): return a<b
        if (c==3):
            fmtr="%d %b %Y %H:%M:%S"
            fmtw="%Y-%m-%d %H:%M:%S"
            ad=time.strptime(str(a),fmtr)
            bd=time.strptime(str(b),fmtr)
            af=time.strftime(fmtw,ad)
            bf=time.strftime(fmtw,bd)
            return af<bf
        if (c==1):
            wg={'MB':1e3,'GB':1e6,'KB':1}
            try:
                (av,au)=a.split()
                (bv,bu)=b.split()
            except ValueError:
                return a<b
            av=float(string.replace(av,',','.'))*wg[au]
            bv=float(string.replace(bv,',','.'))*wg[bu]
            return av<bv
        return 1
   
# -----------------------------------------------------------------
class Q7FileIconProvider(QFileIconProvider):
    slist=['hdf','HDF','cgns','CGNS','adf','ADF']
    def __init__(self):
        super(Q7FileIconProvider, self).__init__()
        self.dir=QIcon(QPixmap(":/images/icons/folder.gif"))
        self.cgns=QIcon(QPixmap(":/images/icons/tree-load.gif"))
        self.empty=QIcon()
    def icon(self,fileinfo):
        if (type(fileinfo) is not QFileInfo): return self.empty
        if (fileinfo.isDir()): return self.dir
        if (fileinfo.suffix() in Q7FileIconProvider.slist): return self.cgns
        return self.empty

# -----------------------------------------------------------------
class Q7File(QWidget,Ui_Q7FileWindow):
    def __init__(self,parent,mode=LOADMODE):
        super(Q7File, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Load/Save")
        self.parent=parent
        self.iconProvider=Q7FileIconProvider()
        self.model = QFileSystemModel()
        self.model.setIconProvider(self.iconProvider)
        self.proxy = Q7FileFilterProxy(self)
        self.proxy.setSourceModel(self.model)
        hlist=self.parent.getHistory()
        flist=[]
        self.fileentries.addItem("")
        for i in hlist.keys():
            if (i != self.parent.getHistoryLastKey()):
                self.direntries.addItem(i)
                flist=flist+hlist[i]
        for i in flist:
            self.fileentries.addItem(i)
        self.treeview.setModel(self.proxy)
        siglist=[
        (self.model,"directoryLoaded(QString)",self.expandCols),
        (self.treeview,"expanded(QModelIndex)",self.expandCols),
        (self.treeview,"clicked(QModelIndex)",self.clickedNode),
        (self.treeview,"doubleClicked(QModelIndex)",self.clickedNodeAndLoad),
        (self.direntries.lineEdit(),"returnPressed()",self.changeDirEdit),
        (self.direntries,"currentIndexChanged(int)",self.changeDirIndex),
        #(self.direntries,"editTextChanged(QString)",self.changeDirText),
        (self.fileentries,"currentIndexChanged(int)",self.changeFile),
        (self.fileentries.lineEdit(),"editingFinished()",self.changeFile),
        (self.tabs,"currentChanged(int)",self.currentTabChanged),
        (self.cShowAll,"stateChanged(int)",self.updateView),
        (self.cShowDirs,"stateChanged(int)",self.updateView)
            ]
        for (o,s,f) in siglist:
            QObject.connect(o,SIGNAL(s),f)
        self.bClose.clicked.connect(self.close)
        self.bBack.clicked.connect(self.backDir)
        self.bInfo.clicked.connect(self.infoFileView)
        self.mode=mode
        if (self.mode==SAVEMODE): self.setMode(False)
        else:                     self.setMode(True)
        self.setBoxes()
        if (self.parent.getHistoryLastKey() in hlist.keys()):
            self.selecteddir=hlist[self.parent.getHistoryLastKey()][0]
            self.selectedfile=hlist[self.parent.getHistoryLastKey()][1]
            #ixd=self.direntries.findText(self.selecteddir)
            self.setCurrentDir(self.selecteddir)
            ixf=self.fileentries.findText(self.selectedfile)
            self.fileentries.setCurrentIndex(ixf)
            self.changeFile()
        else:
            self.selecteddir=os.getcwd()
            self.selectedfile=""
            self.setCurrentDir(self.selecteddir)
    def infoFileView(self):
        self.control.helpWindow('File')
    def updateView(self):
        p=self.direntries.currentText()
        self.setCurrentDir(p)
        self.proxy.reset()
        self.setCurrentDir(p)
        self.updateFileIfFound()
    def currentTabChanged(self,tabno):
        self.expandCols()
        self.setBoxes()
    def getOpt(self,name):
        return getattr(self,'_Ui_Q7FileWindow__O_'+string.lower(name))
    def setBoxes(self):
        ckh=self.getOpt('FilterHDFFiles')
        ckg=self.getOpt('FilterCGNSFiles')
        if (self.parent.getOptionValue('FilterHDFFiles')):
            ckh.setCheckState(Qt.Checked)
        else:
            ckh.setCheckState(Qt.Unchecked)
        if (self.parent.getOptionValue('FilterCGNSFiles')):
            ckg.setCheckState(Qt.Checked)
        else:
            ckg.setCheckState(Qt.Unchecked)
        if  (     (ckh.checkState()==Qt.Unchecked)
              and (ckg.checkState()==Qt.Unchecked)):
          self.cShowAll.setCheckState(Qt.Checked)
    def getBoxes(self):
        if (self.getOpt('FilterHDFFiles').checkState()==Qt.Unchecked):
            self.parent.setOptionValue('FilterHDFFiles',False)
        else: 
            self.parent.setOptionValue('FilterHDFFiles',True)           
        if (self.getOpt('FilterCGNSFiles').checkState()==Qt.Unchecked):
            self.parent.setOptionValue('FilterCGNSFiles',False)
        else: 
            self.parent.setOptionValue('FilterCGNSFiles',True)           
    def expandCols(self,*args):
        self.getBoxes()
        for n in range(3):
            self.treeview.resizeColumnToContents(n)
    def backDir(self,*args):
        p=os.path.split(self.path())[0]
        self.setCurrentDir(p)
    def changeDirEdit(self,*args):
        #print 'EDIT'
        self.changeDir(args)
    def changeDirText(self,*args):
        #print 'TEXT'
        self.changeDir(args)
    def changeDirIndex(self,*args):
        #print 'INDEX'
        self.changeDir(args)
    def changeDir(self,*args):
        p=self.direntries.currentText()
        if (os.path.isdir(p)): self.updateView()
        else:
            reply=MSG.wQuestion(self.parent,'Directory not found...',
                  """The path doesn't exist, do you want to remove<br>
                     it from the history?""")
            if (reply == MSG.OK):
                ix=self.direntries.currentIndex()
                self.direntries.removeItem(ix)
                
    def changeFile(self,*args):
        self.selectedfile=self.fileentries.lineEdit().text()
        d=None
        if (self.cAutoDir.checkState()==Qt.Checked):
            d=self.parent.getHistoryFile(self.selectedfile)
        if (d is not None):
            self.selecteddir=d[0]
            ix=self.direntries.findText(self.selecteddir)
            if (ix!=-1): self.direntries.setCurrentIndex(ix)
        else: self.selecteddir=self.direntries.lineEdit().text()
        self.updateFileIfFound()
    def selectedPath(self):
        return self.selecteddir+'/'+self.selectedfile
    def updateFileIfFound(self):
        filepath=self.selectedPath()
        midx=self.model.index(filepath)
        if (midx.row==1): return
        fidx=self.proxy.mapFromSource(midx)
        if (fidx.row==1): return
        self.treeview.setCurrentIndex(fidx)
        self.treeview.scrollTo(fidx)
    def setCurrentDir(self,newpath):
        self.model.setRootPath(newpath)
        midx=self.model.index(newpath)
        fidx=self.proxy.mapFromSource(midx)
        self.treeview.setRootIndex(fidx)
        self.treeview.setCurrentIndex(fidx)
        self.direntries.setItemText(self.direntries.currentIndex(),newpath)
        self.selecteddir=newpath
    def setMode(self,load=True):
        if (load):
            self.bAction.clicked.connect(self.load)
            self.bAction.setToolTip(LOADBUTTON)
        else:
            self.bAction.clicked.connect(self.save)
            self.bAction.setToolTip(SAVEBUTTON)
    def load(self):
        diag=self.checkTarget(self.selectedPath())
        if (diag is None):
          self.parent.signals.buffer=self.selectedPath()
          self.hide()
          self.parent.signals.loadFile.emit()
          self.close()
        else:
          MSG.message("Load file: %s"%self.selectedPath(),
                      diag,MSG.INFO)
    def save(self):
        diag=self.checkTarget(self.selectedPath(),write=True)
        if (diag is None):
          self.parent.signals.buffer=self.selectedPath()
          self.hide()
          self.parent.signals.saveFile.emit()
          self.close()
        else:
          MSG.message("Save file: %s"%self.selectedPath(),
                      diag,MSG.INFO)
    def checkTarget(self,filename,write=False):
        if (os.path.exists(filename)): pass
        return None
    def path(self,index=None):
        if (index==None):
            idx=self.treeview.currentIndex()
            p=self.model.filePath(self.proxy.mapToSource(idx))
        else:
            p=self.model.filePath(self.proxy.mapToSource(index))
        return p
    def clickedNodeAndLoad(self,index):
        self.clickedNode(index)
        if (self.mode==SAVEMODE): self.save()
        else:                     self.load()
        
    def clickedNode(self,index):
        self.treeview.resizeColumnToContents(0)
        p=self.path(index)
        if (os.path.isdir(p)):
            f=''
            d=p
            self.setCurrentDir(d)
        else:
            f=os.path.basename(self.path(index))
            d=os.path.dirname(self.path(index))
        ix=self.direntries.findText(d)
        if (ix!=-1):
            self.direntries.setCurrentIndex(ix)
        else:
            self.direntries.addItem(d)
            self.direntries.setCurrentIndex(self.direntries.findText(d))
        ix=self.fileentries.findText(f)
        if (ix!=-1):
            self.fileentries.setCurrentIndex(ix)
        else:
            self.fileentries.addItem(f)
            self.fileentries.setCurrentIndex(self.fileentries.findText(f))
        self.selecteddir=self.direntries.lineEdit().text()
        self.selectedfile=self.fileentries.lineEdit().text()

# --- last line
