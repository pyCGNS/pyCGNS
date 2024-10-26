#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

from .moption import Q7OptionContext as OCTXT

import os.path
import stat
import string
import time
import sys

from CGNS.NAV.config import HAS_MSW

from qtpy.QtCore import Qt, QSortFilterProxyModel, QModelIndex, QFileInfo
from qtpy.QtWidgets import (
    QWidget,
    QFileIconProvider,
    QAbstractItemView,
    QFileSystemModel,
)
from qtpy.QtGui import QIcon, QPixmap

from .Q7FileWindow import Ui_Q7FileWindow

from . import wmessages as MSG

LOADBUTTON = ["Load", "Load from selected file"]
SAVEBUTTON = ["Save", "Save to selected file"]
(LOADMODE, SAVEMODE) = (0, 1)


# -----------------------------------------------------------------
def checkFilePermission(path, write=False):
    if HAS_MSW:
        return True
    if not (os.path.isfile(path) or os.path.isdir(path)):
        return False
    r = False
    w = False
    st = os.stat(path)
    m = st.st_mode
    if (st.st_uid == os.getuid()) and (m & stat.S_IRUSR):
        r = True
    if (st.st_gid == os.getgid()) and (m & stat.S_IRGRP):
        r = True
    if m & stat.S_IROTH:
        r = True
    if not r:
        return False
    if (st.st_uid == os.getuid()) and (m & stat.S_IWUSR):
        w = True
    if (st.st_gid == os.getgid()) and (m & stat.S_IWGRP):
        w = True
    if m & stat.S_IWOTH:
        w = True
    if write and not w:
        return False
    return True


# -----------------------------------------------------------------
class Q7FileFilterProxy(QSortFilterProxyModel):
    def __init__(self, parent):
        super(Q7FileFilterProxy, self).__init__(parent)
        # QSortFilterProxyModel.__init__(self, parent)
        self.model = parent.model
        self.treeview = parent.treeview
        self.wparent = parent
        self.setDynamicSortFilter(True)
        import locale

        locale.setlocale(locale.LC_ALL, "C")

    def filterAcceptsRow(self, row, parentindex):
        idx = self.model.index(row, 1, parentindex)
        p = self.model.filePath(idx)
        if isinstance(p, bytes):
            p = str(self.model.filePath(idx).decode("utf-8"))
        if not self.checkPermission(p):
            return False
        if os.path.isdir(p):
            if self.wparent.cShowDirs.checkState() != Qt.Checked:
                if len(p) > len(self.wparent.selecteddir):
                    return False
            return True
        self.wparent.getBoxes()
        # if (self.wparent.cShowAll.checkState()==Qt.Checked): xlist=[]
        r = self.wparent.parent.matchFileExtensions(p)
        return r

    def checkPermission(self, path, write=False):
        return checkFilePermission(path, write)

    def lessThan(self, left, right):
        c = self.sortColumn()
        a = self.model.data(left)
        b = self.model.data(right)
        if c in (0, 2):
            return a < b
        if c == 3:
            fmt_qt = "dd-MM-yyyy hh:mm:ss"
            left_time = self.model.lastModified(left)
            right_time = self.model.lastModified(right)
            a = left_time.toString(fmt_qt)
            b = right_time.toString(fmt_qt)
            fmtr = "%d-%m-%Y %H:%M:%S"
            fmtw = "%Y-%m-%d %H:%M:%S"
            ad = time.strptime(str(a), fmtr)
            bd = time.strptime(str(b), fmtr)
            af = time.strftime(fmtw, ad)
            bf = time.strftime(fmtw, bd)
            return af < bf
        if c == 1:
            wg = {"MB": 1e3, "GB": 1e6, "KB": 1}
            try:
                (av, au) = a.split()
                (bv, bu) = b.split()
            except ValueError:
                return a < b
            av = float(av.replace(",", ".")) * wg[au]
            bv = float(bv.replace(",", ".")) * wg[bu]
            return av < bv
        return 1


# -----------------------------------------------------------------
class Q7FileIconProvider(QFileIconProvider):
    slist = ["hdf", "HDF", "cgns", "CGNS", "adf", "ADF"]

    def __init__(self):
        super(Q7FileIconProvider, self).__init__()
        self.dir = QIcon(QPixmap(":/images/icons/folder.png"))
        self.cgns = QIcon(QPixmap(":/images/icons/tree-load.png"))
        self.empty = QIcon()

    def icon(self, fileinfo):
        if not isinstance(fileinfo, QFileInfo):
            return self.empty
        if fileinfo.isDir():
            return self.dir
        if fileinfo.suffix() in Q7FileIconProvider.slist:
            return self.cgns
        return self.empty


# -----------------------------------------------------------------
class Q7File(QWidget, Ui_Q7FileWindow):
    def __init__(self, parent, mode=LOADMODE):
        super(Q7File, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Load/Save")
        self.parent = parent
        self.iconProvider = Q7FileIconProvider()
        self.model = QFileSystemModel()
        self.model.setIconProvider(self.iconProvider)
        self.proxy = Q7FileFilterProxy(self)
        self.proxy.setSourceModel(self.model)
        self.treeview.setModel(self.proxy)
        self.setAttribute(Qt.WA_DeleteOnClose)
        #
        self.model.directoryLoaded[str].connect(self.expandCols)
        self.treeview.expanded[QModelIndex].connect(self.expandCols)
        self.treeview.clicked[QModelIndex].connect(self.clickedNode)
        self.treeview.doubleClicked[QModelIndex].connect(self.clickedNodeAndLoad)
        self.direntries.lineEdit().returnPressed.connect(self.changeDirEdit)
        self.direntries.currentIndexChanged[int].connect(self.changeDirIndex)
        self.fileentries.currentIndexChanged[int].connect(self.changeFile)
        self.fileentries.lineEdit().editingFinished.connect(self.changeFile)
        self.tabs.currentChanged[int].connect(self.currentTabChanged)
        self.cShowAll.stateChanged[int].connect(self.updateView)
        self.cShowDirs.stateChanged[int].connect(self.updateView)
        self.rClearSelectedDirs.toggled[bool].connect(self.updateClearDirs)
        self.rClearSelectedFiles.toggled[bool].connect(self.updateClearFiles)
        self.rClearAllDirs.toggled[bool].connect(self.updateClearNone)
        self.rClearNoHDF.toggled[bool].connect(self.updateClearNoHDF)
        self.rClearNotFound.toggled[bool].connect(self.updateClearNotFound)
        #
        self.bClose.clicked.connect(self.closeUnlock)
        self.bCurrent.clicked.connect(self.currentDir)
        self.bBack.clicked.connect(self.backDir)
        self.bInfo.clicked.connect(self.infoFileView)
        self.bInfo2.clicked.connect(self.infoFileView)
        self.bClearHistory.clicked.connect(self.doClearHistory)
        self.mode = mode
        if self.mode == SAVEMODE:
            self.setMode(False)
        else:
            self.setMode(True)
        self.setBoxes()
        self.parent.getHistory()
        self.updateHistory()
        self.updateClearNotFound()

    def closeUnlock(self):
        self.parent.signals.cancel.emit()
        self.close()

    def closeEvent(self, event):
        self.closeUnlock()

    def infoFileView(self):
        self.parent.helpWindow("File")

    def updateView(self):
        p = self.direntries.currentText()
        self.setCurrentDir(p)
        print("CURRENT", p)
        # self.proxy.reset()
        self.proxy.beginResetModel()
        self.proxy.endResetModel()
        self.setCurrentDir(p)
        self.updateFileIfFound()

    def currentTabChanged(self, tabno):
        self.expandCols()
        self.setBoxes()

    def getOpt(self, name):
        return getattr(self, "_Ui_Q7FileWindow__O_" + name.lower())

    def setBoxes(self):
        ckh = self.getOpt("FilterHDFFiles")
        ckg = self.getOpt("FilterCGNSFiles")
        if self.parent.getOptionValue("FilterHDFFiles"):
            ckh.setCheckState(Qt.Checked)
        else:
            ckh.setCheckState(Qt.Unchecked)
        if self.parent.getOptionValue("FilterCGNSFiles"):
            ckg.setCheckState(Qt.Checked)
        else:
            ckg.setCheckState(Qt.Unchecked)
        if (ckh.checkState() == Qt.Unchecked) and (ckg.checkState() == Qt.Unchecked):
            self.cShowAll.setCheckState(Qt.Checked)

    def getBoxes(self):
        if self.getOpt("FilterHDFFiles").checkState() == Qt.Unchecked:
            self.parent.setOptionValue("FilterHDFFiles", False)
        else:
            self.parent.setOptionValue("FilterHDFFiles", True)
        if self.getOpt("FilterCGNSFiles").checkState() == Qt.Unchecked:
            self.parent.setOptionValue("FilterCGNSFiles", False)
        else:
            self.parent.setOptionValue("FilterCGNSFiles", True)

    def expandCols(self, *args):
        self.getBoxes()
        for n in range(3):
            self.treeview.resizeColumnToContents(n)

    def currentDir(self, *args):
        # p=os.path.split(self.path())[0]
        p = os.getcwd()
        self.setCurrentDir(p)

    def backDir(self, *args):
        p = os.path.split(self.path())[0]
        self.setCurrentDir(p)

    def changeDirEdit(self, *args):
        self.changeDir(args)

    def changeDirText(self, *args):
        self.changeDir(args)

    def changeDirIndex(self, *args):
        self.changeDir(args)

    def changeDir(self, *args):
        if self.updateMode:
            return
        p = str(self.direntries.currentText())
        if os.path.isdir(p):
            self.updateView()
        else:
            reply = MSG.wQuestion(
                self.parent,
                110,
                "Directory not found...",
                """The path below doesn't exist, do you want to remove<br>
                                     it from the history?<br>%s"""
                % p,
            )
            if reply == MSG.OK:
                ix = self.direntries.currentIndex()
                self.direntries.removeItem(ix)

    def changeFile(self, *args):
        self.selectedfile = str(self.fileentries.lineEdit().text())
        d = None
        if self.cAutoDir.checkState() == Qt.Checked:
            d = self.parent.getHistoryFile(self.selectedfile)
        if d is not None:
            self.selecteddir = d[0]
            ix = self.direntries.findText(self.selecteddir)
            if ix != -1:
                self.direntries.setCurrentIndex(ix)
        else:
            self.selecteddir = self.direntries.lineEdit().text()
        self.updateFileIfFound()

    def selectedPath(self):
        return os.path.join(self.selecteddir, self.selectedfile)

    def updateFileIfFound(self):
        filepath = self.selectedPath()
        midx = self.model.index(filepath)
        if midx.row == 1:
            return
        fidx = self.proxy.mapFromSource(midx)
        if fidx.row == 1:
            return
        self.treeview.setCurrentIndex(fidx)
        self.treeview.scrollTo(fidx)

    def setCurrentDir(self, newpath):
        self.model.setRootPath(newpath)
        midx = self.model.index(newpath)
        fidx = self.proxy.mapFromSource(midx)
        self.treeview.setRootIndex(fidx)
        self.treeview.setCurrentIndex(fidx)
        self.direntries.setItemText(self.direntries.currentIndex(), newpath)
        self.selecteddir = newpath

    def setMode(self, load=True):
        if load:
            self.bAction.clicked.connect(self.load)
            self.bAction.setToolTip(LOADBUTTON[1])
            self.bAction.setText(LOADBUTTON[0])
            self.cOverwrite.setEnabled(False)
            self.cReadOnly.setEnabled(True)
            self.cNoLargeData.setEnabled(True)
            self.cFollowLinks.setEnabled(True)
            self.cDeleteMissing.setEnabled(False)
            if OCTXT.DoNotLoadLargeArrays:
                self.cNoLargeData.setCheckState(Qt.Checked)
            else:
                self.cNoLargeData.setCheckState(Qt.Unchecked)
            if OCTXT.FollowLinksAtLoad:
                self.cFollowLinks.setCheckState(Qt.Checked)
            else:
                self.cFollowLinks.setCheckState(Qt.Unchecked)
        else:
            self.bAction.clicked.connect(self.save)
            self.bAction.setToolTip(SAVEBUTTON[1])
            self.bAction.setText(SAVEBUTTON[0])
            self.cOverwrite.setEnabled(True)
            self.cReadOnly.setEnabled(False)
            self.cNoLargeData.setEnabled(False)
            self.cFollowLinks.setEnabled(False)
            self.cDeleteMissing.setEnabled(True)
            if OCTXT.FileUpdateRemovesChildren:
                self.cDeleteMissing.setCheckState(Qt.Checked)
            else:
                self.cDeleteMissing.setCheckState(Qt.Unchecked)

    def updateHistory(self):
        self.updateMode = True
        self.direntries.clear()
        self.fileentries.clear()
        hlist = self.parent.getHistory(fromfile=False)
        flist = []
        self.fileentries.addItem("")
        for i in list(hlist):
            if i != self.parent.getHistoryLastKey():
                self.direntries.addItem(str(i))
                flist = flist + hlist[i]
        for i in flist:
            self.fileentries.addItem(str(i))
        self.historyfiles = flist
        self.historydirs = list(hlist)
        if self.parent.getHistoryLastKey() in list(hlist):
            self.selecteddir = hlist[self.parent.getHistoryLastKey()][0]
            self.selectedfile = hlist[self.parent.getHistoryLastKey()][1]
            # ixd=self.direntries.findText(self.selecteddir)
            self.setCurrentDir(self.selecteddir)
            ixf = self.fileentries.findText(self.selectedfile)
            self.fileentries.setCurrentIndex(ixf)
            self.changeFile()
        else:
            self.selecteddir = os.getcwd()
            self.selectedfile = ""
            self.setCurrentDir(self.selecteddir)
        self.updateMode = False

    def doClearHistory(self):
        if self.rClearNoHDF.isChecked():
            reply = MSG.wQuestion(
                self.parent,
                120,
                "Clear history",
                """You really want to remove directory entries from history<br>
                                     where no file with defined extensions has been found?<br>""",
            )
            if reply == MSG.OK:
                for d in self.parent.getDirNoHDFFromHistory():
                    self.parent.removeDirFromHistory(d)
                    self.updateHistory()
                    self.lClear.clear()
        if self.rClearNotFound.isChecked():
            reply = MSG.wQuestion(
                self.parent,
                121,
                "Clear history",
                """You really want to remove <b>NOT FOUND</b> entries from<br>
                                     the history of used directories?<br>""",
            )
            if reply == MSG.OK:
                for d in self.parent.getDirNotFoundFromHistory():
                    self.parent.removeDirFromHistory(d)
                    self.updateHistory()
                    self.lClear.clear()
        if self.rClearAllDirs.isChecked():
            reply = MSG.wQuestion(
                self.parent,
                122,
                "Clear history",
                """You really want to remove <b>ALL</b> entries from the<br>
                                     the history of used files and directories?<br>""",
            )
            if reply == MSG.OK:
                self.parent.destroyHistory()
                self.updateHistory()
        if self.rClearSelectedDirs.isChecked():
            for it in self.lClear.selectedItems():
                self.parent.removeDirFromHistory(it.text())
            self.updateHistory()
            self.updateClearDirs()
        if self.rClearSelectedFiles.isChecked():
            for it in self.lClear.selectedItems():
                fd = self.parent.getHistoryFile(it.text())
                if fd is not None:
                    self.parent.removeFileFromHistory(*fd)
            self.updateHistory()
            self.updateClearFiles()

    def updateClearNone(self):
        self.lClear.clear()

    def updateClearNoHDF(self):
        self.lClear.clear()
        self.lClear.setSelectionMode(QAbstractItemView.NoSelection)
        for d in self.parent.getDirNoHDFFromHistory():
            self.lClear.addItem(d)

    def updateClearNotFound(self):
        self.lClear.clear()
        self.lClear.setSelectionMode(QAbstractItemView.NoSelection)
        for d in self.parent.getDirNotFoundFromHistory():
            self.lClear.addItem(d)

    def updateClearDirs(self):
        self.lClear.clear()
        self.lClear.setSelectionMode(QAbstractItemView.MultiSelection)
        for d in self.historydirs:
            self.lClear.addItem(d)

    def updateClearFiles(self):
        self.lClear.clear()
        self.lClear.setSelectionMode(QAbstractItemView.MultiSelection)
        for d in self.historyfiles:
            self.lClear.addItem(d)

    def load(self):
        diag = self.checkTarget(self.selectedPath())
        if diag is None:
            self.parent.signals.buffer = self.selectedPath()
            self.hide()
            self.parent.signals.loadFile.emit()
            self.close()
        else:
            MSG.message("Load file: %s" % self.selectedPath(), diag, MSG.INFO)

    def save(self):
        diag = self.checkTarget(self.selectedPath(), write=True)
        if diag is None:
            self.parent.signals.buffer = self.selectedPath()
            self.hide()
            self.parent.signals.saveFile.emit()
            self.close()
        else:
            MSG.message("Save file: %s" % self.selectedPath(), diag, MSG.INFO)

    def checkTarget(self, filename, write=False):
        if os.path.isfile(filename) and not write:
            return None
        if not os.path.isfile(filename) and write:
            return None
        if self.cOverwrite.isChecked():
            sc = "The file is <b>completely replaced</b> by the current tree"
        else:
            sc = "The file is <b>updated</b> with the current tree contents."
            if self.cDeleteMissing.isChecked():
                sc += "The nodes <b>NOT</b> found in the current tree are removed from the updated file"
            else:
                sc += (
                    "The nodes <b>NOT</b> found in the current tree are kept unchanged"
                )
        reply = MSG.wQuestion(
            self,
            132,
            "Saving on an already existing file",
            """You are going to save into an existing file,
                              based on the current options you have and the target tree you want to save,
                              the result would be the following:<p>%s<p>
                              If this not the way you want the save to operate, please <i>abort</b> this
                              file selection and check the <i>Load/Save option</i> tab.
                              You still want to write on this file?"""
            % sc,
            buttons=("Continue to save on existing file", "Abort save"),
        )
        if reply:
            return None
        return "User Abort"

    def path(self, index=None):
        if index is None:
            idx = self.treeview.currentIndex()
            p = str(self.model.filePath(self.proxy.mapToSource(idx)))
        else:
            p = str(self.model.filePath(self.proxy.mapToSource(index)))
        return p

    def clickedNodeAndLoad(self, index):
        self.clickedNode(index)
        if self.mode == SAVEMODE:
            self.save()
        else:
            self.load()

    def clickedNode(self, index):
        self.treeview.resizeColumnToContents(0)
        p = self.path(index)
        if os.path.isdir(p):
            f = ""
            d = p
            self.setCurrentDir(d)
        else:
            f = os.path.basename(self.path(index))
            d = os.path.dirname(self.path(index))
        ix = self.direntries.findText(d)
        if ix != -1:
            self.direntries.setCurrentIndex(ix)
        else:
            self.direntries.addItem(d)
            self.direntries.setCurrentIndex(self.direntries.findText(d))
        ix = self.fileentries.findText(f)
        if ix != -1:
            self.fileentries.setCurrentIndex(ix)
        else:
            self.fileentries.addItem(f)
            self.fileentries.setCurrentIndex(self.fileentries.findText(f))
        self.selecteddir = self.direntries.lineEdit().text()
        self.selectedfile = self.fileentries.lineEdit().text()


# --- last line
