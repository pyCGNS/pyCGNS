#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
from qtpy.QtCore import Qt, QObject, Signal
from qtpy.QtWidgets import (
    QAction,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QMenu,
    QTableWidgetItem,
    QHeaderView,
)

from CGNS.NAV.moption import Q7OptionContext as OCTXT

import CGNS.config as config

import CGNS.PAT.cgnslib as CGL

from CGNS.NAV.Q7ControlWindow import Ui_Q7ControlWindow
from CGNS.NAV.wfile import Q7File
from CGNS.NAV.winfo import Q7Info
from CGNS.NAV.woption import Q7Option
from CGNS.NAV.wfingerprint import Q7FingerPrint
from CGNS.NAV.wquery import Q7Query
from CGNS.NAV.whelp import Q7Help

import CGNS.NAV.wmessages as MSG

from CGNS.NAV.wfingerprint import Q7Window as QW


# -----------------------------------------------------------------
class Q7SignalPool(QObject):
    loadFile = Signal(name="loadFile")
    saveFile = Signal(name="saveFile")
    cancel = Signal(name="cancel")
    loadCompleted = Signal(name="loadCompleted")
    buffer = None
    fgprint = None
    saveAs = False


# -----------------------------------------------------------------
class Q7ControlItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if index.column() in [0, 1]:
            option.decorationPosition = QStyleOptionViewItem.Top
            QStyledItemDelegate.paint(self, painter, option, index)
        else:
            QStyledItemDelegate.paint(self, painter, option, index)


# -----------------------------------------------------------------
class Q7Main(QW, Ui_Q7ControlWindow):
    verbose = False

    def __init__(self, parent=None):
        self.lastView = None
        self.w = None
        self.fdialog = None
        import platform

        QW.control_log = MSG.Q7Log()
        QW.__init__(self, QW.VIEW_CONTROL, self, None, None)
        self.versions = {
            "pycgnsversion": "pyCGNS v%s" % config.version,
            #                       'chloneversion':'CHLone %s'%config.CHLONE_VERSION,
            "vtkversion": "VTK v%s" % config.VTK_VERSION,
            "cythonversion": "Cython v%s" % config.CYTHON_VERSION,
            "hdf5version": "HDF5 v%s" % config.HDF5_VERSION,
            "numpyversion": "numpy v%s" % config.NUMPY_VERSION,
            "pythonversion": "python v%s" % platform.python_version(),
            "pyqtversion": "PyQt v%s" % config.PYQT_VERSION,
            "qtversion": "Qt v%s" % config.QT_VERSION,
        }
        self.getHistory()
        self.bAbout.clicked.connect(self.about)
        self.bOptionView.clicked.connect(self.option)
        self.bTreeLoadLast.clicked.connect(self.loadlast)
        self.lockable(self.bTreeLoadLast)
        self.bTreeLoad.clicked.connect(self.load)
        self.bLog.clicked.connect(self.logView)
        self.lockable(self.bTreeLoad)
        self.bEditTree.clicked.connect(self.edit)
        self.lockable(self.bEditTree)
        self.bInfo.clicked.connect(self.infoControl)
        self.bPatternView.setDisabled(True)
        # self.bResetScrollBars.clicked.connect(self.resetScrolls)
        self.bClose.clicked.connect(self.close)
        # QObject.connect(self.controlTable,
        #                SIGNAL("cellClicked(int,int)"),
        #                self.clickedLine)
        self.controlTable.cellClicked.connect(self.clickedLine)
        self.initControlTable()
        self.controlTable.setItemDelegate(Q7ControlItemDelegate(self))
        self.signals = Q7SignalPool()
        self.signals.loadFile.connect(self.loadStart)
        self.signals.saveFile.connect(self.saving)
        self.signals.cancel.connect(self.cancelUnlock)
        self.signals.loadCompleted.connect(self.loadCompleted)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.popupmenu = QMenu()
        self.transientRecurse = False
        self.transientVTK = False
        self.copyPasteBuffer = None
        self.wOption = None
        self.selectForLinkDst = None
        self.newtreecount = 1
        self.help = None
        self._patternwindow = None
        self._toolswindow = None
        self.query = None
        Q7Query.loadUserQueries()
        Q7Query.fillQueries()
        Q7Query.loadUserFunctions()

    def clickedLine(self, *args):
        if self.controlTable.lastButton == Qt.LeftButton:
            # Q7FingerPrint.raiseView(self.getIdxFromLine(args[0]))
            pass
        if self.controlTable.lastButton == Qt.RightButton:
            self.updateMenu(self.controlTable.currentIndex())
            self.popupmenu.popup(self.controlTable.lastPos)

    def closeView(self):
        self.updateLastView()
        if self.lastView is not None:
            fg = Q7FingerPrint.getFingerPrint(self.lastView)
            fg.closeView(self.lastView)
            self.lastView = None

    def raiseView(self):
        self.updateLastView()
        if self.lastView is not None:
            Q7FingerPrint.raiseView(self.lastView)

    def logView(self):
        self.control_log.show()

    def infoControl(self):
        self.helpWindow("Control")

    def helpWindowDoc(self, doc):
        if self.help is not None:
            self.help.close()
        self.help = Q7Help(self, doc=doc)
        self.help.show()

    def helpWindow(self, key):
        if self.help is not None:
            self.help.close()
        self.help = Q7Help(self, key)
        self.help.show()

    def info(self):
        self.updateLastView()
        if self.lastView is not None:
            (f, v, d) = Q7FingerPrint.infoView(self.lastView)
            if not f.isFile():
                return
            self.w = Q7Info(self, d, f)
            self.w.show()

    def closeTree(self):
        self.updateLastView()
        if self.lastView is None:
            return
        (f, v, d) = Q7FingerPrint.infoView(self.lastView)
        reply = MSG.wQuestion(
            self,
            101,
            "Double check...",
            """Do you want to close the tree and all its views,<br>
                              and <b>forget unsaved</b> modifications?""",
        )
        if reply:
            f.closeAllViews()

    def closeAllTrees(self):
        reply = MSG.wQuestion(
            self,
            101,
            "Double check...",
            """Do you want to close all the views,<br>
                              and <b>forget unsaved</b> modifications?""",
        )
        if reply:
            Q7FingerPrint.closeAllTrees()

    def updateLastView(self):
        it = self.controlTable.currentItem()
        if it is None:
            self.lastView = None
        else:
            r = it.row()
            self.lastView = self.getIdxFromLine(r)
        return self.lastView

    def updateMenu(self, idx):
        lv = self.getIdxFromLine(idx.row())
        if lv is not None:
            self.lastView = lv
            actlist = (
                ("View information (Enter)", self.info),
                ("Raise selected view (Space)", self.raiseView),
                None,
                ("Close all trees", self.closeAllTrees),
                ("Close selected tree", self.closeTree),
                ("Close selected view (Del)", self.closeView),
            )
            self.popupmenu.clear()
            self.popupmenu.setTitle("Control view menu")
            for aparam in actlist:
                if aparam is None:
                    self.popupmenu.addSeparator()
                else:
                    a = QAction(aparam[0], self)
                    a.triggered.connect(aparam[1])
                    self.popupmenu.addAction(a)

    def loadOptions(self):
        if self.wOption is None:
            self.wOption = Q7Option(self)
        self.wOption.reset()

    def option(self):
        if self.wOption is None:
            self.wOption = Q7Option(self)
        self.wOption.show()

    def about(self):
        MSG.wInfo(
            self,
            100,
            "pyCGNS v%s" % OCTXT._ToolVersion,
            OCTXT._CopyrightNotice % self.versions,
            again=False,
        )

    def closeApplication(self):
        reply = MSG.wQuestion(
            self,
            101,
            "Double check...",
            """Do you want to quit %s,<b>close all views</b>
                              and forget unsaved modifications?"""
            % OCTXT._ToolName,
        )
        if reply == MSG.OK:
            Q7FingerPrint.closeAllTrees()
            if self.help is not None:
                self.help.close()
            if self._patternwindow is not None:
                self._patternwindow.close()
            if self.control_log is not None:
                self.control_log.close()
            if self._toolswindow is not None:
                self._toolswindow.close()
            return True
        else:
            return False

    def closeEvent(self, event):
        if self.closeApplication():
            event.accept()
        # return True
        else:
            event.ignore()
            #            return False

    def resetScrolls(self):
        self.controlTable.verticalScrollBar().setSliderPosition(0)
        self.controlTable.horizontalScrollBar().setSliderPosition(0)

    def initControlTable(self):
        ctw = self.controlTable
        ctw.control = self
        cth = ctw.horizontalHeader()
        ctw.verticalHeader().hide()
        h = ["S", "T", "View", "Dir", "File", "Node"]
        for i in range(len(h)):
            hi = QTableWidgetItem(h[i])
            hi.setFont(OCTXT._Label_Font)
            ctw.setHorizontalHeaderItem(i, hi)
            cth.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        cth.setSectionResizeMode(len(h) - 1, QHeaderView.Stretch)

    def updateViews(self):
        for i in self.getAllIdx():
            f = Q7FingerPrint.getFingerPrint(i)
            v = Q7FingerPrint.getView(i)
            l = self.getLineFromIdx(i)
            self.modifiedLine(l, f._status, f)
            try:
                v.updateTreeStatus()
            except AttributeError:
                pass

    def modifiedLine(self, n, stat, fg):
        if (Q7FingerPrint.STATUS_MODIFIED in stat) and (
            Q7FingerPrint.STATUS_SAVEABLE in stat
        ):
            stitem = QTableWidgetItem(self.IC(QW.I_MOD_SAV), "")
            stitem.setToolTip("Tree modified and saveable")
        if (Q7FingerPrint.STATUS_MODIFIED in stat) and (
            Q7FingerPrint.STATUS_SAVEABLE not in stat
        ):
            stitem = QTableWidgetItem(self.IC(QW.I_MOD_USAV), "")
            stitem.setToolTip("Tree modified but NOT saveable")
        if (Q7FingerPrint.STATUS_MODIFIED not in stat) and (
            Q7FingerPrint.STATUS_SAVEABLE not in stat
        ):
            stitem = QTableWidgetItem(self.IC(QW.I_UMOD_USAV), "")
            stitem.setToolTip("Tree unmodified and NOT saveable")
        if (Q7FingerPrint.STATUS_MODIFIED not in stat) and (
            Q7FingerPrint.STATUS_SAVEABLE in stat
        ):
            stitem = QTableWidgetItem(self.IC(QW.I_UMOD_SAV), "")
            stitem.setToolTip("Tree unmodified and saveable")
        stitem.setTextAlignment(Qt.AlignCenter)
        self.controlTable.setItem(n, 0, stitem)
        self.controlTable.item(n, 3).setText(str(fg.filedir))
        self.controlTable.item(n, 4).setText(str(fg.filename))

    def addLine(self, l, fg):
        ctw = self.controlTable
        ctw.setRowCount(ctw.rowCount() + 1)
        r = ctw.rowCount() - 1
        if l[1] == QW.VIEW_TREE:
            tpitem = QTableWidgetItem(self.IC(QW.I_TREE), "")
        if l[1] == QW.VIEW_FORM:
            tpitem = QTableWidgetItem(self.IC(QW.I_FORM), "")
        if l[1] == QW.VIEW_VTK:
            tpitem = QTableWidgetItem(self.IC(QW.I_VTK), "")
        if l[1] == QW.VIEW_QUERY:
            tpitem = QTableWidgetItem(self.IC(QW.I_QUERY), "")
        if l[1] == QW.VIEW_SELECT:
            tpitem = QTableWidgetItem(self.IC(QW.I_SELECT), "")
        if l[1] == QW.VIEW_DIAG:
            tpitem = QTableWidgetItem(self.IC(QW.I_DIAG), "")
        if l[1] == QW.VIEW_TOOLS:
            tpitem = QTableWidgetItem(self.IC(QW.I_TOOLS), "")
            l = l[0:2] + [None, None, None]
        if l[1] == QW.VIEW_LINK:
            tpitem = QTableWidgetItem(self.IC(QW.I_LINK), "")
        if l[1] == QW.VIEW_DIFF:
            tpitem = QTableWidgetItem(self.IC(QW.I_DIFF), "")
        tpitem.setTextAlignment(Qt.AlignCenter)
        ctw.setItem(r, 1, tpitem)
        for i in range(len(l) - 2):
            it = QTableWidgetItem("%s " % (l[i + 2]))
            if i in [0]:
                it.setTextAlignment(Qt.AlignCenter)
            else:
                it.setTextAlignment(Qt.AlignLeft)
            it.setFont(OCTXT._Table_Font)
            ctw.setItem(r, i + 2, it)
        self.modifiedLine(r, l[0], fg)
        ctw.setColumnWidth(0, 25)
        ctw.setColumnWidth(1, 25)
        ctw.setColumnWidth(2, 50)
        for i in range(self.controlTable.rowCount()):
            ctw.resizeRowToContents(i)

    def selectLine(self, idx):
        i = int(self.getLineFromIdx(idx))
        if i != -1:
            self.controlTable.setCurrentCell(i, 2)

    def delLine(self, idx):
        i = int(self.getLineFromIdx(idx))
        if i != -1:
            self.controlTable.removeRow(i)

    def getIdxFromLine(self, l):
        self.controlTable.setCurrentCell(l, 2)
        it = self.controlTable.currentItem()
        return it.text()

    def getLineFromIdx(self, idx):
        found = -1
        for n in range(self.controlTable.rowCount()):
            if int(idx) == int(self.controlTable.item(n, 2).text()):
                found = n
        return found

    def getAllIdx(self):
        all = []
        for n in range(self.controlTable.rowCount()):
            all.append(self.controlTable.item(n, 2).text())
        return all

    def clearOtherSelections(self):
        if self._patternwindow is not None:
            self._patternwindow.clearSelection()

    def cancelUnlock(self, *args):
        self.lockView(False)

    def loadStart(self, *args):
        self._T("loading: [%s]" % self.signals.buffer)
        self.busyCursor()
        Q7FingerPrint.treeLoad(self, self.signals.buffer)
        Q7FingerPrint.refreshScreen()

    def setDefaults(self):
        self.loadOptions()
        self._application.setStyleSheet(self.wOption._options["UserCSS"])

    def loadCompleted(self, *args, **kwargs):
        self.lockView(False)
        if "dataset_name" in kwargs:
            filedir = kwargs["dataset_base"]
            filename = kwargs["dataset_name"]
            tree = kwargs["dataset_tree"]
            links = kwargs["dataset_references"]
            paths = kwargs["dataset_paths"]
            fgprint = Q7FingerPrint(
                self, filedir, filename, tree, links, paths, **kwargs
            )
        else:
            fgprint = self.signals.fgprint
        if len(fgprint) > 1:
            code = fgprint[1][0]
            msg0 = fgprint[1][1]
            msg1 = fgprint[1][2]
            MSG.wError(self, code, msg0, msg1)
        elif fgprint.tree is None:
            MSG.wError(
                self, 201, "Load error", "Fatal error while loading file, empty tree"
            )
        else:
            child = self.loadQ7Tree(fgprint)
            child.show()
            self.setHistory(fgprint.filedir, fgprint.filename)
            self.updateViews()
            fgprint.getInfo(force=True)
        self.signals.fgprint = None
        Q7FingerPrint.killProxy()
        self.readyCursor()

    def saving(self, *args):
        self._T("saving as: [%s]" % self.signals.buffer)
        self.busyCursor()
        Q7FingerPrint.treeSave(
            self, self.signals.fgprint, self.signals.buffer, self.signals.saveAs
        )
        self.setHistory(self.signals.fgprint.filedir, self.signals.fgprint.filename)
        self.updateViews()
        self.signals.fgprint.getInfo(force=True)
        self.readyCursor()
        self.lockView(False)

    def load(self):
        self.fdialog = Q7File(self)
        self.lockView(True)
        self.fdialog.show()

    def loadlast(self):
        if self.getLastFile() is None:
            return
        self.signals.buffer = self.getLastFile()[0] + "/" + self.getLastFile()[1]
        if self.signals.buffer is None:
            self.load()
        else:
            self.signals.loadFile.emit()

    def loadfile(self, name):
        self.signals.buffer = name
        self.signals.loadFile.emit()

    def save(self, fgprint):
        self.signals.fgprint = fgprint
        self.signals.saveAs = True
        self.fdialog = Q7File(self, 1)
        self.lockView(True)
        self.fdialog.show()

    def savedirect(self, fgprint):
        self.signals.fgprint = fgprint
        self.signals.saveAs = False
        self.signals.buffer = fgprint.filedir + "/" + fgprint.filename
        self.signals.saveFile.emit()

    def edit(self):
        self._T("edit new")
        tree = CGL.newCGNSTree()
        tc = self.newtreecount
        self.busyCursor()
        fgprint = Q7FingerPrint(self, ".", "new#%.3d.hdf" % tc, tree, [], [])
        child = self.loadQ7Tree(fgprint)
        fgprint._status = [Q7FingerPrint.STATUS_MODIFIED]
        self.readyCursor()
        self.newtreecount += 1
        child.show()

    def userFunctionFromPath(self, path, types):
        return Q7Query._userFunction

    def loadQ7Tree(self, fgprint):
        from CGNS.NAV.wtree import Q7Tree
        from CGNS.NAV.mtree import Q7TreeModel

        Q7TreeModel(fgprint.index)
        return Q7Tree(self, "/", fgprint.index)


# -----------------------------------------------------------------
