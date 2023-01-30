#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import sys

import linecache
import numpy
import os

from qtpy.QtCore import (Qt, QModelIndex, QAbstractItemModel, QItemSelectionModel)
from qtpy.QtGui import (QIcon, QPixmap, QColor)
from qtpy.QtWidgets import QTreeView

from . import wmessages as MSG
from ..PAT import cgnskeywords as CGK
from ..PAT import cgnsutils as CGU
from ..VAL.grammars import CGNS_VAL_USER_DEFAULT as CGV
from ..VAL.parse import findgrammar
from ..VAL.parse import messages as CGM
from ..VAL import simplecheck as CGV
from .moption import Q7OptionContext as OCTXT
from .wfingerprint import Q7FingerPrint


def trace(f):
    def globaltrace(frame, why, arg):
        if why == "call":
            return localtrace
        return None

    def localtrace(frame, why, arg):
        if why == "line":
            # record the file name and line number of every trace
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno

            bname = os.path.basename(filename)
            print("{}({}): {}".format(bname,
                                      lineno,
                                      linecache.getline(filename, lineno)), )
        return localtrace

    def _f(*args, **kwds):
        sys.settrace(globaltrace)
        result = f(*args, **kwds)
        sys.settrace(None)
        return result

    return _f


HIDEVALUE = '@@HIDE@@'
LAZYVALUE = '@@LAZY@@'

(COLUMN_NAME,
 COLUMN_SIDS,
 COLUMN_FLAG_LINK,
 COLUMN_FLAG_SELECT,
 COLUMN_FLAG_CHECK,
 COLUMN_FLAG_USER,
 COLUMN_SHAPE,
 COLUMN_DATATYPE,
 COLUMN_VALUE) = range(9)

COLUMN_LAST = COLUMN_VALUE

COLUMN_ICO = [COLUMN_FLAG_LINK, COLUMN_FLAG_SELECT, COLUMN_FLAG_CHECK,
              COLUMN_FLAG_USER, COLUMN_VALUE]
COLUMN_FLAGS = [COLUMN_FLAG_LINK, COLUMN_FLAG_SELECT, COLUMN_FLAG_CHECK,
                COLUMN_FLAG_USER]

COLUMN_EDIT = [COLUMN_NAME, COLUMN_SIDS, COLUMN_DATATYPE, COLUMN_VALUE]

COLUMN_TITLE = ['-'] * (COLUMN_LAST + 1)

COLUMN_TITLE[COLUMN_NAME] = 'Name'
COLUMN_TITLE[COLUMN_SIDS] = 'SIDS type'
COLUMN_TITLE[COLUMN_FLAG_LINK] = 'L'
COLUMN_TITLE[COLUMN_SHAPE] = 'Shape'
COLUMN_TITLE[COLUMN_FLAG_SELECT] = 'M'
COLUMN_TITLE[COLUMN_FLAG_CHECK] = 'C'
COLUMN_TITLE[COLUMN_FLAG_USER] = 'U'
COLUMN_TITLE[COLUMN_DATATYPE] = 'D'
COLUMN_TITLE[COLUMN_VALUE] = 'Value'

STLKTOPOK = '@@LTOPOK@@'  # top link entry ok
STLKCHDOK = '@@LCHDOK@@'  # child link entry ok
STLKTOPBK = '@@LTOPBK@@'  # broken top link entry
STLKTOPNF = '@@LTOPNF@@'  # top link entry ok not followed
STLKNOLNK = '@@LNOLNK@@'  # no link
STLKIGNOD = '@@LIGNOD@@'  # link ignored by load process

STCHKUNKN = '@@CKUNKN@@'  # unknown
STCHKGOOD = '@@CKGOOD@@'  # good, ok
STCHKWARN = '@@CKWARN@@'  # warning
STCHKFAIL = '@@CKFAIL@@'  # fail, bad
STCHKUSER = '@@CKUSER@@'  # user condition

STCHKLIST = (STCHKUNKN, STCHKGOOD, STCHKWARN, STCHKFAIL, STCHKUSER)

STUSR__ = '@@USR_%s@@'
STUSR_P = '@@USR_%d@@'
STUSR_X = '@@USR_X@@'
STUSR_0 = '@@USR_0@@'
STUSR_1 = '@@USR_1@@'
STUSR_2 = '@@USR_2@@'
STUSR_3 = '@@USR_3@@'
STUSR_4 = '@@USR_4@@'
STUSR_5 = '@@USR_5@@'
STUSR_6 = '@@USR_6@@'
STUSR_7 = '@@USR_7@@'
STUSR_8 = '@@USR_8@@'
STUSR_9 = '@@USR_9@@'
STUSR_A = '@@USR_A@@'
STUSR_B = '@@USR_B@@'
STUSR_C = '@@USR_C@@'
STUSR_D = '@@USR_D@@'
STUSR_E = '@@USR_E@@'
STUSR_F = '@@USR_F@@'

USERSTATES = [STUSR_0, STUSR_1, STUSR_2, STUSR_3, STUSR_4,
              STUSR_5, STUSR_6, STUSR_7, STUSR_8, STUSR_9,
              STUSR_A, STUSR_B, STUSR_C, STUSR_D, STUSR_E, STUSR_F]

STMARK_ON = '@@MARK_ON@@'
STMARKOFF = '@@MARKOFF@@'

STSHRUNKN = '@@SHRUNKN@@'

EDITNODE = '@@NODEDIT@@'
NEWCHILDNODE = '@@NODENEWC@@'
NEWBROTHERNODE = '@@NODENEWB@@'
MARKNODE = '@@NODEMARK@@'
DOWNNODE = '@@NODEDOWN@@'
UPNODE = '@@NODEUP@@'
COPY = '@@NODECOPY@@'
CUT = '@@NODECUT@@'
DELETENODE = '@@NODEDELETE@@'
PASTEBROTHER = '@@NODEPASTEB@@'
PASTECHILD = '@@NODEPASTEC@@'
CUTSELECTED = '@@NODECUTS@@'
PASTEBROTHERSELECTED = '@@NODEPASTEBS@@'
PASTECHILDSELECTED = '@@NODEPASTECS@@'
OPENFORM = '@@OPENFORM@@'
OPENVIEW = '@@OPENVIEW@@'
LOADNODEDATASELECTED = '@@LOADNODEDATASELECTED@@'
RELEASENODEDATASELECTED = '@@RELEASENODEDATASELECTED@@'
LOADNODEDATA = '@@LOADNODEDATA@@'
RELEASENODEDATA = '@@RELEASENODEDATA@@'
OPENLINKEDTOFILE = '@@OPENLINKEDTOFILE@@'
EXPANDSUBTREE = '@@EXPANDSUBTREE@@'
COLLAPSESUBTREE = '@@COLLAPSESUBTREE@@'
EXPANDSUBTREESELECTED = '@@EXPANDSUBTREESELECTED@@'
COLLAPSESUBTREESELECTED = '@@COLLAPSESUBTREESELECTED@@'

DT_LARGE = '@@DATALARGE@@'
DT_LAZY = '@@DATALAZY@@'

USERFLAG_0 = '@@USER0@@'
USERFLAG_1 = '@@USER1@@'
USERFLAG_2 = '@@USER2@@'
USERFLAG_3 = '@@USER3@@'
USERFLAG_4 = '@@USER4@@'
USERFLAG_5 = '@@USER5@@'
USERFLAG_6 = '@@USER6@@'
USERFLAG_7 = '@@USER7@@'
USERFLAG_8 = '@@USER8@@'
USERFLAG_9 = '@@USER9@@'
USERFLAG_A = '@@USERA@@'
USERFLAG_B = '@@USERB@@'
USERFLAG_C = '@@USERC@@'
USERFLAG_D = '@@USERD@@'
USERFLAG_E = '@@USERE@@'
USERFLAG_F = '@@USERF@@'

USERFLAGS = [USERFLAG_0, USERFLAG_1, USERFLAG_2, USERFLAG_3, USERFLAG_4,
             USERFLAG_5, USERFLAG_6, USERFLAG_7, USERFLAG_8, USERFLAG_9,
             USERFLAG_A, USERFLAG_B, USERFLAG_C, USERFLAG_D, USERFLAG_E,
             USERFLAG_F,
             ]

ICONMAPPING = {
    STLKNOLNK: ":/images/icons/empty.png",
    STLKTOPOK: ":/images/icons/link.png",
    STLKCHDOK: ":/images/icons/link-child.png",
    STLKTOPBK: ":/images/icons/link-break.png",
    STLKTOPNF: ":/images/icons/link-error.png",
    STLKIGNOD: ":/images/icons/link-ignore.png",

    STCHKUNKN: ":/images/icons/empty.png",
    STCHKGOOD: ":/images/icons/check-ok.png",
    STCHKFAIL: ":/images/icons/check-fail.png",
    STCHKWARN: ":/images/icons/check-warn.png",

    STUSR_X: ":/images/icons/empty.png",
    STSHRUNKN: ":/images/icons/empty.png",
    STMARKOFF: ":/images/icons/empty.png",
    STMARK_ON: ":/images/icons/mark-node.png",

    STUSR_0: ":/images/icons/user-0.png",
    STUSR_1: ":/images/icons/user-1.png",
    STUSR_2: ":/images/icons/user-2.png",
    STUSR_3: ":/images/icons/user-3.png",
    STUSR_4: ":/images/icons/user-4.png",
    STUSR_5: ":/images/icons/user-5.png",
    STUSR_6: ":/images/icons/user-6.png",
    STUSR_7: ":/images/icons/user-7.png",
    STUSR_8: ":/images/icons/user-8.png",
    STUSR_9: ":/images/icons/user-9.png",
    STUSR_A: ":/images/icons/user-A.png",
    STUSR_B: ":/images/icons/user-B.png",
    STUSR_C: ":/images/icons/user-C.png",
    STUSR_D: ":/images/icons/user-D.png",
    STUSR_E: ":/images/icons/user-E.png",
    STUSR_F: ":/images/icons/user-F.png",

    DT_LARGE: ":/images/icons/data-array-large.png",
    DT_LAZY: ":/images/icons/data-array-lazy.png",
}

KEYMAPPING = {
    MARKNODE: Qt.Key_Space,
    UPNODE: Qt.Key_Up,
    DOWNNODE: Qt.Key_Down,
    OPENFORM: Qt.Key_F,
    OPENVIEW: Qt.Key_W,

    NEWCHILDNODE: Qt.Key_A,
    NEWBROTHERNODE: Qt.Key_Z,
    EDITNODE: Qt.Key_Insert,
    COPY: Qt.Key_C,
    CUT: Qt.Key_X,
    DELETENODE: Qt.Key_Delete,
    PASTECHILD: Qt.Key_Y,
    PASTEBROTHER: Qt.Key_V,
    CUTSELECTED: Qt.Key_X,
    PASTECHILDSELECTED: Qt.Key_Y,
    PASTEBROTHERSELECTED: Qt.Key_V,

    EXPANDSUBTREE: Qt.Key_Plus,
    COLLAPSESUBTREE: Qt.Key_Minus,
    EXPANDSUBTREESELECTED: Qt.Key_Plus,
    COLLAPSESUBTREESELECTED: Qt.Key_Minus,

    LOADNODEDATA: Qt.Key_L,
    RELEASENODEDATA: Qt.Key_R,
    LOADNODEDATASELECTED: Qt.Key_L,
    RELEASENODEDATASELECTED: Qt.Key_R,

    OPENLINKEDTOFILE: Qt.Key_O,

    USERFLAG_0: Qt.Key_0,
    USERFLAG_1: Qt.Key_1,
    USERFLAG_2: Qt.Key_2,
    USERFLAG_3: Qt.Key_3,
    USERFLAG_4: Qt.Key_4,
    USERFLAG_5: Qt.Key_5,
    USERFLAG_6: Qt.Key_6,
    USERFLAG_7: Qt.Key_7,
    USERFLAG_8: Qt.Key_8,
    USERFLAG_9: Qt.Key_9,
    USERFLAG_A: Qt.Key_A,
    USERFLAG_B: Qt.Key_B,
    USERFLAG_C: Qt.Key_C,
    USERFLAG_D: Qt.Key_D,
    USERFLAG_E: Qt.Key_E,
    USERFLAG_F: Qt.Key_F,
}

MODIFIERMAPPING = {
    OPENFORM: (Qt.Key_Control,),
    OPENVIEW: (Qt.Key_Control,),

    EXPANDSUBTREE: (Qt.Key_Control,),
    COLLAPSESUBTREE: (Qt.Key_Control,),
    EXPANDSUBTREESELECTED: (Qt.Key_Control, Qt.Key_Shift),
    COLLAPSESUBTREESELECTED: (Qt.Key_Control, Qt.Key_Shift),

    NEWCHILDNODE: (Qt.Key_Control,),
    NEWBROTHERNODE: (Qt.Key_Control,),
    EDITNODE: (Qt.Key_Control,),
    COPY: (Qt.Key_Control,),
    CUT: (Qt.Key_Control,),
    PASTECHILD: (Qt.Key_Control,),
    PASTEBROTHER: (Qt.Key_Control,),
    CUTSELECTED: (Qt.Key_Control, Qt.Key_Shift),
    PASTECHILDSELECTED: (Qt.Key_Control, Qt.Key_Shift),
    PASTEBROTHERSELECTED: (Qt.Key_Control, Qt.Key_Shift),

    LOADNODEDATA: (Qt.Key_Control,),
    RELEASENODEDATA: (Qt.Key_Control,),
    LOADNODEDATASELECTED: (Qt.Key_Control, Qt.Key_Shift),
    RELEASENODEDATASELECTED: (Qt.Key_Control, Qt.Key_Shift),

    OPENLINKEDTOFILE: (Qt.Key_Control,),
}

EDITKEYMAPPINGS = [
    KEYMAPPING[NEWCHILDNODE],
    KEYMAPPING[NEWBROTHERNODE],
    KEYMAPPING[COPY],
    KEYMAPPING[CUT],
    KEYMAPPING[DELETENODE],
    KEYMAPPING[PASTECHILD],
    KEYMAPPING[PASTEBROTHER],
    KEYMAPPING[CUTSELECTED],
    KEYMAPPING[PASTECHILDSELECTED],
    KEYMAPPING[PASTEBROTHERSELECTED],
]

USERKEYMAPPINGS = [KEYMAPPING[USERFLAG_0], KEYMAPPING[USERFLAG_1],
                   KEYMAPPING[USERFLAG_2], KEYMAPPING[USERFLAG_3],
                   KEYMAPPING[USERFLAG_4], KEYMAPPING[USERFLAG_5],
                   KEYMAPPING[USERFLAG_6], KEYMAPPING[USERFLAG_7],
                   KEYMAPPING[USERFLAG_8], KEYMAPPING[USERFLAG_9],
                   KEYMAPPING[USERFLAG_A], KEYMAPPING[USERFLAG_B],
                   KEYMAPPING[USERFLAG_C], KEYMAPPING[USERFLAG_D],
                   KEYMAPPING[USERFLAG_E], KEYMAPPING[USERFLAG_F],
                   ]

ALLKEYMAPPINGS = [KEYMAPPING[v] for v in KEYMAPPING]


# -----------------------------------------------------------------
# class Q7TreeFilterProxy(QSortFilterProxyModel):
#     def __init__(self,parent):
#         QSortFilterProxyModel.__init__(self,parent)
#         self._treeview=parent.treeview
#         self._control=parent.parent
#         self._wparent=parent
#         self.setDynamicSortFilter(False)
#         self.setFilterRole( Qt.EditRole | Qt.DisplayRole )
#     def lessThan(self,i1,i2):
#         fsort=self.sourceModel().getSortedChildRow
#         c1=fsort(i1.internalPointer().sidsPath())
#         c2=fsort(i2.internalPointer().sidsPath())
#         return c2<c1

# -----------------------------------------------------------------
class Q7TreeView(QTreeView):
    def __init__(self, parent):
        QTreeView.__init__(self, None)
        self._parent = parent
        self._control = None
        self._fgindex = -1
        self.setUniformRowHeights(True)

    @property
    def FG(self):
        return Q7FingerPrint.getByIndex(self._fgindex)

    def isLocked(self):
        return self.FG.isLocked()

    def setControlWindow(self, control, fingerprintindex):
        self._control = control
        self._fgindex = fingerprintindex

    def clearLastEntered(self):
        if self._control is not None:
            return self._control.clearLastEntered()
        return None

    def getLastEntered(self):
        if self._control is not None:
            return self._control.getLastEntered()
        return None

    def setLastEntered(self, ix=None):
        if self._control is not None:
            self._control.setLastEntered(ix)

    def selectionChanged(self, old, new):
        # QTreeView.selectionChanged(self,old,new)
        if old.count():
            n = self.modelData(QModelIndex(old[0].topLeft()))
            self._parent.updateStatus(n)

    def modelData(self, idx):
        if self._control is not None:
            return self._control.modelData(idx)
        return None

    def modelIndex(self, idx):
        if self._control is not None:
            return self._control.modelIndex(idx)
        return None

    def mousePressEvent(self, event):
        self.lastPos = event.globalPos()
        self.lastButton = event.button()
        QTreeView.mousePressEvent(self, event)

    def wink(self, index):
        if self.isExpanded(index):
            self.setExpanded(index, False)
            self.setExpanded(index, True)
        else:
            self.setExpanded(index, True)
            self.setExpanded(index, False)

    def keyPressEvent(self, event):
        kmod = event.modifiers()
        kval = event.key()
        if kval not in ALLKEYMAPPINGS:
            return
        last = self.getLastEntered()
        if last is not None:
            lastpath = last.sidsPath()
            nix = self.model().indexByPath(last.sidsPath())
            pix = self.model().indexByPath(last.parentItem().sidsPath())
            if kval in EDITKEYMAPPINGS:
                if (kmod & Qt.ControlModifier and
                        not kmod & Qt.ShiftModifier):
                    if kval == KEYMAPPING[NEWCHILDNODE]:
                        self.model().newNodeChild(last)
                        nix = self.model().indexByPath(last.sidsPath())
                        self.exclusiveSelectRow(nix)
                    if kval == KEYMAPPING[NEWBROTHERNODE]:
                        self.model().newNodeBrother(last)
                        nix = self.model().indexByPath(last.sidsPath())
                        self.exclusiveSelectRow(nix)
                    if kval == KEYMAPPING[COPY]:
                        self.model().copyNode(last)
                        self.exclusiveSelectRow(nix)
                    if kval == KEYMAPPING[CUT]:
                        if self.model().cutNode(last):
                            self.exclusiveSelectRow(pix)
                        else:
                            self.exclusiveSelectRow(nix)
                    if kval == KEYMAPPING[PASTECHILD]:
                        self.model().pasteAsChild(last)
                        nix = self.model().indexByPath(last.sidsPath())
                        self.exclusiveSelectRow(nix)
                    if kval == KEYMAPPING[PASTEBROTHER]:
                        self.model().pasteAsBrother(last)
                        nix = self.model().indexByPath(last.sidsPath())
                        self.exclusiveSelectRow(nix)
                if (kmod & Qt.ControlModifier and
                        kmod & Qt.ShiftModifier):
                    if kval == KEYMAPPING[CUTSELECTED]:
                        self.model().cutAllSelectedNodes()
                        self.exclusiveSelectRow()
                    if kval == KEYMAPPING[PASTECHILDSELECTED]:
                        self.model().pasteAsChildAllSelectedNodes()
                        self.exclusiveSelectRow()
                    if kval == KEYMAPPING[PASTEBROTHERSELECTED]:
                        self.model().pasteAsBrotherAllSelectedNodes()
                        self.exclusiveSelectRow()
                if kval == KEYMAPPING[DELETENODE]:
                    if self.model().deleteNode(last):
                        self.exclusiveSelectRow(pix)
                    else:
                        self.exclusiveSelectRow(nix)
            # --- same keys different meta
            if ((kval in USERKEYMAPPINGS) and not
            (kmod & Qt.ControlModifier) and not
            (kmod & Qt.ShiftModifier)):
                last.setUserState(kval - 48)
                self.exclusiveSelectRow(nix)
            elif ((kval == KEYMAPPING[EXPANDSUBTREE]) and
                  (kmod & Qt.ControlModifier) and not
                  (kmod & Qt.ShiftModifier)):
                self.expand_sb()
            elif ((kval == KEYMAPPING[COLLAPSESUBTREE]) and
                  (kmod & Qt.ControlModifier) and not
                  (kmod & Qt.ShiftModifier)):
                self.collapse_sb()
            elif ((kval == KEYMAPPING[EXPANDSUBTREESELECTED]) and
                  (kmod & Qt.ControlModifier) and
                  (kmod & Qt.ShiftModifier)):
                self.sexpand_sb()
            elif ((kval == KEYMAPPING[COLLAPSESUBTREESELECTED]) and
                  (kmod & Qt.ControlModifier) and
                  (kmod & Qt.ShiftModifier)):
                self.scollapse_sb()
            elif kval == KEYMAPPING[EDITNODE]:
                if kmod & Qt.ControlModifier:
                    eix = self.model().createIndex(nix.row(), COLUMN_SIDS,
                                                   nix.internalPointer())
                    last.setEditCheck(False)
                    self.edit(eix)
                elif kmod & Qt.ShiftModifier:
                    eix = self.model().createIndex(nix.row(), COLUMN_VALUE,
                                                   nix.internalPointer())
                    last.setEditCheck(False)
                    self.edit(eix)
                else:
                    self.edit(nix)
            elif kval == KEYMAPPING[MARKNODE]:
                self.markNode(last)
                self.exclusiveSelectRow(nix)
            elif kval == KEYMAPPING[UPNODE]:
                if kmod & Qt.ControlModifier:
                    self.upRowLevel(nix)
                elif kmod & Qt.ShiftModifier:
                    self.upRowMarked()
                else:
                    QTreeView.keyPressEvent(self, event)
                    lix = self.modelCurrentIndex()
                    self.exclusiveSelectRow(lix)
            elif kval == KEYMAPPING[DOWNNODE]:
                if kmod & Qt.ControlModifier:
                    self.downRowLevel(nix)
                elif kmod & Qt.ShiftModifier:
                    self.downRowMarked()
                else:
                    QTreeView.keyPressEvent(self, event)
                    lix = self.modelCurrentIndex()
                    self.exclusiveSelectRow(lix)
            elif kval == KEYMAPPING[OPENFORM]:
                self._parent.formview()
                self.exclusiveSelectRow(nix)
            elif kval == KEYMAPPING[OPENVIEW]:
                self._parent.openSubTree()
                self.exclusiveSelectRow(nix)
            else:
                QTreeView.keyPressEvent(self, event)

    def modelCurrentIndex(self):
        idx = self.tryToMapTo(self.currentIndex())
        return idx

    def refreshView(self, ixc=None):
        if ixc is not None and ixc.isValid():
            self.model().refreshModel(ixc)
            return
        ixc = self.model().createIndex(0, 0, None)
        self.model().refreshModel(ixc)

    def markNode(self, node):
        node.switchMarked()
        self.model().updateSelected()
        self.changeRow(node)

    def upRowLevel(self, index):
        self.relativeMoveToRow(-1, index)

    def downRowLevel(self, index):
        self.relativeMoveToRow(+1, index)

    def upRowMarked(self):
        self.changeSelectedMark(-1)

    def downRowMarked(self):
        self.changeSelectedMark(+1)

    def relativeMoveToRow(self, shift, index):
        if (index == -1) or (not index.isValid()):
            index = QModelIndex()
        index = self.tryToMapTo(index)
        row = index.row()
        col = index.column()
        if not index.sibling(row + shift, col).isValid():
            return
        parent = index.parent()
        nix = self.model().index(row + shift, col, parent)
        self.exclusiveSelectRow(nix)

    def changeRow(self, nodeitem):
        pix = self.model().indexByPath(nodeitem.sidsPath()).parent()
        row = pix.row()
        ix1 = self.model().createIndex(row, 0, nodeitem)
        ix2 = self.model().createIndex(row, COLUMN_LAST, nodeitem)
        self.model().dataChanged.emit(ix1, ix2)

    def M(self):
        try:
            m = self.model().sourceModel()
        except AttributeError:
            m = self.model()
        return m

    def tryToMapTo(self, idx):
        if idx.model() is None:
            return idx
        if idx.model() != self.M():
            idx = self.model().mapToSource(idx)
        return idx

    def tryToMapFrom(self, idx):
        if idx.model() is None:
            return idx
        if idx.model() != self.M():
            idx = self.model().mapFromSource(idx)
        return idx

    def selectByPath(self, path):
        npath = CGU.getPathNoRoot(path)
        ix = self.M().indexByPath('/CGNSTree' + npath)
        self.exclusiveSelectRow(ix)

    def exclusiveSelectRow(self, index=-1, setlast=True):
        if index == -1:
            index = QModelIndex()
        elif not index.isValid():
            index = QModelIndex()
        mod = QItemSelectionModel.SelectCurrent | QItemSelectionModel.Rows
        if index.internalPointer() is None:
            return
        self.selectionModel().clearSelection()
        pth = index.internalPointer().sidsPath()
        nix = self.M().indexByPath(pth)
        self.selectionModel().setCurrentIndex(index, mod)
        if setlast:
            self.clearLastEntered()
            self.setLastEntered(index)
        self.scrollTo(index)
        self.refreshView(index)

    def changeSelectedMark(self, delta):
        if self.M()._selected == []:
            return
        sidx = self.M()._selectedIndex
        if self.M()._selectedIndex == -1:
            self.M()._selectedIndex = 0
        elif (delta == -1) and (sidx == 0):
            self.M()._selectedIndex = len(self.M()._selected) - 1
        elif delta == -1:
            self.M()._selectedIndex -= 1
        elif (delta == +1) and (sidx == len(self.M()._selected) - 1):
            self.M()._selectedIndex = 0
        elif delta == +1:
            self.M()._selectedIndex += 1
        if ((self.M()._selectedIndex != -1) and
                (self.M()._selectedIndex < len(self.M()._selected))):
            path = self.M()._selected[self.M()._selectedIndex]
            idx = self.M().match(self.M().index(0, 0, QModelIndex()),
                                 Qt.UserRole,
                                 path,
                                 flags=Qt.MatchExactly | Qt.MatchRecursive)
            if idx[0].isValid():
                self.exclusiveSelectRow(idx[0])

    def doRelease(self):
        self._model = None

    def expand_sb(self):
        ix = self.modelCurrentIndex()
        self._expand_sb(ix)
        self.resizeAll()

    def _expand_sb(self, ix):
        self.expand(ix)
        for c in self.modelData(ix).children():
            ix2 = self.M().indexByPath(c.sidsPath())
            self._expand_sb(ix2)

    def collapse_sb(self):
        ix = self.modelCurrentIndex()
        self._collapse_sb(ix)
        self.resizeAll()

    def _collapse_sb(self, ix):
        self.collapse(ix)
        for c in self.modelData(ix).children():
            ix2 = self.M().indexByPath(c.sidsPath())
            self._collapse_sb(ix2)

    def sexpand_sb(self):
        for pth in self.M().selectedNodes:
            ix = self.M().indexByPath(pth)
            self._expand_sb(ix)
        self.resizeAll()

    def scollapse_sb(self):
        for pth in self.M().selectedNodes:
            ix = self.M().indexByPath(pth)
            self._collapse_sb(ix)
        self.resizeAll()

    def resizeAll(self):
        for n in range(COLUMN_LAST + 1):
            self.resizeColumnToContents(n)


# -----------------------------------------------------------------
def __sortItems(i1, i2):
    c1 = i1.model().getSortedChildRow(i1.sidsPath())
    c2 = i2.model().getSortedChildRow(i2.sidsPath())
    return c1 - c2


# -----------------------------------------------------------------
class Q7TreeItem(object):
    stype = {'MT': 0, 'I4': 4, 'I8': 8, 'R4': 4, 'R8': 8, 'C1': 1, 'LK': 0}
    atype = {'I4': 'int32', 'I8': 'int64',
             'R4': 'float32', 'R8': 'float64',
             'LK': 'S', 'MT': 'S', 'C1': 'S'}
    __lastEdited = None

    def __init__(self, fgprintindex, data, model, tag="", parent=None):
        self._model = model
        self._parentitem = parent
        self._itemnode = data
        self._childrenitems = []
        self._title = COLUMN_TITLE
        self._size = None
        self._fgindex = fgprintindex
        self._log = None
        self._checkable = True
        self._diag = None
        self._states = {'mark': STMARKOFF, 'check': STCHKUNKN,
                        'user': STUSR_X, 'shared': STSHRUNKN}
        self._tag = tag
        self._nodes = 0
        if parent is not None:
            self._path = CGU.getPathNormalize(parent.sidsPath() + '/' + data[0])
            self._path_no_root = CGU.getPathNoRoot(self._path)
            if (model is not None):
                self._model._extension[self._path] = self
                self._nodes = len(self._model._extension)
        else:
            self._path = ''
            self._path_no_root = ''
        self._depth = self._path.count('/')
        if self._path in self.FG.lazy:
            self._lazy = True
        else:
            self._lazy = False
        self.set_sidsIsLinkChild()
        self.set_sidsLinkStatus()
        self.set_sidsDataType()

    @property
    def FG(self):
        return Q7FingerPrint.getByIndex(self._fgindex)

    def orderTag(self):
        return self._tag + "0" * (self.FG.depth * 4 - len(self._tag))

    def updateData(self, data):
        self._itemnode[1] = data

    def sidsIsRoot(self):
        if self._parentitem is None:
            return True
        return False

    def sidsIsCGNSTree(self):
        if self._path == '/CGNSTree':
            return True
        return False

    def sidsParent(self):
        return self._parentitem._itemnode

    def sidsPathSet(self, path):
        del self._model._extension[self._path]
        self._path = path
        self._model._extension[self._path] = self

    def sidsPath(self):
        return self._path

    def sidsNode(self):
        return self._itemnode

    def sidsName(self):
        return self._itemnode[0]

    def sidsNameSet(self, name):
        if not isinstance(name, (str, bytes)):
            return False
        try:
            name = str(name)
        except UnicodeEncodeError:
            return False
        if self._checkable and self._itemnode[0] != name:
            if name == '':
                return False
            if not CGU.checkName(name):
                return False
            if (not CGU.checkDuplicatedName(self.sidsParent(),
                                            name, dienow=False)):
                return False
        self._itemnode[0] = name
        self._checkable = True
        return True

    def sidsValue(self):
        return self._itemnode[1]

    def sidsValueArray(self):
        if isinstance(self._itemnode[1], numpy.ndarray):
            return True
        return False

    def sidsValueFortranOrder(self):
        if self.sidsValueArray():
            return numpy.isfortran(self.sidsValue())
        return False

    def sidsValueEnum(self):
        if self.sidsType() in CGK.cgnsenums:
            return CGK.cgnsenums[self.sidsType()]
        return None

    def trySetAs(self, value, dt):
        try:
            aval = eval(value)
            if aval is None:
                self._itemnode[1] = aval
                return True
            if isinstance(aval, list):
                aval = numpy.array(aval, dtype=dt, order='F')
                self._itemnode[1] = aval
                return True
            else:
                aval = numpy.array([aval], dtype=dt, order='F')
                self._itemnode[1] = aval
                return True
        except:
            pass
        return False

    def sidsValueSet(self, value):
        if value == "":
            return False
        odt = self.sidsDataType()
        if odt == CGK.R4:
            return self.trySetAs(value, 'float32')
        if odt == CGK.R8:
            return self.trySetAs(value, 'float64')
        if odt == CGK.I4:
            return self.trySetAs(value, 'int32')
        if odt == CGK.I8:
            return self.trySetAs(value, 'int64')
        if odt == CGK.MT:
            self._itemnode[1] = None
            return True
        if odt == CGK.C1:
            if not isinstance(value, (str, bytes)):
                return False
            try:
                value = str(value)
            except UnicodeEncodeError:
                return False
            self._itemnode[1] = CGU.setStringAsArray(value)
            return True
        return False

    def sidsChildren(self):
        return self._itemnode[2]

    def sidsType(self):
        return self._itemnode[3]

    def sidsTypePath(self):
        return CGU.getPathAsTypes(self.FG.tree, self._path,
                                  legacy=False)

    def sidsTypeSet(self, value):
        if not isinstance(value, (str, bytes)):
            return False
        try:
            value = str(value)
        except UnicodeEncodeError:
            return False
        if self._checkable and (value not in self.sidsTypeList()):
            return False
        self._itemnode[3] = value
        self._checkable = True
        return True

    def set_sidsDataType(self, all=False):
        if self._itemnode is None:
            return None
        self._datatype = CGU.getValueDataType(self._itemnode)
        return self._datatype

    def sidsDataType(self, all=False):
        if all:
            return CGK.adftypes
        return self._datatype

    def sidsDataTypeSet(self, value):
        if value not in CGK.adftypes:
            print('BAD')
            return False
        odt = self.sidsDataType()
        ndt = value
        adt = Q7TreeItem.atype[ndt]
        oval = self._itemnode[1]
        if (oval is None) and (ndt != CGK.MT):
            if ndt == CGK.C1:
                oval = ""
            else:
                oval = (0,)
        if ndt == CGK.MT:
            nval = None
        else:
            if ((odt not in [CGK.I4, CGK.I8, CGK.R4, CGK.R8]) and
                    (ndt in [CGK.I4, CGK.I8, CGK.R4, CGK.R8])):
                oval = (0,)
            if ndt == CGK.C1:
                oval = CGU.setStringAsArray(oval)
            nval = numpy.array(oval, dtype=adt)
        self._itemnode[1] = nval
        self.set_sidsDataType()
        return True

    def sidsDataTypeSize(self):
        return Q7TreeItem.stype[self._datatype]

    def sidsTypeList(self):
        tlist = CGU.getNodeAllowedChildrenTypes(self._parentitem._itemnode,
                                                self._itemnode)
        return tlist

    def sidsDims(self):
        if isinstance(self.sidsValue(), numpy.ndarray):
            return self.sidsValue().shape
        return (0,)

    def set_sidsLinkStatus(self):
        self._link_status = STLKNOLNK
        self._is_link = False
        if self._path_no_root in [lk[-2] for lk in self.FG.links]:
            self._link_status = STLKTOPOK
            self._is_link = True
        elif self.sidsIsLinkChild():
            self._link_status = STLKCHDOK
            self._is_link = True

    def set_sidsIsLinkChild(self):
        self._is_link_child = False
        pit = self.parentItem()
        if (pit is None): return False
        while not pit.sidsIsRoot():
            if pit.sidsIsLink():
                self._is_link_child = True
                return True
            pit = pit.parentItem()
        return False

    def sidsLinkStatus(self):
        return self._link_status

    def sidsIsLink(self):
        return self._is_link

    def sidsIsLinkChild(self):
        return self._is_link_child

    def sidsLinkFilename(self):
        pth = CGU.getPathNoRoot(self.sidsPath())
        for lk in self.FG.links:
            if pth in lk[-2]:
                return "%s/%s" % (lk[0], lk[1])
        return None

    def sidsLinkValue(self):
        pth = CGU.getPathNoRoot(self.sidsPath())
        for lk in self.FG.links:
            if pth in lk[-2]:
                return "[%s/]%s:%s" % (lk[0], lk[1], lk[2])
        return None

    def sidsRemoveChild(self, node):
        children = self.sidsChildren()
        idx = 0
        while idx < len(children):
            childnode = children[idx]
            if (childnode[0] == node[0]) and (childnode[3] == node[3]):
                break
            idx += 1
        if idx < len(children):
            children.pop(idx)

    def sidsAddChild(self, node):
        if node is None:
            ntype = CGK.UserDefinedData_ts
            name = '{%s#%.3d}' % (ntype, 0)
            newtree = CGU.newNode(name, None, [], ntype)
        else:
            newtree = CGU.nodeCopy(node)
        name = newtree[0]
        ntype = newtree[3]
        parent = self._itemnode
        if parent is None:
            parent = self.FG.tree
        count = 0
        while not CGU.checkDuplicatedName(parent, name, dienow=False):
            count += 1
            name = '{%s#%.3d}' % (ntype, count)
        newtree[0] = name
        parent[2].append(newtree)
        newpath = self.sidsPath() + '/%s' % name
        newrow = self._model.getSortedChildRow(newpath)
        return (newtree, newpath, newrow)

    def addChild(self, item, idx):
        self._childrenitems.insert(idx, item)

    def moveChild(self, item, fromrow, torow):
        self._childrenitems.pop(fromrow)
        self._childrenitems.insert(torow, item)

    def delChild(self, item):
        idx = 0
        while idx < self.childrenCount():
            if item == self._childrenitems[idx]:
                break
            idx += 1
        if idx < self.childrenCount():
            self._childrenitems.pop(idx)

    def children(self):
        return self._childrenitems

    def child(self, row):
        return self._childrenitems[row]

    def childRow(self):
        pth = self.sidsPath()
        parentitem = self.parentItem()
        row = 0
        for child in parentitem._childrenitems:
            if child.sidsPath() == pth:
                return row
            row += 1
        return -1

    def hasChildren(self):
        if self.childrenCount() > 0:
            return True
        return False

    def childrenCount(self):
        return len(self._childrenitems)

    def columnCount(self):
        return COLUMN_LAST + 1

    def setEditCheck(self, check=True):
        self._checkable = check

    def hasEditCheck(self):
        return self._checkable

    def dataRelease(self):
        if self._lazy:
            return
        if self.sidsPath() not in self.FG.lazy:
            self.FG.lazy[self.sidsPath()] = 1
        self._lazy = True
        self.updateData(None)

    def dataLoad(self, partialtree):
        if not self._lazy:
            return
        nodetab = self.sidsValue()
        npath = CGU.getPathNoRoot(self.sidsPath())
        if (nodetab is None) and (partialtree[2] is not None):
            nval = CGU.getNodeByPath(partialtree, npath)
            if nval is not None:
                self.updateData(nval[1])
                if npath in self.FG.lazy:
                    del self.FG.lazy.remove[self.sidsPath()]
                self._lazy = False
        return self

    def hasLazyLoad(self):
        return self._lazy

    def hasValueView(self):
        if self._lazy:
            return False
        if self.sidsValue() is None:
            return False
        if isinstance(self.sidsValue(), numpy.ndarray):
            if not self.sidsValue().shape:
                return False
            try:
                vsize = 1
                for x in self.sidsValue().shape:
                    vsize *= x
            except TypeError:
                print('# CGNS.NAV unexpected (mtree.hasValueView)',
                      self.sidsPath())
            if ((vsize > OCTXT.MaxDisplayDataSize) and
                    (OCTXT.MaxDisplayDataSize > 0)):
                return False
            if self.sidsValue().dtype.kind in ['S', 'a']:
                if len(self.sidsValue().shape) == 1:
                    return True
                if len(self.sidsValue().shape) > 2:
                    return False
                return False
            return True
        return False

    def data(self, column):
        if self._itemnode is None:
            return self._title[column]
        if column == COLUMN_NAME:
            return self.sidsName()
        if column == COLUMN_SIDS:
            return self.sidsType()
        if column == COLUMN_FLAG_LINK:
            return self.sidsLinkStatus()
        if column == COLUMN_SHAPE:
            if self.sidsValue() is None:
                return None
            if not isinstance(self.sidsValue(), numpy.ndarray):
                return None
            if not self.sidsValue().shape:
                return '(0,)'
            return str(self.sidsValue().shape)
        if column == COLUMN_FLAG_SELECT:
            return self._states['mark']
        if column == COLUMN_DATATYPE:
            return self.sidsDataType()
        if column == COLUMN_FLAG_CHECK:
            return self._states['check']
        if column == COLUMN_FLAG_USER:
            return self._states['user']
        if column == COLUMN_VALUE:
            if self._lazy:
                return LAZYVALUE
            if self.sidsValue() is None:
                return None
            if isinstance(self.sidsValue(), numpy.ndarray):
                if not self.sidsValue().shape:
                    vsize = 0
                else:
                    try:
                        vsize = 1
                        for x in self.sidsValue().shape:
                            vsize *= x
                    except TypeError:
                        print('# CGNS.NAV unexpected (mtree.data)', self.sidsPath())
                if ((vsize > OCTXT.MaxDisplayDataSize) and
                        (OCTXT.MaxDisplayDataSize > 0)):
                    return HIDEVALUE
                if self.sidsValue().dtype.kind in ['S', 'a']:
                    if vsize == 0:
                        return ''
                    if len(self.sidsValue().shape) == 1:
                        return self.sidsValue().tostring().decode('ascii')
                    if len(self.sidsValue().shape) > 2:
                        return HIDEVALUE
                    # TODO: use qtextedit for multiple lines
                    # v=self.sidsValue().T
                    # v=numpy.append(v,numpy.array([['\n']]*v.shape[0]),1)
                    # v=v.tostring().decode('ascii')
                    # return v
                    return HIDEVALUE
                if (self.sidsValue().shape == (1,)) and OCTXT.Show1DAsPlain:
                    return str(self.sidsValue()[0])
            return str(self.sidsValue().tolist())[:100]
        return None

    def parentItem(self):
        p = self._parentitem
        return p

    def row(self):
        if self._parentitem:
            return self._parentitem._childrenitems.index(self)
        return 0

    def switchMarked(self):
        if self._states['mark'] == STMARK_ON:
            self._states['mark'] = STMARKOFF
        else:
            self._states['mark'] = STMARK_ON

    def setCheck(self, check):
        if check in STCHKLIST:
            self._states['check'] = check

    def setDiag(self, diag):
        self._diag = diag

    def getDiag(self):
        return self._diag

    def userState(self):
        return self._states['user']

    def setUserStatePrivate(self, s):
        self._states['user'] = STUSR__ % s

    def setUserState(self, s):
        if int(s) in range(10):
            state = STUSR_P % int(s)
        else:
            state = STUSR__ % chr(s + 48)  # hmmm quite tricky
        if self._states['user'] == state:
            self._states['user'] = STUSR_X
        else:
            self._states['user'] = state

    def lastEdited(self):
        return Q7TreeItem.__lastEdited

    def setLastEdited(self):
        Q7TreeItem.__lastEdited = self
        return self

    def doRelease(self):
        Q7TreeItem.__lastEdited = None
        for c in list(self.children()):
            if c is not self:
                c.doRelease()
                if c.children() == []:
                    c._itemnode = None
                    c._model = None
                    self.delChild(c)


SORTTAG = "%.4x"


# -----------------------------------------------------------------
class Q7TreeModel(QAbstractItemModel):
    _icons = {}

    @classmethod
    def initIcons(cls):
        for ik in ICONMAPPING:
            Q7TreeModel._icons[ik] = QIcon(QPixmap(ICONMAPPING[ik]))

    def __init__(self, fgprintindex, parent=None):
        super(Q7TreeModel, self).__init__(parent)
        if Q7TreeModel._icons == {}:
            Q7TreeModel.initIcons()
        self._extension = {}
        self._fgindex = fgprintindex
        self._rootitem = Q7TreeItem(fgprintindex, (None), None)
        self._control = self.FG.control
        self._control.loadOptions()
        self._slist = OCTXT._SortedTypeList
        self._count = 0
        self._movedPaths = {}
        self.parseAndUpdate(self._rootitem,
                            self.FG.tree,
                            QModelIndex(), 0)
        self.FG.model = self
        self._selected = []
        self._selectedIndex = -1

    @property
    def FG(self):
        return Q7FingerPrint.getByIndex(self._fgindex)

    @property
    def selectedNodes(self):
        return self._selected

    def modelReset(self):
        # self.reset()
        self.beginResetModel()
        self.endResetModel()
        self._extension = {}
        self._count = 0
        self._movedPaths = {}
        self._rootitem = Q7TreeItem(self.FG.index, (None), None)
        self.parseAndUpdate(self._rootitem, self.FG.tree, QModelIndex(), 0)
        self._selected = []
        self._selectedIndex = -1

    def nodeFromPath(self, path):
        if path in self._extension:
            return self._extension[path]
        return None

    def modifiedPaths(self, node, parentpath, newname, oldname):
        self._movedPaths = {}
        newpath = parentpath + '/' + newname
        oldpath = parentpath + '/' + oldname
        oldplen = len(oldpath)
        node.sidsPathSet(newpath)
        plist = list(self._selected)
        for p in plist:
            if CGU.hasSameRootPath(oldpath, p):
                self._selected.remove(p)
                self._selected += [str(newpath + p[oldplen:])]
        for p in self._extension:
            if CGU.hasSameRootPath(oldpath, p):
                pn = str(newpath + p[oldplen:])
                nd = self._extension[p]
                nd.sidsPathSet(pn)
                self._movedPaths[p] = pn

    def sortNamesAndTypes(self, paths):
        t = []
        if paths is None:
            return []
        for p in paths:
            n = self.nodeFromPath(p)
            if n is not None:
                t += [(n.orderTag(), p)]
        t.sort()
        return [e[1] for e in t]

    def getSelected(self, noroot=False):
        if noroot:
            return [CGU.getPathNoRoot(pth) for pth in self._selected]
        return self._selected

    def getSelectedZones(self):
        zlist = []
        for pth in self._selected:
            spth = pth.split('/')
            if len(spth) > 3:
                zpth = '/'.join([''] + spth[2:4])
                node = CGU.getNodeByPath(self.FG.tree, zpth)
                if ((node is not None) and
                        (node[3] == CGK.Zone_ts) and
                        (zpth not in zlist)):
                    zlist += [zpth]
            if len(spth) == 3:
                bpth = '/'.join([''] + spth[2:])
                for node in CGU.getChildrenByPath(self.FG.tree, bpth):
                    if node[3] == CGK.Zone_ts:
                        zpth = bpth + '/' + node[0]
                        if zpth not in zlist:
                            zlist += [zpth]
        return zlist

    def getSelectedShortCut(self, ulist=None):
        slist = []
        if ulist is None:
            ulist = self._selected
        for pth in ulist:
            if CGU.hasFirstPathItem(pth):
                pth = CGU.removeFirstPathItem(pth)
            slist += [pth]
        return slist

    def updateSelected(self):
        self._selected = []
        self._selectedIndex = -1
        for k in self._extension:
            if self._extension[k]._states['mark'] == STMARK_ON:
                self._selected += [self._extension[k].sidsPath()]
        self._selected = self.sortNamesAndTypes(self._selected)

    def checkSelected(self):
        return self.checkTree(self.FG.tree, self._selected)

    def checkClear(self):
        for k in self._extension:
            self._extension[k]._states['check'] = STCHKUNKN

    def markExtendToList(self, mlist):
        for k in self._extension:
            if k in mlist:
                self._extension[k]._states['mark'] = STMARK_ON

    def markAll(self):
        for k in self._extension:
            self._extension[k]._states['mark'] = STMARK_ON

    def unmarkAll(self):
        for k in self._extension:
            self._extension[k]._states['mark'] = STMARKOFF

    def swapMarks(self):
        for k in self._extension:
            self._extension[k].switchMarked()

    def columnCount(self, parent):
        if parent.isValid():
            return parent.internalPointer().columnCount()
        else:
            return self._rootitem.columnCount()

    def data(self, index, role):
        if not index.isValid():
            return None
        col = index.column()
        item = index.internalPointer()
        if role == Qt.UserRole:
            return item.sidsPath()
        if role == Qt.ToolTipRole:
            if col == COLUMN_FLAG_LINK:
                lk = item.sidsLinkValue()
                if lk is not None:
                    return lk
            if col == COLUMN_FLAG_CHECK:
                dg = item.getDiag()
                if dg is not None:
                    return dg
            return None
        if role not in [Qt.EditRole, Qt.DisplayRole, Qt.DecorationRole]:
            return None
        if ((role == Qt.DecorationRole) and (col not in COLUMN_ICO)):
            return None
        disp = item.data(col)
        if (col == COLUMN_VALUE) and (role == Qt.DecorationRole):
            if disp == HIDEVALUE:
                return Q7TreeModel._icons[DT_LARGE]
            elif disp == LAZYVALUE:
                return Q7TreeModel._icons[DT_LAZY]
            else:
                return None
        if (col == COLUMN_VALUE) and (role == Qt.DisplayRole):
            if disp in [HIDEVALUE, LAZYVALUE]:
                return None
        # if ((index.column()==COLUMN_FLAG_USER) and (role == Qt.DisplayRole)):
        #     return None
        if disp in ICONMAPPING:
            disp = Q7TreeModel._icons[disp]
        return disp

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def headerData(self, section, orientation, role):
        if ((orientation == Qt.Horizontal) and
                (role == Qt.DisplayRole)):
            return self._rootitem.data(section)
        return None

    def indexByPath(self, path, nosort=False):
        if path in self._movedPaths:
            npath = self._movedPaths[path]
            path = npath
        row = self.getSortedChildRow(path, nosort)
        col = COLUMN_NAME
        ix = self.createIndex(row, col, self.nodeFromPath(path))
        if not ix.isValid():
            return QModelIndex()
        return ix

    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        if not parent.isValid():
            parentitem = self._rootitem
        else:
            parentitem = parent.internalPointer()
        childitem = parentitem.child(row)
        if childitem:
            return self.createIndex(row, column, childitem)
        return QModelIndex()

    def parent(self, index):
        if (not index.isValid()) or (index.internalPointer() is None):
            return QModelIndex()
        if index.internalPointer().sidsPath() == '/CGNSTree':
            return QModelIndex()
        childitem = index.internalPointer()
        parentitem = childitem.parentItem()
        if parentitem is None:
            return QModelIndex()
        return self.createIndex(parentitem.row(), 0, parentitem)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0
        if not parent.isValid():
            parentitem = self._rootitem
        else:
            parentitem = parent.internalPointer()
        if isinstance(parentitem, QModelIndex):
            return 0
        return parentitem.childrenCount()

    def getSortedChildRow(self, path, nosort=False):
        trace = 0
        npath = CGU.getPathNoRoot(path)
        if npath == '/':
            return -1
        targetname = CGU.getPathLeaf(npath)
        parentpath = CGU.getPathAncestor(npath)
        node = CGU.getNodeByPath(self.FG.tree, parentpath)
        if node is None:
            node = self.FG.tree
        row = 0
        if nosort:
            for childnode in node[2]:
                if childnode[0] == targetname:
                    return row
                row += 1
            return -1
        for childnode in CGU.getNextChildSortByType(node, criteria=self._slist):
            if childnode[0] == targetname:
                return row
            row += 1
        return -1

    def setData(self, index, value, role):
        if self.FG.isLocked():
            return
        if ((value is None) or
                (role != Qt.EditRole) or
                (not index.isValid()) or
                (index.column() not in COLUMN_EDIT)):
            return False
        node = index.internalPointer()
        if node.sidsIsLinkChild():
            return False
        node.setLastEdited()
        oldpath = node.sidsPath()
        oldname = node.sidsName()
        oldtype = node.sidsType()
        oldindex = self.indexByPath(node.sidsPath())
        parentpath = node.parentItem().sidsPath()
        fromrow = node.row()
        st = False
        if index.column() == COLUMN_NAME:
            newname = value
            st = node.sidsNameSet(value)
            if st:
                newpath = parentpath + '/' + newname
                torow = self.getSortedChildRow(newpath)
                node.parentItem().moveChild(node, fromrow, torow)
                self.modifiedPaths(node, parentpath, newname, oldname)
                newindex = self.indexByPath(newpath)
                self.rowsMoved.emit(newindex.parent(), fromrow, fromrow,
                                    newindex, torow)
                self.refreshModel(newindex.parent())
                self.refreshModel(newindex)
            else:
                MSG.wError(self._control,
                           0, "Cannot change name of node: %s" % oldpath,
                           "The new name [%s] is rejected" % newname)
        if index.column() == COLUMN_SIDS:
            newtype = value
            if not node.hasEditCheck() or (newtype in CGK.cgnstypes):
                st = node.sidsTypeSet(newtype)
            if st:
                torow = self.getSortedChildRow(oldpath)
                node.parentItem().moveChild(node, fromrow, torow)
                newindex = self.indexByPath(oldpath)
                self.rowsMoved.emit(newindex.parent(), fromrow, fromrow,
                                    newindex, torow)
                self.refreshModel(newindex.parent())
                self.refreshModel(newindex)
            else:
                MSG.wError(self._control,
                           0, "Cannot change SIDS type of node: %s" % oldpath,
                           "The new type [%s] is rejected" % newtype)
        if index.column() == COLUMN_VALUE:
            st = node.sidsValueSet(value)
            if not st:
                MSG.wError(self._control,
                           0, "Cannot change value of node: %s" % oldpath,
                           "The value is rejected")
        if index.column() == COLUMN_DATATYPE:
            st = node.sidsDataTypeSet(value)
        if st:
            self.FG.addTreeStatus(Q7FingerPrint.STATUS_MODIFIED)
        return st

    def removeItem(self, parentitem, targetitem, row):
        parentindex = self.indexByPath(parentitem.sidsPath())
        path = targetitem.sidsPath()
        self.beginRemoveRows(parentindex, row, row)
        parentitem.delChild(targetitem)
        del self._extension[path]
        self.endRemoveRows()

    def removeItemTree(self, nodeitem):
        self.parseAndRemove(nodeitem)
        row = nodeitem.childRow()
        self.removeItem(nodeitem.parentItem(), nodeitem, row)

    def parseAndRemove(self, parentitem):
        if not parentitem.hasChildren():
            return
        while parentitem.childrenCount() != 0:
            r = parentitem.childrenCount()
            child = parentitem.child(r - 1)
            self.parseAndRemove(child)
            self.removeItem(parentitem, child, r - 1)

    def itemNew(self, fg, nt, md, tg, pi):
        return Q7TreeItem(fg, nt, md, tg, pi)

    def parseAndUpdate(self, parentItem, node, parentIndex, row, parenttag=""):
        self._count += 1
        tag = parenttag + SORTTAG % self._count
        newItem = self.itemNew(self.FG.index, (node), self, tag, parentItem)
        self.FG.depth = max(newItem._depth, self.FG.depth)
        self.FG.nodes = max(newItem._nodes, self.FG.nodes)
        if not parentIndex.isValid():
            parentIndex = self.createIndex(0, 0, self._rootitem)
        self.rowsAboutToBeInserted.emit(parentIndex, row, row)
        self.beginInsertRows(parentIndex, row, row)
        parentItem.addChild(newItem, row)
        self.endInsertRows()
        self.rowsInserted.emit(parentIndex, row, row)
        newIndex = self.createIndex(row, 0, newItem)
        crow = 0
        for childnode in CGU.getNextChildSortByType(node, criteria=self._slist):
            c = self.parseAndUpdate(newItem, childnode, newIndex, crow, tag)
            self.FG.depth = max(c._depth, self.FG.depth)
            self.FG.nodes = max(c._nodes, self.FG.nodes)
            crow += 1
        self.FG.refreshScreen()
        return newItem

    def refreshModel(self, nodeidx):
        self.dataChanged.emit(QModelIndex(), QModelIndex())
        if not nodeidx.isValid():
            return
        row = nodeidx.row()
        dlt = 2
        parentidx = nodeidx.parent()
        maxrow = self.rowCount(parentidx)
        row1 = min(0, abs(row - dlt))
        row2 = min(row + dlt, maxrow)
        ix1 = self.createIndex(row1, 0, parentidx.internalPointer())
        ix2 = self.createIndex(row2, COLUMN_LAST, parentidx.internalPointer())
        if ix1.isValid() and ix2.isValid():
            self.dataChanged.emit(ix1, ix2)
        self.layoutChanged.emit()

    def newNodeChild(self, nodeitem):
        if nodeitem is None:
            return
        nix = self.indexByPath(nodeitem.sidsPath())
        (ntree, npath, nrow) = nodeitem.sidsAddChild(None)
        self.parseAndUpdate(nodeitem, ntree, nix, nrow, nodeitem._tag)
        nix = self.indexByPath(nodeitem.sidsPath())
        pix = self.indexByPath(CGU.getPathAncestor(npath))
        cix = self.indexByPath(npath)
        self.refreshModel(pix)
        self.refreshModel(nix)
        self.refreshModel(cix)
        self.FG.addTreeStatus(Q7FingerPrint.STATUS_MODIFIED)

    def newNodeBrother(self, nodeitem):
        if nodeitem is None:
            return
        nix = self.indexByPath(nodeitem.sidsPath())
        self.newNodeChild(nix.parent().internalPointer())
        self.FG.addTreeStatus(Q7FingerPrint.STATUS_MODIFIED)

    def copyNodeRaw(self, node):
        if node is None:
            return
        # deep copy here
        self._control.copyPasteBuffer = CGU.nodeCopy(node)
        self._control.clearOtherSelections()

    def copyNode(self, nodeitem):
        if nodeitem is None:
            return
        # deep copy here
        self._control.copyPasteBuffer = CGU.nodeCopy(nodeitem._itemnode)
        self._control.clearOtherSelections()

    def cutAllSelectedNodes(self):
        for pth in self._selected:
            nodeitem = self.nodeFromPath(pth)
            self.cutNode(nodeitem)
        self._control.clearOtherSelections()

    def cutNode(self, nodeitem):
        if nodeitem is None:
            return False
        if nodeitem.sidsIsLink() or nodeitem.sidsIsLinkChild():
            return False
        self._control.copyPasteBuffer = CGU.nodeCopy(nodeitem._itemnode)
        parentitem = nodeitem.parentItem()
        path = CGU.getPathAncestor(nodeitem.sidsPath())
        self.removeItemTree(nodeitem)
        pix = self.indexByPath(path)
        parentitem.sidsRemoveChild(self._control.copyPasteBuffer)
        self.refreshModel(pix)
        self.FG.addTreeStatus(Q7FingerPrint.STATUS_MODIFIED)
        self._control.clearOtherSelections()
        return True

    def deleteNode(self, nodeitem):
        if nodeitem is None:
            return False
        if nodeitem.sidsIsLink() or nodeitem.sidsIsLinkChild():
            return False
        _tmpCopy = CGU.nodeCopy(nodeitem._itemnode)
        parentitem = nodeitem.parentItem()
        path = CGU.getPathAncestor(nodeitem.sidsPath())
        self.removeItemTree(nodeitem)
        pix = self.indexByPath(path)
        parentitem.sidsRemoveChild(_tmpCopy)
        self.refreshModel(pix)
        self.FG.addTreeStatus(Q7FingerPrint.STATUS_MODIFIED)
        self._control.clearOtherSelections()
        return True

    def pasteAsChildAllSelectedNodes(self):
        for pth in self._selected:
            nodeitem = self.nodeFromPath(pth)
            self.pasteAsChild(nodeitem)

    def pasteAsChild(self, nodeitem):
        if nodeitem is None:
            return False
        if self._control.copyPasteBuffer is None:
            return False
        if nodeitem.sidsIsLink() or nodeitem.sidsIsLinkChild():
            return
        # Dirty patch
        # Prevent copying inside subtree based on node name
        if self._control.copyPasteBuffer[0] in nodeitem.sidsPath():
            # TODO
            # Would be nice to compare origin path of PasteBuffer
            # either way we should check path of nodes with same name
            # to ensure they are different !
            return False
        nix = self.indexByPath(nodeitem.sidsPath())
        (ntree, npath, nrow) = nodeitem.sidsAddChild(self._control.copyPasteBuffer)
        self.parseAndUpdate(nodeitem, ntree, nix, nrow, nodeitem._tag)
        nix = self.indexByPath(nodeitem.sidsPath())
        pix = self.indexByPath('/CGNSTree' + CGU.getPathAncestor(npath))
        cix = self.indexByPath(npath)
        self.refreshModel(pix)
        self.refreshModel(nix)
        self.refreshModel(cix)
        self.FG.addTreeStatus(Q7FingerPrint.STATUS_MODIFIED)
        return True

    def pasteAsBrotherAllSelectedNodes(self):
        for pth in self._selected:
            nodeitem = self.nodeFromPath(pth)
            self.pasteAsBrother(nodeitem)

    def pasteAsBrother(self, nodeitem):
        if nodeitem is None:
            return False
        if nodeitem.sidsIsLink() or nodeitem.sidsIsLinkChild():
            return
        nix = self.indexByPath(nodeitem.sidsPath())
        self.pasteAsChild(nix.parent().internalPointer())
        self.FG.addTreeStatus(Q7FingerPrint.STATUS_MODIFIED)
        return True

    def dataLoadSelected(self, single=None):
        udict = {}
        if single is None:
            pthlist = self._selected
        else:
            pthlist = [single]
        for pth in pthlist:
            nodeitem = self.nodeFromPath(pth)
            udict[CGU.getPathNoRoot(nodeitem.sidsPath())] = nodeitem.sidsValue()
        if udict:
            (t, l, p) = self.FG.updateNodeData(udict)
        for pth in pthlist:
            nodeitem = self.nodeFromPath(pth)
            nodeitem.dataLoad(t)

    def dataReleaseSelected(self, single=None):
        if single is None:
            pthlist = self._selected
        else:
            pthlist = [single]
        for pth in pthlist:
            nodeitem = self.nodeFromPath(pth)
            nodeitem.dataRelease()

    def checkTree(self, T, pathlist):
        self.FG.pushGrammarPaths()
        modset = set()
        for tag in self.FG.nextGrammarTag():
            modset.add(findgrammar.importUserGrammars(tag))
        self.FG.popGrammarPaths()
        for mod in modset:
            if mod is None:
                checkdiag = CGV.CGNS_VAL_USER_Checks(None)
            else:
                checkdiag = mod.CGNS_VAL_USER_Checks(None)
            checkdiag.checkTree(T, False)
        if pathlist == []:
            pathlist = list(self._extension)
        for path in pathlist:
            pth = CGU.getPathNoRoot(path)
            if pth in checkdiag.log:
                item = self._extension[path]
                stat = checkdiag.log.getWorstDiag(pth)
                if stat == CGM.CHECK_NONE:
                    item.setCheck(STCHKUNKN)
                if stat == CGM.CHECK_GOOD:
                    item.setCheck(STCHKGOOD)
                if stat == CGM.CHECK_FAIL:
                    item.setCheck(STCHKFAIL)
                    s = ""
                    for e in checkdiag.log[pth]:
                        s += checkdiag.log.message(e) + '\n'
                    item.setDiag(s[:-1])
                if stat == CGM.CHECK_WARN:
                    item.setCheck(STCHKWARN)
                    s = ""
                    for e in checkdiag.log[pth]:
                        s += checkdiag.log.message(e) + '\n'
                    item.setDiag(s[:-1])
                if stat == CGM.CHECK_USER:
                    item.setCheck(STCHKUSER)
                    s = ""
                    for e in checkdiag.log[pth]:
                        s += checkdiag.log.message(e) + '\n'
                    item.setDiag(s[:-1])
        return checkdiag.log

    def hasUserColor(self, k):
        cl = OCTXT.UserColors
        try:
            c = k[-3]
            if cl[int(c)] is None:
                return False
        except:
            return False
        return True

    def getUserColor(self, k):
        cb = Qt.black
        cl = OCTXT.UserColors
        c = int(k[-3])
        return QColor(cl[c])

    def doRelease(self):
        self._rootitem.doRelease()
        # self._rootitem=None
        self._control = None
        self._selected = None
        self.extension = None
        Q7TreeItem._icons = {}
        # self.reset()
        self.beginResetModel()
        self.endResetModel()

# -----------------------------------------------------------------
