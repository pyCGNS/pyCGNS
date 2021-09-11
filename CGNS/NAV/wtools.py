#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QButtonGroup
from qtpy.QtGui import QColor, QPalette

from CGNS.NAV.wfingerprint import Q7FingerPrint
from CGNS.NAV.Q7ToolsWindow import Ui_Q7ToolsWindow
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.wquery import Q7Query, Q7SelectionList
from CGNS.NAV.moption import Q7OptionContext as OCTXT
from CGNS.NAV.diff import diffAB
from CGNS.NAV.merge import mergeAB
from CGNS.NAV.mmergetreeview import Q7TreeMergeModel, TAG_FRONT, TAG_BACK
from CGNS.NAV.wtree import Q7Tree
from CGNS.NAV.wdifftreeview import Q7Diff
from CGNS.NAV.wmergetreeview import Q7Merge
from CGNS.NAV.mquery import Q7QueryEntry

import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnskeywords as CGK


# -----------------------------------------------------------------
class Q7Criteria(object):
    def __init__(self):
        self.name = ""
        self.sidstype = ""
        self.value = ""
        self.shape = ""
        self.rgx_name = False
        self.not_name = False
        self.rgx_sids = False
        self.not_sids = False


# -----------------------------------------------------------------
class Q7ToolsView(Q7Window, Ui_Q7ToolsWindow):
    def __init__(self, parent, fgprint, master):
        Q7Window.__init__(self, Q7Window.VIEW_TOOLS, parent, None, None)
        self.bClose.clicked.connect(self.reject)
        self.bInfo.clicked.connect(self.infoToolsView)
        self.bDiff.clicked.connect(self.diffAB)
        self.bMerge.clicked.connect(self.mergeAB)
        self._master = master
        self._fgprint = fgprint
        self._model = fgprint.model
        self._treeview = master.treeview
        self.cGroup.currentIndexChanged.connect(self.fillqueries)
        self.cQuery.currentTextChanged[str].connect(self.checkquery)
        # QObject.connect(self.cGroup,
        #                SIGNAL("currentIndexChanged(int)"),
        #                self.fillqueries)
        # QObject.connect(self.cQuery,
        #                SIGNAL("valueChanged(int)"),
        #                self.checkquery)
        qlist = Q7Query.queriesNamesList()
        qlist.sort()
        for q in qlist:
            self.cQuery.addItem(q)
        gqlist = Q7Query.queriesGroupsList()
        for gq in gqlist:
            self.cGroup.addItem(gq)
        # ix=-1#self.cQuery.findText()self.querymodel.getCurrentQuery())
        # if (ix!=-1): self.cQuery.setCurrentIndex(ix)
        self.bApply.clicked.connect(self.applyquery)
        pal = self.bApplyBox.palette()
        pal.setColor(QPalette.WindowText, QColor("red"))
        self.bApplyBox.setPalette(pal)
        self.lockable(self.bApply)
        self.bOperateDoc.clicked.connect(self.querydoc)
        self.bRunSearch.clicked.connect(self.runsearch)
        if self._control.query is not None:
            ix = self.cQuery.findText(self._control.query, flags=Qt.MatchStartsWith)
            if ix != -1:
                self.cQuery.setCurrentIndex(ix)
            self.applyquery()
            self.selectionlist()
        self.sLevel.valueChanged[int].connect(self.updateCriteria)
        self.cSIDStype.editTextChanged[str].connect(self.updateSIDStypeEntry)
        # QObject.connect(self.sLevel,
        #                SIGNAL("valueChanged(int)"),
        #                self.updateCriteria)
        # QObject.connect(self.cSIDStype,
        #                SIGNAL("editTextChanged(QString)"),
        #                self.updateSIDStypeEntry)
        self.criteria = []
        for l in range(3):
            self.criteria.append(Q7Criteria())
            self.criteria[-1].name = ""
            self.criteria[-1].sidstype = ""
            self.criteria[-1].value = ""
            self.criteria[-1].shape = ""
            self.criteria[-1].rgx_name = False
            self.criteria[-1].not_name = False
            self.criteria[-1].rgx_sids = False
            self.criteria[-1].not_sids = False
        self.criteria[0].title = "Ancestor criteria"
        self.criteria[1].title = "Current node criteria"
        self.criteria[2].title = "Children criteria"
        self.previousLevel = 1
        self.cSIDStype.addItems([""] + CGK.cgnstypes)
        self.cDataType.addItems([""] + list(CGK.adftypes))
        self.updateCriteria()

    def updateSIDStypeEntry(self):
        print(self.cSIDStype.currentText())

    def updateCriteria(self):
        l = self.sLevel.value()
        p = self.previousLevel
        self.criteria[p].name = self.eName.text()
        self.criteria[p].sidstype = self.cSIDStype.currentText()
        self.criteria[p].rgx_name = self.cRegexpName.isChecked()
        self.criteria[p].not_name = self.cNotName.isChecked()
        self.criteria[p].rgx_sids = self.cRegexpSIDStype.isChecked()
        self.criteria[p].not_sids = self.cNotSIDStype.isChecked()
        self.eName.setText(self.criteria[l].name)
        st = self.criteria[l].sidstype
        self.cSIDStype.setCurrentIndex(self.cSIDStype.findText(st))
        self.cRegexpName.setChecked(self.criteria[l].rgx_name)
        self.cNotName.setChecked(self.criteria[l].not_name)
        self.cRegexpSIDStype.setChecked(self.criteria[l].rgx_sids)
        self.cNotSIDStype.setChecked(self.criteria[l].not_sids)
        self.previousLevel = l
        self.gCriteria.setTitle(self.criteria[l].title)

    def model(self):
        return self._fgprint.model

    def checkquery(self):
        q = self.cQuery.currentText()
        if Q7Query.getQuery(q) is not None and Q7Query.getQuery(q).hasArgs:
            self.eUserVariable.setEnabled(True)
        else:
            self.eUserVariable.setEnabled(False)

    def fillqueries(self):
        g = self.cGroup.currentText()
        self.cQuery.clear()
        for qn in Q7Query.queriesNamesList():
            if (g == "*") or (Q7Query.getQuery(qn).group == g):
                self.cQuery.addItem(qn)

    def querydoc(self):
        q = self.cQuery.currentText()
        if q in Q7Query.queriesNamesList():
            doc = Q7Query.getQuery(q).doc
            self._control.helpWindowDoc(doc)

    def runsearch(self):
        self.updateCriteria()
        s = "n1=True\nn2=True\nn3=True\n"
        s += "t1=True\nt2=True\nt3=True\n"
        if self.criteria[0].name:
            if self.criteria[0].rgx_name:
                s += "n1=(CGU.stringNameMatches(PARENT,'%s'))\n" % self.criteria[0].name
            else:
                s += "n1=(PARENT[0]=='%s')\n" % self.criteria[0].name
            if self.criteria[0].not_name:
                s += "n1=not n1\n"
        if self.criteria[1].name:
            if self.criteria[1].rgx_name:
                s += "n2=(CGU.stringMatches(NAME,'%s'))\n" % self.criteria[1].name
            else:
                s += "n2=(NAME=='%s')\n" % self.criteria[1].name
            if self.criteria[1].not_name:
                s += "n2=not n2\n"
        if self.criteria[2].name:
            if self.criteria[1].rgx_name:
                s += (
                    "for cn in CGU.childrenNames(NODE):\n  n3=n3 or ('%s'==cn[0])\n"
                    % self.criteria[2].name
                )
            else:
                s += "n3=('%s' in CGU.childrenNames(NODE))\n" % self.criteria[2].name
        s += "rn=n1 and n2 and n3\n"
        if self.criteria[0].sidstype:
            if self.criteria[0].rgx_sids:
                s += (
                    "t1=(CGU.stringTypeMatches(PARENT,'%s'))\n"
                    % self.criteria[0].sidstype
                )
            else:
                s += "t1=(PARENT[3]=='%s')\n" % self.criteria[0].sidstype
            if self.criteria[0].not_sids:
                s += "t1=not t1\n"
        if self.criteria[1].sidstype:
            if self.criteria[1].rgx_sids:
                s += (
                    "t2=(CGU.stringTypeMatches(NAME,'%s'))\n"
                    % self.criteria[1].sidstype
                )
            else:
                s += "t2=(SIDSTYPE=='%s')\n" % self.criteria[1].sidstype
            if self.criteria[1].not_sids:
                s += "t2=not t2\n"
        if self.criteria[2].sidstype:
            if self.criteria[1].rgx_sids:
                s += (
                    "t3=False\nfor cn in CHILDREN:\n  t3=t3 or (CGU.stringTypeMatches('%s,cn))\n"
                    % self.criteria[2].sidstype
                )
            else:
                s += (
                    "t3=False\nfor cn in CHILDREN:\n  t3=t3 or ('%s'==cn[3])\n"
                    % self.criteria[2].sidstype
                )
        s += "rt=t1 and t2 and t3\n"
        s += "RESULT=rn and rt\n"
        q = Q7QueryEntry("TMP", script=s)
        skp = list(self._fgprint.lazy)
        sl = q.run(
            self._fgprint.tree,
            self._fgprint.links,
            skp,
            False,
            "",
            self._model.getSelected(),
        )
        self._model.markExtendToList(sl)
        self._model.updateSelected()
        self._treeview.refreshView()

    def applyquery(self):
        q = self.cQuery.currentText()
        v = self.eUserVariable.text()
        if q in ["", " "]:
            self.unmarkall()
            return
        qry = Q7Query
        if q in qry.queriesNamesList():
            sl = qry.getQuery(q).run(
                self._fgprint.tree,
                self._fgprint.links,
                list(self._fgprint.lazy),
                False,
                v,
                self.model().getSelected(),
            )
            self.model().markExtendToList(sl)
            self.model().updateSelected()
            if qry.getQuery(q).requireTreeUpdate():
                self.model().modelReset()
        self._treeview.refreshView()

    def reset(self):
        self.cAncestor.clear()
        self.cVersionA.clear()
        r = self._fgprint.getUniqueTreeViewIdList()
        for i in r:
            self.cAncestor.addItem("%.3d" % i)
            self.cVersionA.addItem("%.3d" % i)
        for i in r:
            self.cTreeA.addItem("%.3d" % i)
            self.cTreeB.addItem("%.3d" % i)
            self.cTreeAncestor.addItem("%.3d" % i)
        self.gForce = QButtonGroup()
        self.gForce.addButton(self.rForceA)
        self.gForce.addButton(self.rForceB)
        self.gForce.addButton(self.rForceNone)
        self.rForceNone.setChecked(True)
        self.gAncestor = QButtonGroup()
        self.gAncestor.addButton(self.rAncestorA)
        self.gAncestor.addButton(self.rAncestorB)
        self.gAncestor.addButton(self.rAncestor)
        self.rAncestor.setChecked(True)
        self.ePrefixA.setText("%sA%s" % (TAG_FRONT, TAG_BACK))
        self.ePrefixB.setText("%sB%s" % (TAG_FRONT, TAG_BACK))

    def infoToolsView(self):
        self._control.helpWindow("Tools")

    def show(self):
        self.reset()
        super(Q7ToolsView, self).show()

    def diffAB(self):
        idx_a = int(self.cAncestor.currentText())
        idx_b = int(self.cVersionA.currentText())
        fpa = self._fgprint.getFingerPrint(idx_a)
        fpb = self._fgprint.getFingerPrint(idx_b)
        diag = {}
        diffAB(fpa.tree, fpb.tree, "", "A", diag, False)
        dw = Q7Diff(self._control, fpa.index, fpb.index, diag)
        dw.show()

    def mergeAB(self):
        idx_a = int(self.cTreeA.currentText())
        idx_b = int(self.cTreeB.currentText())
        fpa = self._fgprint.getFingerPrint(idx_a)
        fpb = self._fgprint.getFingerPrint(idx_b)
        pfx_a = self.ePrefixA.text()
        pfx_b = self.ePrefixB.text()
        tree = CGL.newCGNSTree()
        tc = fpa.control.newtreecount
        fpc = Q7FingerPrint(fpa.control, ".", "new#%.3d.hdf" % tc, tree, [], [])
        Q7TreeMergeModel(fpc)
        self.merge = Q7Tree(fpa.control, "/", fpc)
        fpc._status = [Q7FingerPrint.STATUS_MODIFIED]
        fpa.control.newtreecount += 1
        diag = {}
        diffAB(fpa.tree, fpb.tree, "", "A", diag, False)
        fpc.tree = mergeAB(fpa.tree, fpb.tree, fpc.tree, "C", diag, pfx_a, pfx_b)
        fpc.model.modelReset()
        dw = Q7Merge(self._control, fpc, diag)
        dw.show()
        self.merge.hide()

    def reject(self):
        if self.merge is not None:
            self.merge.show()
            self.merge = None
        if self._master._control._toolswindow is not None:
            self._master._control._toolswindow = None

    def closeEvent(self, event):
        self.reject()
        event.accept()


# -----------------------------------------------------------------
