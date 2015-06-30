#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
import string
from PySide.QtCore  import *
from PySide.QtGui   import *
from CGNS.NAV.wfingerprint import Q7FingerPrint
from CGNS.NAV.Q7ToolsWindow import Ui_Q7ToolsWindow
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.moption import Q7OptionContext  as OCTXT
from CGNS.NAV.diff import diffAB
from CGNS.NAV.merge import mergeAB
from CGNS.NAV.mmergetreeview import Q7TreeMergeModel,TAG_FRONT,TAG_BACK
from CGNS.NAV.wtree import Q7Tree
from CGNS.NAV.wdifftreeview import Q7Diff
from CGNS.NAV.wmergetreeview import Q7Merge

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnslib   as CGL

# a process P is call with:
# import P
# P.run(*ARGS)
PROCESSES=['HybridGenConnect']

# -----------------------------------------------------------------
class Q7ToolsView(Q7Window,Ui_Q7ToolsWindow):
    def __init__(self,parent,fgprint,master):
        Q7Window.__init__(self,Q7Window.VIEW_TOOLS,parent,None,None)
        self.bClose.clicked.connect(self.reject)
        self.bInfo.clicked.connect(self.infoToolsView)
        self.bDiff.clicked.connect(self.diffAB)
        self.bMerge.clicked.connect(self.mergeAB)
        self.bApply.clicked.connect(self.runProcess)
        self._master=master
        self._fgprint=fgprint
    def reset(self):
        self.cAncestor.clear()
        self.cVersionA.clear()
        r=self._fgprint.getUniqueTreeViewIdList()
        for i in r:
            self.cAncestor.addItem('%.3d'%i)
            self.cVersionA.addItem('%.3d'%i)
        for i in r:
            self.cTreeA.addItem('%.3d'%i)
            self.cTreeB.addItem('%.3d'%i)
            self.cTreeAncestor.addItem('%.3d'%i)
        for p in PROCESSES:
            self.cProcess.addItem(p)
        self.gForce=QButtonGroup()
        self.gForce.addButton(self.rForceA)
        self.gForce.addButton(self.rForceB)
        self.gForce.addButton(self.rForceNone)
        self.rForceNone.setChecked(True)
        self.gAncestor=QButtonGroup()
        self.gAncestor.addButton(self.rAncestorA)
        self.gAncestor.addButton(self.rAncestorB)
        self.gAncestor.addButton(self.rAncestor)
        self.rAncestor.setChecked(True)
        self.ePrefixA.setText('%sA%s'%(TAG_FRONT,TAG_BACK))
        self.ePrefixB.setText('%sB%s'%(TAG_FRONT,TAG_BACK))
    def infoToolsView(self):
        self._control.helpWindow('Tools')
    def show(self):
        self.reset()
        super(Q7ToolsView, self).show()
    def diffAB(self):
        idxA=int(self.cAncestor.currentText())
        idxB=int(self.cVersionA.currentText())
        fpa=self._fgprint.getFingerPrint(idxA)
        fpb=self._fgprint.getFingerPrint(idxB)
        diag={}
        diffAB(fpa.tree,fpb.tree,'','A',diag,False)
        dw=Q7Diff(self._control,fpa,fpb,diag)
        dw.show()
    def mergeAB(self):
        idxA=int(self.cTreeA.currentText())
        idxB=int(self.cTreeB.currentText())
        fpa=self._fgprint.getFingerPrint(idxA)
        fpb=self._fgprint.getFingerPrint(idxB)
        pfxA=self.ePrefixA.text()
        pfxB=self.ePrefixB.text()
        tree=CGL.newCGNSTree()
        tc=fpa.control.newtreecount
        fpc=Q7FingerPrint(fpa.control,'.','new#%.3d.hdf'%tc,tree,[],[])
        Q7TreeMergeModel(fpc)
        self.merge=Q7Tree(fpa.control,'/',fpc)
        fpc._status=[Q7FingerPrint.STATUS_MODIFIED]
        fpa.control.newtreecount+=1
        diag={}
        diffAB(fpa.tree,fpb.tree,'','A',diag,False)
        fpc.tree=mergeAB(fpa.tree,fpb.tree,fpc.tree,'C',diag,pfxA,pfxB)
        fpc.model.modelReset()
        dw=Q7Merge(self._control,fpc,diag)
        dw.show()
        self.merge.hide()
    def runProcess(self):
        p=self.cProcess.currentText()
        print 'P:',p
    def reject(self):
        if (self.merge is not None):
            self.merge.show()
            self.merge=None
        if (self._master._control._toolswindow is not None):
            self._master._control._toolswindow=None
    def closeEvent(self, event):
        self.reject()
        event.accept()
         
# -----------------------------------------------------------------
