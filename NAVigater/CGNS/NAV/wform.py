#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
from PySide.QtCore  import *
from PySide.QtGui   import *
from CGNS.NAV.Q7FormWindow import Ui_Q7FormWindow

# -----------------------------------------------------------------
class Q7Form(QWidget,Ui_Q7FormWindow):
    def __init__(self,parent,node):
        super(Q7Form, self).__init__(None)
        self.setupUi(self)
        self.setWindowTitle("Node form")
        self.node=node
        for t in self.node.sidsTypeList():
            self.eType.addItem(t)
        self.bApply.clicked.connect(self.accept)
        self.bClose.clicked.connect(self.reject)
        self._parent=parent
    def accept(self):
        pass
    def reject(self):
        self._parent.form=None
        self.close()
    def show(self):
        self.reset()
        super(Q7Form, self).show()
    def reset(self):
        self.eName.setText(self.node.sidsName())
        self.ePath.setText(self.node.sidsPath())
        self.setCurrentType(self.node.sidsType())
        self.eDims.setText(str(self.node.sidsDims()))
    def setCurrentType(self,ntype):
        idx=self.eType.findText(ntype)
        self.eType.setCurrentIndex(idx)
