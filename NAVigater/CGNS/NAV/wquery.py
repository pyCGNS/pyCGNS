#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
from PySide.QtCore  import *
from PySide.QtGui   import *
from CGNS.NAV.Q7QueryWindow import Ui_Q7QueryWindow
from CGNS.NAV.mquery import Q7QueryTableModel,Q7ComboBoxDelegate
from CGNS.NAV.wfingerprint import Q7Window

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

# -----------------------------------------------------------------
class Q7QueryTableItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        t=index.data(Qt.DisplayRole)
        r=option.rect.adjusted(2, 2, -2, -2);
        painter.drawText(r,Qt.AlignVCenter|Qt.AlignLeft,t, r)

# -----------------------------------------------------------------
class Q7Query(Q7Window,Ui_Q7QueryWindow):
    def __init__(self,control,fgprint,treeview):
        Q7Window.__init__(self,Q7Window.VIEW_QUERY,control,'/',fgprint)
        self.querytablemodel=Q7QueryTableModel(self)
        self.querytableview.setModel(self.querytablemodel)
        self.querytableview.setEditTriggers(QAbstractItemView.CurrentChanged)
        self.querytableview.viewport().installEventFilter(self)
        self.querytablemodel.setDelegates(self.querytableview)
        self.bClose.clicked.connect(self.reject)
        self.showQuery('QUADs')
    def reject(self):
        self.close()
    def reset(self):
        for qn in self.querytablemodel._queries:
            self.cQueryName.addItem(qn)
    def showQuery(self,name):
        if (self.querytablemodel._queries.has_key(name)):
            pass
    def show(self):
        self.reset()
        super(Q7Query, self).show()

# -----------------------------------------------------------------
