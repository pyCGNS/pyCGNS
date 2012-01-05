#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
import numpy

from PySide.QtCore    import *
from PySide.QtGui     import *

import CGNS.PAT.cgnsutils as CT

# -----------------------------------------------------------------
class Q7TableView(QTableView):
    def  __init__(self,parent):
        QTableView.__init__(self,None)
        self._parent=None
    def mousePressEvent(self,event):
       self.lastPos=event.globalPos()
       self.lastButton=event.button()
       QTableView.mousePressEvent(self,event)
       
# -----------------------------------------------------------------
class Q7TableModel(QAbstractTableModel):  
    def __init__(self,node,showparams,parent=None):  
        super(Q7TableModel, self).__init__(parent)
        self.node=node
        self.showmode=showparams['mode']
        self.cs=showparams['cs']
        self.ls=showparams['ls']
        self.fmt="%-s"
        if (self.node.sidsDataType() in ['R4','R8']): self.fmt="% 0.12e"
        if (self.node.sidsDataType() in ['I4','i8']): self.fmt="%12d"
        self.flatarray=self.node.sidsValue().flat
    def columnCount(self, parent):
        return self.cs
    def rowCount(self, parent):
        return self.ls
    def index(self, row, column, parent):
        return self.createIndex(row, column, 0)  
    def data(self, index, role):
        if (role!=Qt.DisplayRole): return None
        return self.fmt%self.flatarray[index.row()*self.cs+index.column()]
    def flags(self, index):  
        if (not index.isValid()):  return Qt.NoItemFlags  
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable  
    #def headerData(self, section, orientation, role):
    #    if (role!=Qt.DisplayRole): return None
    #    return section
    #def parent(self,child):
    #    return self.createIndex(-1,-1,0) 

# -----------------------------------------------------------------
