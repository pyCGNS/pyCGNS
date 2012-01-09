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

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

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
        if (self.node.sidsDataType() in ['I4','I8']): self.fmt="%12d"
        if (self.node.sidsDataType() not in ['MT','LK8']):
            self.flatarray=self.node.sidsValue().flat
        else:
            self.flatarray=None
    def columnCount(self, parent):
        return self.cs
    def rowCount(self, parent):
        return self.ls
    def index(self, row, column, parent):
        return self.createIndex(row, column, 0)  
    def data(self, index, role):
        if (role!=Qt.DisplayRole): return None
        if (self.flatarray is None): return None
        return self.fmt%self.flatarray[index.row()*self.cs+index.column()]
    def flags(self, index):  
        if (not index.isValid()):  return Qt.NoItemFlags  
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable
    def getValue(self,index):
        return self.flatarray[index.row()*self.cs+index.column()]
    def getEnumeratedValueIfPossible(self,index):
        if (self.node.sidsType()==CGK.Elements_ts):
            if ((index.row()==0) and (index.column()==0)):
                ev=CGK.ElementType_l
                et=[CGK.ElementType_ts]
                eti=0
                evi=self.flatarray[index.row()*self.cs+index.column()]
                return (et,ev,eti,evi)
        return None

# -----------------------------------------------------------------
