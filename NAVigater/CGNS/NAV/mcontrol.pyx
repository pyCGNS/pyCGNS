#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

from PySide.QtCore    import *
from PySide.QtGui     import *

# -----------------------------------------------------------------
class Q7ControlTableWidget(QTableWidget):
    def mousePressEvent(self,event):
        self.lastPos=event.globalPos()
        self.lastButton=event.button()
        QTableWidget.mousePressEvent(self,event)
    def keyPressEvent(self,event):
        QTableWidget.keyPressEvent(self,event)
       
# -----------------------------------------------------------------
