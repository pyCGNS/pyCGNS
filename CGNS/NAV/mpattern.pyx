#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# 
from CGNS.NAV.moption import Q7OptionContext as OCTXT

from qtpy.QtCore import *
from qtpy.QtWidgets  import *

COPYPATTERN='@@COPYPATTERN@@'

KEYMAPPING={
 COPYPATTERN:   Qt.Key_Space,
}

# -----------------------------------------------------------------
class Q7PatternTableWidget(QTableWidget):
    def mousePressEvent(self,event):
        self.lastPos=event.globalPos()
        self.lastButton=event.button()
        QTableWidget.mousePressEvent(self,event)
    def keyPressEvent(self,event):
        kmod=event.modifiers()
        kval=event.key()
        if (kval==KEYMAPPING[COPYPATTERN]):
            self._panel.copyPatternInBuffer()
            return
        QTableWidget.keyPressEvent(self,event)
       
# -----------------------------------------------------------------
