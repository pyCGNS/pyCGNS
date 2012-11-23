#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

from PySide.QtCore    import *
from PySide.QtGui     import *

RAISEVIEW='@@RAISEVIEW@@'
CLOSEVIEW='@@CLOSEVIEW@@'
INFO     ='@@INFOVIEW@@'

KEYMAPPING={
 RAISEVIEW:     Qt.Key_Space,
 CLOSEVIEW:     Qt.Key_Delete,
 INFO     :     Qt.Key_Enter,
}

# -----------------------------------------------------------------
class Q7ControlTableWidget(QTableWidget):
    def mousePressEvent(self,event):
        self.lastPos=event.globalPos()
        self.lastButton=event.button()
        QTableWidget.mousePressEvent(self,event)
    def keyPressEvent(self,event):
        kmod=event.modifiers()
        kval=event.key()
        if (kval==KEYMAPPING[RAISEVIEW]):
            self.control.raiseView()
            return
        if (kval==KEYMAPPING[CLOSEVIEW]):
            self.control.closeView()
            return
        if (kval==KEYMAPPING[INFO]):
            self.control.info()
            return
        QTableWidget.keyPressEvent(self,event)
       
# -----------------------------------------------------------------
