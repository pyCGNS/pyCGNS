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
from CGNS.NAV.Q7InfoWindow import Ui_Q7InfoWindow
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.moption import Q7OptionContext  as OCTXT

# -----------------------------------------------------------------
class Q7Info(Q7Window,Ui_Q7InfoWindow):
    def __init__(self,parent,data,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_INFO,parent,None,None)
        self.bClose.clicked.connect(self.reject)
        self.bInfo.clicked.connect(self.infoInfoView)
        self._data=data
    def infoInfoView(self):
        self._control.helpWindow('Info')
    def show(self):
        self.reset()
        super(Q7Info, self).show()
    def reset(self):
        for k in dir(self):
            if self._data.has_key(k):
                v=self._data[k]
                if (type(v) is bool):
                    getattr(self,k).setEnabled(True)
                    if (v): getattr(self,k).setCheckState(Qt.Checked)
                    else:   getattr(self,k).setCheckState(Qt.Unchecked)
                    getattr(self,k).setEnabled(False)
                if (type(v) in [str,unicode,int,float]):
                    getattr(self,k).setText(str(v))
                    #getattr(self,k).setFont(OCTXT.FixedFontTable)
    def reject(self):
        self.close()
         
# -----------------------------------------------------------------
