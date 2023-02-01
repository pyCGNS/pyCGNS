#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
from .moption import Q7OptionContext as OCTXT

from qtpy.QtCore import Qt, QEvent
from qtpy.QtWidgets import QComboBox, QListWidget


# -----------------------------------------------------------------
class Q7ComboBox(QComboBox):
    def __init__(self, arg):
        QComboBox.__init__(self, arg)
        self.actorlist = QListWidget()
        self.setModel(self.actorlist.model())
        self.setView(self.actorlist)
        self.view().installEventFilter(self)
        self.parent = None

    def setParent(self, parent):
        self.parent = parent

    def eventFilter(self, o, e):
        if e.type() == QEvent.KeyPress:
            # kmod = e.modifiers()
            kval = e.key()
            if kval in [Qt.Key_Z]:
                path = self.actorlist.currentItem().text()
                actor = self.parent.findPathObject(path)
                self.parent.changeCurrentActor([path, actor])
                return True
            if kval in [Qt.Key_H]:
                path = self.actorlist.currentItem().text()
                actor = self.parent.findPathObject(path)
                self.parent.changeCurrentActor([path, actor])
                self.parent.hideActor(None)
                return True
        return QComboBox.eventFilter(self, o, e)

    def keyPressEvent(self, event):
        kmod = event.modifiers()
        kval = event.key()
        print(kmod, kval)


# --- last line
