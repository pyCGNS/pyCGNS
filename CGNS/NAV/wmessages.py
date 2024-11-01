#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

from .moption import Q7OptionContext as OCTXT

import sys
import string
import re

from qtpy.QtWidgets import QTextEdit, QDialog
from qtpy.QtGui import QTextCursor, QSyntaxHighlighter, QTextCharFormat, QFont, QColor

from .Q7MessageWindow import Ui_Q7MessageWindow
from .Q7LogWindow import Ui_Q7LogWindow

(INFO, QUESTION, ERROR, WARNING) = (0, 1, 2, 3)

# globals
OK = True
CANCEL = False
ANSWER = True


# -----------------------------------------------------------------
class Q7Log(QDialog, Ui_Q7LogWindow):
    def __init__(self):
        QDialog.__init__(self, None)
        self.setupUi(self)
        self.bClose.clicked.connect(self.leave)
        self.bClear.clicked.connect(self.clear)
        self.setWindowTitle("%s: Log" % OCTXT._ToolName)
        self.eLog.setReadOnly(True)
        self.eLog.setAcceptRichText(False)
        self.eLog.setStyleSheet('font: 12pt "Courier";')

    def write(self, msg):
        self.eLog.insertPlainText(msg)

    def clear(self):
        self.eLog.clear()

    def leave(self, *arg):
        self.hide()


# -----------------------------------------------------------------
class Q7MessageBox(QDialog, Ui_Q7MessageWindow):
    def __init__(self, control, code):
        QDialog.__init__(self, None)
        self.setupUi(self)
        self.bOK.clicked.connect(self.runOK)
        self.bCANCEL.clicked.connect(self.runCANCEL)
        self.bInfo.setDisabled(True)
        self._text = ""
        self._control = control
        self._code = 0

    def setMode(self, cancel=False, again=False):
        if not again:
            self.cNotAgain.hide()
        if not cancel:
            self.bCANCEL.hide()

    def setLayout(
        self, text, btype=INFO, cancel=False, again=False, buttons=("OK", "Cancel")
    ):
        self.bOK.setText(buttons[0])
        if len(buttons) > 1:
            self.bCANCEL.setText(buttons[1])
        self.eMessage.setText(str(text))
        self._text = text
        self.eMessage.setReadOnly(True)
        self.setMode(cancel, again)

    def runOK(self, *arg):
        self.addToSkip()
        self.close()
        self.setResult(QDialog.Accepted)

    def runCANCEL(self, *arg):
        self.addToSkip()
        self.close()
        self.setResult(QDialog.Rejected)

    def addToSkip(self):
        if self.cNotAgain.isChecked():
            if self._code and self._code not in OCTXT.IgnoredMessages:
                OCTXT.IgnoredMessages += [self._code]

    def showAndWait(self):
        ret = self.exec_()
        if ret == QDialog.Accepted:
            return True
        else:
            return False


def wError(control, code, info, error):
    txt = """<img source=":/images/icons/user-G.png">  <big>ERROR #%d</big><hr>
         %s<br>%s""" % (
        code,
        error,
        info,
    )
    if code in OCTXT.IgnoredMessages:
        return True
    msg = Q7MessageBox(control, code)
    msg.setWindowTitle("%s: Error" % OCTXT._ToolName)
    msg.setLayout(txt, btype=ERROR, cancel=False, again=True, buttons=("Close",))
    return msg.showAndWait()


def wQuestion(control, code, title, question, again=True, buttons=("OK", "Cancel")):
    txt = """<img source=":/images/icons/user-M.png">
         <b> <big>%s</big></b><hr>%s""" % (
        title,
        question,
    )
    if code in OCTXT.IgnoredMessages:
        return True
    msg = Q7MessageBox(control, code)
    msg.setWindowTitle("%s: Question" % OCTXT._ToolName)
    msg.setLayout(txt, btype=QUESTION, cancel=True, again=again, buttons=buttons)
    return msg.showAndWait()


def wInfo(control, code, title, info, again=True, buttons=("Close",)):
    txt = """<img source=":/images/icons/user-S.png">
         <b> <big>%s</big></b><hr>%s""" % (
        title,
        info,
    )
    if code in OCTXT.IgnoredMessages:
        return True
    msg = Q7MessageBox(control, code)
    msg.setWindowTitle("%s: Info" % OCTXT._ToolName)
    msg.setLayout(txt, btype=INFO, cancel=False, again=again, buttons=buttons)
    return msg.showAndWait()


# --- last line
