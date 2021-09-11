#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

from CGNS.NAV.moption import Q7OptionContext as OCTXT

import sys
import string
import re

from qtpy.QtWidgets import QTextEdit, QDialog
from qtpy.QtGui import QTextCursor, QSyntaxHighlighter, QTextCharFormat, QFont, QColor


from CGNS.NAV.Q7MessageWindow import Ui_Q7MessageWindow
from CGNS.NAV.Q7LogWindow import Ui_Q7LogWindow

(INFO, QUESTION, ERROR, WARNING) = (0, 1, 2, 3)

# globals
OK = True
CANCEL = False
ANSWER = True


# -----------------------------------------------------------------
class Q7DocEditor(QTextEdit):
    def __init__(self, parent=None):
        QTextEdit.__init__(self, parent)

    def initText(self, text):
        self.clear()
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.Start, QTextCursor.MoveAnchor)
        self.ensureCursorVisible()


# -----------------------------------------------------------------
class Q7PythonEditor(QTextEdit):
    def __init__(self, parent=None):
        QTextEdit.__init__(self, parent)
        self.highlighter = Q7PythonEditorHighlighter(self.document())
        fsz = 12
        ffm = "Courier"
        self.setStyleSheet('font: %dpt "%s";' % (fsz, ffm))

    def initText(self, text):
        self.clear()
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.Start, QTextCursor.MoveAnchor)
        self.ensureCursorVisible()


# -----------------------------------------------------------------
class Q7PythonEditorHighlighter(QSyntaxHighlighter):
    def __init__(self, *args):
        keywords = r"\bimport\b|\bas\b|\bfor\b|\bdef\b|\bpass\b"
        keywords += r"|\bclass\b|\bfrom\b|\bif\b|\bthen\b|\belse\b"
        keywords += r"|\belif\b|\btry\b|\bexcept\b|\bfinally\b|\braise\b"
        keywords += r"|\bprint\b|\bin\b|\bnot\b|\band\b|\bor\b|\bcontinue\b"
        keywords += r"|\bwhile\b|\breturn\b|\blambda\b|\bwith\b|\del\b"
        keywords += r"|\bglobal\b|\byield\b|\bexec\b|\bassert\b|\break\b"
        constants = r"\bNone\b|\bTrue\b|\bFalse\b|\bself\b"
        autovars = r"\bNODE\b|\bNAME\b|\bVALUE\b|\bSIDSTYPE\b|\bCHILDREN\b"
        autovars += r"|\bTREE\b|\bPATH\b|\bRESULT\b|\bARGS\b|\bPARENT\b"
        autovars += r"|\bLINKS\b|\bSKIPS\b"
        numbers = r"[-+]?\d+" + "|[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
        comment = r"\#.*$"
        textstring = r'"[^"]*?"|\'[^\']*?\''
        self.f_keywords = QTextCharFormat()
        self.f_keywords.setFontWeight(QFont.Bold)
        self.f_keywords.setForeground(QColor("Blue"))
        self.f_comment = QTextCharFormat()
        self.f_comment.setFontWeight(QFont.Light)
        self.f_comment.setForeground(QColor("Red"))
        self.f_constants = QTextCharFormat()
        self.f_constants.setFontWeight(QFont.Light)
        self.f_constants.setForeground(QColor("Navy"))
        self.f_autovars = QTextCharFormat()
        self.f_autovars.setFontWeight(QFont.Bold)
        self.f_autovars.setForeground(QColor("Green"))
        self.f_textstring = QTextCharFormat()
        self.f_textstring.setFontWeight(QFont.Bold)
        self.f_textstring.setForeground(QColor("Coral"))
        self.f_numbers = QTextCharFormat()
        self.f_numbers.setFontWeight(QFont.Light)
        self.f_numbers.setForeground(QColor("Gray"))
        self.r_keywords = re.compile(keywords)
        self.r_numbers = re.compile(numbers)
        self.r_comment = re.compile(comment)
        self.r_constants = re.compile(constants)
        self.r_autovars = re.compile(autovars)
        self.r_textstring = re.compile(textstring)
        self.in_comment = False
        self.in_string = False
        self.r_syntax = []
        self.r_syntax.append((self.r_keywords, self.f_keywords))
        self.r_syntax.append((self.r_numbers, self.f_numbers))
        self.r_syntax.append((self.r_autovars, self.f_autovars))
        self.r_syntax.append((self.r_constants, self.f_constants))
        self.r_syntax.append((self.r_comment, self.f_comment))
        self.r_syntax.append((self.r_textstring, self.f_textstring))
        QSyntaxHighlighter.__init__(self, *args)

    def highlightBlock(self, textblock):
        text = str(textblock)
        fdef = self.format(0)
        fmax = len(text)
        for (rex, fmt) in self.r_syntax:
            mit = rex.finditer(text)
            for m in mit:
                self.setFormat(m.span()[0], m.span()[1], fmt)
                self.setFormat(m.span()[1], fmax, fdef)


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
