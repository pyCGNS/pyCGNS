#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

# from CGNS.NAV.moption import Q7OptionContext as OCTXT

import re

from qtpy.QtWidgets import QTextEdit
from qtpy.QtGui import QColor, QSyntaxHighlighter, QTextCursor, QFont, QTextCharFormat


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
        autovars += r"|\bLINKS\b|\bSKIPS\b|\bSELECTED\b"
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


# --- last line
