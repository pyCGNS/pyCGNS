#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

from CGNS.NAV.moption import Q7OptionContext as OCTXT

import string

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

from CGNS.NAV.Q7OptionsWindow import Ui_Q7OptionsWindow
from CGNS.NAV.wfingerprint import Q7Window

combonames = []
for tn in ["label", "edit", "table", "button", "rname", "nname"]:
    combonames += [
        "__O_%s_family" % tn,
        "__O_%s_size" % tn,
        "__O_%s_bold" % tn,
        "__O_%s_italic" % tn,
    ]


# -----------------------------------------------------------------
class Q7Option(Q7Window, Ui_Q7OptionsWindow):
    labdict = {
        "Label": ["QLabel", "QTabWidget", "QGroupBox", "QCheckBox", "QRadioButton"],
        "Button": ["QPushButton"],
        "Edit": ["QLineEdit", "QSpinBox", "QComboBox"],
        "RName": ["Q7TreeView"],
        "NName": [],
        "Table": ["Q7TableView"],
    }
    combos = combonames

    def __init__(self, parent):
        Q7Window.__init__(self, Q7Window.VIEW_OPTION, parent, None, None)
        self.bApply.clicked.connect(self.accept)
        self.bInfo.clicked.connect(self.infoOptionView)
        self.bClose.clicked.connect(self.close)
        self.bReset.clicked.connect(self.reset)
        self.bResetIgnored.clicked.connect(self.resetIgnored)
        self.bResetFonts.clicked.connect(self.resetFonts)
        self.getOptions()

    def infoOptionView(self):
        self._control.helpWindow("Option")

    def getopt(self, name):
        if name[0] == "_":
            return None
        try:
            a = getattr(self, "_Ui_Q7OptionsWindow__O_" + name.lower())
        except AttributeError:
            return None
        return a

    def checkDeps(self):
        for dep in OCTXT._depends:
            for chks in OCTXT._depends[dep]:
                if self.getopt(chks) is not None:
                    if not self.getopt(chks).isChecked():
                        self.getopt(dep).setDisabled(True)
                    else:
                        self.getopt(dep).setEnabled(True)
                else:
                    print("CGNS.NAV (debug) NO OPTION :", chks)

    def resetIgnored(self):
        OCTXT.IgnoredMessages = []

    def reset(self):
        self.getOptions()
        data = self._options
        for k in data:
            if (k[0] != "_") and (self.getopt(k) is not None):
                setattr(OCTXT, k, data[k])
                if self.getopt(k).objectName() in Q7Option.combos:
                    pass
                elif isinstance(data[k], bool):
                    if data[k]:
                        self.getopt(k).setCheckState(Qt.Checked)
                    else:
                        self.getopt(k).setCheckState(Qt.Unchecked)
                elif isinstance(data[k], int):
                    self.getopt(k).setValue(data[k])
                elif isinstance(data[k], str):
                    try:
                        self.getopt(k).setText(data[k])
                    except AttributeError:
                        pass  # self.getopt(k).setEditText(data[k])
                elif isinstance(data[k], list):
                    s = ""
                    for l in data[k]:
                        s += "%s\n" % l
                    self.getopt(k).setPlainText(s)
        self.checkDeps()
        self.updateFonts()

    def resetFonts(self):
        for k in OCTXT._Default_Fonts:
            self.setOptionValue(k, OCTXT._Default_Fonts[k])
        self.updateFonts()
        self.accept()

    def show(self):
        self.reset()
        super(Q7Option, self).show()

    def accept(self):
        data = self._options
        for k in data:
            if (k[0] != "_") and (self.getopt(k) is not None):
                if self.getopt(k).objectName() in Q7Option.combos:
                    pass
                elif isinstance(data[k], bool):
                    if self.getopt(k).isChecked():
                        data[k] = True
                    else:
                        data[k] = False
                elif isinstance(data[k], int):
                    v = self.getopt(k).value()
                    if self.validateOption(k, v):
                        data[k] = self.getopt(k).value()
                elif isinstance(data[k], str):
                    v = self.getopt(k).text()
                    if self.validateOption(k, v):
                        data[k] = self.getopt(k).text()
                elif isinstance(data[k], list):
                    s = self.getopt(k).toPlainText()
                    cset = []
                    for l in s.split("\n"):
                        if l and l not in cset:
                            cset.append(l)
                    if self.validateOption(k, cset):
                        data[k] = cset
        self.updateFonts(update=True)
        self.setOptions()
        self.reset()

    def updateFonts(self, update=False):
        data = self._options
        scss = ""
        for kfont in Q7Option.labdict:
            if update:
                fm = self.getopt("%s_Family" % kfont).currentFont().family()
                it = self.getopt("%s_Italic" % kfont).isChecked()
                bd = self.getopt("%s_Bold" % kfont).isChecked()
                sz = int(self.getopt("%s_Size" % kfont).text())
                data["%s_Family" % kfont] = fm
                data["%s_Size" % kfont] = sz
                data["%s_Bold" % kfont] = bd
                data["%s_Italic" % kfont] = it
            fm = self._options["%s_Family" % kfont]
            sz = self._options["%s_Size" % kfont]
            bd = self._options["%s_Bold" % kfont]
            it = self._options["%s_Italic" % kfont]
            if bd:
                wg = QFont.Bold
                self.getopt("%s_Bold" % kfont).setCheckState(Qt.Checked)
            else:
                wg = QFont.Normal
                self.getopt("%s_Bold" % kfont).setCheckState(Qt.Unchecked)
            if it:
                self.getopt("%s_Italic" % kfont).setCheckState(Qt.Checked)
            else:
                self.getopt("%s_Italic" % kfont).setCheckState(Qt.Unchecked)
            self.getopt("%s_Size" % kfont).setValue(sz)
            qf = QFont(fm, italic=it, weight=wg, pointSize=sz)
            self._options["_%s_Font" % kfont] = qf
            self.getopt("%s_Family" % kfont).setCurrentFont(qf)
            for wtype in Q7Option.labdict[kfont]:
                bf = ""
                tf = ""
                if bd:
                    bf = "bold"
                if it:
                    tf = "italic"
                scss += """%s { font:  %s %s %dpx "%s" }\n""" % (wtype, bf, tf, sz, fm)
        self._control._application.setStyleSheet(scss)
        self._options["UserCSS"] = scss


# -----------------------------------------------------------------
