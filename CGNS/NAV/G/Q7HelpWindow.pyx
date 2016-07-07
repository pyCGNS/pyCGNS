# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7HelpWindow.ui'
#
# Created: Thu Jul  7 14:45:18 2016
#      by: PyQt4 UI code generator 4.11.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Q7HelpWindow(object):
    def setupUi(self, Q7HelpWindow):
        Q7HelpWindow.setObjectName(_fromUtf8("Q7HelpWindow"))
        Q7HelpWindow.resize(715, 350)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7HelpWindow.sizePolicy().hasHeightForWidth())
        Q7HelpWindow.setSizePolicy(sizePolicy)
        Q7HelpWindow.setMinimumSize(QtCore.QSize(715, 350))
        Q7HelpWindow.setMaximumSize(QtCore.QSize(1200, 350))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/cgSpy.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7HelpWindow.setWindowIcon(icon)
        self.gridLayout = QtGui.QGridLayout(Q7HelpWindow)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.eHelp = QtGui.QTextEdit(Q7HelpWindow)
        self.eHelp.setObjectName(_fromUtf8("eHelp"))
        self.verticalLayout.addWidget(self.eHelp)
        self.gridLayout.addLayout(self.verticalLayout, 3, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.bBackControl = QtGui.QPushButton(Q7HelpWindow)
        self.bBackControl.setEnabled(False)
        self.bBackControl.setMinimumSize(QtCore.QSize(25, 25))
        self.bBackControl.setMaximumSize(QtCore.QSize(25, 25))
        self.bBackControl.setText(_fromUtf8(""))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/top.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bBackControl.setIcon(icon1)
        self.bBackControl.setObjectName(_fromUtf8("bBackControl"))
        self.horizontalLayout_2.addWidget(self.bBackControl)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_2, 4, 0, 1, 1)

        self.retranslateUi(Q7HelpWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7HelpWindow)

    def retranslateUi(self, Q7HelpWindow):
        Q7HelpWindow.setWindowTitle(_translate("Q7HelpWindow", "Form", None))

import Res_rc
