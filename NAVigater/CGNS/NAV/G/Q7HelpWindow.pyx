# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7HelpWindow.ui'
#
# Created: Wed Apr 24 10:21:26 2013
#      by: pyside-uic 0.2.13 running on PySide 1.0.9
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Q7HelpWindow(object):
    def setupUi(self, Q7HelpWindow):
        Q7HelpWindow.setObjectName("Q7HelpWindow")
        Q7HelpWindow.resize(715, 350)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7HelpWindow.sizePolicy().hasHeightForWidth())
        Q7HelpWindow.setSizePolicy(sizePolicy)
        Q7HelpWindow.setMinimumSize(QtCore.QSize(715, 350))
        Q7HelpWindow.setMaximumSize(QtCore.QSize(1200, 350))
        self.gridLayout = QtGui.QGridLayout(Q7HelpWindow)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.eHelp = QtGui.QTextEdit(Q7HelpWindow)
        self.eHelp.setObjectName("eHelp")
        self.verticalLayout.addWidget(self.eHelp)
        self.gridLayout.addLayout(self.verticalLayout, 3, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bBackControl = QtGui.QPushButton(Q7HelpWindow)
        self.bBackControl.setEnabled(False)
        self.bBackControl.setMinimumSize(QtCore.QSize(25, 25))
        self.bBackControl.setMaximumSize(QtCore.QSize(25, 25))
        self.bBackControl.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/top.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bBackControl.setIcon(icon)
        self.bBackControl.setObjectName("bBackControl")
        self.horizontalLayout_2.addWidget(self.bBackControl)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_2, 4, 0, 1, 1)

        self.retranslateUi(Q7HelpWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7HelpWindow)

    def retranslateUi(self, Q7HelpWindow):
        Q7HelpWindow.setWindowTitle(QtGui.QApplication.translate("Q7HelpWindow", "Form", None, QtGui.QApplication.UnicodeUTF8))

import Res_rc
