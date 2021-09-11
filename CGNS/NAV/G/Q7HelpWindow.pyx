# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7HelpWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Q7HelpWindow(object):
    def setupUi(self, Q7HelpWindow):
        Q7HelpWindow.setObjectName("Q7HelpWindow")
        Q7HelpWindow.resize(715, 350)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7HelpWindow.sizePolicy().hasHeightForWidth())
        Q7HelpWindow.setSizePolicy(sizePolicy)
        Q7HelpWindow.setMinimumSize(QtCore.QSize(715, 350))
        Q7HelpWindow.setMaximumSize(QtCore.QSize(1200, 350))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/cgSpy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7HelpWindow.setWindowIcon(icon)
        self.gridLayout = QtWidgets.QGridLayout(Q7HelpWindow)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.eHelp = QtWidgets.QTextEdit(Q7HelpWindow)
        self.eHelp.setObjectName("eHelp")
        self.verticalLayout.addWidget(self.eHelp)
        self.gridLayout.addLayout(self.verticalLayout, 3, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bBackControl = QtWidgets.QPushButton(Q7HelpWindow)
        self.bBackControl.setEnabled(False)
        self.bBackControl.setMinimumSize(QtCore.QSize(25, 25))
        self.bBackControl.setMaximumSize(QtCore.QSize(25, 25))
        self.bBackControl.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icons/top.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bBackControl.setIcon(icon1)
        self.bBackControl.setObjectName("bBackControl")
        self.horizontalLayout_2.addWidget(self.bBackControl)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_2, 4, 0, 1, 1)

        self.retranslateUi(Q7HelpWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7HelpWindow)

    def retranslateUi(self, Q7HelpWindow):
        _translate = QtCore.QCoreApplication.translate
        Q7HelpWindow.setWindowTitle(_translate("Q7HelpWindow", "Form"))
from . import Res_rc
