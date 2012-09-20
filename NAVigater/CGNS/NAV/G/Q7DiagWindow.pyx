# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7DiagWindow.ui'
#
# Created: Thu Sep 20 10:44:07 2012
#      by: pyside-uic 0.2.13 running on PySide 1.0.9
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Q7DiagWindow(object):
    def setupUi(self, Q7DiagWindow):
        Q7DiagWindow.setObjectName("Q7DiagWindow")
        Q7DiagWindow.resize(715, 350)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7DiagWindow.sizePolicy().hasHeightForWidth())
        Q7DiagWindow.setSizePolicy(sizePolicy)
        Q7DiagWindow.setMinimumSize(QtCore.QSize(715, 350))
        Q7DiagWindow.setMaximumSize(QtCore.QSize(1200, 350))
        self.gridLayout = QtGui.QGridLayout(Q7DiagWindow)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bBackControl = QtGui.QPushButton(Q7DiagWindow)
        self.bBackControl.setMinimumSize(QtCore.QSize(25, 25))
        self.bBackControl.setMaximumSize(QtCore.QSize(25, 25))
        self.bBackControl.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/top.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bBackControl.setIcon(icon)
        self.bBackControl.setObjectName("bBackControl")
        self.horizontalLayout_2.addWidget(self.bBackControl)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bClose = QtGui.QPushButton(Q7DiagWindow)
        self.bClose.setObjectName("bClose")
        self.horizontalLayout_2.addWidget(self.bClose)
        self.gridLayout.addLayout(self.horizontalLayout_2, 5, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.bSave = QtGui.QPushButton(Q7DiagWindow)
        self.bSave.setMinimumSize(QtCore.QSize(25, 25))
        self.bSave.setMaximumSize(QtCore.QSize(25, 25))
        self.bSave.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icons/save.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bSave.setIcon(icon1)
        self.bSave.setObjectName("bSave")
        self.horizontalLayout.addWidget(self.bSave)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.diagTable = QtGui.QTableWidget(Q7DiagWindow)
        self.diagTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.diagTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.diagTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.diagTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.diagTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.diagTable.setColumnCount(4)
        self.diagTable.setObjectName("diagTable")
        self.diagTable.setColumnCount(4)
        self.diagTable.setRowCount(0)
        self.verticalLayout.addWidget(self.diagTable)
        self.gridLayout.addLayout(self.verticalLayout, 4, 0, 1, 1)

        self.retranslateUi(Q7DiagWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7DiagWindow)

    def retranslateUi(self, Q7DiagWindow):
        Q7DiagWindow.setWindowTitle(QtGui.QApplication.translate("Q7DiagWindow", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.bClose.setText(QtGui.QApplication.translate("Q7DiagWindow", "Close", None, QtGui.QApplication.UnicodeUTF8))
        self.diagTable.setSortingEnabled(True)

import Res_rc
