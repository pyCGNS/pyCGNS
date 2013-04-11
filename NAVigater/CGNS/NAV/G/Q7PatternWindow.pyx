# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7PatternWindow.ui'
#
# Created: Tue Apr  9 11:00:20 2013
#      by: pyside-uic 0.2.13 running on PySide 1.0.9
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Q7PatternWindow(object):
    def setupUi(self, Q7PatternWindow):
        Q7PatternWindow.setObjectName("Q7PatternWindow")
        Q7PatternWindow.resize(715, 350)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7PatternWindow.sizePolicy().hasHeightForWidth())
        Q7PatternWindow.setSizePolicy(sizePolicy)
        Q7PatternWindow.setMinimumSize(QtCore.QSize(715, 350))
        Q7PatternWindow.setMaximumSize(QtCore.QSize(1200, 350))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/cgSpy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7PatternWindow.setWindowIcon(icon)
        self.gridLayout = QtGui.QGridLayout(Q7PatternWindow)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bInfo = QtGui.QPushButton(Q7PatternWindow)
        self.bInfo.setMinimumSize(QtCore.QSize(25, 25))
        self.bInfo.setMaximumSize(QtCore.QSize(25, 25))
        self.bInfo.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icons/help-view.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bInfo.setIcon(icon1)
        self.bInfo.setObjectName("bInfo")
        self.horizontalLayout_2.addWidget(self.bInfo)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bClose = QtGui.QPushButton(Q7PatternWindow)
        self.bClose.setObjectName("bClose")
        self.horizontalLayout_2.addWidget(self.bClose)
        self.gridLayout.addLayout(self.horizontalLayout_2, 5, 0, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.bAdd = QtGui.QPushButton(Q7PatternWindow)
        self.bAdd.setMinimumSize(QtCore.QSize(25, 25))
        self.bAdd.setMaximumSize(QtCore.QSize(25, 25))
        self.bAdd.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/icons/pattern-open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bAdd.setIcon(icon2)
        self.bAdd.setObjectName("bAdd")
        self.horizontalLayout.addWidget(self.bAdd)
        self.bDelete = QtGui.QPushButton(Q7PatternWindow)
        self.bDelete.setMinimumSize(QtCore.QSize(25, 25))
        self.bDelete.setMaximumSize(QtCore.QSize(25, 25))
        self.bDelete.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/icons/pattern-close.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bDelete.setIcon(icon3)
        self.bDelete.setObjectName("bDelete")
        self.horizontalLayout.addWidget(self.bDelete)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.bCopy = QtGui.QPushButton(Q7PatternWindow)
        self.bCopy.setMinimumSize(QtCore.QSize(25, 25))
        self.bCopy.setMaximumSize(QtCore.QSize(25, 25))
        self.bCopy.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/images/icons/mark-node.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bCopy.setIcon(icon4)
        self.bCopy.setObjectName("bCopy")
        self.horizontalLayout.addWidget(self.bCopy)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.bSave = QtGui.QPushButton(Q7PatternWindow)
        self.bSave.setMinimumSize(QtCore.QSize(25, 25))
        self.bSave.setMaximumSize(QtCore.QSize(25, 25))
        self.bSave.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/images/icons/pattern-save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bSave.setIcon(icon5)
        self.bSave.setObjectName("bSave")
        self.horizontalLayout.addWidget(self.bSave)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.patternTable = Q7PatternTableWidget(Q7PatternWindow)
        self.patternTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.patternTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.patternTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.patternTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.patternTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.patternTable.setColumnCount(4)
        self.patternTable.setObjectName("patternTable")
        self.patternTable.setColumnCount(4)
        self.patternTable.setRowCount(0)
        self.patternTable.horizontalHeader().setStretchLastSection(True)
        self.verticalLayout.addWidget(self.patternTable)
        self.gridLayout.addLayout(self.verticalLayout, 4, 0, 1, 1)

        self.retranslateUi(Q7PatternWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7PatternWindow)

    def retranslateUi(self, Q7PatternWindow):
        Q7PatternWindow.setWindowTitle(QtGui.QApplication.translate("Q7PatternWindow", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.bClose.setText(QtGui.QApplication.translate("Q7PatternWindow", "Close", None, QtGui.QApplication.UnicodeUTF8))
        self.patternTable.setSortingEnabled(True)

from CGNS.NAV.mpattern import Q7PatternTableWidget
import Res_rc