# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7DiagWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Q7DiagWindow(object):
    def setupUi(self, Q7DiagWindow):
        Q7DiagWindow.setObjectName("Q7DiagWindow")
        Q7DiagWindow.resize(715, 375)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7DiagWindow.sizePolicy().hasHeightForWidth())
        Q7DiagWindow.setSizePolicy(sizePolicy)
        Q7DiagWindow.setMinimumSize(QtCore.QSize(715, 350))
        Q7DiagWindow.setMaximumSize(QtCore.QSize(1200, 900))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/cgSpy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7DiagWindow.setWindowIcon(icon)
        self.gridLayout = QtWidgets.QGridLayout(Q7DiagWindow)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bBackControl = QtWidgets.QPushButton(Q7DiagWindow)
        self.bBackControl.setMinimumSize(QtCore.QSize(25, 25))
        self.bBackControl.setMaximumSize(QtCore.QSize(25, 25))
        self.bBackControl.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icons/top.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bBackControl.setIcon(icon1)
        self.bBackControl.setObjectName("bBackControl")
        self.horizontalLayout_2.addWidget(self.bBackControl)
        self.bInfo = QtWidgets.QPushButton(Q7DiagWindow)
        self.bInfo.setMinimumSize(QtCore.QSize(25, 25))
        self.bInfo.setMaximumSize(QtCore.QSize(25, 25))
        self.bInfo.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/icons/help-view.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bInfo.setIcon(icon2)
        self.bInfo.setObjectName("bInfo")
        self.horizontalLayout_2.addWidget(self.bInfo)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bClose = QtWidgets.QPushButton(Q7DiagWindow)
        self.bClose.setObjectName("bClose")
        self.horizontalLayout_2.addWidget(self.bClose)
        self.gridLayout.addLayout(self.horizontalLayout_2, 5, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.bExpandAll = QtWidgets.QPushButton(Q7DiagWindow)
        self.bExpandAll.setMinimumSize(QtCore.QSize(25, 25))
        self.bExpandAll.setMaximumSize(QtCore.QSize(25, 25))
        self.bExpandAll.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/icons/level-in.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bExpandAll.setIcon(icon3)
        self.bExpandAll.setObjectName("bExpandAll")
        self.horizontalLayout.addWidget(self.bExpandAll)
        self.bCollapseAll = QtWidgets.QPushButton(Q7DiagWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(25)
        sizePolicy.setVerticalStretch(25)
        sizePolicy.setHeightForWidth(self.bCollapseAll.sizePolicy().hasHeightForWidth())
        self.bCollapseAll.setSizePolicy(sizePolicy)
        self.bCollapseAll.setMinimumSize(QtCore.QSize(25, 25))
        self.bCollapseAll.setMaximumSize(QtCore.QSize(25, 25))
        self.bCollapseAll.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/images/icons/level-out.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bCollapseAll.setIcon(icon4)
        self.bCollapseAll.setObjectName("bCollapseAll")
        self.horizontalLayout.addWidget(self.bCollapseAll)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.bPrevious = QtWidgets.QPushButton(Q7DiagWindow)
        self.bPrevious.setMinimumSize(QtCore.QSize(25, 25))
        self.bPrevious.setMaximumSize(QtCore.QSize(25, 25))
        self.bPrevious.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/images/icons/node-sids-opened.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bPrevious.setIcon(icon5)
        self.bPrevious.setObjectName("bPrevious")
        self.horizontalLayout.addWidget(self.bPrevious)
        self.eCount = QtWidgets.QLineEdit(Q7DiagWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.eCount.sizePolicy().hasHeightForWidth())
        self.eCount.setSizePolicy(sizePolicy)
        self.eCount.setMaximumSize(QtCore.QSize(30, 16777215))
        self.eCount.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.eCount.setReadOnly(True)
        self.eCount.setObjectName("eCount")
        self.horizontalLayout.addWidget(self.eCount)
        self.bNext = QtWidgets.QPushButton(Q7DiagWindow)
        self.bNext.setMinimumSize(QtCore.QSize(25, 25))
        self.bNext.setMaximumSize(QtCore.QSize(25, 25))
        self.bNext.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/images/icons/selected.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bNext.setIcon(icon6)
        self.bNext.setObjectName("bNext")
        self.horizontalLayout.addWidget(self.bNext)
        self.cFilter = QtWidgets.QComboBox(Q7DiagWindow)
        self.cFilter.setObjectName("cFilter")
        self.horizontalLayout.addWidget(self.cFilter)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.bWhich = QtWidgets.QPushButton(Q7DiagWindow)
        self.bWhich.setMinimumSize(QtCore.QSize(25, 25))
        self.bWhich.setMaximumSize(QtCore.QSize(25, 25))
        self.bWhich.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/images/icons/check-grammars.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bWhich.setIcon(icon7)
        self.bWhich.setIconSize(QtCore.QSize(24, 24))
        self.bWhich.setObjectName("bWhich")
        self.horizontalLayout.addWidget(self.bWhich)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.cWarnings = QtWidgets.QCheckBox(Q7DiagWindow)
        self.cWarnings.setEnabled(True)
        self.cWarnings.setChecked(True)
        self.cWarnings.setObjectName("cWarnings")
        self.horizontalLayout.addWidget(self.cWarnings)
        self.cDiagFirst = QtWidgets.QCheckBox(Q7DiagWindow)
        self.cDiagFirst.setEnabled(True)
        self.cDiagFirst.setObjectName("cDiagFirst")
        self.horizontalLayout.addWidget(self.cDiagFirst)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.bSave = QtWidgets.QPushButton(Q7DiagWindow)
        self.bSave.setMinimumSize(QtCore.QSize(25, 25))
        self.bSave.setMaximumSize(QtCore.QSize(25, 25))
        self.bSave.setText("")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/images/icons/check-save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bSave.setIcon(icon8)
        self.bSave.setObjectName("bSave")
        self.horizontalLayout.addWidget(self.bSave)
        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.diagTable = QtWidgets.QTreeWidget(Q7DiagWindow)
        self.diagTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.diagTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.diagTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.diagTable.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.diagTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.diagTable.setObjectName("diagTable")
        self.diagTable.headerItem().setText(0, "1")
        self.verticalLayout.addWidget(self.diagTable)
        self.gridLayout.addLayout(self.verticalLayout, 4, 0, 1, 1)

        self.retranslateUi(Q7DiagWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7DiagWindow)

    def retranslateUi(self, Q7DiagWindow):
        _translate = QtCore.QCoreApplication.translate
        Q7DiagWindow.setWindowTitle(_translate("Q7DiagWindow", "Form"))
        self.bClose.setText(_translate("Q7DiagWindow", "Close"))
        self.cWarnings.setText(_translate("Q7DiagWindow", "Warnings"))
        self.cDiagFirst.setText(_translate("Q7DiagWindow", "Diagnostics first"))
from . import Res_rc
