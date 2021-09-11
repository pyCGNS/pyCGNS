# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7LogWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Q7LogWindow(object):
    def setupUi(self, Q7LogWindow):
        Q7LogWindow.setObjectName("Q7LogWindow")
        Q7LogWindow.resize(600, 400)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7LogWindow.sizePolicy().hasHeightForWidth())
        Q7LogWindow.setSizePolicy(sizePolicy)
        Q7LogWindow.setMinimumSize(QtCore.QSize(500, 140))
        Q7LogWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/cgSpy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7LogWindow.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(Q7LogWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.eLog = Q7PythonEditor(Q7LogWindow)
        self.eLog.setMinimumSize(QtCore.QSize(0, 0))
        self.eLog.setFrameShadow(QtWidgets.QFrame.Raised)
        self.eLog.setLineWidth(0)
        self.eLog.setObjectName("eLog")
        self.verticalLayout.addWidget(self.eLog)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(868, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bClear = QtWidgets.QPushButton(Q7LogWindow)
        self.bClear.setObjectName("bClear")
        self.horizontalLayout_2.addWidget(self.bClear)
        self.bClose = QtWidgets.QPushButton(Q7LogWindow)
        self.bClose.setObjectName("bClose")
        self.horizontalLayout_2.addWidget(self.bClose)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Q7LogWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7LogWindow)

    def retranslateUi(self, Q7LogWindow):
        _translate = QtCore.QCoreApplication.translate
        Q7LogWindow.setWindowTitle(_translate("Q7LogWindow", "Dialog"))
        self.bClear.setText(_translate("Q7LogWindow", "Clear"))
        self.bClose.setText(_translate("Q7LogWindow", "Hide"))
from CGNS.NAV.weditors import Q7PythonEditor
from . import Res_rc
