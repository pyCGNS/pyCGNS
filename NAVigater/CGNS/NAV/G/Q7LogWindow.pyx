# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7LogWindow.ui'
#
# Created: Fri Apr 19 13:45:09 2013
#      by: pyside-uic 0.2.13 running on PySide 1.0.9
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Q7LogWindow(object):
    def setupUi(self, Q7LogWindow):
        Q7LogWindow.setObjectName("Q7LogWindow")
        Q7LogWindow.resize(500, 140)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7LogWindow.sizePolicy().hasHeightForWidth())
        Q7LogWindow.setSizePolicy(sizePolicy)
        Q7LogWindow.setMinimumSize(QtCore.QSize(500, 140))
        Q7LogWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/cgSpy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7LogWindow.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(Q7LogWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.eLog = Q7PythonEditor(Q7LogWindow)
        self.eLog.setMinimumSize(QtCore.QSize(0, 0))
        self.eLog.setFrameShadow(QtGui.QFrame.Raised)
        self.eLog.setLineWidth(0)
        self.eLog.setObjectName("eLog")
        self.verticalLayout.addWidget(self.eLog)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bInfo = QtGui.QPushButton(Q7LogWindow)
        self.bInfo.setMinimumSize(QtCore.QSize(25, 25))
        self.bInfo.setMaximumSize(QtCore.QSize(25, 25))
        self.bInfo.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icons/help-view.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bInfo.setIcon(icon1)
        self.bInfo.setObjectName("bInfo")
        self.horizontalLayout_2.addWidget(self.bInfo)
        spacerItem = QtGui.QSpacerItem(868, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bClear = QtGui.QPushButton(Q7LogWindow)
        self.bClear.setObjectName("bClear")
        self.horizontalLayout_2.addWidget(self.bClear)
        self.bClose = QtGui.QPushButton(Q7LogWindow)
        self.bClose.setObjectName("bClose")
        self.horizontalLayout_2.addWidget(self.bClose)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Q7LogWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7LogWindow)

    def retranslateUi(self, Q7LogWindow):
        Q7LogWindow.setWindowTitle(QtGui.QApplication.translate("Q7LogWindow", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.bClear.setText(QtGui.QApplication.translate("Q7LogWindow", "Clear", None, QtGui.QApplication.UnicodeUTF8))
        self.bClose.setText(QtGui.QApplication.translate("Q7LogWindow", "Hide", None, QtGui.QApplication.UnicodeUTF8))

from CGNS.NAV.weditors import Q7PythonEditor
import Res_rc
