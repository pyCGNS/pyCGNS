# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7LogWindow.ui'
#
# Created: Fri Jun 24 13:58:11 2016
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

class Ui_Q7LogWindow(object):
    def setupUi(self, Q7LogWindow):
        Q7LogWindow.setObjectName(_fromUtf8("Q7LogWindow"))
        Q7LogWindow.resize(600, 400)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7LogWindow.sizePolicy().hasHeightForWidth())
        Q7LogWindow.setSizePolicy(sizePolicy)
        Q7LogWindow.setMinimumSize(QtCore.QSize(500, 140))
        Q7LogWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/icons/cgSpy.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7LogWindow.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(Q7LogWindow)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.eLog = Q7PythonEditor(Q7LogWindow)
        self.eLog.setMinimumSize(QtCore.QSize(0, 0))
        self.eLog.setFrameShadow(QtGui.QFrame.Raised)
        self.eLog.setLineWidth(0)
        self.eLog.setObjectName(_fromUtf8("eLog"))
        self.verticalLayout.addWidget(self.eLog)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem = QtGui.QSpacerItem(868, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bClear = QtGui.QPushButton(Q7LogWindow)
        self.bClear.setObjectName(_fromUtf8("bClear"))
        self.horizontalLayout_2.addWidget(self.bClear)
        self.bClose = QtGui.QPushButton(Q7LogWindow)
        self.bClose.setObjectName(_fromUtf8("bClose"))
        self.horizontalLayout_2.addWidget(self.bClose)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Q7LogWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7LogWindow)

    def retranslateUi(self, Q7LogWindow):
        Q7LogWindow.setWindowTitle(_translate("Q7LogWindow", "Dialog", None))
        self.bClear.setText(_translate("Q7LogWindow", "Clear", None))
        self.bClose.setText(_translate("Q7LogWindow", "Hide", None))

from CGNS.NAV.weditors import Q7PythonEditor
import Res_rc
