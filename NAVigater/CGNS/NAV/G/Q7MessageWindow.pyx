# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7MessageWindow.ui'
#
# Created: Tue Apr  9 11:00:24 2013
#      by: pyside-uic 0.2.13 running on PySide 1.0.9
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Q7MessageWindow(object):
    def setupUi(self, Q7MessageWindow):
        Q7MessageWindow.setObjectName("Q7MessageWindow")
        Q7MessageWindow.resize(500, 140)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7MessageWindow.sizePolicy().hasHeightForWidth())
        Q7MessageWindow.setSizePolicy(sizePolicy)
        Q7MessageWindow.setMinimumSize(QtCore.QSize(500, 140))
        Q7MessageWindow.setMaximumSize(QtCore.QSize(500, 140))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/cgSpy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7MessageWindow.setWindowIcon(icon)
        self.eMessage = QtGui.QTextEdit(Q7MessageWindow)
        self.eMessage.setGeometry(QtCore.QRect(9, 9, 482, 91))
        self.eMessage.setMinimumSize(QtCore.QSize(0, 0))
        self.eMessage.setFrameShadow(QtGui.QFrame.Plain)
        self.eMessage.setLineWidth(0)
        self.eMessage.setUndoRedoEnabled(False)
        self.eMessage.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.eMessage.setObjectName("eMessage")
        self.verticalLayoutWidget = QtGui.QWidget(Q7MessageWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(6, 106, 486, 31))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.verticalLayoutWidget)
        self.horizontalLayout_2.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bInfo = QtGui.QPushButton(self.verticalLayoutWidget)
        self.bInfo.setMinimumSize(QtCore.QSize(25, 25))
        self.bInfo.setMaximumSize(QtCore.QSize(25, 25))
        self.bInfo.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icons/help-view.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bInfo.setIcon(icon1)
        self.bInfo.setObjectName("bInfo")
        self.horizontalLayout_2.addWidget(self.bInfo)
        self.cNotAgain = QtGui.QCheckBox(self.verticalLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cNotAgain.sizePolicy().hasHeightForWidth())
        self.cNotAgain.setSizePolicy(sizePolicy)
        self.cNotAgain.setObjectName("cNotAgain")
        self.horizontalLayout_2.addWidget(self.cNotAgain)
        spacerItem = QtGui.QSpacerItem(868, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bCANCEL = QtGui.QPushButton(self.verticalLayoutWidget)
        self.bCANCEL.setObjectName("bCANCEL")
        self.horizontalLayout_2.addWidget(self.bCANCEL)
        self.bOK = QtGui.QPushButton(self.verticalLayoutWidget)
        self.bOK.setObjectName("bOK")
        self.horizontalLayout_2.addWidget(self.bOK)

        self.retranslateUi(Q7MessageWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7MessageWindow)

    def retranslateUi(self, Q7MessageWindow):
        Q7MessageWindow.setWindowTitle(QtGui.QApplication.translate("Q7MessageWindow", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.cNotAgain.setText(QtGui.QApplication.translate("Q7MessageWindow", "Don\'t show this message again", None, QtGui.QApplication.UnicodeUTF8))
        self.bCANCEL.setText(QtGui.QApplication.translate("Q7MessageWindow", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.bOK.setText(QtGui.QApplication.translate("Q7MessageWindow", "OK", None, QtGui.QApplication.UnicodeUTF8))

import Res_rc
