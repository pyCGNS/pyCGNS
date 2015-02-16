# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7MessageWindow.ui'
#
# Created: Fri Jan 23 10:40:28 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.1
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Q7MessageWindow(object):
    def setupUi(self, Q7MessageWindow):
        Q7MessageWindow.setObjectName("Q7MessageWindow")
        Q7MessageWindow.resize(500, 200)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7MessageWindow.sizePolicy().hasHeightForWidth())
        Q7MessageWindow.setSizePolicy(sizePolicy)
        Q7MessageWindow.setMinimumSize(QtCore.QSize(500, 200))
        Q7MessageWindow.setMaximumSize(QtCore.QSize(1200, 600))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/cgSpy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7MessageWindow.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(Q7MessageWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.eMessage = QtGui.QTextEdit(Q7MessageWindow)
        self.eMessage.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans Mono")
        self.eMessage.setFont(font)
        self.eMessage.setFrameShadow(QtGui.QFrame.Plain)
        self.eMessage.setLineWidth(0)
        self.eMessage.setUndoRedoEnabled(False)
        self.eMessage.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.eMessage.setObjectName("eMessage")
        self.verticalLayout.addWidget(self.eMessage)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bInfo = QtGui.QPushButton(Q7MessageWindow)
        self.bInfo.setMinimumSize(QtCore.QSize(25, 25))
        self.bInfo.setMaximumSize(QtCore.QSize(25, 25))
        self.bInfo.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icons/help-view.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bInfo.setIcon(icon1)
        self.bInfo.setObjectName("bInfo")
        self.horizontalLayout_2.addWidget(self.bInfo)
        self.cNotAgain = QtGui.QCheckBox(Q7MessageWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cNotAgain.sizePolicy().hasHeightForWidth())
        self.cNotAgain.setSizePolicy(sizePolicy)
        self.cNotAgain.setObjectName("cNotAgain")
        self.horizontalLayout_2.addWidget(self.cNotAgain)
        spacerItem = QtGui.QSpacerItem(868, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bCANCEL = QtGui.QPushButton(Q7MessageWindow)
        self.bCANCEL.setObjectName("bCANCEL")
        self.horizontalLayout_2.addWidget(self.bCANCEL)
        self.bOK = QtGui.QPushButton(Q7MessageWindow)
        self.bOK.setObjectName("bOK")
        self.horizontalLayout_2.addWidget(self.bOK)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Q7MessageWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7MessageWindow)

    def retranslateUi(self, Q7MessageWindow):
        Q7MessageWindow.setWindowTitle(QtGui.QApplication.translate("Q7MessageWindow", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.cNotAgain.setText(QtGui.QApplication.translate("Q7MessageWindow", "Don\'t show this message again", None, QtGui.QApplication.UnicodeUTF8))
        self.bCANCEL.setText(QtGui.QApplication.translate("Q7MessageWindow", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.bOK.setText(QtGui.QApplication.translate("Q7MessageWindow", "OK", None, QtGui.QApplication.UnicodeUTF8))

import Res_rc
