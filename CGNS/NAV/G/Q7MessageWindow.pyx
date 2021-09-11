# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7MessageWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Q7MessageWindow(object):
    def setupUi(self, Q7MessageWindow):
        Q7MessageWindow.setObjectName("Q7MessageWindow")
        Q7MessageWindow.resize(500, 200)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7MessageWindow.sizePolicy().hasHeightForWidth())
        Q7MessageWindow.setSizePolicy(sizePolicy)
        Q7MessageWindow.setMinimumSize(QtCore.QSize(500, 200))
        Q7MessageWindow.setMaximumSize(QtCore.QSize(1200, 600))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/cgSpy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7MessageWindow.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(Q7MessageWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.eMessage = QtWidgets.QTextEdit(Q7MessageWindow)
        self.eMessage.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans Mono")
        self.eMessage.setFont(font)
        self.eMessage.setFrameShadow(QtWidgets.QFrame.Plain)
        self.eMessage.setLineWidth(0)
        self.eMessage.setUndoRedoEnabled(False)
        self.eMessage.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.eMessage.setObjectName("eMessage")
        self.verticalLayout.addWidget(self.eMessage)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bInfo = QtWidgets.QPushButton(Q7MessageWindow)
        self.bInfo.setMinimumSize(QtCore.QSize(25, 25))
        self.bInfo.setMaximumSize(QtCore.QSize(25, 25))
        self.bInfo.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icons/help-view.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bInfo.setIcon(icon1)
        self.bInfo.setObjectName("bInfo")
        self.horizontalLayout_2.addWidget(self.bInfo)
        self.cNotAgain = QtWidgets.QCheckBox(Q7MessageWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cNotAgain.sizePolicy().hasHeightForWidth())
        self.cNotAgain.setSizePolicy(sizePolicy)
        self.cNotAgain.setObjectName("cNotAgain")
        self.horizontalLayout_2.addWidget(self.cNotAgain)
        spacerItem = QtWidgets.QSpacerItem(868, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bCANCEL = QtWidgets.QPushButton(Q7MessageWindow)
        self.bCANCEL.setObjectName("bCANCEL")
        self.horizontalLayout_2.addWidget(self.bCANCEL)
        self.bOK = QtWidgets.QPushButton(Q7MessageWindow)
        self.bOK.setObjectName("bOK")
        self.horizontalLayout_2.addWidget(self.bOK)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Q7MessageWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7MessageWindow)

    def retranslateUi(self, Q7MessageWindow):
        _translate = QtCore.QCoreApplication.translate
        Q7MessageWindow.setWindowTitle(_translate("Q7MessageWindow", "Dialog"))
        self.cNotAgain.setText(_translate("Q7MessageWindow", "Don\'t show this message again"))
        self.bCANCEL.setText(_translate("Q7MessageWindow", "Cancel"))
        self.bOK.setText(_translate("Q7MessageWindow", "OK"))
from . import Res_rc
