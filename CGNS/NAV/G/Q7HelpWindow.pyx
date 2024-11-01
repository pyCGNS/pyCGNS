# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7HelpWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QPushButton,
    QSizePolicy, QSpacerItem, QTextEdit, QVBoxLayout,
    QWidget)
from . import Res_rc

class Ui_Q7HelpWindow(object):
    def setupUi(self, Q7HelpWindow):
        if not Q7HelpWindow.objectName():
            Q7HelpWindow.setObjectName(u"Q7HelpWindow")
        Q7HelpWindow.resize(715, 350)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7HelpWindow.sizePolicy().hasHeightForWidth())
        Q7HelpWindow.setSizePolicy(sizePolicy)
        Q7HelpWindow.setMinimumSize(QSize(715, 350))
        Q7HelpWindow.setMaximumSize(QSize(1200, 350))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7HelpWindow.setWindowIcon(icon)
        self.gridLayout = QGridLayout(Q7HelpWindow)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.eHelp = QTextEdit(Q7HelpWindow)
        self.eHelp.setObjectName(u"eHelp")

        self.verticalLayout.addWidget(self.eHelp)


        self.gridLayout.addLayout(self.verticalLayout, 3, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.bBackControl = QPushButton(Q7HelpWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setEnabled(False)
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon1)

        self.horizontalLayout_2.addWidget(self.bBackControl)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.gridLayout.addLayout(self.horizontalLayout_2, 4, 0, 1, 1)


        self.retranslateUi(Q7HelpWindow)

        QMetaObject.connectSlotsByName(Q7HelpWindow)
    # setupUi

    def retranslateUi(self, Q7HelpWindow):
        Q7HelpWindow.setWindowTitle(QCoreApplication.translate("Q7HelpWindow", u"Form", None))
        self.bBackControl.setText("")
    # retranslateUi

