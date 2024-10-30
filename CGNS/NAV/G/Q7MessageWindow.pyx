# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7MessageWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFrame,
    QHBoxLayout, QLayout, QPushButton, QSizePolicy,
    QSpacerItem, QTextEdit, QVBoxLayout, QWidget)
from . import Res_rc

class Ui_Q7MessageWindow(object):
    def setupUi(self, Q7MessageWindow):
        if not Q7MessageWindow.objectName():
            Q7MessageWindow.setObjectName(u"Q7MessageWindow")
        Q7MessageWindow.resize(500, 200)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7MessageWindow.sizePolicy().hasHeightForWidth())
        Q7MessageWindow.setSizePolicy(sizePolicy)
        Q7MessageWindow.setMinimumSize(QSize(500, 200))
        Q7MessageWindow.setMaximumSize(QSize(1200, 600))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7MessageWindow.setWindowIcon(icon)
        self.verticalLayout = QVBoxLayout(Q7MessageWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.eMessage = QTextEdit(Q7MessageWindow)
        self.eMessage.setObjectName(u"eMessage")
        self.eMessage.setMinimumSize(QSize(0, 0))
        font = QFont()
        font.setFamilies([u"DejaVu Sans Mono"])
        self.eMessage.setFont(font)
        self.eMessage.setFrameShadow(QFrame.Plain)
        self.eMessage.setLineWidth(0)
        self.eMessage.setUndoRedoEnabled(False)
        self.eMessage.setTextInteractionFlags(Qt.NoTextInteraction)

        self.verticalLayout.addWidget(self.eMessage)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setSizeConstraint(QLayout.SetNoConstraint)
        self.bInfo = QPushButton(Q7MessageWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon1)

        self.horizontalLayout_2.addWidget(self.bInfo)

        self.cNotAgain = QCheckBox(Q7MessageWindow)
        self.cNotAgain.setObjectName(u"cNotAgain")
        sizePolicy.setHeightForWidth(self.cNotAgain.sizePolicy().hasHeightForWidth())
        self.cNotAgain.setSizePolicy(sizePolicy)

        self.horizontalLayout_2.addWidget(self.cNotAgain)

        self.horizontalSpacer_2 = QSpacerItem(868, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.bCANCEL = QPushButton(Q7MessageWindow)
        self.bCANCEL.setObjectName(u"bCANCEL")

        self.horizontalLayout_2.addWidget(self.bCANCEL)

        self.bOK = QPushButton(Q7MessageWindow)
        self.bOK.setObjectName(u"bOK")

        self.horizontalLayout_2.addWidget(self.bOK)


        self.verticalLayout.addLayout(self.horizontalLayout_2)


        self.retranslateUi(Q7MessageWindow)

        QMetaObject.connectSlotsByName(Q7MessageWindow)
    # setupUi

    def retranslateUi(self, Q7MessageWindow):
        Q7MessageWindow.setWindowTitle(QCoreApplication.translate("Q7MessageWindow", u"Dialog", None))
        self.bInfo.setText("")
        self.cNotAgain.setText(QCoreApplication.translate("Q7MessageWindow", u"Don't show this message again", None))
        self.bCANCEL.setText(QCoreApplication.translate("Q7MessageWindow", u"Cancel", None))
        self.bOK.setText(QCoreApplication.translate("Q7MessageWindow", u"OK", None))
    # retranslateUi

