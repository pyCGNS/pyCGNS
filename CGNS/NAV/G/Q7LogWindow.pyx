# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7LogWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout,
    QLayout, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

from CGNS.NAV.weditors import Q7PythonEditor
from . import Res_rc

class Ui_Q7LogWindow(object):
    def setupUi(self, Q7LogWindow):
        if not Q7LogWindow.objectName():
            Q7LogWindow.setObjectName(u"Q7LogWindow")
        Q7LogWindow.resize(600, 400)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7LogWindow.sizePolicy().hasHeightForWidth())
        Q7LogWindow.setSizePolicy(sizePolicy)
        Q7LogWindow.setMinimumSize(QSize(500, 140))
        Q7LogWindow.setMaximumSize(QSize(16777215, 16777215))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7LogWindow.setWindowIcon(icon)
        self.verticalLayout = QVBoxLayout(Q7LogWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.eLog = Q7PythonEditor(Q7LogWindow)
        self.eLog.setObjectName(u"eLog")
        self.eLog.setMinimumSize(QSize(0, 0))
        self.eLog.setFrameShadow(QFrame.Raised)
        self.eLog.setLineWidth(0)

        self.verticalLayout.addWidget(self.eLog)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setSizeConstraint(QLayout.SetNoConstraint)
        self.horizontalSpacer_2 = QSpacerItem(868, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.bClear = QPushButton(Q7LogWindow)
        self.bClear.setObjectName(u"bClear")

        self.horizontalLayout_2.addWidget(self.bClear)

        self.bClose = QPushButton(Q7LogWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout_2.addWidget(self.bClose)


        self.verticalLayout.addLayout(self.horizontalLayout_2)


        self.retranslateUi(Q7LogWindow)

        QMetaObject.connectSlotsByName(Q7LogWindow)
    # setupUi

    def retranslateUi(self, Q7LogWindow):
        Q7LogWindow.setWindowTitle(QCoreApplication.translate("Q7LogWindow", u"Dialog", None))
        self.bClear.setText(QCoreApplication.translate("Q7LogWindow", u"Clear", None))
        self.bClose.setText(QCoreApplication.translate("Q7LogWindow", u"Hide", None))
    # retranslateUi

