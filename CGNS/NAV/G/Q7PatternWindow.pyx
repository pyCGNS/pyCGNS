# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7PatternWindow.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QGridLayout, QHBoxLayout,
    QHeaderView, QPushButton, QSizePolicy, QSpacerItem,
    QTableWidgetItem, QVBoxLayout, QWidget)

from CGNS.NAV.mpattern import Q7PatternTableWidget
from . import Res_rc

class Ui_Q7PatternWindow(object):
    def setupUi(self, Q7PatternWindow):
        if not Q7PatternWindow.objectName():
            Q7PatternWindow.setObjectName(u"Q7PatternWindow")
        Q7PatternWindow.resize(715, 350)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7PatternWindow.sizePolicy().hasHeightForWidth())
        Q7PatternWindow.setSizePolicy(sizePolicy)
        Q7PatternWindow.setMinimumSize(QSize(715, 350))
        Q7PatternWindow.setMaximumSize(QSize(1200, 350))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7PatternWindow.setWindowIcon(icon)
        self.gridLayout = QGridLayout(Q7PatternWindow)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.bInfo = QPushButton(Q7PatternWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon1)

        self.horizontalLayout_2.addWidget(self.bInfo)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.bClose = QPushButton(Q7PatternWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout_2.addWidget(self.bClose)


        self.gridLayout.addLayout(self.horizontalLayout_2, 5, 0, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bAdd = QPushButton(Q7PatternWindow)
        self.bAdd.setObjectName(u"bAdd")
        self.bAdd.setMinimumSize(QSize(25, 25))
        self.bAdd.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/pattern-open.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bAdd.setIcon(icon2)

        self.horizontalLayout.addWidget(self.bAdd)

        self.bDelete = QPushButton(Q7PatternWindow)
        self.bDelete.setObjectName(u"bDelete")
        self.bDelete.setMinimumSize(QSize(25, 25))
        self.bDelete.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/pattern-close.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bDelete.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bDelete)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bCopy = QPushButton(Q7PatternWindow)
        self.bCopy.setObjectName(u"bCopy")
        self.bCopy.setMinimumSize(QSize(25, 25))
        self.bCopy.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/mark-node.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bCopy.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bCopy)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.bSave = QPushButton(Q7PatternWindow)
        self.bSave.setObjectName(u"bSave")
        self.bSave.setMinimumSize(QSize(25, 25))
        self.bSave.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/pattern-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSave.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bSave)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.patternTable = Q7PatternTableWidget(Q7PatternWindow)
        if (self.patternTable.columnCount() < 4):
            self.patternTable.setColumnCount(4)
        self.patternTable.setObjectName(u"patternTable")
        self.patternTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.patternTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.patternTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.patternTable.setSelectionMode(QAbstractItemView.SingleSelection)
        self.patternTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.patternTable.setSortingEnabled(True)
        self.patternTable.setColumnCount(4)
        self.patternTable.horizontalHeader().setStretchLastSection(True)

        self.verticalLayout.addWidget(self.patternTable)


        self.gridLayout.addLayout(self.verticalLayout, 4, 0, 1, 1)


        self.retranslateUi(Q7PatternWindow)

        QMetaObject.connectSlotsByName(Q7PatternWindow)
    # setupUi

    def retranslateUi(self, Q7PatternWindow):
        Q7PatternWindow.setWindowTitle(QCoreApplication.translate("Q7PatternWindow", u"Form", None))
        self.bInfo.setText("")
        self.bClose.setText(QCoreApplication.translate("Q7PatternWindow", u"Close", None))
        self.bAdd.setText("")
        self.bDelete.setText("")
        self.bCopy.setText("")
        self.bSave.setText("")
    # retranslateUi

