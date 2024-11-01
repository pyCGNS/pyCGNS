# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7DiagWindow.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QGridLayout, QHBoxLayout, QHeaderView, QLineEdit,
    QPushButton, QSizePolicy, QSpacerItem, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget)
from . import Res_rc

class Ui_Q7DiagWindow(object):
    def setupUi(self, Q7DiagWindow):
        if not Q7DiagWindow.objectName():
            Q7DiagWindow.setObjectName(u"Q7DiagWindow")
        Q7DiagWindow.resize(715, 375)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7DiagWindow.sizePolicy().hasHeightForWidth())
        Q7DiagWindow.setSizePolicy(sizePolicy)
        Q7DiagWindow.setMinimumSize(QSize(715, 350))
        Q7DiagWindow.setMaximumSize(QSize(1200, 900))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7DiagWindow.setWindowIcon(icon)
        self.gridLayout = QGridLayout(Q7DiagWindow)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.bBackControl = QPushButton(Q7DiagWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon1)

        self.horizontalLayout_2.addWidget(self.bBackControl)

        self.bInfo = QPushButton(Q7DiagWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon2)

        self.horizontalLayout_2.addWidget(self.bInfo)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.bClose = QPushButton(Q7DiagWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout_2.addWidget(self.bClose)


        self.gridLayout.addLayout(self.horizontalLayout_2, 5, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bExpandAll = QPushButton(Q7DiagWindow)
        self.bExpandAll.setObjectName(u"bExpandAll")
        self.bExpandAll.setMinimumSize(QSize(25, 25))
        self.bExpandAll.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/level-in.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bExpandAll.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bExpandAll)

        self.bCollapseAll = QPushButton(Q7DiagWindow)
        self.bCollapseAll.setObjectName(u"bCollapseAll")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(25)
        sizePolicy1.setVerticalStretch(25)
        sizePolicy1.setHeightForWidth(self.bCollapseAll.sizePolicy().hasHeightForWidth())
        self.bCollapseAll.setSizePolicy(sizePolicy1)
        self.bCollapseAll.setMinimumSize(QSize(25, 25))
        self.bCollapseAll.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/level-out.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bCollapseAll.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bCollapseAll)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.bPrevious = QPushButton(Q7DiagWindow)
        self.bPrevious.setObjectName(u"bPrevious")
        self.bPrevious.setMinimumSize(QSize(25, 25))
        self.bPrevious.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/node-sids-opened.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bPrevious.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bPrevious)

        self.eCount = QLineEdit(Q7DiagWindow)
        self.eCount.setObjectName(u"eCount")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.eCount.sizePolicy().hasHeightForWidth())
        self.eCount.setSizePolicy(sizePolicy2)
        self.eCount.setMaximumSize(QSize(30, 16777215))
        self.eCount.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.eCount.setReadOnly(True)

        self.horizontalLayout.addWidget(self.eCount)

        self.bNext = QPushButton(Q7DiagWindow)
        self.bNext.setObjectName(u"bNext")
        self.bNext.setMinimumSize(QSize(25, 25))
        self.bNext.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/selected.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bNext.setIcon(icon6)

        self.horizontalLayout.addWidget(self.bNext)

        self.cFilter = QComboBox(Q7DiagWindow)
        self.cFilter.setObjectName(u"cFilter")

        self.horizontalLayout.addWidget(self.cFilter)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bWhich = QPushButton(Q7DiagWindow)
        self.bWhich.setObjectName(u"bWhich")
        self.bWhich.setMinimumSize(QSize(25, 25))
        self.bWhich.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/check-grammars.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bWhich.setIcon(icon7)
        self.bWhich.setIconSize(QSize(24, 24))

        self.horizontalLayout.addWidget(self.bWhich)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_5)

        self.cWarnings = QCheckBox(Q7DiagWindow)
        self.cWarnings.setObjectName(u"cWarnings")
        self.cWarnings.setEnabled(True)
        self.cWarnings.setChecked(True)

        self.horizontalLayout.addWidget(self.cWarnings)

        self.cDiagFirst = QCheckBox(Q7DiagWindow)
        self.cDiagFirst.setObjectName(u"cDiagFirst")
        self.cDiagFirst.setEnabled(True)

        self.horizontalLayout.addWidget(self.cDiagFirst)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.bSave = QPushButton(Q7DiagWindow)
        self.bSave.setObjectName(u"bSave")
        self.bSave.setMinimumSize(QSize(25, 25))
        self.bSave.setMaximumSize(QSize(25, 25))
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/check-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSave.setIcon(icon8)

        self.horizontalLayout.addWidget(self.bSave)


        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.diagTable = QTreeWidget(Q7DiagWindow)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.diagTable.setHeaderItem(__qtreewidgetitem)
        self.diagTable.setObjectName(u"diagTable")
        self.diagTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.diagTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.diagTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.diagTable.setSelectionMode(QAbstractItemView.SingleSelection)
        self.diagTable.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.verticalLayout.addWidget(self.diagTable)


        self.gridLayout.addLayout(self.verticalLayout, 4, 0, 1, 1)


        self.retranslateUi(Q7DiagWindow)

        QMetaObject.connectSlotsByName(Q7DiagWindow)
    # setupUi

    def retranslateUi(self, Q7DiagWindow):
        Q7DiagWindow.setWindowTitle(QCoreApplication.translate("Q7DiagWindow", u"Form", None))
        self.bBackControl.setText("")
        self.bInfo.setText("")
        self.bClose.setText(QCoreApplication.translate("Q7DiagWindow", u"Close", None))
        self.bExpandAll.setText("")
        self.bCollapseAll.setText("")
        self.bPrevious.setText("")
        self.bNext.setText("")
        self.bWhich.setText("")
        self.cWarnings.setText(QCoreApplication.translate("Q7DiagWindow", u"Warnings", None))
        self.cDiagFirst.setText(QCoreApplication.translate("Q7DiagWindow", u"Diagnostics first", None))
        self.bSave.setText("")
    # retranslateUi

