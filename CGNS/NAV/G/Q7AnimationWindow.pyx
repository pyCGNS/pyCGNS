# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7AnimationWindow.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QGridLayout,
    QHBoxLayout, QHeaderView, QPushButton, QSizePolicy,
    QSpacerItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget)
from . import Res_rc

class Ui_Q7AnimationWindow(object):
    def setupUi(self, Q7AnimationWindow):
        if not Q7AnimationWindow.objectName():
            Q7AnimationWindow.setObjectName(u"Q7AnimationWindow")
        Q7AnimationWindow.resize(715, 350)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7AnimationWindow.sizePolicy().hasHeightForWidth())
        Q7AnimationWindow.setSizePolicy(sizePolicy)
        Q7AnimationWindow.setMinimumSize(QSize(715, 350))
        Q7AnimationWindow.setMaximumSize(QSize(1200, 350))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7AnimationWindow.setWindowIcon(icon)
        self.gridLayout = QGridLayout(Q7AnimationWindow)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.bBackControl = QPushButton(Q7AnimationWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon1)

        self.horizontalLayout_2.addWidget(self.bBackControl)

        self.bInfo = QPushButton(Q7AnimationWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon2)

        self.horizontalLayout_2.addWidget(self.bInfo)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.bClose = QPushButton(Q7AnimationWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout_2.addWidget(self.bClose)


        self.gridLayout.addLayout(self.horizontalLayout_2, 6, 0, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.diagTable = QTreeWidget(Q7AnimationWindow)
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


        self.gridLayout.addLayout(self.verticalLayout, 5, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bExpandAll = QPushButton(Q7AnimationWindow)
        self.bExpandAll.setObjectName(u"bExpandAll")
        self.bExpandAll.setMinimumSize(QSize(25, 25))
        self.bExpandAll.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/level-in.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bExpandAll.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bExpandAll)

        self.bCollapseAll = QPushButton(Q7AnimationWindow)
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

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bFrameAdd = QPushButton(Q7AnimationWindow)
        self.bFrameAdd.setObjectName(u"bFrameAdd")
        self.bFrameAdd.setMinimumSize(QSize(25, 25))
        self.bFrameAdd.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/anim-add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bFrameAdd.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bFrameAdd)

        self.bFrameDel = QPushButton(Q7AnimationWindow)
        self.bFrameDel.setObjectName(u"bFrameDel")
        self.bFrameDel.setMinimumSize(QSize(25, 25))
        self.bFrameDel.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/anim-del.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bFrameDel.setIcon(icon6)

        self.horizontalLayout.addWidget(self.bFrameDel)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.pushButton_7 = QPushButton(Q7AnimationWindow)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setMinimumSize(QSize(25, 25))
        self.pushButton_7.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/anim-auto-sids.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_7.setIcon(icon7)

        self.horizontalLayout.addWidget(self.pushButton_7)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.bSave = QPushButton(Q7AnimationWindow)
        self.bSave.setObjectName(u"bSave")
        self.bSave.setMinimumSize(QSize(25, 25))
        self.bSave.setMaximumSize(QSize(25, 25))
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/anim-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSave.setIcon(icon8)

        self.horizontalLayout.addWidget(self.bSave)


        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.pushButton = QPushButton(Q7AnimationWindow)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setMinimumSize(QSize(25, 25))
        self.pushButton.setMaximumSize(QSize(25, 25))
        icon9 = QIcon()
        icon9.addFile(u":/images/icons/anim-item-first.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton.setIcon(icon9)

        self.horizontalLayout_3.addWidget(self.pushButton)

        self.pushButton_2 = QPushButton(Q7AnimationWindow)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setMinimumSize(QSize(25, 25))
        self.pushButton_2.setMaximumSize(QSize(25, 25))
        icon10 = QIcon()
        icon10.addFile(u":/images/icons/anim-item-prev.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_2.setIcon(icon10)

        self.horizontalLayout_3.addWidget(self.pushButton_2)

        self.pushButton_5 = QPushButton(Q7AnimationWindow)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setMinimumSize(QSize(25, 25))
        self.pushButton_5.setMaximumSize(QSize(25, 25))
        icon11 = QIcon()
        icon11.addFile(u":/images/icons/anim-item-next.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_5.setIcon(icon11)

        self.horizontalLayout_3.addWidget(self.pushButton_5)

        self.pushButton_6 = QPushButton(Q7AnimationWindow)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setMinimumSize(QSize(25, 25))
        self.pushButton_6.setMaximumSize(QSize(25, 25))
        icon12 = QIcon()
        icon12.addFile(u":/images/icons/anim-item-last.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_6.setIcon(icon12)

        self.horizontalLayout_3.addWidget(self.pushButton_6)

        self.cActors = QComboBox(Q7AnimationWindow)
        self.cActors.setObjectName(u"cActors")

        self.horizontalLayout_3.addWidget(self.cActors)

        self.pushButton_3 = QPushButton(Q7AnimationWindow)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setMinimumSize(QSize(25, 25))
        self.pushButton_3.setMaximumSize(QSize(25, 25))
        icon13 = QIcon()
        icon13.addFile(u":/images/icons/anim-item-add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_3.setIcon(icon13)

        self.horizontalLayout_3.addWidget(self.pushButton_3)

        self.pushButton_4 = QPushButton(Q7AnimationWindow)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setMinimumSize(QSize(25, 25))
        self.pushButton_4.setMaximumSize(QSize(25, 25))
        icon14 = QIcon()
        icon14.addFile(u":/images/icons/anim-item-del.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_4.setIcon(icon14)

        self.horizontalLayout_3.addWidget(self.pushButton_4)


        self.gridLayout.addLayout(self.horizontalLayout_3, 4, 0, 1, 1)


        self.retranslateUi(Q7AnimationWindow)

        QMetaObject.connectSlotsByName(Q7AnimationWindow)
    # setupUi

    def retranslateUi(self, Q7AnimationWindow):
        Q7AnimationWindow.setWindowTitle(QCoreApplication.translate("Q7AnimationWindow", u"Form", None))
        self.bBackControl.setText("")
        self.bInfo.setText("")
        self.bClose.setText(QCoreApplication.translate("Q7AnimationWindow", u"Close", None))
        self.bExpandAll.setText("")
        self.bCollapseAll.setText("")
        self.bFrameAdd.setText("")
        self.bFrameDel.setText("")
        self.pushButton_7.setText("")
        self.bSave.setText("")
        self.pushButton.setText("")
        self.pushButton_2.setText("")
        self.pushButton_5.setText("")
        self.pushButton_6.setText("")
        self.pushButton_3.setText("")
        self.pushButton_4.setText("")
    # retranslateUi

