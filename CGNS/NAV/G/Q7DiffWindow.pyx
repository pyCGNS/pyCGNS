# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7DiffWindow.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QHBoxLayout, QHeaderView,
    QPushButton, QScrollBar, QSizePolicy, QSpacerItem,
    QToolButton, QVBoxLayout, QWidget)

from CGNS.NAV.mdifftreeview import Q7DiffTreeView
from . import Res_rc

class Ui_Q7DiffWindow(object):
    def setupUi(self, Q7DiffWindow):
        if not Q7DiffWindow.objectName():
            Q7DiffWindow.setObjectName(u"Q7DiffWindow")
        Q7DiffWindow.resize(1124, 300)
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7DiffWindow.setWindowIcon(icon)
        self.verticalLayout_2 = QVBoxLayout(Q7DiffWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bLockScroll = QPushButton(Q7DiffWindow)
        self.bLockScroll.setObjectName(u"bLockScroll")
        self.bLockScroll.setMinimumSize(QSize(25, 25))
        self.bLockScroll.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/lock-scroll.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bLockScroll.setIcon(icon1)
        self.bLockScroll.setCheckable(True)
        self.bLockScroll.setChecked(True)

        self.horizontalLayout.addWidget(self.bLockScroll)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bZoomOut = QToolButton(Q7DiffWindow)
        self.bZoomOut.setObjectName(u"bZoomOut")
        self.bZoomOut.setMinimumSize(QSize(25, 25))
        self.bZoomOut.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/level-out.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bZoomOut.setIcon(icon2)

        self.horizontalLayout.addWidget(self.bZoomOut)

        self.bZoomAll = QPushButton(Q7DiffWindow)
        self.bZoomAll.setObjectName(u"bZoomAll")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bZoomAll.sizePolicy().hasHeightForWidth())
        self.bZoomAll.setSizePolicy(sizePolicy)
        self.bZoomAll.setMinimumSize(QSize(25, 25))
        self.bZoomAll.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/level-all.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bZoomAll.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bZoomAll)

        self.bZoomIn = QToolButton(Q7DiffWindow)
        self.bZoomIn.setObjectName(u"bZoomIn")
        self.bZoomIn.setMinimumSize(QSize(25, 25))
        self.bZoomIn.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/level-in.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bZoomIn.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bZoomIn)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_11)

        self.bSaveDiff = QToolButton(Q7DiffWindow)
        self.bSaveDiff.setObjectName(u"bSaveDiff")
        self.bSaveDiff.setMinimumSize(QSize(25, 25))
        self.bSaveDiff.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/select-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSaveDiff.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bSaveDiff)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.treeviewA = Q7DiffTreeView(Q7DiffWindow)
        self.treeviewA.setObjectName(u"treeviewA")
        self.treeviewA.viewport().setProperty(u"cursor", QCursor(Qt.CursorShape.CrossCursor))
        self.treeviewA.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.treeviewA.setAutoScroll(False)
        self.treeviewA.setProperty(u"showDropIndicator", False)
        self.treeviewA.setHorizontalScrollMode(QAbstractItemView.ScrollPerItem)
        self.treeviewA.setIndentation(16)
        self.treeviewA.setRootIsDecorated(True)
        self.treeviewA.setUniformRowHeights(True)
        self.treeviewA.setExpandsOnDoubleClick(False)

        self.horizontalLayout_2.addWidget(self.treeviewA)

        self.verticalScrollBarA = QScrollBar(Q7DiffWindow)
        self.verticalScrollBarA.setObjectName(u"verticalScrollBarA")
        self.verticalScrollBarA.setOrientation(Qt.Vertical)

        self.horizontalLayout_2.addWidget(self.verticalScrollBarA)

        self.verticalScrollBarB = QScrollBar(Q7DiffWindow)
        self.verticalScrollBarB.setObjectName(u"verticalScrollBarB")
        self.verticalScrollBarB.setOrientation(Qt.Vertical)

        self.horizontalLayout_2.addWidget(self.verticalScrollBarB)

        self.treeviewB = Q7DiffTreeView(Q7DiffWindow)
        self.treeviewB.setObjectName(u"treeviewB")
        self.treeviewB.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.horizontalLayout_2.addWidget(self.treeviewB)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.bBackControl = QPushButton(Q7DiffWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon6)

        self.horizontalLayout_3.addWidget(self.bBackControl)

        self.bInfo = QPushButton(Q7DiffWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon7)

        self.horizontalLayout_3.addWidget(self.bInfo)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_6)

        self.bPreviousMark = QToolButton(Q7DiffWindow)
        self.bPreviousMark.setObjectName(u"bPreviousMark")
        self.bPreviousMark.setEnabled(True)
        self.bPreviousMark.setMinimumSize(QSize(25, 25))
        self.bPreviousMark.setMaximumSize(QSize(25, 25))
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/node-sids-opened.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bPreviousMark.setIcon(icon8)

        self.horizontalLayout_3.addWidget(self.bPreviousMark)

        self.bUnmarkAll_1 = QToolButton(Q7DiffWindow)
        self.bUnmarkAll_1.setObjectName(u"bUnmarkAll_1")
        self.bUnmarkAll_1.setMinimumSize(QSize(25, 25))
        self.bUnmarkAll_1.setMaximumSize(QSize(25, 25))
        icon9 = QIcon()
        icon9.addFile(u":/images/icons/node-sids-leaf.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bUnmarkAll_1.setIcon(icon9)

        self.horizontalLayout_3.addWidget(self.bUnmarkAll_1)

        self.bNextMark = QToolButton(Q7DiffWindow)
        self.bNextMark.setObjectName(u"bNextMark")
        self.bNextMark.setMinimumSize(QSize(25, 25))
        self.bNextMark.setMaximumSize(QSize(25, 25))
        icon10 = QIcon()
        icon10.addFile(u":/images/icons/node-sids-closed.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bNextMark.setIcon(icon10)

        self.horizontalLayout_3.addWidget(self.bNextMark)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_8)

        self.bClose = QPushButton(Q7DiffWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout_3.addWidget(self.bClose)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)


        self.retranslateUi(Q7DiffWindow)

        QMetaObject.connectSlotsByName(Q7DiffWindow)
    # setupUi

    def retranslateUi(self, Q7DiffWindow):
        Q7DiffWindow.setWindowTitle(QCoreApplication.translate("Q7DiffWindow", u"Form", None))
#if QT_CONFIG(tooltip)
        self.bLockScroll.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Lock scrollbars together", None))
#endif // QT_CONFIG(tooltip)
        self.bLockScroll.setText("")
#if QT_CONFIG(tooltip)
        self.bZoomOut.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Collapse lowest tree level", None))
#endif // QT_CONFIG(tooltip)
        self.bZoomOut.setText(QCoreApplication.translate("Q7DiffWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bZoomAll.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Expand all tree", None))
#endif // QT_CONFIG(tooltip)
        self.bZoomAll.setText("")
#if QT_CONFIG(tooltip)
        self.bZoomIn.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Expand lowest tree level", None))
#endif // QT_CONFIG(tooltip)
        self.bZoomIn.setText(QCoreApplication.translate("Q7DiffWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bSaveDiff.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Save tree view snapshot", None))
#endif // QT_CONFIG(tooltip)
        self.bSaveDiff.setText(QCoreApplication.translate("Q7DiffWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.verticalScrollBarA.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"DiffA file", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.verticalScrollBarB.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"DiffB file", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.bBackControl.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Raise Control window", None))
#endif // QT_CONFIG(tooltip)
        self.bBackControl.setText("")
#if QT_CONFIG(tooltip)
        self.bInfo.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Contextual help", None))
#endif // QT_CONFIG(tooltip)
        self.bInfo.setText("")
#if QT_CONFIG(tooltip)
        self.bPreviousMark.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Select previous marked node", None))
#endif // QT_CONFIG(tooltip)
        self.bPreviousMark.setText(QCoreApplication.translate("Q7DiffWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bUnmarkAll_1.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Unmark all nodes", None))
#endif // QT_CONFIG(tooltip)
        self.bUnmarkAll_1.setText(QCoreApplication.translate("Q7DiffWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bNextMark.setToolTip(QCoreApplication.translate("Q7DiffWindow", u"Select next marked node", None))
#endif // QT_CONFIG(tooltip)
        self.bNextMark.setText(QCoreApplication.translate("Q7DiffWindow", u"...", None))
        self.bClose.setText(QCoreApplication.translate("Q7DiffWindow", u"Close", None))
    # retranslateUi

