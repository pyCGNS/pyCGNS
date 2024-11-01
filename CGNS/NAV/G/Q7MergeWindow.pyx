# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7MergeWindow.ui'
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
    QPushButton, QSizePolicy, QSpacerItem, QToolButton,
    QVBoxLayout, QWidget)

from CGNS.NAV.mdifftreeview import Q7DiffTreeView
from . import Res_rc

class Ui_Q7MergeWindow(object):
    def setupUi(self, Q7MergeWindow):
        if not Q7MergeWindow.objectName():
            Q7MergeWindow.setObjectName(u"Q7MergeWindow")
        Q7MergeWindow.resize(1124, 300)
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7MergeWindow.setWindowIcon(icon)
        self.verticalLayout_2 = QVBoxLayout(Q7MergeWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bSelectA = QPushButton(Q7MergeWindow)
        self.bSelectA.setObjectName(u"bSelectA")
        self.bSelectA.setMinimumSize(QSize(24, 24))
        self.bSelectA.setMaximumSize(QSize(24, 24))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/user-A.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSelectA.setIcon(icon1)
        self.bSelectA.setCheckable(True)
        self.bSelectA.setChecked(True)

        self.horizontalLayout.addWidget(self.bSelectA)

        self.bSelectOrderSwap = QPushButton(Q7MergeWindow)
        self.bSelectOrderSwap.setObjectName(u"bSelectOrderSwap")
        self.bSelectOrderSwap.setMinimumSize(QSize(24, 24))
        self.bSelectOrderSwap.setMaximumSize(QSize(24, 24))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/reverse.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSelectOrderSwap.setIcon(icon2)

        self.horizontalLayout.addWidget(self.bSelectOrderSwap)

        self.bSelectB = QPushButton(Q7MergeWindow)
        self.bSelectB.setObjectName(u"bSelectB")
        self.bSelectB.setMinimumSize(QSize(24, 24))
        self.bSelectB.setMaximumSize(QSize(24, 24))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/user-B.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSelectB.setIcon(icon3)
        self.bSelectB.setCheckable(True)
        self.bSelectB.setChecked(True)

        self.horizontalLayout.addWidget(self.bSelectB)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bZoomOut = QToolButton(Q7MergeWindow)
        self.bZoomOut.setObjectName(u"bZoomOut")
        self.bZoomOut.setMinimumSize(QSize(25, 25))
        self.bZoomOut.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/level-out.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bZoomOut.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bZoomOut)

        self.bZoomAll = QPushButton(Q7MergeWindow)
        self.bZoomAll.setObjectName(u"bZoomAll")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bZoomAll.sizePolicy().hasHeightForWidth())
        self.bZoomAll.setSizePolicy(sizePolicy)
        self.bZoomAll.setMinimumSize(QSize(25, 25))
        self.bZoomAll.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/level-all.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bZoomAll.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bZoomAll)

        self.bZoomIn = QToolButton(Q7MergeWindow)
        self.bZoomIn.setObjectName(u"bZoomIn")
        self.bZoomIn.setMinimumSize(QSize(25, 25))
        self.bZoomIn.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/level-in.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bZoomIn.setIcon(icon6)

        self.horizontalLayout.addWidget(self.bZoomIn)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_11)

        self.bSaveDiff = QToolButton(Q7MergeWindow)
        self.bSaveDiff.setObjectName(u"bSaveDiff")
        self.bSaveDiff.setMinimumSize(QSize(25, 25))
        self.bSaveDiff.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/select-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSaveDiff.setIcon(icon7)

        self.horizontalLayout.addWidget(self.bSaveDiff)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.treeview = Q7DiffTreeView(Q7MergeWindow)
        self.treeview.setObjectName(u"treeview")
        self.treeview.viewport().setProperty(u"cursor", QCursor(Qt.CursorShape.CrossCursor))
        self.treeview.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.treeview.setAutoScroll(False)
        self.treeview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.treeview.setProperty(u"showDropIndicator", False)
        self.treeview.setHorizontalScrollMode(QAbstractItemView.ScrollPerItem)
        self.treeview.setIndentation(16)
        self.treeview.setRootIsDecorated(True)
        self.treeview.setUniformRowHeights(True)
        self.treeview.setExpandsOnDoubleClick(False)

        self.horizontalLayout_2.addWidget(self.treeview)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.bBackControl = QPushButton(Q7MergeWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon8)

        self.horizontalLayout_3.addWidget(self.bBackControl)

        self.bInfo = QPushButton(Q7MergeWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon9 = QIcon()
        icon9.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon9)

        self.horizontalLayout_3.addWidget(self.bInfo)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_6)

        self.bPreviousMark = QToolButton(Q7MergeWindow)
        self.bPreviousMark.setObjectName(u"bPreviousMark")
        self.bPreviousMark.setMinimumSize(QSize(25, 25))
        self.bPreviousMark.setMaximumSize(QSize(25, 25))
        icon10 = QIcon()
        icon10.addFile(u":/images/icons/node-sids-opened.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bPreviousMark.setIcon(icon10)

        self.horizontalLayout_3.addWidget(self.bPreviousMark)

        self.bUnmarkAll_1 = QToolButton(Q7MergeWindow)
        self.bUnmarkAll_1.setObjectName(u"bUnmarkAll_1")
        self.bUnmarkAll_1.setMinimumSize(QSize(25, 25))
        self.bUnmarkAll_1.setMaximumSize(QSize(25, 25))
        icon11 = QIcon()
        icon11.addFile(u":/images/icons/node-sids-leaf.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bUnmarkAll_1.setIcon(icon11)

        self.horizontalLayout_3.addWidget(self.bUnmarkAll_1)

        self.bNextMark = QToolButton(Q7MergeWindow)
        self.bNextMark.setObjectName(u"bNextMark")
        self.bNextMark.setMinimumSize(QSize(25, 25))
        self.bNextMark.setMaximumSize(QSize(25, 25))
        icon12 = QIcon()
        icon12.addFile(u":/images/icons/node-sids-closed.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bNextMark.setIcon(icon12)

        self.horizontalLayout_3.addWidget(self.bNextMark)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_8)

        self.bClose = QPushButton(Q7MergeWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout_3.addWidget(self.bClose)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)


        self.retranslateUi(Q7MergeWindow)

        QMetaObject.connectSlotsByName(Q7MergeWindow)
    # setupUi

    def retranslateUi(self, Q7MergeWindow):
        Q7MergeWindow.setWindowTitle(QCoreApplication.translate("Q7MergeWindow", u"Form", None))
        self.bSelectA.setText("")
#if QT_CONFIG(tooltip)
        self.bSelectOrderSwap.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Reverse precedence order of merge", None))
#endif // QT_CONFIG(tooltip)
        self.bSelectOrderSwap.setText("")
        self.bSelectB.setText("")
#if QT_CONFIG(tooltip)
        self.bZoomOut.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Collapse lowest tree level", None))
#endif // QT_CONFIG(tooltip)
        self.bZoomOut.setText(QCoreApplication.translate("Q7MergeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bZoomAll.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Expand all tree", None))
#endif // QT_CONFIG(tooltip)
        self.bZoomAll.setText("")
#if QT_CONFIG(tooltip)
        self.bZoomIn.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Expand lowest tree level", None))
#endif // QT_CONFIG(tooltip)
        self.bZoomIn.setText(QCoreApplication.translate("Q7MergeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bSaveDiff.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Save tree view snapshot", None))
#endif // QT_CONFIG(tooltip)
        self.bSaveDiff.setText(QCoreApplication.translate("Q7MergeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bBackControl.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Raise Control window", None))
#endif // QT_CONFIG(tooltip)
        self.bBackControl.setText("")
#if QT_CONFIG(tooltip)
        self.bInfo.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Contextual help", None))
#endif // QT_CONFIG(tooltip)
        self.bInfo.setText("")
#if QT_CONFIG(tooltip)
        self.bPreviousMark.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Select previous marked node", None))
#endif // QT_CONFIG(tooltip)
        self.bPreviousMark.setText(QCoreApplication.translate("Q7MergeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bUnmarkAll_1.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Unmark all nodes", None))
#endif // QT_CONFIG(tooltip)
        self.bUnmarkAll_1.setText(QCoreApplication.translate("Q7MergeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bNextMark.setToolTip(QCoreApplication.translate("Q7MergeWindow", u"Select next marked node", None))
#endif // QT_CONFIG(tooltip)
        self.bNextMark.setText(QCoreApplication.translate("Q7MergeWindow", u"...", None))
        self.bClose.setText(QCoreApplication.translate("Q7MergeWindow", u"Close", None))
    # retranslateUi

