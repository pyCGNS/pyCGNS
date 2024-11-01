# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7ControlWindow.ui'
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
    QPushButton, QSizePolicy, QSpacerItem, QTableWidgetItem,
    QToolButton, QVBoxLayout, QWidget)

from CGNS.NAV.mcontrol import Q7ControlTableWidget
from . import Res_rc

class Ui_Q7ControlWindow(object):
    def setupUi(self, Q7ControlWindow):
        if not Q7ControlWindow.objectName():
            Q7ControlWindow.setObjectName(u"Q7ControlWindow")
        Q7ControlWindow.resize(799, 232)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7ControlWindow.sizePolicy().hasHeightForWidth())
        Q7ControlWindow.setSizePolicy(sizePolicy)
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7ControlWindow.setWindowIcon(icon)
        self.verticalLayout_2 = QVBoxLayout(Q7ControlWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bTreeLoadLast = QToolButton(Q7ControlWindow)
        self.bTreeLoadLast.setObjectName(u"bTreeLoadLast")
        self.bTreeLoadLast.setMinimumSize(QSize(25, 25))
        self.bTreeLoadLast.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/tree-load-g.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bTreeLoadLast.setIcon(icon1)

        self.horizontalLayout.addWidget(self.bTreeLoadLast)

        self.bTreeLoad = QToolButton(Q7ControlWindow)
        self.bTreeLoad.setObjectName(u"bTreeLoad")
        self.bTreeLoad.setMinimumSize(QSize(25, 25))
        self.bTreeLoad.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/tree-load.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bTreeLoad.setIcon(icon2)

        self.horizontalLayout.addWidget(self.bTreeLoad)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bEditTree = QToolButton(Q7ControlWindow)
        self.bEditTree.setObjectName(u"bEditTree")
        self.bEditTree.setMinimumSize(QSize(25, 25))
        self.bEditTree.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/tree-new.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bEditTree.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bEditTree)

        self.bPatternView = QToolButton(Q7ControlWindow)
        self.bPatternView.setObjectName(u"bPatternView")
        self.bPatternView.setMinimumSize(QSize(25, 25))
        self.bPatternView.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/pattern.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bPatternView.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bPatternView)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.bLog = QPushButton(Q7ControlWindow)
        self.bLog.setObjectName(u"bLog")
        self.bLog.setMinimumSize(QSize(25, 25))
        self.bLog.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/subtree-sids-warning.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bLog.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bLog)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.bOptionView = QToolButton(Q7ControlWindow)
        self.bOptionView.setObjectName(u"bOptionView")
        self.bOptionView.setMinimumSize(QSize(25, 25))
        self.bOptionView.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/options-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bOptionView.setIcon(icon6)

        self.horizontalLayout.addWidget(self.bOptionView)

        self.bInfo = QPushButton(Q7ControlWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon7)

        self.horizontalLayout.addWidget(self.bInfo)

        self.bAbout = QToolButton(Q7ControlWindow)
        self.bAbout.setObjectName(u"bAbout")
        self.bAbout.setMinimumSize(QSize(25, 25))
        self.bAbout.setMaximumSize(QSize(25, 25))
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/view-help.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bAbout.setIcon(icon8)

        self.horizontalLayout.addWidget(self.bAbout)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.bClose = QPushButton(Q7ControlWindow)
        self.bClose.setObjectName(u"bClose")
        self.bClose.setMinimumSize(QSize(25, 25))
        self.bClose.setMaximumSize(QSize(25, 25))
        icon9 = QIcon()
        icon9.addFile(u":/images/icons/close-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bClose.setIcon(icon9)

        self.horizontalLayout.addWidget(self.bClose)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.controlTable = Q7ControlTableWidget(Q7ControlWindow)
        if (self.controlTable.columnCount() < 6):
            self.controlTable.setColumnCount(6)
        self.controlTable.setObjectName(u"controlTable")
        sizePolicy.setHeightForWidth(self.controlTable.sizePolicy().hasHeightForWidth())
        self.controlTable.setSizePolicy(sizePolicy)
        self.controlTable.setMinimumSize(QSize(781, 181))
        self.controlTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.controlTable.setTabKeyNavigation(False)
        self.controlTable.setProperty(u"showDropIndicator", False)
        self.controlTable.setDragDropOverwriteMode(False)
        self.controlTable.setSelectionMode(QAbstractItemView.SingleSelection)
        self.controlTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.controlTable.setSortingEnabled(True)
        self.controlTable.setCornerButtonEnabled(False)
        self.controlTable.setRowCount(0)
        self.controlTable.setColumnCount(6)
        self.controlTable.horizontalHeader().setCascadingSectionResizes(True)
        self.controlTable.horizontalHeader().setStretchLastSection(True)

        self.verticalLayout_2.addWidget(self.controlTable)


        self.retranslateUi(Q7ControlWindow)

        QMetaObject.connectSlotsByName(Q7ControlWindow)
    # setupUi

    def retranslateUi(self, Q7ControlWindow):
        Q7ControlWindow.setWindowTitle(QCoreApplication.translate("Q7ControlWindow", u"Form", None))
#if QT_CONFIG(tooltip)
        self.bTreeLoadLast.setToolTip(QCoreApplication.translate("Q7ControlWindow", u"Load the last used CGNS file", None))
#endif // QT_CONFIG(tooltip)
        self.bTreeLoadLast.setText("")
#if QT_CONFIG(tooltip)
        self.bTreeLoad.setToolTip(QCoreApplication.translate("Q7ControlWindow", u"Load an existing CGNS file", None))
#endif // QT_CONFIG(tooltip)
        self.bTreeLoad.setText("")
#if QT_CONFIG(tooltip)
        self.bEditTree.setToolTip(QCoreApplication.translate("Q7ControlWindow", u"Create a new CGNS tree from scratch", None))
#endif // QT_CONFIG(tooltip)
        self.bEditTree.setText(QCoreApplication.translate("Q7ControlWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bPatternView.setToolTip(QCoreApplication.translate("Q7ControlWindow", u"Open CGNS/SIDS sub-trees database", None))
#endif // QT_CONFIG(tooltip)
        self.bPatternView.setText(QCoreApplication.translate("Q7ControlWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bLog.setToolTip(QCoreApplication.translate("Q7ControlWindow", u"Log window", None))
#endif // QT_CONFIG(tooltip)
        self.bLog.setText("")
#if QT_CONFIG(tooltip)
        self.bOptionView.setToolTip(QCoreApplication.translate("Q7ControlWindow", u"Set user defined options", None))
#endif // QT_CONFIG(tooltip)
        self.bOptionView.setText(QCoreApplication.translate("Q7ControlWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bInfo.setToolTip(QCoreApplication.translate("Q7ControlWindow", u"Help window", None))
#endif // QT_CONFIG(tooltip)
        self.bInfo.setText("")
#if QT_CONFIG(tooltip)
        self.bAbout.setToolTip(QCoreApplication.translate("Q7ControlWindow", u"About....", None))
#endif // QT_CONFIG(tooltip)
        self.bAbout.setText(QCoreApplication.translate("Q7ControlWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bClose.setToolTip(QCoreApplication.translate("Q7ControlWindow", u"Close all CGNS.NAV windows", None))
#endif // QT_CONFIG(tooltip)
        self.bClose.setText("")
    # retranslateUi

