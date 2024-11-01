# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7TreeWindow.ui'
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
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QToolButton, QVBoxLayout, QWidget)

from CGNS.NAV.mtree import Q7TreeView
from . import Res_rc

class Ui_Q7TreeWindow(object):
    def setupUi(self, Q7TreeWindow):
        if not Q7TreeWindow.objectName():
            Q7TreeWindow.setObjectName(u"Q7TreeWindow")
        Q7TreeWindow.resize(1124, 300)
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7TreeWindow.setWindowIcon(icon)
        self.verticalLayout_2 = QVBoxLayout(Q7TreeWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bSave = QToolButton(Q7TreeWindow)
        self.bSave.setObjectName(u"bSave")
        self.bSave.setMinimumSize(QSize(25, 25))
        self.bSave.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSave.setIcon(icon1)

        self.horizontalLayout.addWidget(self.bSave)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.bSaveAs = QToolButton(Q7TreeWindow)
        self.bSaveAs.setObjectName(u"bSaveAs")
        self.bSaveAs.setMinimumSize(QSize(25, 25))
        self.bSaveAs.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/tree-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSaveAs.setIcon(icon2)

        self.horizontalLayout.addWidget(self.bSaveAs)

        self.bPatternDB = QToolButton(Q7TreeWindow)
        self.bPatternDB.setObjectName(u"bPatternDB")
        self.bPatternDB.setMinimumSize(QSize(25, 25))
        self.bPatternDB.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/pattern-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bPatternDB.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bPatternDB)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.bZoomOut = QToolButton(Q7TreeWindow)
        self.bZoomOut.setObjectName(u"bZoomOut")
        self.bZoomOut.setMinimumSize(QSize(25, 25))
        self.bZoomOut.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/level-out.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bZoomOut.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bZoomOut)

        self.bZoomAll = QPushButton(Q7TreeWindow)
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

        self.bZoomIn = QToolButton(Q7TreeWindow)
        self.bZoomIn.setObjectName(u"bZoomIn")
        self.bZoomIn.setMinimumSize(QSize(25, 25))
        self.bZoomIn.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/level-in.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bZoomIn.setIcon(icon6)

        self.horizontalLayout.addWidget(self.bZoomIn)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bMarkAll = QToolButton(Q7TreeWindow)
        self.bMarkAll.setObjectName(u"bMarkAll")
        self.bMarkAll.setMinimumSize(QSize(25, 25))
        self.bMarkAll.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/flag-all.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bMarkAll.setIcon(icon7)

        self.horizontalLayout.addWidget(self.bMarkAll)

        self.bSwapMarks = QToolButton(Q7TreeWindow)
        self.bSwapMarks.setObjectName(u"bSwapMarks")
        self.bSwapMarks.setMinimumSize(QSize(25, 25))
        self.bSwapMarks.setMaximumSize(QSize(25, 25))
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/flag-revert.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSwapMarks.setIcon(icon8)

        self.horizontalLayout.addWidget(self.bSwapMarks)

        self.bUnmarkAll_2 = QPushButton(Q7TreeWindow)
        self.bUnmarkAll_2.setObjectName(u"bUnmarkAll_2")
        self.bUnmarkAll_2.setMinimumSize(QSize(25, 25))
        self.bUnmarkAll_2.setMaximumSize(QSize(25, 25))
        icon9 = QIcon()
        icon9.addFile(u":/images/icons/flag-none.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bUnmarkAll_2.setIcon(icon9)

        self.horizontalLayout.addWidget(self.bUnmarkAll_2)

        self.bMarksAsList = QToolButton(Q7TreeWindow)
        self.bMarksAsList.setObjectName(u"bMarksAsList")
        self.bMarksAsList.setMinimumSize(QSize(25, 25))
        self.bMarksAsList.setMaximumSize(QSize(25, 25))
        icon10 = QIcon()
        icon10.addFile(u":/images/icons/operate-list.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bMarksAsList.setIcon(icon10)

        self.horizontalLayout.addWidget(self.bMarksAsList)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.bCheck = QToolButton(Q7TreeWindow)
        self.bCheck.setObjectName(u"bCheck")
        self.bCheck.setMinimumSize(QSize(25, 25))
        self.bCheck.setMaximumSize(QSize(25, 25))
        icon11 = QIcon()
        icon11.addFile(u":/images/icons/check-all.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bCheck.setIcon(icon11)

        self.horizontalLayout.addWidget(self.bCheck)

        self.bClearChecks = QToolButton(Q7TreeWindow)
        self.bClearChecks.setObjectName(u"bClearChecks")
        self.bClearChecks.setMinimumSize(QSize(25, 25))
        self.bClearChecks.setMaximumSize(QSize(25, 25))
        icon12 = QIcon()
        icon12.addFile(u":/images/icons/check-clear.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bClearChecks.setIcon(icon12)

        self.horizontalLayout.addWidget(self.bClearChecks)

        self.bCheckList = QToolButton(Q7TreeWindow)
        self.bCheckList.setObjectName(u"bCheckList")
        self.bCheckList.setMinimumSize(QSize(25, 25))
        self.bCheckList.setMaximumSize(QSize(25, 25))
        icon13 = QIcon()
        icon13.addFile(u":/images/icons/check-list.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bCheckList.setIcon(icon13)

        self.horizontalLayout.addWidget(self.bCheckList)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_5)

        self.bSelectLinkSrc = QPushButton(Q7TreeWindow)
        self.bSelectLinkSrc.setObjectName(u"bSelectLinkSrc")
        self.bSelectLinkSrc.setMinimumSize(QSize(25, 25))
        self.bSelectLinkSrc.setMaximumSize(QSize(25, 25))
        icon14 = QIcon()
        icon14.addFile(u":/images/icons/link-src.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSelectLinkSrc.setIcon(icon14)
        self.bSelectLinkSrc.setCheckable(True)

        self.horizontalLayout.addWidget(self.bSelectLinkSrc)

        self.bSelectLinkDst = QPushButton(Q7TreeWindow)
        self.bSelectLinkDst.setObjectName(u"bSelectLinkDst")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(25)
        sizePolicy1.setVerticalStretch(25)
        sizePolicy1.setHeightForWidth(self.bSelectLinkDst.sizePolicy().hasHeightForWidth())
        self.bSelectLinkDst.setSizePolicy(sizePolicy1)
        self.bSelectLinkDst.setMinimumSize(QSize(25, 25))
        self.bSelectLinkDst.setMaximumSize(QSize(25, 25))
        icon15 = QIcon()
        icon15.addFile(u":/images/icons/link-dst.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSelectLinkDst.setIcon(icon15)
        self.bSelectLinkDst.setCheckable(True)

        self.horizontalLayout.addWidget(self.bSelectLinkDst)

        self.bAddLink = QPushButton(Q7TreeWindow)
        self.bAddLink.setObjectName(u"bAddLink")
        self.bAddLink.setMinimumSize(QSize(25, 25))
        self.bAddLink.setMaximumSize(QSize(25, 25))
        icon16 = QIcon()
        icon16.addFile(u":/images/icons/link-add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bAddLink.setIcon(icon16)

        self.horizontalLayout.addWidget(self.bAddLink)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_11)

        self.bToolsView = QPushButton(Q7TreeWindow)
        self.bToolsView.setObjectName(u"bToolsView")
        self.bToolsView.setMinimumSize(QSize(25, 25))
        self.bToolsView.setMaximumSize(QSize(25, 25))
        icon17 = QIcon()
        icon17.addFile(u":/images/icons/tools.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bToolsView.setIcon(icon17)

        self.horizontalLayout.addWidget(self.bToolsView)

        self.bFormView = QPushButton(Q7TreeWindow)
        self.bFormView.setObjectName(u"bFormView")
        sizePolicy.setHeightForWidth(self.bFormView.sizePolicy().hasHeightForWidth())
        self.bFormView.setSizePolicy(sizePolicy)
        self.bFormView.setMinimumSize(QSize(25, 25))
        self.bFormView.setMaximumSize(QSize(25, 25))
        icon18 = QIcon()
        icon18.addFile(u":/images/icons/form-open.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bFormView.setIcon(icon18)

        self.horizontalLayout.addWidget(self.bFormView)

        self.bVTKView = QToolButton(Q7TreeWindow)
        self.bVTKView.setObjectName(u"bVTKView")
        self.bVTKView.setMinimumSize(QSize(25, 25))
        self.bVTKView.setMaximumSize(QSize(25, 25))
        icon19 = QIcon()
        icon19.addFile(u":/images/icons/vtk.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bVTKView.setIcon(icon19)

        self.horizontalLayout.addWidget(self.bVTKView)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_10)

        self.bPatternView = QPushButton(Q7TreeWindow)
        self.bPatternView.setObjectName(u"bPatternView")
        self.bPatternView.setMinimumSize(QSize(25, 25))
        self.bPatternView.setMaximumSize(QSize(25, 25))
        icon20 = QIcon()
        icon20.addFile(u":/images/icons/pattern-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bPatternView.setIcon(icon20)

        self.horizontalLayout.addWidget(self.bPatternView)

        self.bLinkView = QToolButton(Q7TreeWindow)
        self.bLinkView.setObjectName(u"bLinkView")
        self.bLinkView.setMinimumSize(QSize(25, 25))
        self.bLinkView.setMaximumSize(QSize(25, 25))
        icon21 = QIcon()
        icon21.addFile(u":/images/icons/link-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bLinkView.setIcon(icon21)

        self.horizontalLayout.addWidget(self.bLinkView)

        self.bQueryView = QToolButton(Q7TreeWindow)
        self.bQueryView.setObjectName(u"bQueryView")
        self.bQueryView.setMinimumSize(QSize(25, 25))
        self.bQueryView.setMaximumSize(QSize(25, 25))
        icon22 = QIcon()
        icon22.addFile(u":/images/icons/operate-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bQueryView.setIcon(icon22)

        self.horizontalLayout.addWidget(self.bQueryView)

        self.bCheckView = QToolButton(Q7TreeWindow)
        self.bCheckView.setObjectName(u"bCheckView")
        self.bCheckView.setMinimumSize(QSize(25, 25))
        self.bCheckView.setMaximumSize(QSize(25, 25))
        icon23 = QIcon()
        icon23.addFile(u":/images/icons/check-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bCheckView.setIcon(icon23)

        self.horizontalLayout.addWidget(self.bCheckView)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_7)

        self.bScreenShot = QToolButton(Q7TreeWindow)
        self.bScreenShot.setObjectName(u"bScreenShot")
        self.bScreenShot.setMinimumSize(QSize(25, 25))
        self.bScreenShot.setMaximumSize(QSize(25, 25))
        icon24 = QIcon()
        icon24.addFile(u":/images/icons/snapshot.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bScreenShot.setIcon(icon24)

        self.horizontalLayout.addWidget(self.bScreenShot)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.treeview = Q7TreeView(Q7TreeWindow)
        self.treeview.setObjectName(u"treeview")
        self.treeview.viewport().setProperty(u"cursor", QCursor(Qt.CursorShape.CrossCursor))
        self.treeview.setAutoScroll(False)
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
        self.bBackControl = QPushButton(Q7TreeWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon25 = QIcon()
        icon25.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon25)

        self.horizontalLayout_3.addWidget(self.bBackControl)

        self.bInfo = QPushButton(Q7TreeWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon26 = QIcon()
        icon26.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon26)

        self.horizontalLayout_3.addWidget(self.bInfo)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_8)

        self.lineEditLock = QPushButton(Q7TreeWindow)
        self.lineEditLock.setObjectName(u"lineEditLock")
        self.lineEditLock.setMinimumSize(QSize(25, 25))
        self.lineEditLock.setMaximumSize(QSize(25, 25))
        icon27 = QIcon()
        icon27.addFile(u":/images/icons/optional-sids-node.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon27.addFile(u":/images/icons/lock-scroll.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.lineEditLock.setIcon(icon27)
        self.lineEditLock.setCheckable(True)

        self.horizontalLayout_3.addWidget(self.lineEditLock)

        self.lineEdit = QLineEdit(Q7TreeWindow)
        self.lineEdit.setObjectName(u"lineEdit")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy2)
        self.lineEdit.setMinimumSize(QSize(0, 0))

        self.horizontalLayout_3.addWidget(self.lineEdit)

        self.bPreviousMark = QToolButton(Q7TreeWindow)
        self.bPreviousMark.setObjectName(u"bPreviousMark")
        self.bPreviousMark.setMinimumSize(QSize(25, 25))
        self.bPreviousMark.setMaximumSize(QSize(25, 25))
        icon28 = QIcon()
        icon28.addFile(u":/images/icons/node-sids-opened.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bPreviousMark.setIcon(icon28)

        self.horizontalLayout_3.addWidget(self.bPreviousMark)

        self.bUnmarkAll_1 = QToolButton(Q7TreeWindow)
        self.bUnmarkAll_1.setObjectName(u"bUnmarkAll_1")
        self.bUnmarkAll_1.setMinimumSize(QSize(25, 25))
        self.bUnmarkAll_1.setMaximumSize(QSize(25, 25))
        icon29 = QIcon()
        icon29.addFile(u":/images/icons/node-sids-leaf.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bUnmarkAll_1.setIcon(icon29)

        self.horizontalLayout_3.addWidget(self.bUnmarkAll_1)

        self.bNextMark = QToolButton(Q7TreeWindow)
        self.bNextMark.setObjectName(u"bNextMark")
        self.bNextMark.setMinimumSize(QSize(25, 25))
        self.bNextMark.setMaximumSize(QSize(25, 25))
        icon30 = QIcon()
        icon30.addFile(u":/images/icons/node-sids-closed.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bNextMark.setIcon(icon30)

        self.horizontalLayout_3.addWidget(self.bNextMark)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)


        self.retranslateUi(Q7TreeWindow)

        QMetaObject.connectSlotsByName(Q7TreeWindow)
    # setupUi

    def retranslateUi(self, Q7TreeWindow):
        Q7TreeWindow.setWindowTitle(QCoreApplication.translate("Q7TreeWindow", u"Form", None))
#if QT_CONFIG(tooltip)
        self.bSave.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Save tree (overwrite existing file)", None))
#endif // QT_CONFIG(tooltip)
        self.bSave.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bSaveAs.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Save tree as (creates a new file)", None))
#endif // QT_CONFIG(tooltip)
        self.bSaveAs.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bPatternDB.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Save tree as profile", None))
#endif // QT_CONFIG(tooltip)
        self.bPatternDB.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bZoomOut.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Collapse lowest tree level", None))
#endif // QT_CONFIG(tooltip)
        self.bZoomOut.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bZoomAll.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Expand all tree", None))
#endif // QT_CONFIG(tooltip)
        self.bZoomAll.setText("")
#if QT_CONFIG(tooltip)
        self.bZoomIn.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Expand lowest tree level", None))
#endif // QT_CONFIG(tooltip)
        self.bZoomIn.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bMarkAll.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Mark all nodes", None))
#endif // QT_CONFIG(tooltip)
        self.bMarkAll.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bSwapMarks.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Swap marked/unmarked nodes", None))
#endif // QT_CONFIG(tooltip)
        self.bSwapMarks.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
        self.bUnmarkAll_2.setText("")
#if QT_CONFIG(tooltip)
        self.bMarksAsList.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Open selected nodes list", None))
#endif // QT_CONFIG(tooltip)
        self.bMarksAsList.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bCheck.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Check selected nodes", None))
#endif // QT_CONFIG(tooltip)
        self.bCheck.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bClearChecks.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Remove check labels", None))
#endif // QT_CONFIG(tooltip)
        self.bClearChecks.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bCheckList.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Open diagnostics list", None))
#endif // QT_CONFIG(tooltip)
        self.bCheckList.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bSelectLinkSrc.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Add selected node as parent to link source", None))
#endif // QT_CONFIG(tooltip)
        self.bSelectLinkSrc.setText("")
#if QT_CONFIG(tooltip)
        self.bSelectLinkDst.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Add selected node as link destination", None))
#endif // QT_CONFIG(tooltip)
        self.bSelectLinkDst.setText("")
#if QT_CONFIG(tooltip)
        self.bAddLink.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Create a link using selected link source/destination", None))
#endif // QT_CONFIG(tooltip)
        self.bAddLink.setText("")
#if QT_CONFIG(tooltip)
        self.bToolsView.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Open Tools view", None))
#endif // QT_CONFIG(tooltip)
        self.bToolsView.setText("")
#if QT_CONFIG(tooltip)
        self.bFormView.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Open Form view on selected node", None))
#endif // QT_CONFIG(tooltip)
        self.bFormView.setText("")
#if QT_CONFIG(tooltip)
        self.bVTKView.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Open VTK view", None))
#endif // QT_CONFIG(tooltip)
        self.bVTKView.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bPatternView.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Open Patterns view", None))
#endif // QT_CONFIG(tooltip)
        self.bPatternView.setText("")
#if QT_CONFIG(tooltip)
        self.bLinkView.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Open Links view", None))
#endif // QT_CONFIG(tooltip)
        self.bLinkView.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bQueryView.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Open Queries view", None))
#endif // QT_CONFIG(tooltip)
        self.bQueryView.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bCheckView.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Open Check view", None))
#endif // QT_CONFIG(tooltip)
        self.bCheckView.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bScreenShot.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Save tree view snapshot", None))
#endif // QT_CONFIG(tooltip)
        self.bScreenShot.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bBackControl.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Raise Control window", None))
#endif // QT_CONFIG(tooltip)
        self.bBackControl.setText("")
        self.bInfo.setText("")
#if QT_CONFIG(tooltip)
        self.lineEditLock.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Toggle tree selection updates the edit line", None))
#endif // QT_CONFIG(tooltip)
        self.lineEditLock.setText("")
#if QT_CONFIG(tooltip)
        self.bPreviousMark.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Select previous marked node", None))
#endif // QT_CONFIG(tooltip)
        self.bPreviousMark.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bUnmarkAll_1.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Unmark all nodes", None))
#endif // QT_CONFIG(tooltip)
        self.bUnmarkAll_1.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
#if QT_CONFIG(tooltip)
        self.bNextMark.setToolTip(QCoreApplication.translate("Q7TreeWindow", u"Select next marked node", None))
#endif // QT_CONFIG(tooltip)
        self.bNextMark.setText(QCoreApplication.translate("Q7TreeWindow", u"...", None))
    # retranslateUi

