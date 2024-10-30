# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7VTKWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QHBoxLayout, QPushButton, QSizePolicy, QSpacerItem,
    QSpinBox, QVBoxLayout, QWidget)

from CGNS.NAV.Q7VTKRenderWindowInteractor import Q7VTKRenderWindowInteractor
from CGNS.NAV.wvtkutils import Q7ComboBox
from . import Res_rc

class Ui_Q7VTKWindow(object):
    def setupUi(self, Q7VTKWindow):
        if not Q7VTKWindow.objectName():
            Q7VTKWindow.setObjectName(u"Q7VTKWindow")
        Q7VTKWindow.resize(700, 402)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7VTKWindow.sizePolicy().hasHeightForWidth())
        Q7VTKWindow.setSizePolicy(sizePolicy)
        Q7VTKWindow.setMinimumSize(QSize(700, 400))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7VTKWindow.setWindowIcon(icon)
        self.verticalLayout = QVBoxLayout(Q7VTKWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.cViews = QComboBox(Q7VTKWindow)
        self.cViews.setObjectName(u"cViews")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.cViews.sizePolicy().hasHeightForWidth())
        self.cViews.setSizePolicy(sizePolicy1)
        self.cViews.setEditable(True)
        self.cViews.setMaxCount(16)
        self.cViews.setInsertPolicy(QComboBox.InsertAtTop)

        self.horizontalLayout.addWidget(self.cViews)

        self.bAddView = QPushButton(Q7VTKWindow)
        self.bAddView.setObjectName(u"bAddView")
        self.bAddView.setMinimumSize(QSize(25, 25))
        self.bAddView.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/camera-add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bAddView.setIcon(icon1)

        self.horizontalLayout.addWidget(self.bAddView)

        self.bSaveView = QPushButton(Q7VTKWindow)
        self.bSaveView.setObjectName(u"bSaveView")
        self.bSaveView.setMinimumSize(QSize(25, 25))
        self.bSaveView.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/camera-snap.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSaveView.setIcon(icon2)

        self.horizontalLayout.addWidget(self.bSaveView)

        self.bRemoveView = QPushButton(Q7VTKWindow)
        self.bRemoveView.setObjectName(u"bRemoveView")
        self.bRemoveView.setMinimumSize(QSize(25, 25))
        self.bRemoveView.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/camera-remove.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bRemoveView.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bRemoveView)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_9)

        self.bColorMapMin = QPushButton(Q7VTKWindow)
        self.bColorMapMin.setObjectName(u"bColorMapMin")
        self.bColorMapMin.setMinimumSize(QSize(25, 25))
        self.bColorMapMin.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/colors-first.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bColorMapMin.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bColorMapMin)

        self.bColorMapMax = QPushButton(Q7VTKWindow)
        self.bColorMapMax.setObjectName(u"bColorMapMax")
        self.bColorMapMax.setMinimumSize(QSize(25, 25))
        self.bColorMapMax.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/colors-last.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bColorMapMax.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bColorMapMax)

        self.cVariables = QComboBox(Q7VTKWindow)
        self.cVariables.setObjectName(u"cVariables")

        self.horizontalLayout.addWidget(self.cVariables)

        self.cColorSpace = QComboBox(Q7VTKWindow)
        self.cColorSpace.setObjectName(u"cColorSpace")
        self.cColorSpace.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout.addWidget(self.cColorSpace)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_6)

        self.bSaveVTK = QPushButton(Q7VTKWindow)
        self.bSaveVTK.setObjectName(u"bSaveVTK")
        self.bSaveVTK.setMinimumSize(QSize(25, 25))
        self.bSaveVTK.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSaveVTK.setIcon(icon6)

        self.horizontalLayout.addWidget(self.bSaveVTK)

        self.bScreenShot = QPushButton(Q7VTKWindow)
        self.bScreenShot.setObjectName(u"bScreenShot")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.bScreenShot.sizePolicy().hasHeightForWidth())
        self.bScreenShot.setSizePolicy(sizePolicy2)
        self.bScreenShot.setMinimumSize(QSize(25, 25))
        self.bScreenShot.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/snapshot.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bScreenShot.setIcon(icon7)

        self.horizontalLayout.addWidget(self.bScreenShot)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.line = QFrame(Q7VTKWindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.bX = QPushButton(Q7VTKWindow)
        self.bX.setObjectName(u"bX")
        sizePolicy2.setHeightForWidth(self.bX.sizePolicy().hasHeightForWidth())
        self.bX.setSizePolicy(sizePolicy2)
        self.bX.setMinimumSize(QSize(25, 25))
        self.bX.setMaximumSize(QSize(25, 25))

        self.horizontalLayout_3.addWidget(self.bX)

        self.bY = QPushButton(Q7VTKWindow)
        self.bY.setObjectName(u"bY")
        sizePolicy2.setHeightForWidth(self.bY.sizePolicy().hasHeightForWidth())
        self.bY.setSizePolicy(sizePolicy2)
        self.bY.setMinimumSize(QSize(25, 25))
        self.bY.setMaximumSize(QSize(25, 25))

        self.horizontalLayout_3.addWidget(self.bY)

        self.bZ = QPushButton(Q7VTKWindow)
        self.bZ.setObjectName(u"bZ")
        sizePolicy2.setHeightForWidth(self.bZ.sizePolicy().hasHeightForWidth())
        self.bZ.setSizePolicy(sizePolicy2)
        self.bZ.setMinimumSize(QSize(25, 25))
        self.bZ.setMaximumSize(QSize(25, 25))

        self.horizontalLayout_3.addWidget(self.bZ)

        self.cMirror = QCheckBox(Q7VTKWindow)
        self.cMirror.setObjectName(u"cMirror")

        self.horizontalLayout_3.addWidget(self.cMirror)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_10)

        self.bZoom = QPushButton(Q7VTKWindow)
        self.bZoom.setObjectName(u"bZoom")
        self.bZoom.setMinimumSize(QSize(25, 25))
        self.bZoom.setMaximumSize(QSize(25, 25))
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/zoompoint.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bZoom.setIcon(icon8)
        self.bZoom.setCheckable(True)

        self.horizontalLayout_3.addWidget(self.bZoom)

        self.selectable = QPushButton(Q7VTKWindow)
        self.selectable.setObjectName(u"selectable")
        self.selectable.setMinimumSize(QSize(25, 25))
        self.selectable.setMaximumSize(QSize(25, 25))
        self.selectable.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        icon9 = QIcon()
        icon9.addFile(u":/images/icons/lock-legend.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.selectable.setIcon(icon9)
        self.selectable.setCheckable(True)

        self.horizontalLayout_3.addWidget(self.selectable)

        self.cShowValue = QPushButton(Q7VTKWindow)
        self.cShowValue.setObjectName(u"cShowValue")
        self.cShowValue.setMinimumSize(QSize(25, 25))
        self.cShowValue.setMaximumSize(QSize(25, 25))
        icon10 = QIcon()
        icon10.addFile(u":/images/icons/value.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.cShowValue.setIcon(icon10)
        self.cShowValue.setCheckable(True)

        self.horizontalLayout_3.addWidget(self.cShowValue)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_3)

        self.bSuffleColors = QPushButton(Q7VTKWindow)
        self.bSuffleColors.setObjectName(u"bSuffleColors")
        sizePolicy2.setHeightForWidth(self.bSuffleColors.sizePolicy().hasHeightForWidth())
        self.bSuffleColors.setSizePolicy(sizePolicy2)
        self.bSuffleColors.setMinimumSize(QSize(25, 25))
        self.bSuffleColors.setMaximumSize(QSize(25, 25))
        icon11 = QIcon()
        icon11.addFile(u":/images/icons/colors.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSuffleColors.setIcon(icon11)

        self.horizontalLayout_3.addWidget(self.bSuffleColors)

        self.bBlackColor = QPushButton(Q7VTKWindow)
        self.bBlackColor.setObjectName(u"bBlackColor")
        sizePolicy2.setHeightForWidth(self.bBlackColor.sizePolicy().hasHeightForWidth())
        self.bBlackColor.setSizePolicy(sizePolicy2)
        self.bBlackColor.setMinimumSize(QSize(25, 25))
        self.bBlackColor.setMaximumSize(QSize(25, 25))
        icon12 = QIcon()
        icon12.addFile(u":/images/icons/colors-bw.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBlackColor.setIcon(icon12)

        self.horizontalLayout_3.addWidget(self.bBlackColor)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_4)

        self.bAnimateWindow = QPushButton(Q7VTKWindow)
        self.bAnimateWindow.setObjectName(u"bAnimateWindow")
        self.bAnimateWindow.setMinimumSize(QSize(25, 25))
        self.bAnimateWindow.setMaximumSize(QSize(25, 25))
        icon13 = QIcon()
        icon13.addFile(u":/images/icons/anim-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bAnimateWindow.setIcon(icon13)

        self.horizontalLayout_3.addWidget(self.bAnimateWindow)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_7)

        self.bResetCamera = QPushButton(Q7VTKWindow)
        self.bResetCamera.setObjectName(u"bResetCamera")
        self.bResetCamera.setMinimumSize(QSize(25, 25))
        self.bResetCamera.setMaximumSize(QSize(25, 25))
        icon14 = QIcon()
        icon14.addFile(u":/images/icons/zoom-actor.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bResetCamera.setIcon(icon14)

        self.horizontalLayout_3.addWidget(self.bResetCamera)

        self.cRotationAxis = QComboBox(Q7VTKWindow)
        self.cRotationAxis.setObjectName(u"cRotationAxis")

        self.horizontalLayout_3.addWidget(self.cRotationAxis)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.display = Q7VTKRenderWindowInteractor(Q7VTKWindow)
        self.display.setObjectName(u"display")
        sizePolicy.setHeightForWidth(self.display.sizePolicy().hasHeightForWidth())
        self.display.setSizePolicy(sizePolicy)

        self.verticalLayout_2.addWidget(self.display)


        self.verticalLayout.addLayout(self.verticalLayout_2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.bUpdateFromTree = QPushButton(Q7VTKWindow)
        self.bUpdateFromTree.setObjectName(u"bUpdateFromTree")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(25)
        sizePolicy3.setVerticalStretch(25)
        sizePolicy3.setHeightForWidth(self.bUpdateFromTree.sizePolicy().hasHeightForWidth())
        self.bUpdateFromTree.setSizePolicy(sizePolicy3)
        self.bUpdateFromTree.setMinimumSize(QSize(25, 25))
        self.bUpdateFromTree.setMaximumSize(QSize(25, 25))
        icon15 = QIcon()
        icon15.addFile(u":/images/icons/user-U.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bUpdateFromTree.setIcon(icon15)

        self.horizontalLayout_4.addWidget(self.bUpdateFromTree)

        self.cCurrentPath = Q7ComboBox(Q7VTKWindow)
        self.cCurrentPath.setObjectName(u"cCurrentPath")
        sizePolicy1.setHeightForWidth(self.cCurrentPath.sizePolicy().hasHeightForWidth())
        self.cCurrentPath.setSizePolicy(sizePolicy1)
        self.cCurrentPath.setEditable(False)
        self.cCurrentPath.setMinimumContentsLength(100)

        self.horizontalLayout_4.addWidget(self.cCurrentPath)

        self.bUpdateFromVTK = QPushButton(Q7VTKWindow)
        self.bUpdateFromVTK.setObjectName(u"bUpdateFromVTK")
        self.bUpdateFromVTK.setMinimumSize(QSize(25, 25))
        self.bUpdateFromVTK.setMaximumSize(QSize(25, 25))
        self.bUpdateFromVTK.setIcon(icon15)

        self.horizontalLayout_4.addWidget(self.bUpdateFromVTK)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.line_2 = QFrame(Q7VTKWindow)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.bBackControl = QPushButton(Q7VTKWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon16 = QIcon()
        icon16.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon16)

        self.horizontalLayout_2.addWidget(self.bBackControl)

        self.bInfo = QPushButton(Q7VTKWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon17 = QIcon()
        icon17.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon17)

        self.horizontalLayout_2.addWidget(self.bInfo)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_5)

        self.sIndex1 = QSpinBox(Q7VTKWindow)
        self.sIndex1.setObjectName(u"sIndex1")

        self.horizontalLayout_2.addWidget(self.sIndex1)

        self.sIndex2 = QSpinBox(Q7VTKWindow)
        self.sIndex2.setObjectName(u"sIndex2")

        self.horizontalLayout_2.addWidget(self.sIndex2)

        self.sIndex3 = QSpinBox(Q7VTKWindow)
        self.sIndex3.setObjectName(u"sIndex3")

        self.horizontalLayout_2.addWidget(self.sIndex3)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.cShowFamily = QCheckBox(Q7VTKWindow)
        self.cShowFamily.setObjectName(u"cShowFamily")
        self.cShowFamily.setChecked(True)

        self.horizontalLayout_2.addWidget(self.cShowFamily)

        self.cShowZone = QCheckBox(Q7VTKWindow)
        self.cShowZone.setObjectName(u"cShowZone")
        self.cShowZone.setChecked(True)

        self.horizontalLayout_2.addWidget(self.cShowZone)

        self.cShowBC = QCheckBox(Q7VTKWindow)
        self.cShowBC.setObjectName(u"cShowBC")
        self.cShowBC.setChecked(False)

        self.horizontalLayout_2.addWidget(self.cShowBC)

        self.cShowCT = QCheckBox(Q7VTKWindow)
        self.cShowCT.setObjectName(u"cShowCT")
        self.cShowCT.setChecked(False)

        self.horizontalLayout_2.addWidget(self.cShowCT)

        self.bPrevious = QPushButton(Q7VTKWindow)
        self.bPrevious.setObjectName(u"bPrevious")
        self.bPrevious.setMinimumSize(QSize(25, 25))
        self.bPrevious.setMaximumSize(QSize(25, 25))
        icon18 = QIcon()
        icon18.addFile(u":/images/icons/control.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bPrevious.setIcon(icon18)

        self.horizontalLayout_2.addWidget(self.bPrevious)

        self.bReset = QPushButton(Q7VTKWindow)
        self.bReset.setObjectName(u"bReset")
        self.bReset.setMinimumSize(QSize(25, 25))
        self.bReset.setMaximumSize(QSize(25, 25))
        icon19 = QIcon()
        icon19.addFile(u":/images/icons/node-sids-leaf.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bReset.setIcon(icon19)

        self.horizontalLayout_2.addWidget(self.bReset)

        self.bNext = QPushButton(Q7VTKWindow)
        self.bNext.setObjectName(u"bNext")
        self.bNext.setMinimumSize(QSize(25, 25))
        self.bNext.setMaximumSize(QSize(25, 25))
        icon20 = QIcon()
        icon20.addFile(u":/images/icons/node-sids-closed.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bNext.setIcon(icon20)

        self.horizontalLayout_2.addWidget(self.bNext)


        self.verticalLayout.addLayout(self.horizontalLayout_2)


        self.retranslateUi(Q7VTKWindow)

        QMetaObject.connectSlotsByName(Q7VTKWindow)
    # setupUi

    def retranslateUi(self, Q7VTKWindow):
        Q7VTKWindow.setWindowTitle(QCoreApplication.translate("Q7VTKWindow", u"Form", None))
#if QT_CONFIG(tooltip)
        Q7VTKWindow.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Show value", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.bAddView.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Add current view to view list", None))
#endif // QT_CONFIG(tooltip)
        self.bAddView.setText("")
#if QT_CONFIG(tooltip)
        self.bSaveView.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Write view list into a file", None))
#endif // QT_CONFIG(tooltip)
        self.bSaveView.setText("")
#if QT_CONFIG(tooltip)
        self.bRemoveView.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Remove current view from view list", None))
#endif // QT_CONFIG(tooltip)
        self.bRemoveView.setText("")
#if QT_CONFIG(tooltip)
        self.bColorMapMin.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Set color for palette high bound", None))
#endif // QT_CONFIG(tooltip)
        self.bColorMapMin.setText("")
#if QT_CONFIG(tooltip)
        self.bColorMapMax.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Set color for palette low bound", None))
#endif // QT_CONFIG(tooltip)
        self.bColorMapMax.setText("")
#if QT_CONFIG(tooltip)
        self.bSaveVTK.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Save VTK data into a file", None))
#endif // QT_CONFIG(tooltip)
        self.bSaveVTK.setText("")
#if QT_CONFIG(tooltip)
        self.bScreenShot.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Save view snapshot into a file", None))
#endif // QT_CONFIG(tooltip)
        self.bScreenShot.setText("")
#if QT_CONFIG(tooltip)
        self.bX.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Show Y/Z plane", None))
#endif // QT_CONFIG(tooltip)
        self.bX.setText(QCoreApplication.translate("Q7VTKWindow", u"X", None))
#if QT_CONFIG(tooltip)
        self.bY.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Show X/Z plane", None))
#endif // QT_CONFIG(tooltip)
        self.bY.setText(QCoreApplication.translate("Q7VTKWindow", u"Y", None))
#if QT_CONFIG(tooltip)
        self.bZ.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Show X/Y plane", None))
#endif // QT_CONFIG(tooltip)
        self.bZ.setText(QCoreApplication.translate("Q7VTKWindow", u"Z", None))
        self.cMirror.setText(QCoreApplication.translate("Q7VTKWindow", u"Mirror", None))
#if QT_CONFIG(tooltip)
        self.bZoom.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Activate Mouse Zoom mode", None))
#endif // QT_CONFIG(tooltip)
        self.bZoom.setText("")
#if QT_CONFIG(tooltip)
        self.selectable.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Acticate Move Legend Mode", None))
#endif // QT_CONFIG(tooltip)
        self.selectable.setText("")
#if QT_CONFIG(tooltip)
        self.cShowValue.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Activate Show Value Mode", None))
#endif // QT_CONFIG(tooltip)
        self.cShowValue.setText("")
#if QT_CONFIG(tooltip)
        self.bSuffleColors.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Change colors to random", None))
#endif // QT_CONFIG(tooltip)
        self.bSuffleColors.setText("")
#if QT_CONFIG(tooltip)
        self.bBlackColor.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Switch black/white colors", None))
#endif // QT_CONFIG(tooltip)
        self.bBlackColor.setText("")
        self.bAnimateWindow.setText("")
#if QT_CONFIG(tooltip)
        self.bResetCamera.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Zoom and center on selected", None))
#endif // QT_CONFIG(tooltip)
        self.bResetCamera.setText("")
#if QT_CONFIG(tooltip)
        self.cRotationAxis.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Rotation axis", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.bUpdateFromTree.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Reset VTK view selection from tree view FIRST selected node", None))
#endif // QT_CONFIG(tooltip)
        self.bUpdateFromTree.setText("")
#if QT_CONFIG(tooltip)
        self.bUpdateFromVTK.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Reset tree view selection from ALL VTK view selected objects", None))
#endif // QT_CONFIG(tooltip)
        self.bUpdateFromVTK.setText("")
#if QT_CONFIG(tooltip)
        self.bBackControl.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Raise CGNS.NAV control window", None))
#endif // QT_CONFIG(tooltip)
        self.bBackControl.setText("")
        self.bInfo.setText("")
#if QT_CONFIG(tooltip)
        self.sIndex3.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Get/Set index for third dim", None))
#endif // QT_CONFIG(tooltip)
        self.cShowFamily.setText(QCoreApplication.translate("Q7VTKWindow", u"Families", None))
        self.cShowZone.setText(QCoreApplication.translate("Q7VTKWindow", u"Zones", None))
        self.cShowBC.setText(QCoreApplication.translate("Q7VTKWindow", u"BCs", None))
        self.cShowCT.setText(QCoreApplication.translate("Q7VTKWindow", u"CTs", None))
#if QT_CONFIG(tooltip)
        self.bPrevious.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Highlight previous selected item", None))
#endif // QT_CONFIG(tooltip)
        self.bPrevious.setText("")
#if QT_CONFIG(tooltip)
        self.bReset.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Clear selection list", None))
#endif // QT_CONFIG(tooltip)
        self.bReset.setText("")
#if QT_CONFIG(tooltip)
        self.bNext.setToolTip(QCoreApplication.translate("Q7VTKWindow", u"Highlight next selected item", None))
#endif // QT_CONFIG(tooltip)
        self.bNext.setText("")
    # retranslateUi

