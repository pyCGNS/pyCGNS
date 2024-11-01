# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7VTKPlotWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
    QPushButton, QSizePolicy, QSpacerItem, QSpinBox,
    QVBoxLayout, QWidget)

from CGNS.NAV.Q7VTKRenderWindowInteractor import Q7VTKRenderWindowInteractor
from . import Res_rc

class Ui_Q7VTKPlotWindow(object):
    def setupUi(self, Q7VTKPlotWindow):
        if not Q7VTKPlotWindow.objectName():
            Q7VTKPlotWindow.setObjectName(u"Q7VTKPlotWindow")
        Q7VTKPlotWindow.resize(803, 679)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7VTKPlotWindow.sizePolicy().hasHeightForWidth())
        Q7VTKPlotWindow.setSizePolicy(sizePolicy)
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7VTKPlotWindow.setWindowIcon(icon)
        self.verticalLayout = QVBoxLayout(Q7VTKPlotWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.cViews = QComboBox(Q7VTKPlotWindow)
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

        self.bAddView = QPushButton(Q7VTKPlotWindow)
        self.bAddView.setObjectName(u"bAddView")
        self.bAddView.setMinimumSize(QSize(25, 25))
        self.bAddView.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/camera-add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bAddView.setIcon(icon1)

        self.horizontalLayout.addWidget(self.bAddView)

        self.bSaveView = QPushButton(Q7VTKPlotWindow)
        self.bSaveView.setObjectName(u"bSaveView")
        self.bSaveView.setMinimumSize(QSize(25, 25))
        self.bSaveView.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/camera-snap.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSaveView.setIcon(icon2)

        self.horizontalLayout.addWidget(self.bSaveView)

        self.bRemoveView = QPushButton(Q7VTKPlotWindow)
        self.bRemoveView.setObjectName(u"bRemoveView")
        self.bRemoveView.setMinimumSize(QSize(25, 25))
        self.bRemoveView.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/camera-remove.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bRemoveView.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bRemoveView)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.bSaveVTK = QPushButton(Q7VTKPlotWindow)
        self.bSaveVTK.setObjectName(u"bSaveVTK")
        self.bSaveVTK.setMinimumSize(QSize(25, 25))
        self.bSaveVTK.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSaveVTK.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bSaveVTK)

        self.bScreenShot = QPushButton(Q7VTKPlotWindow)
        self.bScreenShot.setObjectName(u"bScreenShot")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.bScreenShot.sizePolicy().hasHeightForWidth())
        self.bScreenShot.setSizePolicy(sizePolicy2)
        self.bScreenShot.setMinimumSize(QSize(25, 25))
        self.bScreenShot.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/snapshot.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bScreenShot.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bScreenShot)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.line = QFrame(Q7VTKPlotWindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.bX = QPushButton(Q7VTKPlotWindow)
        self.bX.setObjectName(u"bX")
        sizePolicy2.setHeightForWidth(self.bX.sizePolicy().hasHeightForWidth())
        self.bX.setSizePolicy(sizePolicy2)
        self.bX.setMinimumSize(QSize(25, 25))
        self.bX.setMaximumSize(QSize(25, 25))

        self.horizontalLayout_3.addWidget(self.bX)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_7)

        self.cVariableX = QComboBox(Q7VTKPlotWindow)
        self.cVariableX.setObjectName(u"cVariableX")

        self.horizontalLayout_3.addWidget(self.cVariableX)

        self.cVariableY = QComboBox(Q7VTKPlotWindow)
        self.cVariableY.setObjectName(u"cVariableY")

        self.horizontalLayout_3.addWidget(self.cVariableY)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_6)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.display = Q7VTKRenderWindowInteractor(Q7VTKPlotWindow)
        self.display.setObjectName(u"display")
        sizePolicy.setHeightForWidth(self.display.sizePolicy().hasHeightForWidth())
        self.display.setSizePolicy(sizePolicy)

        self.verticalLayout_2.addWidget(self.display)


        self.verticalLayout.addLayout(self.verticalLayout_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.bBackControl = QPushButton(Q7VTKPlotWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon6)

        self.horizontalLayout_2.addWidget(self.bBackControl)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_5)

        self.sIndex1 = QSpinBox(Q7VTKPlotWindow)
        self.sIndex1.setObjectName(u"sIndex1")

        self.horizontalLayout_2.addWidget(self.sIndex1)

        self.sIndex2 = QSpinBox(Q7VTKPlotWindow)
        self.sIndex2.setObjectName(u"sIndex2")

        self.horizontalLayout_2.addWidget(self.sIndex2)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout_2)


        self.retranslateUi(Q7VTKPlotWindow)

        QMetaObject.connectSlotsByName(Q7VTKPlotWindow)
    # setupUi

    def retranslateUi(self, Q7VTKPlotWindow):
        Q7VTKPlotWindow.setWindowTitle(QCoreApplication.translate("Q7VTKPlotWindow", u"Form", None))
#if QT_CONFIG(tooltip)
        self.bAddView.setToolTip(QCoreApplication.translate("Q7VTKPlotWindow", u"Add current view to view list", None))
#endif // QT_CONFIG(tooltip)
        self.bAddView.setText("")
#if QT_CONFIG(tooltip)
        self.bSaveView.setToolTip(QCoreApplication.translate("Q7VTKPlotWindow", u"Write view list into a file", None))
#endif // QT_CONFIG(tooltip)
        self.bSaveView.setText("")
#if QT_CONFIG(tooltip)
        self.bRemoveView.setToolTip(QCoreApplication.translate("Q7VTKPlotWindow", u"Remove current view from view list", None))
#endif // QT_CONFIG(tooltip)
        self.bRemoveView.setText("")
#if QT_CONFIG(tooltip)
        self.bSaveVTK.setToolTip(QCoreApplication.translate("Q7VTKPlotWindow", u"Save VTK data into a file", None))
#endif // QT_CONFIG(tooltip)
        self.bSaveVTK.setText("")
#if QT_CONFIG(tooltip)
        self.bScreenShot.setToolTip(QCoreApplication.translate("Q7VTKPlotWindow", u"Save view snapshot into a file", None))
#endif // QT_CONFIG(tooltip)
        self.bScreenShot.setText("")
#if QT_CONFIG(tooltip)
        self.bX.setToolTip(QCoreApplication.translate("Q7VTKPlotWindow", u"Show Y/Z plane", None))
#endif // QT_CONFIG(tooltip)
        self.bX.setText(QCoreApplication.translate("Q7VTKPlotWindow", u"X/Y", None))
#if QT_CONFIG(tooltip)
        self.bBackControl.setToolTip(QCoreApplication.translate("Q7VTKPlotWindow", u"Raise CGNS.NAV control window", None))
#endif // QT_CONFIG(tooltip)
        self.bBackControl.setText("")
    # retranslateUi

