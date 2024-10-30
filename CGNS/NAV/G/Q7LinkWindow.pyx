# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7LinkWindow.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSpacerItem, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)
from . import Res_rc

class Ui_Q7LinkWindow(object):
    def setupUi(self, Q7LinkWindow):
        if not Q7LinkWindow.objectName():
            Q7LinkWindow.setObjectName(u"Q7LinkWindow")
        Q7LinkWindow.resize(715, 350)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7LinkWindow.sizePolicy().hasHeightForWidth())
        Q7LinkWindow.setSizePolicy(sizePolicy)
        Q7LinkWindow.setMinimumSize(QSize(715, 350))
        Q7LinkWindow.setMaximumSize(QSize(3000, 750))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7LinkWindow.setWindowIcon(icon)
        self.verticalLayout_2 = QVBoxLayout(Q7LinkWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bCheckLink = QPushButton(Q7LinkWindow)
        self.bCheckLink.setObjectName(u"bCheckLink")
        self.bCheckLink.setEnabled(True)
        self.bCheckLink.setMinimumSize(QSize(25, 25))
        self.bCheckLink.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/link-check.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bCheckLink.setIcon(icon1)

        self.horizontalLayout.addWidget(self.bCheckLink)

        self.bLoadTree = QPushButton(Q7LinkWindow)
        self.bLoadTree.setObjectName(u"bLoadTree")
        self.bLoadTree.setMinimumSize(QSize(25, 25))
        self.bLoadTree.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/tree-load-g.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bLoadTree.setIcon(icon2)

        self.horizontalLayout.addWidget(self.bLoadTree)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bAddLink = QPushButton(Q7LinkWindow)
        self.bAddLink.setObjectName(u"bAddLink")
        self.bAddLink.setMinimumSize(QSize(25, 25))
        self.bAddLink.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/link-add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bAddLink.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bAddLink)

        self.bDuplicateLink = QPushButton(Q7LinkWindow)
        self.bDuplicateLink.setObjectName(u"bDuplicateLink")
        self.bDuplicateLink.setMinimumSize(QSize(25, 25))
        self.bDuplicateLink.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/link-dup.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bDuplicateLink.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bDuplicateLink)

        self.bFromSelection = QPushButton(Q7LinkWindow)
        self.bFromSelection.setObjectName(u"bFromSelection")
        self.bFromSelection.setMinimumSize(QSize(25, 25))
        self.bFromSelection.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/link-slist.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bFromSelection.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bFromSelection)

        self.bDeleteLink = QPushButton(Q7LinkWindow)
        self.bDeleteLink.setObjectName(u"bDeleteLink")
        self.bDeleteLink.setMinimumSize(QSize(25, 25))
        self.bDeleteLink.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/link-delete.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bDeleteLink.setIcon(icon6)

        self.horizontalLayout.addWidget(self.bDeleteLink)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.bSave = QPushButton(Q7LinkWindow)
        self.bSave.setObjectName(u"bSave")
        self.bSave.setMinimumSize(QSize(25, 25))
        self.bSave.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/link-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSave.setIcon(icon7)

        self.horizontalLayout.addWidget(self.bSave)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.line = QFrame(Q7LinkWindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_2.addWidget(self.line)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_2 = QLabel(Q7LinkWindow)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setEnabled(True)

        self.horizontalLayout_4.addWidget(self.label_2)

        self.cUnreachable = QCheckBox(Q7LinkWindow)
        self.cUnreachable.setObjectName(u"cUnreachable")
        self.cUnreachable.setEnabled(True)

        self.horizontalLayout_4.addWidget(self.cUnreachable)

        self.cDuplicates = QCheckBox(Q7LinkWindow)
        self.cDuplicates.setObjectName(u"cDuplicates")
        self.cDuplicates.setEnabled(True)

        self.horizontalLayout_4.addWidget(self.cDuplicates)

        self.cBad = QCheckBox(Q7LinkWindow)
        self.cBad.setObjectName(u"cBad")
        self.cBad.setEnabled(True)

        self.horizontalLayout_4.addWidget(self.cBad)

        self.cExternal = QCheckBox(Q7LinkWindow)
        self.cExternal.setObjectName(u"cExternal")
        self.cExternal.setEnabled(True)
        palette = QPalette()
        brush = QBrush(QColor(255, 0, 0, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush1 = QBrush(QColor(0, 0, 0, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Text, brush1)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
        palette.setBrush(QPalette.Active, QPalette.NoRole, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.NoRole, brush1)
        brush2 = QBrush(QColor(118, 116, 113, 255))
        brush2.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.NoRole, brush1)
        self.cExternal.setPalette(palette)
        font = QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setStrikeOut(False)
        self.cExternal.setFont(font)

        self.horizontalLayout_4.addWidget(self.cExternal)

        self.cInternal = QCheckBox(Q7LinkWindow)
        self.cInternal.setObjectName(u"cInternal")
        self.cInternal.setEnabled(True)
        palette1 = QPalette()
        palette1.setBrush(QPalette.Active, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.WindowText, brush2)
        self.cInternal.setPalette(palette1)

        self.horizontalLayout_4.addWidget(self.cInternal)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_4)

        self.cApplyToAll = QCheckBox(Q7LinkWindow)
        self.cApplyToAll.setObjectName(u"cApplyToAll")
        self.cApplyToAll.setEnabled(True)
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/user-G.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.cApplyToAll.setIcon(icon8)

        self.horizontalLayout_4.addWidget(self.cApplyToAll)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.linkTable = QTableWidget(Q7LinkWindow)
        if (self.linkTable.columnCount() < 5):
            self.linkTable.setColumnCount(5)
        self.linkTable.setObjectName(u"linkTable")
        self.linkTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.linkTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.linkTable.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.linkTable.setSelectionMode(QAbstractItemView.MultiSelection)
        self.linkTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.linkTable.setSortingEnabled(True)
        self.linkTable.setColumnCount(5)

        self.verticalLayout.addWidget(self.linkTable)


        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.bBackControl = QPushButton(Q7LinkWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon9 = QIcon()
        icon9.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon9)

        self.horizontalLayout_2.addWidget(self.bBackControl)

        self.bInfo = QPushButton(Q7LinkWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon10 = QIcon()
        icon10.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon10)

        self.horizontalLayout_2.addWidget(self.bInfo)

        self.eDirSource = QLineEdit(Q7LinkWindow)
        self.eDirSource.setObjectName(u"eDirSource")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.eDirSource.sizePolicy().hasHeightForWidth())
        self.eDirSource.setSizePolicy(sizePolicy1)

        self.horizontalLayout_2.addWidget(self.eDirSource)

        self.label = QLabel(Q7LinkWindow)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.eFileSource = QLineEdit(Q7LinkWindow)
        self.eFileSource.setObjectName(u"eFileSource")
        sizePolicy.setHeightForWidth(self.eFileSource.sizePolicy().hasHeightForWidth())
        self.eFileSource.setSizePolicy(sizePolicy)

        self.horizontalLayout_2.addWidget(self.eFileSource)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.bClose = QPushButton(Q7LinkWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout_2.addWidget(self.bClose)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)


        self.retranslateUi(Q7LinkWindow)

        QMetaObject.connectSlotsByName(Q7LinkWindow)
    # setupUi

    def retranslateUi(self, Q7LinkWindow):
        Q7LinkWindow.setWindowTitle(QCoreApplication.translate("Q7LinkWindow", u"Form", None))
#if QT_CONFIG(tooltip)
        self.bCheckLink.setToolTip(QCoreApplication.translate("Q7LinkWindow", u"check and fix link list", None))
#endif // QT_CONFIG(tooltip)
        self.bCheckLink.setText("")
#if QT_CONFIG(tooltip)
        self.bLoadTree.setToolTip(QCoreApplication.translate("Q7LinkWindow", u"open link target file", None))
#endif // QT_CONFIG(tooltip)
        self.bLoadTree.setText("")
#if QT_CONFIG(tooltip)
        self.bAddLink.setToolTip(QCoreApplication.translate("Q7LinkWindow", u"add new link entry", None))
#endif // QT_CONFIG(tooltip)
        self.bAddLink.setText("")
#if QT_CONFIG(tooltip)
        self.bDuplicateLink.setToolTip(QCoreApplication.translate("Q7LinkWindow", u"duplicate link entry", None))
#endif // QT_CONFIG(tooltip)
        self.bDuplicateLink.setText("")
#if QT_CONFIG(tooltip)
        self.bFromSelection.setToolTip(QCoreApplication.translate("Q7LinkWindow", u"insert link from selection", None))
#endif // QT_CONFIG(tooltip)
        self.bFromSelection.setText("")
#if QT_CONFIG(tooltip)
        self.bDeleteLink.setToolTip(QCoreApplication.translate("Q7LinkWindow", u"delete link entry", None))
#endif // QT_CONFIG(tooltip)
        self.bDeleteLink.setText("")
        self.bSave.setText("")
        self.label_2.setText(QCoreApplication.translate("Q7LinkWindow", u"Select", None))
        self.cUnreachable.setText(QCoreApplication.translate("Q7LinkWindow", u"unreachable", None))
        self.cDuplicates.setText(QCoreApplication.translate("Q7LinkWindow", u"duplicates", None))
        self.cBad.setText(QCoreApplication.translate("Q7LinkWindow", u"bad links", None))
        self.cExternal.setText(QCoreApplication.translate("Q7LinkWindow", u"external links", None))
        self.cInternal.setText(QCoreApplication.translate("Q7LinkWindow", u"internal links", None))
        self.cApplyToAll.setText(QCoreApplication.translate("Q7LinkWindow", u"Apply to All Selected", None))
        self.bBackControl.setText("")
        self.bInfo.setText("")
        self.label.setText(QCoreApplication.translate("Q7LinkWindow", u"/", None))
        self.bClose.setText(QCoreApplication.translate("Q7LinkWindow", u"Close", None))
    # retranslateUi

