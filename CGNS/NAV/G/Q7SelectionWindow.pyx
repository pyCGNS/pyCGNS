# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7SelectionWindow.ui'
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
    QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QSizePolicy, QSpacerItem, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)
from . import Res_rc

class Ui_Q7SelectionWindow(object):
    def setupUi(self, Q7SelectionWindow):
        if not Q7SelectionWindow.objectName():
            Q7SelectionWindow.setObjectName(u"Q7SelectionWindow")
        Q7SelectionWindow.resize(715, 350)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7SelectionWindow.sizePolicy().hasHeightForWidth())
        Q7SelectionWindow.setSizePolicy(sizePolicy)
        Q7SelectionWindow.setMinimumSize(QSize(0, 0))
        Q7SelectionWindow.setMaximumSize(QSize(90000, 90000))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7SelectionWindow.setWindowIcon(icon)
        self.gridLayout = QGridLayout(Q7SelectionWindow)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.bBackControl = QPushButton(Q7SelectionWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon1)

        self.horizontalLayout_2.addWidget(self.bBackControl)

        self.bInfo = QPushButton(Q7SelectionWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon2)

        self.horizontalLayout_2.addWidget(self.bInfo)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.bClose = QPushButton(Q7SelectionWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout_2.addWidget(self.bClose)


        self.gridLayout.addLayout(self.horizontalLayout_2, 7, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bPrevious = QPushButton(Q7SelectionWindow)
        self.bPrevious.setObjectName(u"bPrevious")
        self.bPrevious.setEnabled(False)
        self.bPrevious.setMinimumSize(QSize(25, 25))
        self.bPrevious.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/control.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bPrevious.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bPrevious)

        self.bFirst = QPushButton(Q7SelectionWindow)
        self.bFirst.setObjectName(u"bFirst")
        self.bFirst.setEnabled(False)
        self.bFirst.setMinimumSize(QSize(25, 25))
        self.bFirst.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/node-sids-leaf.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bFirst.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bFirst)

        self.bNext = QPushButton(Q7SelectionWindow)
        self.bNext.setObjectName(u"bNext")
        self.bNext.setEnabled(False)
        self.bNext.setMinimumSize(QSize(25, 25))
        self.bNext.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/node-sids-closed.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bNext.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bNext)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bSelectAll = QPushButton(Q7SelectionWindow)
        self.bSelectAll.setObjectName(u"bSelectAll")
        self.bSelectAll.setMinimumSize(QSize(25, 25))
        self.bSelectAll.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/select-add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSelectAll.setIcon(icon6)

        self.horizontalLayout.addWidget(self.bSelectAll)

        self.bReverse = QPushButton(Q7SelectionWindow)
        self.bReverse.setObjectName(u"bReverse")
        self.bReverse.setMinimumSize(QSize(25, 25))
        self.bReverse.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/flag-revert.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bReverse.setIcon(icon7)

        self.horizontalLayout.addWidget(self.bReverse)

        self.bUnselectAll = QPushButton(Q7SelectionWindow)
        self.bUnselectAll.setObjectName(u"bUnselectAll")
        self.bUnselectAll.setMinimumSize(QSize(25, 25))
        self.bUnselectAll.setMaximumSize(QSize(25, 25))
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/select-delete.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bUnselectAll.setIcon(icon8)

        self.horizontalLayout.addWidget(self.bUnselectAll)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_5)

        self.bRemoveToSelect = QPushButton(Q7SelectionWindow)
        self.bRemoveToSelect.setObjectName(u"bRemoveToSelect")
        icon9 = QIcon()
        icon9.addFile(u":/images/icons/flag-none.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bRemoveToSelect.setIcon(icon9)

        self.horizontalLayout.addWidget(self.bRemoveToSelect)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.bSave = QPushButton(Q7SelectionWindow)
        self.bSave.setObjectName(u"bSave")
        self.bSave.setMinimumSize(QSize(25, 25))
        self.bSave.setMaximumSize(QSize(25, 25))
        icon10 = QIcon()
        icon10.addFile(u":/images/icons/select-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSave.setIcon(icon10)

        self.horizontalLayout.addWidget(self.bSave)


        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.selectionTable = QTableWidget(Q7SelectionWindow)
        if (self.selectionTable.columnCount() < 1):
            self.selectionTable.setColumnCount(1)
        self.selectionTable.setObjectName(u"selectionTable")
        sizePolicy.setHeightForWidth(self.selectionTable.sizePolicy().hasHeightForWidth())
        self.selectionTable.setSizePolicy(sizePolicy)
        self.selectionTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.selectionTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.selectionTable.setAutoScroll(False)
        self.selectionTable.setEditTriggers(QAbstractItemView.DoubleClicked|QAbstractItemView.EditKeyPressed)
        self.selectionTable.setDragDropOverwriteMode(False)
        self.selectionTable.setAlternatingRowColors(False)
        self.selectionTable.setSelectionMode(QAbstractItemView.MultiSelection)
        self.selectionTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.selectionTable.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.selectionTable.setSortingEnabled(True)
        self.selectionTable.setColumnCount(1)

        self.verticalLayout.addWidget(self.selectionTable)


        self.gridLayout.addLayout(self.verticalLayout, 6, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label = QLabel(Q7SelectionWindow)
        self.label.setObjectName(u"label")

        self.horizontalLayout_3.addWidget(self.label)

        self.cShowSIDS = QCheckBox(Q7SelectionWindow)
        self.cShowSIDS.setObjectName(u"cShowSIDS")
        self.cShowSIDS.setChecked(False)

        self.horizontalLayout_3.addWidget(self.cShowSIDS)

        self.cShowPath = QCheckBox(Q7SelectionWindow)
        self.cShowPath.setObjectName(u"cShowPath")
        self.cShowPath.setChecked(True)

        self.horizontalLayout_3.addWidget(self.cShowPath)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_3)

        self.cForce = QCheckBox(Q7SelectionWindow)
        self.cForce.setObjectName(u"cForce")
        self.cForce.setEnabled(True)
        self.cForce.setCheckable(True)
        self.cForce.setChecked(True)

        self.horizontalLayout_3.addWidget(self.cForce)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_6)

        self.bApply = QPushButton(Q7SelectionWindow)
        self.bApply.setObjectName(u"bApply")
        self.bApply.setMinimumSize(QSize(25, 25))
        icon11 = QIcon()
        icon11.addFile(u":/images/icons/operate-execute.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bApply.setIcon(icon11)

        self.horizontalLayout_3.addWidget(self.bApply)

        self.cApplyToAll = QCheckBox(Q7SelectionWindow)
        self.cApplyToAll.setObjectName(u"cApplyToAll")
        self.cApplyToAll.setEnabled(True)
        icon12 = QIcon()
        icon12.addFile(u":/images/icons/user-G.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.cApplyToAll.setIcon(icon12)

        self.horizontalLayout_3.addWidget(self.cApplyToAll)


        self.gridLayout.addLayout(self.horizontalLayout_3, 5, 0, 1, 1)

        self.line = QFrame(Q7SelectionWindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout.addWidget(self.line, 4, 0, 1, 1)


        self.retranslateUi(Q7SelectionWindow)

        QMetaObject.connectSlotsByName(Q7SelectionWindow)
    # setupUi

    def retranslateUi(self, Q7SelectionWindow):
        Q7SelectionWindow.setWindowTitle(QCoreApplication.translate("Q7SelectionWindow", u"Form", None))
        self.bBackControl.setText("")
        self.bInfo.setText("")
        self.bClose.setText(QCoreApplication.translate("Q7SelectionWindow", u"Close", None))
        self.bPrevious.setText("")
        self.bFirst.setText("")
        self.bNext.setText("")
#if QT_CONFIG(tooltip)
        self.bSelectAll.setToolTip(QCoreApplication.translate("Q7SelectionWindow", u"XOR", None))
#endif // QT_CONFIG(tooltip)
        self.bSelectAll.setText("")
        self.bReverse.setText("")
        self.bUnselectAll.setText("")
        self.bRemoveToSelect.setText("")
        self.bSave.setText("")
        self.label.setText(QCoreApplication.translate("Q7SelectionWindow", u"Show:", None))
        self.cShowSIDS.setText(QCoreApplication.translate("Q7SelectionWindow", u"SIDS", None))
        self.cShowPath.setText(QCoreApplication.translate("Q7SelectionWindow", u"Path", None))
        self.cForce.setText(QCoreApplication.translate("Q7SelectionWindow", u"Force value check", None))
#if QT_CONFIG(tooltip)
        self.bApply.setToolTip(QCoreApplication.translate("Q7SelectionWindow", u"Apply changes", None))
#endif // QT_CONFIG(tooltip)
        self.bApply.setText("")
        self.cApplyToAll.setText(QCoreApplication.translate("Q7SelectionWindow", u"Apply to All Selected", None))
    # retranslateUi

