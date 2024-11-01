# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7FileWindow.ui'
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
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLayout, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QRadioButton, QSizePolicy,
    QSpacerItem, QSpinBox, QTabWidget, QTreeView,
    QVBoxLayout, QWidget)
from . import Res_rc

class Ui_Q7FileWindow(object):
    def setupUi(self, Q7FileWindow):
        if not Q7FileWindow.objectName():
            Q7FileWindow.setObjectName(u"Q7FileWindow")
        Q7FileWindow.setWindowModality(Qt.ApplicationModal)
        Q7FileWindow.resize(728, 450)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7FileWindow.sizePolicy().hasHeightForWidth())
        Q7FileWindow.setSizePolicy(sizePolicy)
        Q7FileWindow.setMinimumSize(QSize(728, 450))
        Q7FileWindow.setMaximumSize(QSize(800, 600))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7FileWindow.setWindowIcon(icon)
        self.verticalLayout = QVBoxLayout(Q7FileWindow)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(4, 0, 4, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, -1, 0, 0)
        self.tabs = QTabWidget(Q7FileWindow)
        self.tabs.setObjectName(u"tabs")
        self.tabs.setEnabled(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tabs.sizePolicy().hasHeightForWidth())
        self.tabs.setSizePolicy(sizePolicy1)
        self.tabs.setMinimumSize(QSize(0, 0))
        self.tabs.setMaximumSize(QSize(800, 440))
        self.Selection = QWidget()
        self.Selection.setObjectName(u"Selection")
        sizePolicy1.setHeightForWidth(self.Selection.sizePolicy().hasHeightForWidth())
        self.Selection.setSizePolicy(sizePolicy1)
        self.Selection.setMinimumSize(QSize(0, 0))
        self.Selection.setMaximumSize(QSize(700, 400))
        self.selectionVL = QVBoxLayout(self.Selection)
        self.selectionVL.setSpacing(6)
        self.selectionVL.setObjectName(u"selectionVL")
        self.selectionVL.setContentsMargins(8, 8, 8, 8)
        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(3)
        self.gridLayout.setObjectName(u"gridLayout")
        self.fileentries = QComboBox(self.Selection)
        self.fileentries.setObjectName(u"fileentries")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.fileentries.sizePolicy().hasHeightForWidth())
        self.fileentries.setSizePolicy(sizePolicy2)
        self.fileentries.setEditable(True)

        self.gridLayout.addWidget(self.fileentries, 5, 2, 1, 2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.cNoLargeData = QCheckBox(self.Selection)
        self.cNoLargeData.setObjectName(u"cNoLargeData")
        font = QFont()
        font.setBold(False)
        self.cNoLargeData.setFont(font)

        self.horizontalLayout_4.addWidget(self.cNoLargeData)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)


        self.gridLayout.addLayout(self.horizontalLayout_4, 1, 2, 1, 3)

        self.direntries = QComboBox(self.Selection)
        self.direntries.setObjectName(u"direntries")
        sizePolicy2.setHeightForWidth(self.direntries.sizePolicy().hasHeightForWidth())
        self.direntries.setSizePolicy(sizePolicy2)
        self.direntries.setEditable(True)

        self.gridLayout.addWidget(self.direntries, 3, 2, 1, 3)

        self.cShowDirs = QCheckBox(self.Selection)
        self.cShowDirs.setObjectName(u"cShowDirs")
        self.cShowDirs.setLayoutDirection(Qt.LeftToRight)
        self.cShowDirs.setChecked(True)

        self.gridLayout.addWidget(self.cShowDirs, 5, 4, 1, 1)

        self.bCurrent = QPushButton(self.Selection)
        self.bCurrent.setObjectName(u"bCurrent")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.bCurrent.sizePolicy().hasHeightForWidth())
        self.bCurrent.setSizePolicy(sizePolicy3)
        self.bCurrent.setMinimumSize(QSize(25, 25))
        self.bCurrent.setMaximumSize(QSize(25, 25))
        self.bCurrent.setLayoutDirection(Qt.LeftToRight)
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/local-dir.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bCurrent.setIcon(icon1)

        self.gridLayout.addWidget(self.bCurrent, 3, 1, 1, 1)

        self.bBack = QPushButton(self.Selection)
        self.bBack.setObjectName(u"bBack")
        sizePolicy3.setHeightForWidth(self.bBack.sizePolicy().hasHeightForWidth())
        self.bBack.setSizePolicy(sizePolicy3)
        self.bBack.setMinimumSize(QSize(25, 25))
        self.bBack.setMaximumSize(QSize(25, 25))
        font1 = QFont()
        font1.setPointSize(14)
        font1.setBold(True)
        self.bBack.setFont(font1)
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/parent-dir.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBack.setIcon(icon2)
        self.bBack.setIconSize(QSize(16, 16))

        self.gridLayout.addWidget(self.bBack, 3, 0, 1, 1)


        self.selectionVL.addLayout(self.gridLayout)

        self.treeview = QTreeView(self.Selection)
        self.treeview.setObjectName(u"treeview")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.treeview.sizePolicy().hasHeightForWidth())
        self.treeview.setSizePolicy(sizePolicy4)
        self.treeview.setMinimumSize(QSize(650, 235))
        self.treeview.setMaximumSize(QSize(720, 450))
        self.treeview.setEditTriggers(QAbstractItemView.EditKeyPressed|QAbstractItemView.SelectedClicked)
        self.treeview.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.treeview.setUniformRowHeights(True)
        self.treeview.setSortingEnabled(True)
        self.treeview.setAllColumnsShowFocus(True)

        self.selectionVL.addWidget(self.treeview)

        self.bottomAction = QHBoxLayout()
        self.bottomAction.setSpacing(6)
        self.bottomAction.setObjectName(u"bottomAction")
        self.bottomAction.setSizeConstraint(QLayout.SetFixedSize)
        self.bottomAction.setContentsMargins(-1, -1, 0, -1)
        self.bInfo = QPushButton(self.Selection)
        self.bInfo.setObjectName(u"bInfo")
        sizePolicy3.setHeightForWidth(self.bInfo.sizePolicy().hasHeightForWidth())
        self.bInfo.setSizePolicy(sizePolicy3)
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon3)

        self.bottomAction.addWidget(self.bInfo)

        self.horizontalSpacer_3 = QSpacerItem(40, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottomAction.addItem(self.horizontalSpacer_3)

        self.bAction = QPushButton(self.Selection)
        self.bAction.setObjectName(u"bAction")
        sizePolicy3.setHeightForWidth(self.bAction.sizePolicy().hasHeightForWidth())
        self.bAction.setSizePolicy(sizePolicy3)
        self.bAction.setMaximumSize(QSize(70, 16777215))
        self.bAction.setFont(font)

        self.bottomAction.addWidget(self.bAction)

        self.bClose = QPushButton(self.Selection)
        self.bClose.setObjectName(u"bClose")
        sizePolicy3.setHeightForWidth(self.bClose.sizePolicy().hasHeightForWidth())
        self.bClose.setSizePolicy(sizePolicy3)
        self.bClose.setMinimumSize(QSize(0, 0))
        self.bClose.setMaximumSize(QSize(70, 16777215))

        self.bottomAction.addWidget(self.bClose)


        self.selectionVL.addLayout(self.bottomAction)

        self.tabs.addTab(self.Selection, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.load_save_Lay = QGridLayout(self.tab)
        self.load_save_Lay.setSpacing(2)
        self.load_save_Lay.setObjectName(u"load_save_Lay")
        self.load_save_Lay.setContentsMargins(8, 8, 8, 8)
        self.groupBox_4 = QGroupBox(self.tab)
        self.groupBox_4.setObjectName(u"groupBox_4")
        sizePolicy1.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy1)
        self.groupBox_4.setMaximumSize(QSize(16777215, 16777215))
        self.saveVL = QVBoxLayout(self.groupBox_4)
        self.saveVL.setSpacing(2)
        self.saveVL.setObjectName(u"saveVL")
        self.saveVL.setContentsMargins(8, 8, 8, 8)
        self.cOverwrite = QCheckBox(self.groupBox_4)
        self.cOverwrite.setObjectName(u"cOverwrite")
        self.cOverwrite.setFont(font)

        self.saveVL.addWidget(self.cOverwrite)

        self.cDeleteMissing = QCheckBox(self.groupBox_4)
        self.cDeleteMissing.setObjectName(u"cDeleteMissing")

        self.saveVL.addWidget(self.cDeleteMissing)

        self.cSkipEmpty = QCheckBox(self.groupBox_4)
        self.cSkipEmpty.setObjectName(u"cSkipEmpty")

        self.saveVL.addWidget(self.cSkipEmpty)

        self.cSaveWithoutLinks = QCheckBox(self.groupBox_4)
        self.cSaveWithoutLinks.setObjectName(u"cSaveWithoutLinks")

        self.saveVL.addWidget(self.cSaveWithoutLinks)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.saveVL.addItem(self.verticalSpacer_2)


        self.load_save_Lay.addWidget(self.groupBox_4, 0, 1, 1, 1)

        self.groupBox_5 = QGroupBox(self.tab)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setMaximumSize(QSize(16777215, 16777215))
        self.CHLoneOptionGrid = QGridLayout(self.groupBox_5)
        self.CHLoneOptionGrid.setSpacing(2)
        self.CHLoneOptionGrid.setObjectName(u"CHLoneOptionGrid")
        self.CHLoneOptionGrid.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.CHLoneOptionGrid.setContentsMargins(8, 8, 8, 8)
        self.checkBox = QCheckBox(self.groupBox_5)
        self.checkBox.setObjectName(u"checkBox")

        self.CHLoneOptionGrid.addWidget(self.checkBox, 0, 0, 1, 1)

        self.checkBox_2 = QCheckBox(self.groupBox_5)
        self.checkBox_2.setObjectName(u"checkBox_2")

        self.CHLoneOptionGrid.addWidget(self.checkBox_2, 2, 0, 1, 2)

        self.line = QFrame(self.groupBox_5)
        self.line.setObjectName(u"line")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy5)
        self.line.setFrameShape(QFrame.Shape.VLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.CHLoneOptionGrid.addWidget(self.line, 0, 2, 7, 1)

        self.horizontalSpacer_5 = QSpacerItem(298, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.CHLoneOptionGrid.addItem(self.horizontalSpacer_5, 0, 3, 1, 1)

        self.checkBox_3 = QCheckBox(self.groupBox_5)
        self.checkBox_3.setObjectName(u"checkBox_3")

        self.CHLoneOptionGrid.addWidget(self.checkBox_3, 3, 0, 1, 1)

        self.lineEdit_2 = QLineEdit(self.groupBox_5)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        sizePolicy2.setHeightForWidth(self.lineEdit_2.sizePolicy().hasHeightForWidth())
        self.lineEdit_2.setSizePolicy(sizePolicy2)
        self.lineEdit_2.setMinimumSize(QSize(0, 0))
        self.lineEdit_2.setMaximumSize(QSize(16777215, 16777215))
        self.lineEdit_2.setAcceptDrops(False)
        self.lineEdit_2.setMaxLength(32760)
        self.lineEdit_2.setFrame(True)

        self.CHLoneOptionGrid.addWidget(self.lineEdit_2, 4, 0, 1, 2)

        self.horizontalSpacer_7 = QSpacerItem(200, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.CHLoneOptionGrid.addItem(self.horizontalSpacer_7, 5, 0, 1, 1)


        self.load_save_Lay.addWidget(self.groupBox_5, 1, 0, 1, 2)

        self.groupBox_3 = QGroupBox(self.tab)
        self.groupBox_3.setObjectName(u"groupBox_3")
        sizePolicy1.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy1)
        self.groupBox_3.setMinimumSize(QSize(0, 0))
        self.groupBox_3.setMaximumSize(QSize(364, 16777215))
        self.LoadVL = QVBoxLayout(self.groupBox_3)
        self.LoadVL.setSpacing(2)
        self.LoadVL.setObjectName(u"LoadVL")
        self.LoadVL.setContentsMargins(8, 8, 8, 8)
        self.cNoLargeData_2 = QCheckBox(self.groupBox_3)
        self.cNoLargeData_2.setObjectName(u"cNoLargeData_2")
        self.cNoLargeData_2.setFont(font)

        self.LoadVL.addWidget(self.cNoLargeData_2)

        self.cFollowLinks = QCheckBox(self.groupBox_3)
        self.cFollowLinks.setObjectName(u"cFollowLinks")

        self.LoadVL.addWidget(self.cFollowLinks)

        self.cReadOnly = QCheckBox(self.groupBox_3)
        self.cReadOnly.setObjectName(u"cReadOnly")
        self.cReadOnly.setEnabled(True)

        self.LoadVL.addWidget(self.cReadOnly)

        self.cLoadSubPath = QCheckBox(self.groupBox_3)
        self.cLoadSubPath.setObjectName(u"cLoadSubPath")

        self.LoadVL.addWidget(self.cLoadSubPath)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.boxSpacer = QSpacerItem(20, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.boxSpacer)

        self.lineEdit = QLineEdit(self.groupBox_3)
        self.lineEdit.setObjectName(u"lineEdit")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy6)

        self.horizontalLayout_2.addWidget(self.lineEdit)


        self.LoadVL.addLayout(self.horizontalLayout_2)

        self.LimitDepthHLay = QHBoxLayout()
        self.LimitDepthHLay.setObjectName(u"LimitDepthHLay")
        self.LimitDepthHLay.setSizeConstraint(QLayout.SetFixedSize)
        self.cLimitDepth = QCheckBox(self.groupBox_3)
        self.cLimitDepth.setObjectName(u"cLimitDepth")

        self.LimitDepthHLay.addWidget(self.cLimitDepth)

        self.spinBox = QSpinBox(self.groupBox_3)
        self.spinBox.setObjectName(u"spinBox")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.spinBox.sizePolicy().hasHeightForWidth())
        self.spinBox.setSizePolicy(sizePolicy7)
        self.spinBox.setMinimumSize(QSize(60, 0))

        self.LimitDepthHLay.addWidget(self.spinBox)

        self.horizontalSpacer_4 = QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.LimitDepthHLay.addItem(self.horizontalSpacer_4)


        self.LoadVL.addLayout(self.LimitDepthHLay)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.LoadVL.addItem(self.verticalSpacer_3)


        self.load_save_Lay.addWidget(self.groupBox_3, 0, 0, 1, 1)

        self.tabs.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.tab_2.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.tab_2.sizePolicy().hasHeightForWidth())
        self.tab_2.setSizePolicy(sizePolicy1)
        self.tab_2.setMinimumSize(QSize(695, 390))
        self.tab_2.setMaximumSize(QSize(720, 420))
        self.verticalLayout_3 = QVBoxLayout(self.tab_2)
        self.verticalLayout_3.setSpacing(2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(6, 6, 6, 6)
        self.topHL = QHBoxLayout()
        self.topHL.setSpacing(0)
        self.topHL.setObjectName(u"topHL")
        self.cActivate = QCheckBox(self.tab_2)
        self.cActivate.setObjectName(u"cActivate")
        self.cActivate.setEnabled(False)
        self.cActivate.setMaximumSize(QSize(260, 16777215))
        self.cActivate.setChecked(True)

        self.topHL.addWidget(self.cActivate)

        self.cAutoDir = QCheckBox(self.tab_2)
        self.cAutoDir.setObjectName(u"cAutoDir")

        self.topHL.addWidget(self.cAutoDir)


        self.verticalLayout_3.addLayout(self.topHL)

        self.downHL = QHBoxLayout()
        self.downHL.setObjectName(u"downHL")
        self.groupBox_2 = QGroupBox(self.tab_2)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy8)
        self.groupBox_2.setMaximumSize(QSize(190, 16777215))
        self.showVL = QVBoxLayout(self.groupBox_2)
        self.showVL.setSpacing(4)
        self.showVL.setObjectName(u"showVL")
        self.showVL.setContentsMargins(4, 4, 4, 4)
        self.cShowAll = QCheckBox(self.groupBox_2)
        self.cShowAll.setObjectName(u"cShowAll")

        self.showVL.addWidget(self.cShowAll)

        self.__O_filterhdffiles = QCheckBox(self.groupBox_2)
        self.__O_filterhdffiles.setObjectName(u"__O_filterhdffiles")
        self.__O_filterhdffiles.setChecked(True)

        self.showVL.addWidget(self.__O_filterhdffiles)

        self.__O_filtercgnsfiles = QCheckBox(self.groupBox_2)
        self.__O_filtercgnsfiles.setObjectName(u"__O_filtercgnsfiles")
        self.__O_filtercgnsfiles.setChecked(True)

        self.showVL.addWidget(self.__O_filtercgnsfiles)

        self.verticalSpacer_4 = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.showVL.addItem(self.verticalSpacer_4)

        self.cShowOwnExt = QCheckBox(self.groupBox_2)
        self.cShowOwnExt.setObjectName(u"cShowOwnExt")

        self.showVL.addWidget(self.cShowOwnExt)

        self.lOwnExt = QListWidget(self.groupBox_2)
        self.lOwnExt.setObjectName(u"lOwnExt")

        self.showVL.addWidget(self.lOwnExt)


        self.downHL.addWidget(self.groupBox_2)

        self.groupBox = QGroupBox(self.tab_2)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy9.setHorizontalStretch(0)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy9)
        self.verticalLayout_4 = QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setSpacing(1)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(4, 4, 4, 4)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.rClearNotFound = QRadioButton(self.groupBox)
        self.rClearNotFound.setObjectName(u"rClearNotFound")
        self.rClearNotFound.setChecked(True)

        self.horizontalLayout_3.addWidget(self.rClearNotFound)

        self.horizontalSpacer_6 = QSpacerItem(10, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_6)

        self.bInfo2 = QPushButton(self.groupBox)
        self.bInfo2.setObjectName(u"bInfo2")
        self.bInfo2.setMinimumSize(QSize(25, 25))
        self.bInfo2.setMaximumSize(QSize(25, 25))
        self.bInfo2.setIcon(icon3)

        self.horizontalLayout_3.addWidget(self.bInfo2)

        self.bClearHistory = QPushButton(self.groupBox)
        self.bClearHistory.setObjectName(u"bClearHistory")
        self.bClearHistory.setMaximumSize(QSize(70, 25))

        self.horizontalLayout_3.addWidget(self.bClearHistory)


        self.verticalLayout_4.addLayout(self.horizontalLayout_3)

        self.rClearNoHDF = QRadioButton(self.groupBox)
        self.rClearNoHDF.setObjectName(u"rClearNoHDF")

        self.verticalLayout_4.addWidget(self.rClearNoHDF)

        self.rClearSelectedDirs = QRadioButton(self.groupBox)
        self.rClearSelectedDirs.setObjectName(u"rClearSelectedDirs")

        self.verticalLayout_4.addWidget(self.rClearSelectedDirs)

        self.rClearSelectedFiles = QRadioButton(self.groupBox)
        self.rClearSelectedFiles.setObjectName(u"rClearSelectedFiles")

        self.verticalLayout_4.addWidget(self.rClearSelectedFiles)

        self.rClearAllDirs = QRadioButton(self.groupBox)
        self.rClearAllDirs.setObjectName(u"rClearAllDirs")

        self.verticalLayout_4.addWidget(self.rClearAllDirs)

        self.lClear = QListWidget(self.groupBox)
        self.lClear.setObjectName(u"lClear")
        self.lClear.setMinimumSize(QSize(0, 100))
        self.lClear.setSelectionMode(QAbstractItemView.MultiSelection)

        self.verticalLayout_4.addWidget(self.lClear)


        self.downHL.addWidget(self.groupBox)


        self.verticalLayout_3.addLayout(self.downHL)

        self.tabs.addTab(self.tab_2, "")

        self.horizontalLayout.addWidget(self.tabs)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(Q7FileWindow)

        self.tabs.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Q7FileWindow)
    # setupUi

    def retranslateUi(self, Q7FileWindow):
        Q7FileWindow.setWindowTitle(QCoreApplication.translate("Q7FileWindow", u"Form", None))
#if QT_CONFIG(tooltip)
        self.cNoLargeData.setToolTip(QCoreApplication.translate("Q7FileWindow", u"Nodes with large data are read but their data is not", None))
#endif // QT_CONFIG(tooltip)
        self.cNoLargeData.setText(QCoreApplication.translate("Q7FileWindow", u"Do not load large data", None))
        self.cShowDirs.setText(QCoreApplication.translate("Q7FileWindow", u"Show directories", None))
#if QT_CONFIG(tooltip)
        self.bCurrent.setToolTip(QCoreApplication.translate("Q7FileWindow", u"Go to launch directory", None))
#endif // QT_CONFIG(tooltip)
        self.bCurrent.setText("")
#if QT_CONFIG(tooltip)
        self.bBack.setToolTip(QCoreApplication.translate("Q7FileWindow", u"Go back to parent directory", None))
#endif // QT_CONFIG(tooltip)
        self.bBack.setText("")
        self.bInfo.setText("")
        self.bAction.setText(QCoreApplication.translate("Q7FileWindow", u"LOAD", None))
        self.bClose.setText(QCoreApplication.translate("Q7FileWindow", u"Cancel", None))
        self.tabs.setTabText(self.tabs.indexOf(self.Selection), QCoreApplication.translate("Q7FileWindow", u"Selection", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Q7FileWindow", u"Save", None))
#if QT_CONFIG(tooltip)
        self.cOverwrite.setToolTip(QCoreApplication.translate("Q7FileWindow", u"Overwrite an existing file with new contents", None))
#endif // QT_CONFIG(tooltip)
        self.cOverwrite.setText(QCoreApplication.translate("Q7FileWindow", u"Overwrite", None))
#if QT_CONFIG(tooltip)
        self.cDeleteMissing.setToolTip(QCoreApplication.translate("Q7FileWindow", u"Children found in existing file but not in current tree are removed", None))
#endif // QT_CONFIG(tooltip)
        self.cDeleteMissing.setText(QCoreApplication.translate("Q7FileWindow", u"Delete missing", None))
        self.cSkipEmpty.setText(QCoreApplication.translate("Q7FileWindow", u"Skip empty", None))
        self.cSaveWithoutLinks.setText(QCoreApplication.translate("Q7FileWindow", u"Do not save with links", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("Q7FileWindow", u"CHLone options (both load and save)", None))
        self.checkBox.setText(QCoreApplication.translate("Q7FileWindow", u"Trace", None))
        self.checkBox_2.setText(QCoreApplication.translate("Q7FileWindow", u"Debug (quite large output)", None))
        self.checkBox_3.setText(QCoreApplication.translate("Q7FileWindow", u"Send output to:", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Q7FileWindow", u"Load", None))
#if QT_CONFIG(tooltip)
        self.cNoLargeData_2.setToolTip(QCoreApplication.translate("Q7FileWindow", u"Nodes with large data are read but their data is not", None))
#endif // QT_CONFIG(tooltip)
        self.cNoLargeData_2.setText(QCoreApplication.translate("Q7FileWindow", u"Do not load large data", None))
        self.cFollowLinks.setText(QCoreApplication.translate("Q7FileWindow", u"Follow links", None))
        self.cReadOnly.setText(QCoreApplication.translate("Q7FileWindow", u"Open as read-only", None))
        self.cLoadSubPath.setText(QCoreApplication.translate("Q7FileWindow", u"Load sub-tree with path:", None))
        self.cLimitDepth.setText(QCoreApplication.translate("Q7FileWindow", u"Limit depth to:", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab), QCoreApplication.translate("Q7FileWindow", u"Load/Save options", None))
        self.cActivate.setText(QCoreApplication.translate("Q7FileWindow", u"Activate directory/file history", None))
        self.cAutoDir.setText(QCoreApplication.translate("Q7FileWindow", u"Auto-Change directory", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Q7FileWindow", u"Show", None))
        self.cShowAll.setText(QCoreApplication.translate("Q7FileWindow", u"All file extensions", None))
        self.__O_filterhdffiles.setText(QCoreApplication.translate("Q7FileWindow", u".hdf files", None))
        self.__O_filtercgnsfiles.setText(QCoreApplication.translate("Q7FileWindow", u".cgns/.adf files", None))
        self.cShowOwnExt.setText(QCoreApplication.translate("Q7FileWindow", u"own extension:", None))
        self.groupBox.setTitle(QCoreApplication.translate("Q7FileWindow", u"Clear history", None))
        self.rClearNotFound.setText(QCoreApplication.translate("Q7FileWindow", u"Not found directory and file entries", None))
        self.bInfo2.setText("")
        self.bClearHistory.setText(QCoreApplication.translate("Q7FileWindow", u"Clear", None))
        self.rClearNoHDF.setText(QCoreApplication.translate("Q7FileWindow", u"Directory entries without correct file e&xtension", None))
        self.rClearSelectedDirs.setText(QCoreApplication.translate("Q7FileWindow", u"Selected directory entries", None))
        self.rClearSelectedFiles.setText(QCoreApplication.translate("Q7FileWindow", u"Selected file entries", None))
        self.rClearAllDirs.setText(QCoreApplication.translate("Q7FileWindow", u"All director&y and file entries", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_2), QCoreApplication.translate("Q7FileWindow", u"History/Filter", None))
    # retranslateUi

