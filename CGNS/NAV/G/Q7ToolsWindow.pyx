# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7ToolsWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Q7ToolsWindow(object):
    def setupUi(self, Q7ToolsWindow):
        Q7ToolsWindow.setObjectName("Q7ToolsWindow")
        Q7ToolsWindow.setWindowModality(QtCore.Qt.NonModal)
        Q7ToolsWindow.resize(580, 364)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7ToolsWindow.sizePolicy().hasHeightForWidth())
        Q7ToolsWindow.setSizePolicy(sizePolicy)
        Q7ToolsWindow.setMinimumSize(QtCore.QSize(580, 364))
        Q7ToolsWindow.setMaximumSize(QtCore.QSize(580, 364))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/cgSpy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Q7ToolsWindow.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(Q7ToolsWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(Q7ToolsWindow)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setObjectName("tabWidget")
        self.Queries = QtWidgets.QWidget()
        self.Queries.setObjectName("Queries")
        self.label_15 = QtWidgets.QLabel(self.Queries)
        self.label_15.setGeometry(QtCore.QRect(50, 25, 37, 25))
        self.label_15.setObjectName("label_15")
        self.cGroup = QtWidgets.QComboBox(self.Queries)
        self.cGroup.setEnabled(True)
        self.cGroup.setGeometry(QtCore.QRect(95, 25, 74, 21))
        self.cGroup.setObjectName("cGroup")
        self.label_16 = QtWidgets.QLabel(self.Queries)
        self.label_16.setGeometry(QtCore.QRect(50, 55, 36, 25))
        self.label_16.setObjectName("label_16")
        self.cQuery = QtWidgets.QComboBox(self.Queries)
        self.cQuery.setGeometry(QtCore.QRect(95, 55, 378, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cQuery.sizePolicy().hasHeightForWidth())
        self.cQuery.setSizePolicy(sizePolicy)
        self.cQuery.setObjectName("cQuery")
        self.bOperateDoc = QtWidgets.QPushButton(self.Queries)
        self.bOperateDoc.setGeometry(QtCore.QRect(15, 55, 25, 25))
        self.bOperateDoc.setMinimumSize(QtCore.QSize(25, 25))
        self.bOperateDoc.setMaximumSize(QtCore.QSize(25, 25))
        self.bOperateDoc.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icons/operate-doc.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bOperateDoc.setIcon(icon1)
        self.bOperateDoc.setObjectName("bOperateDoc")
        self.spinBox = QtWidgets.QSpinBox(self.Queries)
        self.spinBox.setGeometry(QtCore.QRect(190, 235, 53, 22))
        self.spinBox.setObjectName("spinBox")
        self.checkBox = QtWidgets.QCheckBox(self.Queries)
        self.checkBox.setEnabled(False)
        self.checkBox.setGeometry(QtCore.QRect(20, 240, 171, 20))
        self.checkBox.setObjectName("checkBox")
        self.groupBox_2 = QtWidgets.QGroupBox(self.Queries)
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setGeometry(QtCore.QRect(15, 94, 521, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton.setEnabled(False)
        self.radioButton.setGeometry(QtCore.QRect(15, 20, 261, 20))
        self.radioButton.setObjectName("radioButton")
        self.eUserVariable = QtWidgets.QLineEdit(self.groupBox_2)
        self.eUserVariable.setGeometry(QtCore.QRect(130, 70, 376, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.eUserVariable.sizePolicy().hasHeightForWidth())
        self.eUserVariable.setSizePolicy(sizePolicy)
        self.eUserVariable.setObjectName("eUserVariable")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_2.setEnabled(True)
        self.radioButton_2.setGeometry(QtCore.QRect(15, 70, 92, 20))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_3.setEnabled(False)
        self.radioButton_3.setGeometry(QtCore.QRect(15, 95, 131, 20))
        self.radioButton_3.setObjectName("radioButton_3")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setEnabled(False)
        self.lineEdit.setGeometry(QtCore.QRect(130, 95, 376, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_4.setEnabled(False)
        self.radioButton_4.setGeometry(QtCore.QRect(15, 45, 426, 20))
        self.radioButton_4.setObjectName("radioButton_4")
        self.pushButton = QtWidgets.QPushButton(self.Queries)
        self.pushButton.setGeometry(QtCore.QRect(510, 55, 25, 25))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(25)
        sizePolicy.setVerticalStretch(25)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMinimumSize(QtCore.QSize(25, 25))
        self.pushButton.setMaximumSize(QtCore.QSize(25, 25))
        self.pushButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/icons/flag-none.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon2)
        self.pushButton.setObjectName("pushButton")
        self.bApplyBox = QtWidgets.QFrame(self.Queries)
        self.bApplyBox.setGeometry(QtCore.QRect(495, 235, 36, 36))
        self.bApplyBox.setFrameShape(QtWidgets.QFrame.Box)
        self.bApplyBox.setFrameShadow(QtWidgets.QFrame.Plain)
        self.bApplyBox.setLineWidth(2)
        self.bApplyBox.setObjectName("bApplyBox")
        self.bApply = QtWidgets.QPushButton(self.bApplyBox)
        self.bApply.setGeometry(QtCore.QRect(5, 5, 25, 25))
        self.bApply.setMinimumSize(QtCore.QSize(25, 25))
        self.bApply.setMaximumSize(QtCore.QSize(25, 25))
        self.bApply.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/icons/operate-execute.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bApply.setIcon(icon3)
        self.bApply.setObjectName("bApply")
        self.tabWidget.addTab(self.Queries, "")
        self.search = QtWidgets.QWidget()
        self.search.setObjectName("search")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.search)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(5, 5, 541, 271))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.bSaveAsQuery = QtWidgets.QPushButton(self.groupBox)
        self.bSaveAsQuery.setGeometry(QtCore.QRect(350, 235, 24, 24))
        self.bSaveAsQuery.setMinimumSize(QtCore.QSize(24, 24))
        self.bSaveAsQuery.setMaximumSize(QtCore.QSize(24, 24))
        self.bSaveAsQuery.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/images/icons/operate-save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bSaveAsQuery.setIcon(icon4)
        self.bSaveAsQuery.setObjectName("bSaveAsQuery")
        self.eSaveAsQuery = QtWidgets.QLineEdit(self.groupBox)
        self.eSaveAsQuery.setGeometry(QtCore.QRect(80, 235, 261, 21))
        self.eSaveAsQuery.setObjectName("eSaveAsQuery")
        self.sLevel = QtWidgets.QSlider(self.groupBox)
        self.sLevel.setGeometry(QtCore.QRect(420, 100, 16, 56))
        self.sLevel.setMaximum(2)
        self.sLevel.setSliderPosition(1)
        self.sLevel.setOrientation(QtCore.Qt.Vertical)
        self.sLevel.setInvertedAppearance(True)
        self.sLevel.setInvertedControls(True)
        self.sLevel.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.sLevel.setObjectName("sLevel")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(360, 95, 53, 15))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(360, 120, 53, 15))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_14 = QtWidgets.QLabel(self.groupBox)
        self.label_14.setGeometry(QtCore.QRect(360, 145, 53, 15))
        self.label_14.setObjectName("label_14")
        self.gCriteria = QtWidgets.QGroupBox(self.groupBox)
        self.gCriteria.setGeometry(QtCore.QRect(10, 10, 336, 196))
        self.gCriteria.setObjectName("gCriteria")
        self.label_9 = QtWidgets.QLabel(self.gCriteria)
        self.label_9.setGeometry(QtCore.QRect(15, 30, 76, 16))
        self.label_9.setObjectName("label_9")
        self.eName = QtWidgets.QLineEdit(self.gCriteria)
        self.eName.setGeometry(QtCore.QRect(75, 25, 181, 21))
        self.eName.setObjectName("eName")
        self.cDataCheck = QtWidgets.QGroupBox(self.gCriteria)
        self.cDataCheck.setEnabled(True)
        self.cDataCheck.setGeometry(QtCore.QRect(10, 85, 316, 106))
        self.cDataCheck.setCheckable(True)
        self.cDataCheck.setChecked(False)
        self.cDataCheck.setObjectName("cDataCheck")
        self.cDataType = QtWidgets.QComboBox(self.cDataCheck)
        self.cDataType.setGeometry(QtCore.QRect(65, 40, 41, 22))
        self.cDataType.setObjectName("cDataType")
        self.label_11 = QtWidgets.QLabel(self.cDataCheck)
        self.label_11.setGeometry(QtCore.QRect(10, 45, 76, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.cDataCheck)
        self.label_12.setGeometry(QtCore.QRect(10, 75, 76, 16))
        self.label_12.setObjectName("label_12")
        self.eValue = QtWidgets.QLineEdit(self.cDataCheck)
        self.eValue.setGeometry(QtCore.QRect(65, 15, 181, 21))
        self.eValue.setObjectName("eValue")
        self.label_13 = QtWidgets.QLabel(self.cDataCheck)
        self.label_13.setGeometry(QtCore.QRect(10, 20, 76, 16))
        self.label_13.setObjectName("label_13")
        self.eMaxSize = QtWidgets.QSpinBox(self.cDataCheck)
        self.eMaxSize.setGeometry(QtCore.QRect(65, 70, 81, 22))
        self.eMaxSize.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.eMaxSize.setProperty("value", 65)
        self.eMaxSize.setObjectName("eMaxSize")
        self.cRegexpValue = QtWidgets.QPushButton(self.cDataCheck)
        self.cRegexpValue.setGeometry(QtCore.QRect(250, 15, 24, 24))
        self.cRegexpValue.setMinimumSize(QtCore.QSize(24, 24))
        self.cRegexpValue.setMaximumSize(QtCore.QSize(24, 24))
        self.cRegexpValue.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/images/icons/regexp.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cRegexpValue.setIcon(icon5)
        self.cRegexpValue.setCheckable(True)
        self.cRegexpValue.setObjectName("cRegexpValue")
        self.cNotValue = QtWidgets.QPushButton(self.cDataCheck)
        self.cNotValue.setGeometry(QtCore.QRect(280, 15, 24, 24))
        self.cNotValue.setMinimumSize(QtCore.QSize(24, 24))
        self.cNotValue.setMaximumSize(QtCore.QSize(24, 24))
        self.cNotValue.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/images/icons/delete.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cNotValue.setIcon(icon6)
        self.cNotValue.setCheckable(True)
        self.cNotValue.setObjectName("cNotValue")
        self.cNotDataType = QtWidgets.QPushButton(self.cDataCheck)
        self.cNotDataType.setGeometry(QtCore.QRect(280, 45, 24, 24))
        self.cNotDataType.setMinimumSize(QtCore.QSize(24, 24))
        self.cNotDataType.setMaximumSize(QtCore.QSize(24, 24))
        self.cNotDataType.setText("")
        self.cNotDataType.setIcon(icon6)
        self.cNotDataType.setCheckable(True)
        self.cNotDataType.setObjectName("cNotDataType")
        self.label_10 = QtWidgets.QLabel(self.gCriteria)
        self.label_10.setGeometry(QtCore.QRect(15, 60, 76, 16))
        self.label_10.setObjectName("label_10")
        self.cSIDStype = QtWidgets.QComboBox(self.gCriteria)
        self.cSIDStype.setGeometry(QtCore.QRect(75, 55, 181, 22))
        self.cSIDStype.setEditable(True)
        self.cSIDStype.setObjectName("cSIDStype")
        self.cRegexpName = QtWidgets.QPushButton(self.gCriteria)
        self.cRegexpName.setGeometry(QtCore.QRect(260, 25, 24, 24))
        self.cRegexpName.setMinimumSize(QtCore.QSize(24, 24))
        self.cRegexpName.setMaximumSize(QtCore.QSize(24, 24))
        self.cRegexpName.setText("")
        self.cRegexpName.setIcon(icon5)
        self.cRegexpName.setCheckable(True)
        self.cRegexpName.setObjectName("cRegexpName")
        self.cNotName = QtWidgets.QPushButton(self.gCriteria)
        self.cNotName.setGeometry(QtCore.QRect(290, 25, 24, 24))
        self.cNotName.setMinimumSize(QtCore.QSize(24, 24))
        self.cNotName.setMaximumSize(QtCore.QSize(24, 24))
        self.cNotName.setText("")
        self.cNotName.setIcon(icon6)
        self.cNotName.setCheckable(True)
        self.cNotName.setObjectName("cNotName")
        self.cRegexpSIDStype = QtWidgets.QPushButton(self.gCriteria)
        self.cRegexpSIDStype.setGeometry(QtCore.QRect(260, 55, 24, 24))
        self.cRegexpSIDStype.setMinimumSize(QtCore.QSize(24, 24))
        self.cRegexpSIDStype.setMaximumSize(QtCore.QSize(24, 24))
        self.cRegexpSIDStype.setText("")
        self.cRegexpSIDStype.setIcon(icon5)
        self.cRegexpSIDStype.setCheckable(True)
        self.cRegexpSIDStype.setObjectName("cRegexpSIDStype")
        self.cNotSIDStype = QtWidgets.QPushButton(self.gCriteria)
        self.cNotSIDStype.setGeometry(QtCore.QRect(290, 55, 24, 24))
        self.cNotSIDStype.setMinimumSize(QtCore.QSize(24, 24))
        self.cNotSIDStype.setMaximumSize(QtCore.QSize(24, 24))
        self.cNotSIDStype.setText("")
        self.cNotSIDStype.setIcon(icon6)
        self.cNotSIDStype.setCheckable(True)
        self.cNotSIDStype.setObjectName("cNotSIDStype")
        self.rAddToCurrent = QtWidgets.QRadioButton(self.groupBox)
        self.rAddToCurrent.setGeometry(QtCore.QRect(355, 15, 176, 20))
        self.rAddToCurrent.setChecked(True)
        self.rAddToCurrent.setObjectName("rAddToCurrent")
        self.rClearCurrent = QtWidgets.QRadioButton(self.groupBox)
        self.rClearCurrent.setGeometry(QtCore.QRect(355, 55, 171, 20))
        self.rClearCurrent.setObjectName("rClearCurrent")
        self.rWithinCurrent = QtWidgets.QRadioButton(self.groupBox)
        self.rWithinCurrent.setGeometry(QtCore.QRect(355, 35, 156, 20))
        self.rWithinCurrent.setObjectName("rWithinCurrent")
        self.bRunSearch = QtWidgets.QPushButton(self.groupBox)
        self.bRunSearch.setGeometry(QtCore.QRect(350, 175, 24, 24))
        self.bRunSearch.setMinimumSize(QtCore.QSize(24, 24))
        self.bRunSearch.setMaximumSize(QtCore.QSize(24, 24))
        self.bRunSearch.setText("")
        self.bRunSearch.setIcon(icon3)
        self.bRunSearch.setObjectName("bRunSearch")
        self.line = QtWidgets.QFrame(self.groupBox)
        self.line.setGeometry(QtCore.QRect(10, 215, 516, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_19 = QtWidgets.QLabel(self.groupBox)
        self.label_19.setGeometry(QtCore.QRect(40, 240, 53, 15))
        self.label_19.setObjectName("label_19")
        self.bFindQuery = QtWidgets.QPushButton(self.groupBox)
        self.bFindQuery.setGeometry(QtCore.QRect(10, 235, 24, 24))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(24)
        sizePolicy.setVerticalStretch(24)
        sizePolicy.setHeightForWidth(self.bFindQuery.sizePolicy().hasHeightForWidth())
        self.bFindQuery.setSizePolicy(sizePolicy)
        self.bFindQuery.setMinimumSize(QtCore.QSize(24, 24))
        self.bFindQuery.setMaximumSize(QtCore.QSize(24, 24))
        self.bFindQuery.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/images/icons/zoompoint.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bFindQuery.setIcon(icon7)
        self.bFindQuery.setObjectName("bFindQuery")
        self.eWithLinks = QtWidgets.QCheckBox(self.groupBox)
        self.eWithLinks.setGeometry(QtCore.QRect(385, 180, 111, 20))
        self.eWithLinks.setChecked(True)
        self.eWithLinks.setObjectName("eWithLinks")
        self.verticalLayout_2.addWidget(self.groupBox)
        self.tabWidget.addTab(self.search, "")
        self.diff = QtWidgets.QWidget()
        self.diff.setObjectName("diff")
        self.gdiff = QtWidgets.QGroupBox(self.diff)
        self.gdiff.setGeometry(QtCore.QRect(5, 5, 541, 266))
        self.gdiff.setObjectName("gdiff")
        self.label = QtWidgets.QLabel(self.gdiff)
        self.label.setGeometry(QtCore.QRect(25, 25, 58, 16))
        self.label.setObjectName("label")
        self.cAncestor = QtWidgets.QComboBox(self.gdiff)
        self.cAncestor.setGeometry(QtCore.QRect(15, 45, 81, 22))
        self.cAncestor.setObjectName("cAncestor")
        self.cVersionA = QtWidgets.QComboBox(self.gdiff)
        self.cVersionA.setGeometry(QtCore.QRect(140, 45, 81, 22))
        self.cVersionA.setObjectName("cVersionA")
        self.label_2 = QtWidgets.QLabel(self.gdiff)
        self.label_2.setGeometry(QtCore.QRect(150, 25, 58, 16))
        self.label_2.setObjectName("label_2")
        self.cShowA = QtWidgets.QCheckBox(self.gdiff)
        self.cShowA.setEnabled(False)
        self.cShowA.setGeometry(QtCore.QRect(235, 45, 176, 21))
        self.cShowA.setCheckable(True)
        self.cShowA.setChecked(False)
        self.cShowA.setObjectName("cShowA")
        self.bDiff = QtWidgets.QPushButton(self.gdiff)
        self.bDiff.setGeometry(QtCore.QRect(395, 45, 25, 25))
        self.bDiff.setMinimumSize(QtCore.QSize(25, 25))
        self.bDiff.setMaximumSize(QtCore.QSize(25, 25))
        self.bDiff.setText("")
        self.bDiff.setIcon(icon3)
        self.bDiff.setObjectName("bDiff")
        self.tabWidget.addTab(self.diff, "")
        self.merge = QtWidgets.QWidget()
        self.merge.setObjectName("merge")
        self.gdiff_2 = QtWidgets.QGroupBox(self.merge)
        self.gdiff_2.setEnabled(False)
        self.gdiff_2.setGeometry(QtCore.QRect(5, 5, 541, 266))
        self.gdiff_2.setObjectName("gdiff_2")
        self.bMerge = QtWidgets.QPushButton(self.gdiff_2)
        self.bMerge.setGeometry(QtCore.QRect(415, 130, 25, 25))
        self.bMerge.setMinimumSize(QtCore.QSize(25, 25))
        self.bMerge.setMaximumSize(QtCore.QSize(25, 25))
        self.bMerge.setText("")
        self.bMerge.setIcon(icon3)
        self.bMerge.setObjectName("bMerge")
        self.groupBox_5 = QtWidgets.QGroupBox(self.gdiff_2)
        self.groupBox_5.setGeometry(QtCore.QRect(150, 20, 386, 81))
        self.groupBox_5.setObjectName("groupBox_5")
        self.cTreeA = QtWidgets.QComboBox(self.groupBox_5)
        self.cTreeA.setGeometry(QtCore.QRect(10, 20, 81, 22))
        self.cTreeA.setObjectName("cTreeA")
        self.rForceA = QtWidgets.QRadioButton(self.groupBox_5)
        self.rForceA.setGeometry(QtCore.QRect(160, 15, 201, 21))
        self.rForceA.setObjectName("rForceA")
        self.rAncestorA = QtWidgets.QRadioButton(self.groupBox_5)
        self.rAncestorA.setGeometry(QtCore.QRect(160, 35, 226, 21))
        self.rAncestorA.setObjectName("rAncestorA")
        self.ePrefixA = QtWidgets.QLineEdit(self.groupBox_5)
        self.ePrefixA.setGeometry(QtCore.QRect(10, 50, 76, 22))
        self.ePrefixA.setObjectName("ePrefixA")
        self.label_5 = QtWidgets.QLabel(self.groupBox_5)
        self.label_5.setGeometry(QtCore.QRect(90, 50, 58, 16))
        self.label_5.setObjectName("label_5")
        self.groupBox_6 = QtWidgets.QGroupBox(self.gdiff_2)
        self.groupBox_6.setGeometry(QtCore.QRect(150, 180, 386, 81))
        self.groupBox_6.setObjectName("groupBox_6")
        self.cTreeB = QtWidgets.QComboBox(self.groupBox_6)
        self.cTreeB.setGeometry(QtCore.QRect(10, 20, 81, 22))
        self.cTreeB.setObjectName("cTreeB")
        self.rForceB = QtWidgets.QRadioButton(self.groupBox_6)
        self.rForceB.setGeometry(QtCore.QRect(155, 15, 201, 21))
        self.rForceB.setObjectName("rForceB")
        self.rAncestorB = QtWidgets.QRadioButton(self.groupBox_6)
        self.rAncestorB.setGeometry(QtCore.QRect(155, 35, 226, 21))
        self.rAncestorB.setObjectName("rAncestorB")
        self.ePrefixB = QtWidgets.QLineEdit(self.groupBox_6)
        self.ePrefixB.setGeometry(QtCore.QRect(10, 50, 76, 22))
        self.ePrefixB.setObjectName("ePrefixB")
        self.label_6 = QtWidgets.QLabel(self.groupBox_6)
        self.label_6.setGeometry(QtCore.QRect(90, 55, 58, 16))
        self.label_6.setObjectName("label_6")
        self.groupBox_7 = QtWidgets.QGroupBox(self.gdiff_2)
        self.groupBox_7.setGeometry(QtCore.QRect(5, 100, 381, 76))
        self.groupBox_7.setObjectName("groupBox_7")
        self.cTreeAncestor = QtWidgets.QComboBox(self.groupBox_7)
        self.cTreeAncestor.setEnabled(False)
        self.cTreeAncestor.setGeometry(QtCore.QRect(10, 30, 81, 22))
        self.cTreeAncestor.setObjectName("cTreeAncestor")
        self.rForceNone = QtWidgets.QRadioButton(self.groupBox_7)
        self.rForceNone.setGeometry(QtCore.QRect(115, 10, 216, 21))
        self.rForceNone.setObjectName("rForceNone")
        self.rAncestor = QtWidgets.QRadioButton(self.groupBox_7)
        self.rAncestor.setGeometry(QtCore.QRect(115, 30, 211, 21))
        self.rAncestor.setObjectName("rAncestor")
        self.cAutoMerge = QtWidgets.QCheckBox(self.gdiff_2)
        self.cAutoMerge.setGeometry(QtCore.QRect(450, 130, 86, 21))
        self.cAutoMerge.setObjectName("cAutoMerge")
        self.tabWidget.addTab(self.merge, "")
        self.horizontalLayout_2.addWidget(self.tabWidget)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.bInfo = QtWidgets.QPushButton(Q7ToolsWindow)
        self.bInfo.setMinimumSize(QtCore.QSize(25, 25))
        self.bInfo.setMaximumSize(QtCore.QSize(25, 25))
        self.bInfo.setText("")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/images/icons/help-view.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bInfo.setIcon(icon8)
        self.bInfo.setObjectName("bInfo")
        self.horizontalLayout.addWidget(self.bInfo)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.bClose = QtWidgets.QPushButton(Q7ToolsWindow)
        self.bClose.setObjectName("bClose")
        self.horizontalLayout.addWidget(self.bClose)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Q7ToolsWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Q7ToolsWindow)

    def retranslateUi(self, Q7ToolsWindow):
        _translate = QtCore.QCoreApplication.translate
        Q7ToolsWindow.setWindowTitle(_translate("Q7ToolsWindow", "Form"))
        self.label_15.setText(_translate("Q7ToolsWindow", "Group:"))
        self.label_16.setText(_translate("Q7ToolsWindow", "Query:"))
        self.bOperateDoc.setToolTip(_translate("Q7ToolsWindow", "Show Query doc"))
        self.checkBox.setText(_translate("Q7ToolsWindow", "Show predefined query #"))
        self.groupBox_2.setTitle(_translate("Q7ToolsWindow", "Query arguments :"))
        self.radioButton.setText(_translate("Q7ToolsWindow", "Use last selected value as arg"))
        self.radioButton_2.setText(_translate("Q7ToolsWindow", "Plain value"))
        self.radioButton_3.setText(_translate("Q7ToolsWindow", "Regexp value"))
        self.radioButton_4.setText(_translate("Q7ToolsWindow", "Use ALL selected value as arg (loop on values and add selections)"))
        self.bApply.setToolTip(_translate("Q7ToolsWindow", "Run Query"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Queries), _translate("Q7ToolsWindow", "Queries"))
        self.bSaveAsQuery.setToolTip(_translate("Q7ToolsWindow", "Save as a query"))
        self.label_7.setText(_translate("Q7ToolsWindow", "Ancestor"))
        self.label_8.setText(_translate("Q7ToolsWindow", "Current"))
        self.label_14.setText(_translate("Q7ToolsWindow", "Children"))
        self.gCriteria.setTitle(_translate("Q7ToolsWindow", "###"))
        self.label_9.setText(_translate("Q7ToolsWindow", "name:"))
        self.cDataCheck.setTitle(_translate("Q7ToolsWindow", "Data"))
        self.label_11.setText(_translate("Q7ToolsWindow", "type:"))
        self.label_12.setText(_translate("Q7ToolsWindow", "max size:"))
        self.label_13.setText(_translate("Q7ToolsWindow", "value:"))
        self.cRegexpValue.setToolTip(_translate("Q7ToolsWindow", "String is a regular expression"))
        self.cNotValue.setToolTip(_translate("Q7ToolsWindow", "Use all strings but this one"))
        self.cNotDataType.setToolTip(_translate("Q7ToolsWindow", "Use all strings but this one"))
        self.label_10.setText(_translate("Q7ToolsWindow", "SIDS type:"))
        self.cRegexpName.setToolTip(_translate("Q7ToolsWindow", "String is a regular expression"))
        self.cNotName.setToolTip(_translate("Q7ToolsWindow", "Use all strings but this one"))
        self.cRegexpSIDStype.setToolTip(_translate("Q7ToolsWindow", "String is a regular expression"))
        self.cNotSIDStype.setToolTip(_translate("Q7ToolsWindow", "Use all strings but this one"))
        self.rAddToCurrent.setText(_translate("Q7ToolsWindow", "Add to current selection"))
        self.rClearCurrent.setText(_translate("Q7ToolsWindow", "Clear current selection first"))
        self.rWithinCurrent.setText(_translate("Q7ToolsWindow", "Within current selection"))
        self.bRunSearch.setToolTip(_translate("Q7ToolsWindow", "Run current selection criteria"))
        self.label_19.setText(_translate("Q7ToolsWindow", "name:"))
        self.bFindQuery.setToolTip(_translate("Q7ToolsWindow", "Load search query with this name"))
        self.eWithLinks.setText(_translate("Q7ToolsWindow", "Include links"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.search), _translate("Q7ToolsWindow", "Search"))
        self.gdiff.setTitle(_translate("Q7ToolsWindow", "All tree identifications use view number"))
        self.label.setText(_translate("Q7ToolsWindow", "Version A"))
        self.cAncestor.setToolTip(_translate("Q7ToolsWindow", "Select first tree to diff"))
        self.label_2.setText(_translate("Q7ToolsWindow", "Version B"))
        self.cShowA.setText(_translate("Q7ToolsWindow", "Show result on B view"))
        self.bDiff.setToolTip(_translate("Q7ToolsWindow", "Process the diff"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.diff), _translate("Q7ToolsWindow", "Diff"))
        self.gdiff_2.setTitle(_translate("Q7ToolsWindow", "All tree identifications use view number"))
        self.groupBox_5.setTitle(_translate("Q7ToolsWindow", "Version A"))
        self.rForceA.setText(_translate("Q7ToolsWindow", "Force A values when different"))
        self.rAncestorA.setText(_translate("Q7ToolsWindow", "A replaces the common ancestor"))
        self.label_5.setText(_translate("Q7ToolsWindow", "Prefix"))
        self.groupBox_6.setTitle(_translate("Q7ToolsWindow", "Version B"))
        self.rForceB.setText(_translate("Q7ToolsWindow", "Force B values when different"))
        self.rAncestorB.setText(_translate("Q7ToolsWindow", "B replaces the common ancestor"))
        self.label_6.setText(_translate("Q7ToolsWindow", "Prefix"))
        self.groupBox_7.setTitle(_translate("Q7ToolsWindow", "Ancestor"))
        self.rForceNone.setText(_translate("Q7ToolsWindow", "Keep both values when different"))
        self.rAncestor.setText(_translate("Q7ToolsWindow", "Use common ancestor"))
        self.cAutoMerge.setText(_translate("Q7ToolsWindow", "Auto"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.merge), _translate("Q7ToolsWindow", "Merge"))
        self.bClose.setText(_translate("Q7ToolsWindow", "Close"))
from . import Res_rc
