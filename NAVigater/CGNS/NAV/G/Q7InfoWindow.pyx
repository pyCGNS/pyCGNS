# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CGNS/NAV/T/Q7InfoWindow.ui'
#
# Created: Fri Oct 12 15:26:37 2012
#      by: pyside-uic 0.2.13 running on PySide 1.0.9
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Q7InfoWindow(object):
    def setupUi(self, Q7InfoWindow):
        Q7InfoWindow.setObjectName("Q7InfoWindow")
        Q7InfoWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        Q7InfoWindow.resize(715, 390)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7InfoWindow.sizePolicy().hasHeightForWidth())
        Q7InfoWindow.setSizePolicy(sizePolicy)
        Q7InfoWindow.setMinimumSize(QtCore.QSize(715, 390))
        Q7InfoWindow.setMaximumSize(QtCore.QSize(715, 390))
        self.gridLayout = QtGui.QGridLayout(Q7InfoWindow)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bClose = QtGui.QPushButton(Q7InfoWindow)
        self.bClose.setObjectName("bClose")
        self.horizontalLayout_2.addWidget(self.bClose)
        self.gridLayout.addLayout(self.horizontalLayout_2, 7, 0, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.groupBox_2 = QtGui.QGroupBox(Q7InfoWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setMinimumSize(QtCore.QSize(200, 0))
        self.groupBox_2.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_4 = QtGui.QGroupBox(self.groupBox_2)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 110, 181, 81))
        self.groupBox_4.setObjectName("groupBox_4")
        self.cHDF5 = QtGui.QCheckBox(self.groupBox_4)
        self.cHDF5.setGeometry(QtCore.QRect(10, 20, 94, 21))
        self.cHDF5.setCheckable(True)
        self.cHDF5.setObjectName("cHDF5")
        self.cADF = QtGui.QCheckBox(self.groupBox_4)
        self.cADF.setGeometry(QtCore.QRect(80, 20, 94, 21))
        self.cADF.setCheckable(True)
        self.cADF.setObjectName("cADF")
        self.label_14 = QtGui.QLabel(self.groupBox_4)
        self.label_14.setGeometry(QtCore.QRect(10, 50, 91, 16))
        self.label_14.setObjectName("label_14")
        self.eVersionHDF5 = QtGui.QLineEdit(self.groupBox_4)
        self.eVersionHDF5.setGeometry(QtCore.QRect(100, 50, 71, 22))
        self.eVersionHDF5.setReadOnly(True)
        self.eVersionHDF5.setObjectName("eVersionHDF5")
        self.label_4 = QtGui.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(40, 20, 58, 16))
        self.label_4.setObjectName("label_4")
        self.eVersion = QtGui.QLineEdit(self.groupBox_2)
        self.eVersion.setGeometry(QtCore.QRect(90, 20, 71, 22))
        self.eVersion.setReadOnly(True)
        self.eVersion.setObjectName("eVersion")
        self.groupBox_5 = QtGui.QGroupBox(self.groupBox_2)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 40, 181, 71))
        self.groupBox_5.setObjectName("groupBox_5")
        self.cREAD = QtGui.QCheckBox(self.groupBox_5)
        self.cREAD.setGeometry(QtCore.QRect(10, 20, 94, 21))
        self.cREAD.setCheckable(True)
        self.cREAD.setObjectName("cREAD")
        self.cMODIFY = QtGui.QCheckBox(self.groupBox_5)
        self.cMODIFY.setGeometry(QtCore.QRect(10, 40, 94, 21))
        self.cMODIFY.setCheckable(True)
        self.cMODIFY.setObjectName("cMODIFY")
        self.cConverted = QtGui.QCheckBox(self.groupBox_2)
        self.cConverted.setGeometry(QtCore.QRect(10, 200, 161, 21))
        self.cConverted.setCheckable(True)
        self.cConverted.setObjectName("cConverted")
        self.pushButton_3 = QtGui.QPushButton(self.groupBox_2)
        self.pushButton_3.setGeometry(QtCore.QRect(170, 200, 25, 25))
        self.pushButton_3.setMinimumSize(QtCore.QSize(25, 25))
        self.pushButton_3.setMaximumSize(QtCore.QSize(25, 25))
        self.pushButton_3.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/help-view.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_3.setIcon(icon)
        self.pushButton_3.setFlat(True)
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_12 = QtGui.QLabel(self.groupBox_2)
        self.label_12.setGeometry(QtCore.QRect(30, 230, 161, 31))
        self.label_12.setWordWrap(True)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_3.addWidget(self.groupBox_2)
        self.groupBox_6 = QtGui.QGroupBox(Q7InfoWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_6.sizePolicy().hasHeightForWidth())
        self.groupBox_6.setSizePolicy(sizePolicy)
        self.groupBox_6.setMinimumSize(QtCore.QSize(220, 0))
        self.groupBox_6.setObjectName("groupBox_6")
        self.groupBox = QtGui.QGroupBox(self.groupBox_6)
        self.groupBox.setGeometry(QtCore.QRect(10, 130, 201, 131))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(150, 0))
        self.groupBox.setObjectName("groupBox")
        self.cHasLinks = QtGui.QCheckBox(self.groupBox)
        self.cHasLinks.setGeometry(QtCore.QRect(10, 20, 86, 21))
        self.cHasLinks.setCheckable(False)
        self.cHasLinks.setObjectName("cHasLinks")
        self.cSameFS = QtGui.QCheckBox(self.groupBox)
        self.cSameFS.setGeometry(QtCore.QRect(10, 80, 161, 21))
        self.cSameFS.setCheckable(False)
        self.cSameFS.setObjectName("cSameFS")
        self.cBadLinks = QtGui.QCheckBox(self.groupBox)
        self.cBadLinks.setGeometry(QtCore.QRect(10, 100, 141, 20))
        self.cBadLinks.setCheckable(False)
        self.cBadLinks.setObjectName("cBadLinks")
        self.cModeProp = QtGui.QCheckBox(self.groupBox)
        self.cModeProp.setGeometry(QtCore.QRect(10, 60, 131, 21))
        self.cModeProp.setCheckable(False)
        self.cModeProp.setObjectName("cModeProp")
        self.pushButton = QtGui.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(150, 100, 25, 25))
        self.pushButton.setMinimumSize(QtCore.QSize(25, 25))
        self.pushButton.setMaximumSize(QtCore.QSize(25, 25))
        self.pushButton.setText("")
        self.pushButton.setIcon(icon)
        self.pushButton.setFlat(True)
        self.pushButton.setObjectName("pushButton")
        self.cHasInt64 = QtGui.QCheckBox(self.groupBox_6)
        self.cHasInt64.setGeometry(QtCore.QRect(10, 40, 201, 21))
        self.cHasInt64.setCheckable(True)
        self.cHasInt64.setObjectName("cHasInt64")
        self.cNODATA = QtGui.QCheckBox(self.groupBox_6)
        self.cNODATA.setGeometry(QtCore.QRect(10, 20, 86, 21))
        self.cNODATA.setCheckable(True)
        self.cNODATA.setObjectName("cNODATA")
        self.cNoFollow = QtGui.QCheckBox(self.groupBox_6)
        self.cNoFollow.setGeometry(QtCore.QRect(20, 170, 191, 21))
        self.cNoFollow.setCheckable(False)
        self.cNoFollow.setObjectName("cNoFollow")
        self.pushButton_2 = QtGui.QPushButton(self.groupBox_6)
        self.pushButton_2.setGeometry(QtCore.QRect(100, 20, 25, 25))
        self.pushButton_2.setMinimumSize(QtCore.QSize(25, 25))
        self.pushButton_2.setMaximumSize(QtCore.QSize(25, 25))
        self.pushButton_2.setText("")
        self.pushButton_2.setIcon(icon)
        self.pushButton_2.setFlat(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_9 = QtGui.QLabel(self.groupBox_6)
        self.label_9.setGeometry(QtCore.QRect(10, 70, 111, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtGui.QLabel(self.groupBox_6)
        self.label_10.setGeometry(QtCore.QRect(10, 100, 81, 16))
        self.label_10.setObjectName("label_10")
        self.eNodes = QtGui.QLineEdit(self.groupBox_6)
        self.eNodes.setGeometry(QtCore.QRect(120, 70, 91, 22))
        self.eNodes.setReadOnly(True)
        self.eNodes.setObjectName("eNodes")
        self.eDepth = QtGui.QLineEdit(self.groupBox_6)
        self.eDepth.setGeometry(QtCore.QRect(120, 100, 91, 22))
        self.eDepth.setReadOnly(True)
        self.eDepth.setObjectName("eDepth")
        self.horizontalLayout_3.addWidget(self.groupBox_6)
        self.groupBox_3 = QtGui.QGroupBox(Q7InfoWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setMinimumSize(QtCore.QSize(250, 0))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_3 = QtGui.QLabel(self.groupBox_3)
        self.label_3.setGeometry(QtCore.QRect(10, 50, 81, 16))
        self.label_3.setObjectName("label_3")
        self.eFileSize = QtGui.QLineEdit(self.groupBox_3)
        self.eFileSize.setGeometry(QtCore.QRect(90, 20, 161, 22))
        self.eFileSize.setReadOnly(True)
        self.eFileSize.setObjectName("eFileSize")
        self.label_2 = QtGui.QLabel(self.groupBox_3)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 58, 16))
        self.label_2.setObjectName("label_2")
        self.eMergeSize = QtGui.QLineEdit(self.groupBox_3)
        self.eMergeSize.setGeometry(QtCore.QRect(90, 50, 161, 22))
        self.eMergeSize.setReadOnly(True)
        self.eMergeSize.setObjectName("eMergeSize")
        self.eLastDate = QtGui.QLineEdit(self.groupBox_3)
        self.eLastDate.setGeometry(QtCore.QRect(90, 90, 161, 22))
        self.eLastDate.setReadOnly(True)
        self.eLastDate.setObjectName("eLastDate")
        self.label_5 = QtGui.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(10, 90, 81, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtGui.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(10, 110, 71, 31))
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtGui.QLabel(self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(10, 160, 58, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtGui.QLabel(self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(10, 220, 58, 16))
        self.label_8.setObjectName("label_8")
        self.eModifDate = QtGui.QLineEdit(self.groupBox_3)
        self.eModifDate.setGeometry(QtCore.QRect(90, 120, 161, 22))
        self.eModifDate.setReadOnly(True)
        self.eModifDate.setObjectName("eModifDate")
        self.eOwner = QtGui.QLineEdit(self.groupBox_3)
        self.eOwner.setGeometry(QtCore.QRect(90, 160, 113, 22))
        self.eOwner.setReadOnly(True)
        self.eOwner.setObjectName("eOwner")
        self.eRights = QtGui.QLineEdit(self.groupBox_3)
        self.eRights.setGeometry(QtCore.QRect(90, 220, 113, 22))
        self.eRights.setReadOnly(True)
        self.eRights.setObjectName("eRights")
        self.eGroup = QtGui.QLineEdit(self.groupBox_3)
        self.eGroup.setGeometry(QtCore.QRect(90, 190, 113, 22))
        self.eGroup.setReadOnly(True)
        self.eGroup.setObjectName("eGroup")
        self.label_11 = QtGui.QLabel(self.groupBox_3)
        self.label_11.setGeometry(QtCore.QRect(10, 190, 58, 16))
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_3.addWidget(self.groupBox_3)
        self.gridLayout.addLayout(self.horizontalLayout_3, 4, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtGui.QLabel(Q7InfoWindow)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.eFilename = QtGui.QLineEdit(Q7InfoWindow)
        self.eFilename.setReadOnly(True)
        self.eFilename.setObjectName("eFilename")
        self.horizontalLayout.addWidget(self.eFilename)
        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.eTmpFile = QtGui.QLineEdit(Q7InfoWindow)
        self.eTmpFile.setReadOnly(True)
        self.eTmpFile.setObjectName("eTmpFile")
        self.gridLayout_2.addWidget(self.eTmpFile, 0, 1, 1, 1)
        self.label_13 = QtGui.QLabel(Q7InfoWindow)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 6, 0, 1, 1)

        self.retranslateUi(Q7InfoWindow)
        QtCore.QMetaObject.connectSlotsByName(Q7InfoWindow)

    def retranslateUi(self, Q7InfoWindow):
        Q7InfoWindow.setWindowTitle(QtGui.QApplication.translate("Q7InfoWindow", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.bClose.setText(QtGui.QApplication.translate("Q7InfoWindow", "Close", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_2.setTitle(QtGui.QApplication.translate("Q7InfoWindow", "CGNS", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_4.setTitle(QtGui.QApplication.translate("Q7InfoWindow", "Format", None, QtGui.QApplication.UnicodeUTF8))
        self.cHDF5.setText(QtGui.QApplication.translate("Q7InfoWindow", "HDF5", None, QtGui.QApplication.UnicodeUTF8))
        self.cADF.setText(QtGui.QApplication.translate("Q7InfoWindow", "ADF", None, QtGui.QApplication.UnicodeUTF8))
        self.label_14.setText(QtGui.QApplication.translate("Q7InfoWindow", "HDF5 Version", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Q7InfoWindow", "Version:", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_5.setTitle(QtGui.QApplication.translate("Q7InfoWindow", "Mode", None, QtGui.QApplication.UnicodeUTF8))
        self.cREAD.setText(QtGui.QApplication.translate("Q7InfoWindow", "READ only", None, QtGui.QApplication.UnicodeUTF8))
        self.cMODIFY.setText(QtGui.QApplication.translate("Q7InfoWindow", "MODIFY", None, QtGui.QApplication.UnicodeUTF8))
        self.cConverted.setText(QtGui.QApplication.translate("Q7InfoWindow", "Is converted from ADF", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("Q7InfoWindow", "The TMP file name below is the actual converted file.", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_6.setTitle(QtGui.QApplication.translate("Q7InfoWindow", "Contents", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Q7InfoWindow", "Links", None, QtGui.QApplication.UnicodeUTF8))
        self.cHasLinks.setText(QtGui.QApplication.translate("Q7InfoWindow", "Has links", None, QtGui.QApplication.UnicodeUTF8))
        self.cSameFS.setText(QtGui.QApplication.translate("Q7InfoWindow", "Same file system", None, QtGui.QApplication.UnicodeUTF8))
        self.cBadLinks.setText(QtGui.QApplication.translate("Q7InfoWindow", "Bad links detected", None, QtGui.QApplication.UnicodeUTF8))
        self.cModeProp.setText(QtGui.QApplication.translate("Q7InfoWindow", "Mode propagated", None, QtGui.QApplication.UnicodeUTF8))
        self.cHasInt64.setText(QtGui.QApplication.translate("Q7InfoWindow", "Has double integers (int64)", None, QtGui.QApplication.UnicodeUTF8))
        self.cNODATA.setText(QtGui.QApplication.translate("Q7InfoWindow", "NO DATA", None, QtGui.QApplication.UnicodeUTF8))
        self.cNoFollow.setText(QtGui.QApplication.translate("Q7InfoWindow", "Links not followed", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("Q7InfoWindow", "Number of nodes", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("Q7InfoWindow", "Max depth", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_3.setTitle(QtGui.QApplication.translate("Q7InfoWindow", "Stat", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Q7InfoWindow", "Merge size", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Q7InfoWindow", "File size", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Q7InfoWindow", "Last access", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Q7InfoWindow", "Last modification", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Q7InfoWindow", "Owner", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("Q7InfoWindow", "Rights", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("Q7InfoWindow", "Group", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Q7InfoWindow", "File: ", None, QtGui.QApplication.UnicodeUTF8))
        self.label_13.setText(QtGui.QApplication.translate("Q7InfoWindow", "TMP:", None, QtGui.QApplication.UnicodeUTF8))

import Res_rc
