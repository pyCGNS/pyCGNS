# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7QueryWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpacerItem, QTabWidget, QWidget)

from CGNS.NAV.weditors import (Q7DocEditor, Q7PythonEditor)
from . import Res_rc

class Ui_Q7QueryWindow(object):
    def setupUi(self, Q7QueryWindow):
        if not Q7QueryWindow.objectName():
            Q7QueryWindow.setObjectName(u"Q7QueryWindow")
        Q7QueryWindow.resize(715, 350)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7QueryWindow.sizePolicy().hasHeightForWidth())
        Q7QueryWindow.setSizePolicy(sizePolicy)
        Q7QueryWindow.setMinimumSize(QSize(715, 350))
        Q7QueryWindow.setMaximumSize(QSize(715, 350))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7QueryWindow.setWindowIcon(icon)
        self.gridLayout = QGridLayout(Q7QueryWindow)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.bBackControl = QPushButton(Q7QueryWindow)
        self.bBackControl.setObjectName(u"bBackControl")
        self.bBackControl.setMinimumSize(QSize(25, 25))
        self.bBackControl.setMaximumSize(QSize(25, 25))
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/top.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bBackControl.setIcon(icon1)

        self.horizontalLayout_2.addWidget(self.bBackControl)

        self.bInfo = QPushButton(Q7QueryWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon2)

        self.horizontalLayout_2.addWidget(self.bInfo)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.bClose = QPushButton(Q7QueryWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout_2.addWidget(self.bClose)


        self.gridLayout.addLayout(self.horizontalLayout_2, 7, 1, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_2 = QLabel(Q7QueryWindow)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.cQueryGroup = QComboBox(Q7QueryWindow)
        self.cQueryGroup.setObjectName(u"cQueryGroup")
        self.cQueryGroup.setEnabled(True)

        self.horizontalLayout.addWidget(self.cQueryGroup)

        self.label = QLabel(Q7QueryWindow)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.cQueryName = QComboBox(Q7QueryWindow)
        self.cQueryName.setObjectName(u"cQueryName")
        self.cQueryName.setMinimumSize(QSize(300, 0))
        self.cQueryName.setEditable(True)

        self.horizontalLayout.addWidget(self.cQueryName)

        self.bAdd = QPushButton(Q7QueryWindow)
        self.bAdd.setObjectName(u"bAdd")
        self.bAdd.setMinimumSize(QSize(25, 25))
        self.bAdd.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/images/icons/operate-add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bAdd.setIcon(icon3)

        self.horizontalLayout.addWidget(self.bAdd)

        self.bDel = QPushButton(Q7QueryWindow)
        self.bDel.setObjectName(u"bDel")
        self.bDel.setMinimumSize(QSize(25, 25))
        self.bDel.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/images/icons/operate-delete.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bDel.setIcon(icon4)

        self.horizontalLayout.addWidget(self.bDel)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bSaveAsScript = QPushButton(Q7QueryWindow)
        self.bSaveAsScript.setObjectName(u"bSaveAsScript")
        self.bSaveAsScript.setMinimumSize(QSize(25, 25))
        self.bSaveAsScript.setMaximumSize(QSize(25, 25))
        icon5 = QIcon()
        icon5.addFile(u":/images/icons/operate-edit.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSaveAsScript.setIcon(icon5)

        self.horizontalLayout.addWidget(self.bSaveAsScript)

        self.bSave = QPushButton(Q7QueryWindow)
        self.bSave.setObjectName(u"bSave")
        self.bSave.setMinimumSize(QSize(25, 25))
        self.bSave.setMaximumSize(QSize(25, 25))
        icon6 = QIcon()
        icon6.addFile(u":/images/icons/operate-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bSave.setIcon(icon6)

        self.horizontalLayout.addWidget(self.bSave)


        self.gridLayout.addLayout(self.horizontalLayout, 3, 1, 1, 1)

        self.tabWidget = QTabWidget(Q7QueryWindow)
        self.tabWidget.setObjectName(u"tabWidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy1)
        self.tabWidget.setMinimumSize(QSize(695, 265))
        self.tabWidget.setMaximumSize(QSize(695, 265))
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.eText = Q7PythonEditor(self.tab)
        self.eText.setObjectName(u"eText")
        self.eText.setGeometry(QRect(10, 40, 671, 191))
        self.horizontalLayoutWidget_2 = QWidget(self.tab)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(10, 10, 671, 28))
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.cRequireUpdate = QCheckBox(self.horizontalLayoutWidget_2)
        self.cRequireUpdate.setObjectName(u"cRequireUpdate")

        self.horizontalLayout_4.addWidget(self.cRequireUpdate)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_5)

        self.bRevert = QPushButton(self.horizontalLayoutWidget_2)
        self.bRevert.setObjectName(u"bRevert")
        self.bRevert.setEnabled(True)
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(25)
        sizePolicy2.setVerticalStretch(25)
        sizePolicy2.setHeightForWidth(self.bRevert.sizePolicy().hasHeightForWidth())
        self.bRevert.setSizePolicy(sizePolicy2)
        self.bRevert.setMinimumSize(QSize(25, 25))
        self.bRevert.setMaximumSize(QSize(25, 25))
        icon7 = QIcon()
        icon7.addFile(u":/images/icons/undo-at-most.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bRevert.setIcon(icon7)

        self.horizontalLayout_4.addWidget(self.bRevert)

        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.eQueryDoc = Q7DocEditor(self.tab_3)
        self.eQueryDoc.setObjectName(u"eQueryDoc")
        self.eQueryDoc.setGeometry(QRect(10, 40, 671, 191))
        self.horizontalLayoutWidget_3 = QWidget(self.tab_3)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(10, 10, 671, 28))
        self.horizontalLayout_5 = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_6)

        self.bRevertDoc = QPushButton(self.horizontalLayoutWidget_3)
        self.bRevertDoc.setObjectName(u"bRevertDoc")
        self.bRevertDoc.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.bRevertDoc.sizePolicy().hasHeightForWidth())
        self.bRevertDoc.setSizePolicy(sizePolicy2)
        self.bRevertDoc.setMinimumSize(QSize(25, 25))
        self.bRevertDoc.setMaximumSize(QSize(25, 25))
        self.bRevertDoc.setIcon(icon7)

        self.horizontalLayout_5.addWidget(self.bRevertDoc)

        self.bCommitDoc = QPushButton(self.horizontalLayoutWidget_3)
        self.bCommitDoc.setObjectName(u"bCommitDoc")
        self.bCommitDoc.setEnabled(True)
        self.bCommitDoc.setMinimumSize(QSize(25, 25))
        self.bCommitDoc.setMaximumSize(QSize(25, 25))
        icon8 = QIcon()
        icon8.addFile(u":/images/icons/save-log.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bCommitDoc.setIcon(icon8)

        self.horizontalLayout_5.addWidget(self.bCommitDoc)

        self.tabWidget.addTab(self.tab_3, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.horizontalLayoutWidget = QWidget(self.tab_2)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(10, 10, 671, 28))
        self.horizontalLayout_3 = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.bRun = QPushButton(self.horizontalLayoutWidget)
        self.bRun.setObjectName(u"bRun")
        sizePolicy2.setHeightForWidth(self.bRun.sizePolicy().hasHeightForWidth())
        self.bRun.setSizePolicy(sizePolicy2)
        self.bRun.setMinimumSize(QSize(25, 25))
        self.bRun.setMaximumSize(QSize(25, 25))
        icon9 = QIcon()
        icon9.addFile(u":/images/icons/operate-execute.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bRun.setIcon(icon9)

        self.horizontalLayout_3.addWidget(self.bRun)

        self.label_3 = QLabel(self.horizontalLayoutWidget)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_3.addWidget(self.label_3)

        self.eUserVariable = QLineEdit(self.horizontalLayoutWidget)
        self.eUserVariable.setObjectName(u"eUserVariable")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.eUserVariable.sizePolicy().hasHeightForWidth())
        self.eUserVariable.setSizePolicy(sizePolicy3)

        self.horizontalLayout_3.addWidget(self.eUserVariable)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_4)

        self.pushButton_2 = QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setMinimumSize(QSize(25, 25))
        self.pushButton_2.setMaximumSize(QSize(25, 25))
        icon10 = QIcon()
        icon10.addFile(u":/images/icons/select-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_2.setIcon(icon10)

        self.horizontalLayout_3.addWidget(self.pushButton_2)

        self.eResult = Q7PythonEditor(self.tab_2)
        self.eResult.setObjectName(u"eResult")
        self.eResult.setGeometry(QRect(10, 40, 671, 191))
        self.tabWidget.addTab(self.tab_2, "")

        self.gridLayout.addWidget(self.tabWidget, 5, 1, 1, 1)


        self.retranslateUi(Q7QueryWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Q7QueryWindow)
    # setupUi

    def retranslateUi(self, Q7QueryWindow):
        Q7QueryWindow.setWindowTitle(QCoreApplication.translate("Q7QueryWindow", u"Form", None))
        self.bBackControl.setText("")
        self.bInfo.setText("")
        self.bClose.setText(QCoreApplication.translate("Q7QueryWindow", u"Close", None))
        self.label_2.setText(QCoreApplication.translate("Q7QueryWindow", u"Group:", None))
#if QT_CONFIG(tooltip)
        self.cQueryGroup.setToolTip(QCoreApplication.translate("Q7QueryWindow", u"Label to caracterize a set of queries", None))
#endif // QT_CONFIG(tooltip)
        self.label.setText(QCoreApplication.translate("Q7QueryWindow", u"Query:", None))
#if QT_CONFIG(tooltip)
        self.cQueryName.setToolTip(QCoreApplication.translate("Q7QueryWindow", u"Query name", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.bAdd.setToolTip(QCoreApplication.translate("Q7QueryWindow", u"Add as a new query", None))
#endif // QT_CONFIG(tooltip)
        self.bAdd.setText("")
#if QT_CONFIG(tooltip)
        self.bDel.setToolTip(QCoreApplication.translate("Q7QueryWindow", u"Delete the query", None))
#endif // QT_CONFIG(tooltip)
        self.bDel.setText("")
#if QT_CONFIG(tooltip)
        self.bSaveAsScript.setToolTip(QCoreApplication.translate("Q7QueryWindow", u"Write query as single stand-alone python script", None))
#endif // QT_CONFIG(tooltip)
        self.bSaveAsScript.setText("")
#if QT_CONFIG(tooltip)
        self.bSave.setToolTip(QCoreApplication.translate("Q7QueryWindow", u"Save all queries in user profile directory", None))
#endif // QT_CONFIG(tooltip)
        self.bSave.setText("")
        self.cRequireUpdate.setText(QCoreApplication.translate("Q7QueryWindow", u"Script modifies tree and requires all views to update", None))
#if QT_CONFIG(tooltip)
        self.bRevert.setToolTip(QCoreApplication.translate("Q7QueryWindow", u"Revert to last saved text", None))
#endif // QT_CONFIG(tooltip)
        self.bRevert.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("Q7QueryWindow", u"Python", None))
        self.bRevertDoc.setText("")
        self.bCommitDoc.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("Q7QueryWindow", u"Documentation", None))
#if QT_CONFIG(tooltip)
        self.bRun.setToolTip(QCoreApplication.translate("Q7QueryWindow", u"Run query with args", None))
#endif // QT_CONFIG(tooltip)
        self.bRun.setText("")
        self.label_3.setText(QCoreApplication.translate("Q7QueryWindow", u"Args:", None))
        self.pushButton_2.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("Q7QueryWindow", u"Result", None))
    # retranslateUi

