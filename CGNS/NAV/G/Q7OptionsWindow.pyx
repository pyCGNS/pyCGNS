# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Q7OptionsWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QFontComboBox, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSizePolicy, QSpacerItem,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget)
from . import Res_rc

class Ui_Q7OptionsWindow(object):
    def setupUi(self, Q7OptionsWindow):
        if not Q7OptionsWindow.objectName():
            Q7OptionsWindow.setObjectName(u"Q7OptionsWindow")
        Q7OptionsWindow.setWindowModality(Qt.ApplicationModal)
        Q7OptionsWindow.resize(700, 410)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Q7OptionsWindow.sizePolicy().hasHeightForWidth())
        Q7OptionsWindow.setSizePolicy(sizePolicy)
        Q7OptionsWindow.setMinimumSize(QSize(650, 364))
        Q7OptionsWindow.setMaximumSize(QSize(700, 410))
        icon = QIcon()
        icon.addFile(u":/images/icons/cgSpy.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Q7OptionsWindow.setWindowIcon(icon)
        self.verticalLayout = QVBoxLayout(Q7OptionsWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.tabs = QTabWidget(Q7OptionsWindow)
        self.tabs.setObjectName(u"tabs")
        sizePolicy.setHeightForWidth(self.tabs.sizePolicy().hasHeightForWidth())
        self.tabs.setSizePolicy(sizePolicy)
        self.tabs.setMinimumSize(QSize(630, 310))
        self.tabs.setMaximumSize(QSize(700, 310))
        self.tab_1 = QWidget()
        self.tab_1.setObjectName(u"tab_1")
        self.__O_recursivetreedisplay = QCheckBox(self.tab_1)
        self.__O_recursivetreedisplay.setObjectName(u"__O_recursivetreedisplay")
        self.__O_recursivetreedisplay.setGeometry(QRect(10, 15, 274, 22))
        self.__O_donotdisplaylargedata = QCheckBox(self.tab_1)
        self.__O_donotdisplaylargedata.setObjectName(u"__O_donotdisplaylargedata")
        self.__O_donotdisplaylargedata.setGeometry(QRect(10, 155, 274, 22))
        self.__O_donotloadlargearrays = QCheckBox(self.tab_1)
        self.__O_donotloadlargearrays.setObjectName(u"__O_donotloadlargearrays")
        self.__O_donotloadlargearrays.setGeometry(QRect(10, 80, 274, 22))
        self.label_1 = QLabel(self.tab_1)
        self.label_1.setObjectName(u"label_1")
        self.label_1.setGeometry(QRect(10, 35, 217, 27))
        self.__O_maxrecursionlevel = QSpinBox(self.tab_1)
        self.__O_maxrecursionlevel.setObjectName(u"__O_maxrecursionlevel")
        self.__O_maxrecursionlevel.setGeometry(QRect(265, 35, 71, 27))
        font = QFont()
        font.setFamilies([u"Courier"])
        self.__O_maxrecursionlevel.setFont(font)
        self.__O_maxrecursionlevel.setMinimum(1)
        self.__O_maxrecursionlevel.setMaximum(99)
        self.__O_maxrecursionlevel.setValue(7)
        self.__O_maxloaddatasize = QSpinBox(self.tab_1)
        self.__O_maxloaddatasize.setObjectName(u"__O_maxloaddatasize")
        self.__O_maxloaddatasize.setEnabled(True)
        self.__O_maxloaddatasize.setGeometry(QRect(265, 100, 71, 27))
        font1 = QFont()
        font1.setFamilies([u"Courier"])
        font1.setPointSize(10)
        self.__O_maxloaddatasize.setFont(font1)
        self.__O_maxloaddatasize.setMinimum(-1)
        self.__O_maxloaddatasize.setMaximum(65535)
        self.__O_maxloaddatasize.setValue(-1)
        self.__O_filterhdffiles = QCheckBox(self.tab_1)
        self.__O_filterhdffiles.setObjectName(u"__O_filterhdffiles")
        self.__O_filterhdffiles.setGeometry(QRect(365, 15, 274, 22))
        self.__O_filtercgnsfiles = QCheckBox(self.tab_1)
        self.__O_filtercgnsfiles.setObjectName(u"__O_filtercgnsfiles")
        self.__O_filtercgnsfiles.setGeometry(QRect(365, 95, 274, 22))
        self.line = QFrame(self.tab_1)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(335, 10, 20, 261))
        self.line.setFrameShape(QFrame.Shape.VLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)
        self.__O_cgnsfileextension = QPlainTextEdit(self.tab_1)
        self.__O_cgnsfileextension.setObjectName(u"__O_cgnsfileextension")
        self.__O_cgnsfileextension.setGeometry(QRect(380, 120, 211, 41))
        self.__O_cgnsfileextension.setFont(font)
        self.__O_hdffileextension = QPlainTextEdit(self.tab_1)
        self.__O_hdffileextension.setObjectName(u"__O_hdffileextension")
        self.__O_hdffileextension.setGeometry(QRect(380, 40, 211, 41))
        self.__O_hdffileextension.setFont(font)
        self.label_23 = QLabel(self.tab_1)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(10, 175, 271, 27))
        self.__O_maxdisplaydatasize = QSpinBox(self.tab_1)
        self.__O_maxdisplaydatasize.setObjectName(u"__O_maxdisplaydatasize")
        self.__O_maxdisplaydatasize.setGeometry(QRect(265, 175, 71, 27))
        self.__O_maxdisplaydatasize.setFont(font)
        self.__O_maxdisplaydatasize.setMinimum(-1)
        self.__O_maxdisplaydatasize.setMaximum(1000)
        self.__O_maxdisplaydatasize.setValue(-1)
        self.label_9 = QLabel(self.tab_1)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(10, 100, 225, 27))
        self.tabs.addTab(self.tab_1, "")
        self.tab_8 = QWidget()
        self.tab_8.setObjectName(u"tab_8")
        self.__O_fileupdateremoveschildren = QCheckBox(self.tab_8)
        self.__O_fileupdateremoveschildren.setObjectName(u"__O_fileupdateremoveschildren")
        self.__O_fileupdateremoveschildren.setGeometry(QRect(10, 10, 300, 21))
        self.tabs.addTab(self.tab_8, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.__O_addcurrentdirinsearch = QCheckBox(self.tab_2)
        self.__O_addcurrentdirinsearch.setObjectName(u"__O_addcurrentdirinsearch")
        self.__O_addcurrentdirinsearch.setGeometry(QRect(5, 10, 274, 22))
        self.__O_addrootdirinsearch = QCheckBox(self.tab_2)
        self.__O_addrootdirinsearch.setObjectName(u"__O_addrootdirinsearch")
        self.__O_addrootdirinsearch.setGeometry(QRect(5, 35, 340, 22))
        self.groupBox_2 = QGroupBox(self.tab_2)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(5, 105, 546, 171))
        self.__O_linksearchpathlist = QPlainTextEdit(self.groupBox_2)
        self.__O_linksearchpathlist.setObjectName(u"__O_linksearchpathlist")
        self.__O_linksearchpathlist.setGeometry(QRect(0, 25, 541, 61))
        self.__O_linksearchpathlist.setFont(font)
        self.__O_followlinksatload = QCheckBox(self.groupBox_2)
        self.__O_followlinksatload.setObjectName(u"__O_followlinksatload")
        self.__O_followlinksatload.setGeometry(QRect(0, 95, 274, 22))
        self.__O_stoploadbrokenlinks = QCheckBox(self.groupBox_2)
        self.__O_stoploadbrokenlinks.setObjectName(u"__O_stoploadbrokenlinks")
        self.__O_stoploadbrokenlinks.setGeometry(QRect(0, 120, 274, 22))
        self.__O_donotfollowlinksatsave = QCheckBox(self.groupBox_2)
        self.__O_donotfollowlinksatsave.setObjectName(u"__O_donotfollowlinksatsave")
        self.__O_donotfollowlinksatsave.setEnabled(False)
        self.__O_donotfollowlinksatsave.setGeometry(QRect(0, 145, 274, 22))
        self.pushButton = QPushButton(self.tab_2)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setEnabled(False)
        self.pushButton.setGeometry(QRect(365, 15, 200, 26))
        self.tabs.addTab(self.tab_2, "")
        self.tab_7 = QWidget()
        self.tab_7.setObjectName(u"tab_7")
        self.groupBox_4 = QGroupBox(self.tab_7)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(5, 5, 546, 96))
        self.__O_profilesearchpathlist = QPlainTextEdit(self.groupBox_4)
        self.__O_profilesearchpathlist.setObjectName(u"__O_profilesearchpathlist")
        self.__O_profilesearchpathlist.setGeometry(QRect(0, 25, 541, 66))
        self.__O_profilesearchpathlist.setFont(font)
        self.__O_recursivesidspatternsload = QCheckBox(self.tab_7)
        self.__O_recursivesidspatternsload.setObjectName(u"__O_recursivesidspatternsload")
        self.__O_recursivesidspatternsload.setGeometry(QRect(5, 105, 274, 22))
        self.tabs.addTab(self.tab_7, "")
        self.tab_6 = QWidget()
        self.tab_6.setObjectName(u"tab_6")
        self.__O_checkonthefly = QCheckBox(self.tab_6)
        self.__O_checkonthefly.setObjectName(u"__O_checkonthefly")
        self.__O_checkonthefly.setEnabled(False)
        self.__O_checkonthefly.setGeometry(QRect(10, 10, 274, 22))
        self.__O_checkonthefly.setCheckable(True)
        self.__O_forcesidslegacymapping = QCheckBox(self.tab_6)
        self.__O_forcesidslegacymapping.setObjectName(u"__O_forcesidslegacymapping")
        self.__O_forcesidslegacymapping.setEnabled(False)
        self.__O_forcesidslegacymapping.setGeometry(QRect(10, 30, 274, 22))
        self.__O_forcefortranflag = QCheckBox(self.tab_6)
        self.__O_forcefortranflag.setObjectName(u"__O_forcefortranflag")
        self.__O_forcefortranflag.setEnabled(False)
        self.__O_forcefortranflag.setGeometry(QRect(10, 50, 274, 22))
        self.groupBox_3 = QGroupBox(self.tab_6)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(5, 75, 546, 201))
        self.label_31 = QLabel(self.groupBox_3)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setGeometry(QRect(10, 25, 300, 21))
        self.cRecurseGrammarSearch = QCheckBox(self.groupBox_3)
        self.cRecurseGrammarSearch.setObjectName(u"cRecurseGrammarSearch")
        self.cRecurseGrammarSearch.setEnabled(False)
        self.cRecurseGrammarSearch.setGeometry(QRect(10, 50, 301, 21))
        self.cRecurseGrammarSearch.setCheckable(False)
        self.__O_grammarsearchpathlist = QPlainTextEdit(self.groupBox_3)
        self.__O_grammarsearchpathlist.setObjectName(u"__O_grammarsearchpathlist")
        self.__O_grammarsearchpathlist.setGeometry(QRect(0, 130, 536, 66))
        self.__O_grammarsearchpathlist.setFont(font)
        self.label_24 = QLabel(self.groupBox_3)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setGeometry(QRect(0, 95, 147, 51))
        self.label_25 = QLabel(self.tab_6)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setGeometry(QRect(385, 10, 111, 16))
        self.__O_valkeylist = QPlainTextEdit(self.tab_6)
        self.__O_valkeylist.setObjectName(u"__O_valkeylist")
        self.__O_valkeylist.setGeometry(QRect(385, 30, 156, 156))
        self.__O_valkeylist.setFont(font)
        self.tabs.addTab(self.tab_6, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.__O_label_size = QSpinBox(self.tab_3)
        self.__O_label_size.setObjectName(u"__O_label_size")
        self.__O_label_size.setGeometry(QRect(395, 20, 46, 27))
        self.__O_label_size.setMinimum(6)
        self.__O_label_size.setMaximum(18)
        self.__O_label_size.setValue(10)
        self.label = QLabel(self.tab_3)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 25, 166, 17))
        self.__O_label_bold = QCheckBox(self.tab_3)
        self.__O_label_bold.setObjectName(u"__O_label_bold")
        self.__O_label_bold.setGeometry(QRect(455, 25, 86, 21))
        self.__O_label_italic = QCheckBox(self.tab_3)
        self.__O_label_italic.setObjectName(u"__O_label_italic")
        self.__O_label_italic.setGeometry(QRect(500, 25, 86, 21))
        self.__O_label_family = QFontComboBox(self.tab_3)
        self.__O_label_family.setObjectName(u"__O_label_family")
        self.__O_label_family.setGeometry(QRect(180, 20, 216, 26))
        self.label_3 = QLabel(self.tab_3)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 55, 161, 17))
        self.label_12 = QLabel(self.tab_3)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(10, 85, 161, 17))
        self.label_26 = QLabel(self.tab_3)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setGeometry(QRect(10, 115, 161, 17))
        self.label_27 = QLabel(self.tab_3)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setGeometry(QRect(10, 145, 166, 17))
        self.label_28 = QLabel(self.tab_3)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setGeometry(QRect(10, 175, 166, 17))
        self.label_29 = QLabel(self.tab_3)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setGeometry(QRect(445, 5, 58, 16))
        self.label_30 = QLabel(self.tab_3)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setGeometry(QRect(495, 5, 58, 16))
        self.__O_button_family = QFontComboBox(self.tab_3)
        self.__O_button_family.setObjectName(u"__O_button_family")
        self.__O_button_family.setGeometry(QRect(180, 50, 216, 26))
        self.__O_edit_family = QFontComboBox(self.tab_3)
        self.__O_edit_family.setObjectName(u"__O_edit_family")
        self.__O_edit_family.setGeometry(QRect(180, 80, 216, 26))
        self.__O_table_family = QFontComboBox(self.tab_3)
        self.__O_table_family.setObjectName(u"__O_table_family")
        self.__O_table_family.setGeometry(QRect(180, 110, 216, 26))
        self.__O_nname_family = QFontComboBox(self.tab_3)
        self.__O_nname_family.setObjectName(u"__O_nname_family")
        self.__O_nname_family.setEnabled(False)
        self.__O_nname_family.setGeometry(QRect(180, 140, 216, 26))
        self.__O_rname_family = QFontComboBox(self.tab_3)
        self.__O_rname_family.setObjectName(u"__O_rname_family")
        self.__O_rname_family.setGeometry(QRect(180, 170, 216, 26))
        self.__O_button_size = QSpinBox(self.tab_3)
        self.__O_button_size.setObjectName(u"__O_button_size")
        self.__O_button_size.setGeometry(QRect(395, 50, 46, 27))
        self.__O_button_size.setMinimum(6)
        self.__O_button_size.setMaximum(18)
        self.__O_button_size.setValue(10)
        self.__O_edit_size = QSpinBox(self.tab_3)
        self.__O_edit_size.setObjectName(u"__O_edit_size")
        self.__O_edit_size.setGeometry(QRect(395, 80, 46, 27))
        self.__O_edit_size.setMinimum(6)
        self.__O_edit_size.setMaximum(18)
        self.__O_edit_size.setValue(10)
        self.__O_table_size = QSpinBox(self.tab_3)
        self.__O_table_size.setObjectName(u"__O_table_size")
        self.__O_table_size.setGeometry(QRect(395, 110, 46, 27))
        self.__O_table_size.setMinimum(6)
        self.__O_table_size.setMaximum(18)
        self.__O_table_size.setValue(10)
        self.__O_nname_size = QSpinBox(self.tab_3)
        self.__O_nname_size.setObjectName(u"__O_nname_size")
        self.__O_nname_size.setEnabled(False)
        self.__O_nname_size.setGeometry(QRect(395, 140, 46, 27))
        self.__O_nname_size.setMinimum(6)
        self.__O_nname_size.setMaximum(18)
        self.__O_nname_size.setValue(10)
        self.__O_rname_size = QSpinBox(self.tab_3)
        self.__O_rname_size.setObjectName(u"__O_rname_size")
        self.__O_rname_size.setGeometry(QRect(395, 170, 46, 27))
        self.__O_rname_size.setMinimum(6)
        self.__O_rname_size.setMaximum(18)
        self.__O_rname_size.setValue(10)
        self.__O_button_bold = QCheckBox(self.tab_3)
        self.__O_button_bold.setObjectName(u"__O_button_bold")
        self.__O_button_bold.setGeometry(QRect(455, 55, 86, 21))
        self.__O_edit_bold = QCheckBox(self.tab_3)
        self.__O_edit_bold.setObjectName(u"__O_edit_bold")
        self.__O_edit_bold.setGeometry(QRect(455, 85, 86, 21))
        self.__O_table_bold = QCheckBox(self.tab_3)
        self.__O_table_bold.setObjectName(u"__O_table_bold")
        self.__O_table_bold.setGeometry(QRect(455, 115, 86, 21))
        self.__O_nname_bold = QCheckBox(self.tab_3)
        self.__O_nname_bold.setObjectName(u"__O_nname_bold")
        self.__O_nname_bold.setEnabled(False)
        self.__O_nname_bold.setGeometry(QRect(455, 145, 86, 21))
        self.__O_rname_bold = QCheckBox(self.tab_3)
        self.__O_rname_bold.setObjectName(u"__O_rname_bold")
        self.__O_rname_bold.setGeometry(QRect(455, 175, 86, 21))
        self.__O_button_italic = QCheckBox(self.tab_3)
        self.__O_button_italic.setObjectName(u"__O_button_italic")
        self.__O_button_italic.setGeometry(QRect(500, 55, 86, 21))
        self.__O_edit_italic = QCheckBox(self.tab_3)
        self.__O_edit_italic.setObjectName(u"__O_edit_italic")
        self.__O_edit_italic.setGeometry(QRect(500, 85, 86, 21))
        self.__O_table_italic = QCheckBox(self.tab_3)
        self.__O_table_italic.setObjectName(u"__O_table_italic")
        self.__O_table_italic.setGeometry(QRect(500, 115, 86, 21))
        self.__O_nname_italic = QCheckBox(self.tab_3)
        self.__O_nname_italic.setObjectName(u"__O_nname_italic")
        self.__O_nname_italic.setEnabled(False)
        self.__O_nname_italic.setGeometry(QRect(500, 145, 86, 21))
        self.__O_rname_italic = QCheckBox(self.tab_3)
        self.__O_rname_italic.setObjectName(u"__O_rname_italic")
        self.__O_rname_italic.setGeometry(QRect(500, 175, 86, 21))
        self.bResetFonts = QPushButton(self.tab_3)
        self.bResetFonts.setObjectName(u"bResetFonts")
        self.bResetFonts.setGeometry(QRect(10, 215, 25, 25))
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(25)
        sizePolicy1.setVerticalStretch(25)
        sizePolicy1.setHeightForWidth(self.bResetFonts.sizePolicy().hasHeightForWidth())
        self.bResetFonts.setSizePolicy(sizePolicy1)
        self.bResetFonts.setMinimumSize(QSize(25, 25))
        self.bResetFonts.setMaximumSize(QSize(25, 25))
        self.bResetFonts.setText(u"")
        icon1 = QIcon()
        icon1.addFile(u":/images/icons/undo-last-modification.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bResetFonts.setIcon(icon1)
        self.tabs.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.__O_showtableindex = QCheckBox(self.tab_4)
        self.__O_showtableindex.setObjectName(u"__O_showtableindex")
        self.__O_showtableindex.setGeometry(QRect(320, 100, 274, 22))
        self.__O_oneviewpertreenode = QCheckBox(self.tab_4)
        self.__O_oneviewpertreenode.setObjectName(u"__O_oneviewpertreenode")
        self.__O_oneviewpertreenode.setGeometry(QRect(320, 175, 274, 22))
        self.__O_showhelpballoons = QCheckBox(self.tab_4)
        self.__O_showhelpballoons.setObjectName(u"__O_showhelpballoons")
        self.__O_showhelpballoons.setGeometry(QRect(320, 77, 274, 22))
        self.__O_show1dasplain = QCheckBox(self.tab_4)
        self.__O_show1dasplain.setObjectName(u"__O_show1dasplain")
        self.__O_show1dasplain.setGeometry(QRect(320, 125, 310, 22))
        self.__O_transposearrayforview = QCheckBox(self.tab_4)
        self.__O_transposearrayforview.setObjectName(u"__O_transposearrayforview")
        self.__O_transposearrayforview.setGeometry(QRect(320, 30, 274, 22))
        self.__O_autoexpand = QCheckBox(self.tab_4)
        self.__O_autoexpand.setObjectName(u"__O_autoexpand")
        self.__O_autoexpand.setGeometry(QRect(320, 55, 200, 21))
        self.groupBox_5 = QGroupBox(self.tab_4)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(5, 10, 306, 261))
        self.groupBox_5.setCheckable(True)
        self.__O_showsidscolumn = QCheckBox(self.groupBox_5)
        self.__O_showsidscolumn.setObjectName(u"__O_showsidscolumn")
        self.__O_showsidscolumn.setGeometry(QRect(25, 25, 116, 21))
        self.__O_showlinkcolumn = QCheckBox(self.groupBox_5)
        self.__O_showlinkcolumn.setObjectName(u"__O_showlinkcolumn")
        self.__O_showlinkcolumn.setGeometry(QRect(25, 50, 84, 21))
        self.__O_showselectcolumn = QCheckBox(self.groupBox_5)
        self.__O_showselectcolumn.setObjectName(u"__O_showselectcolumn")
        self.__O_showselectcolumn.setGeometry(QRect(25, 75, 84, 21))
        self.__O_showcheckcolumn = QCheckBox(self.groupBox_5)
        self.__O_showcheckcolumn.setObjectName(u"__O_showcheckcolumn")
        self.__O_showcheckcolumn.setGeometry(QRect(25, 100, 84, 21))
        self.__O_showusercolumn = QCheckBox(self.groupBox_5)
        self.__O_showusercolumn.setObjectName(u"__O_showusercolumn")
        self.__O_showusercolumn.setGeometry(QRect(25, 125, 84, 21))
        self.__O_showshapecolumn = QCheckBox(self.groupBox_5)
        self.__O_showshapecolumn.setObjectName(u"__O_showshapecolumn")
        self.__O_showshapecolumn.setGeometry(QRect(25, 150, 84, 21))
        self.__O_showdatatypecolumn = QCheckBox(self.groupBox_5)
        self.__O_showdatatypecolumn.setObjectName(u"__O_showdatatypecolumn")
        self.__O_showdatatypecolumn.setGeometry(QRect(25, 175, 116, 21))
        self.tabs.addTab(self.tab_4, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.groupBox = QGroupBox(self.tab)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 100, 536, 171))
        self.__O_querynoexception = QCheckBox(self.groupBox)
        self.__O_querynoexception.setObjectName(u"__O_querynoexception")
        self.__O_querynoexception.setGeometry(QRect(5, 30, 246, 20))
        self.checkBox = QCheckBox(self.tab)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setEnabled(False)
        self.checkBox.setGeometry(QRect(15, 10, 400, 20))
        self.checkBox_2 = QCheckBox(self.tab)
        self.checkBox_2.setObjectName(u"checkBox_2")
        self.checkBox_2.setEnabled(False)
        self.checkBox_2.setGeometry(QRect(15, 35, 350, 20))
        self.tabs.addTab(self.tab, "")
        self.tab_5 = QWidget()
        self.tab_5.setObjectName(u"tab_5")
        self.label_4 = QLabel(self.tab_5)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(10, 15, 147, 27))
        self.label_5 = QLabel(self.tab_5)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(10, 45, 147, 27))
        self.__O_snapshotdirectory = QLineEdit(self.tab_5)
        self.__O_snapshotdirectory.setObjectName(u"__O_snapshotdirectory")
        self.__O_snapshotdirectory.setGeometry(QRect(160, 45, 391, 27))
        self.__O_snapshotdirectory.setFont(font)
        self.__O_queriesdirectory = QLineEdit(self.tab_5)
        self.__O_queriesdirectory.setObjectName(u"__O_queriesdirectory")
        self.__O_queriesdirectory.setGeometry(QRect(160, 15, 391, 27))
        self.__O_queriesdirectory.setFont(font)
        self.label_6 = QLabel(self.tab_5)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(10, 75, 147, 27))
        self.__O_selectionlistdirectory = QLineEdit(self.tab_5)
        self.__O_selectionlistdirectory.setObjectName(u"__O_selectionlistdirectory")
        self.__O_selectionlistdirectory.setGeometry(QRect(160, 75, 391, 27))
        self.__O_selectionlistdirectory.setFont(font)
        self.label_10 = QLabel(self.tab_5)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(10, 115, 147, 27))
        self.__O_adfconversioncom = QLineEdit(self.tab_5)
        self.__O_adfconversioncom.setObjectName(u"__O_adfconversioncom")
        self.__O_adfconversioncom.setGeometry(QRect(160, 115, 391, 27))
        self.__O_adfconversioncom.setFont(font)
        self.label_11 = QLabel(self.tab_5)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(10, 145, 151, 27))
        self.__O_temporarydirectory = QLineEdit(self.tab_5)
        self.__O_temporarydirectory.setObjectName(u"__O_temporarydirectory")
        self.__O_temporarydirectory.setGeometry(QRect(160, 145, 391, 27))
        self.__O_temporarydirectory.setFont(font)
        self.line_2 = QFrame(self.tab_5)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(10, 180, 536, 16))
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)
        self.__O_activatemultithreading = QCheckBox(self.tab_5)
        self.__O_activatemultithreading.setObjectName(u"__O_activatemultithreading")
        self.__O_activatemultithreading.setGeometry(QRect(375, 195, 200, 21))
        self.__O_navtrace = QCheckBox(self.tab_5)
        self.__O_navtrace.setObjectName(u"__O_navtrace")
        self.__O_navtrace.setGeometry(QRect(10, 195, 200, 21))
        self.__O_chlonetrace = QCheckBox(self.tab_5)
        self.__O_chlonetrace.setObjectName(u"__O_chlonetrace")
        self.__O_chlonetrace.setGeometry(QRect(10, 215, 130, 21))
        self.bResetIgnored = QPushButton(self.tab_5)
        self.bResetIgnored.setObjectName(u"bResetIgnored")
        self.bResetIgnored.setGeometry(QRect(15, 240, 200, 28))
        self.tabs.addTab(self.tab_5, "")

        self.horizontalLayout_2.addWidget(self.tabs)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.bReset = QPushButton(Q7OptionsWindow)
        self.bReset.setObjectName(u"bReset")

        self.horizontalLayout.addWidget(self.bReset)

        self.bInfo = QPushButton(Q7OptionsWindow)
        self.bInfo.setObjectName(u"bInfo")
        self.bInfo.setMinimumSize(QSize(25, 25))
        self.bInfo.setMaximumSize(QSize(25, 25))
        icon2 = QIcon()
        icon2.addFile(u":/images/icons/help-view.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.bInfo.setIcon(icon2)

        self.horizontalLayout.addWidget(self.bInfo)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.bApply = QPushButton(Q7OptionsWindow)
        self.bApply.setObjectName(u"bApply")

        self.horizontalLayout.addWidget(self.bApply)

        self.bClose = QPushButton(Q7OptionsWindow)
        self.bClose.setObjectName(u"bClose")

        self.horizontalLayout.addWidget(self.bClose)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(Q7OptionsWindow)

        self.tabs.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(Q7OptionsWindow)
    # setupUi

    def retranslateUi(self, Q7OptionsWindow):
        Q7OptionsWindow.setWindowTitle(QCoreApplication.translate("Q7OptionsWindow", u"Form", None))
        self.__O_recursivetreedisplay.setText(QCoreApplication.translate("Q7OptionsWindow", u"Recursive tree display", None))
        self.__O_donotdisplaylargedata.setText(QCoreApplication.translate("Q7OptionsWindow", u"Do not dispay large data", None))
        self.__O_donotloadlargearrays.setText(QCoreApplication.translate("Q7OptionsWindow", u"Do not load large data arrays", None))
        self.label_1.setText(QCoreApplication.translate("Q7OptionsWindow", u"Max tree parse recursion level:", None))
        self.__O_filterhdffiles.setText(QCoreApplication.translate("Q7OptionsWindow", u"filter *.hdf files", None))
        self.__O_filtercgnsfiles.setText(QCoreApplication.translate("Q7OptionsWindow", u"filter *.cgns files", None))
        self.label_23.setText(QCoreApplication.translate("Q7OptionsWindow", u"Display nodes with data size below:", None))
        self.label_9.setText(QCoreApplication.translate("Q7OptionsWindow", u"Do not load node data if above:", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_1), QCoreApplication.translate("Q7OptionsWindow", u"Load", None))
        self.__O_fileupdateremoveschildren.setText(QCoreApplication.translate("Q7OptionsWindow", u"File update removes missing children", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_8), QCoreApplication.translate("Q7OptionsWindow", u"Save", None))
#if QT_CONFIG(tooltip)
        self.__O_addcurrentdirinsearch.setToolTip(QCoreApplication.translate("Q7OptionsWindow", u"Set to always add the current directory as the first in the linked-to files search.", u"Set to always add the current directory as the first in the linked-to files search."))
#endif // QT_CONFIG(tooltip)
        self.__O_addcurrentdirinsearch.setText(QCoreApplication.translate("Q7OptionsWindow", u"Add current dir in link search path", None))
        self.__O_addrootdirinsearch.setText(QCoreApplication.translate("Q7OptionsWindow", u"Add file system root dir in link search path", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Q7OptionsWindow", u"Search paths", None))
        self.__O_followlinksatload.setText(QCoreApplication.translate("Q7OptionsWindow", u"Follow links during file load", None))
        self.__O_stoploadbrokenlinks.setText(QCoreApplication.translate("Q7OptionsWindow", u"Stop loading on broken link", None))
        self.__O_donotfollowlinksatsave.setText(QCoreApplication.translate("Q7OptionsWindow", u"Do not follow links during file save", None))
        self.pushButton.setText(QCoreApplication.translate("Q7OptionsWindow", u"Remove unreachable paths", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_2), QCoreApplication.translate("Q7OptionsWindow", u"Links", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Q7OptionsWindow", u"Search paths", None))
        self.__O_recursivesidspatternsload.setText(QCoreApplication.translate("Q7OptionsWindow", u"Recursive SIDS patterns load", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_7), QCoreApplication.translate("Q7OptionsWindow", u"Profiles", None))
        self.__O_checkonthefly.setText(QCoreApplication.translate("Q7OptionsWindow", u"Check on the fly", None))
        self.__O_forcesidslegacymapping.setText(QCoreApplication.translate("Q7OptionsWindow", u"Force SIDS legacy mapping", None))
        self.__O_forcefortranflag.setText(QCoreApplication.translate("Q7OptionsWindow", u"Force fortran flag in numpy arrays", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Q7OptionsWindow", u"CGNS.VAL parameters", None))
        self.label_31.setText(QCoreApplication.translate("Q7OptionsWindow", u"See also: Paths tab and grammars paths", None))
        self.cRecurseGrammarSearch.setText(QCoreApplication.translate("Q7OptionsWindow", u"Activate recursion search for grammars (long)", None))
        self.label_24.setText(QCoreApplication.translate("Q7OptionsWindow", u"<html><head/><body><p><span style=\" font-weight:600;\">Search paths</span></p></body></html>", None))
        self.label_25.setText(QCoreApplication.translate("Q7OptionsWindow", u"Grammar keys", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_6), QCoreApplication.translate("Q7OptionsWindow", u"Checks", None))
        self.label.setText(QCoreApplication.translate("Q7OptionsWindow", u"label", None))
        self.__O_label_bold.setText("")
        self.__O_label_italic.setText("")
        self.label_3.setText(QCoreApplication.translate("Q7OptionsWindow", u"button", None))
        self.label_12.setText(QCoreApplication.translate("Q7OptionsWindow", u"text edit", None))
        self.label_26.setText(QCoreApplication.translate("Q7OptionsWindow", u"table (Form view)", None))
        self.label_27.setText(QCoreApplication.translate("Q7OptionsWindow", u"graphic (VTK view)", None))
        self.label_28.setText(QCoreApplication.translate("Q7OptionsWindow", u"node name (Tree view)", None))
        self.label_29.setText(QCoreApplication.translate("Q7OptionsWindow", u"Bold", None))
        self.label_30.setText(QCoreApplication.translate("Q7OptionsWindow", u"Italic", None))
        self.__O_button_bold.setText("")
        self.__O_edit_bold.setText("")
        self.__O_table_bold.setText("")
        self.__O_nname_bold.setText("")
        self.__O_rname_bold.setText("")
        self.__O_button_italic.setText("")
        self.__O_edit_italic.setText("")
        self.__O_table_italic.setText("")
        self.__O_nname_italic.setText("")
        self.__O_rname_italic.setText("")
#if QT_CONFIG(tooltip)
        self.bResetFonts.setToolTip(QCoreApplication.translate("Q7OptionsWindow", u"Reset all fonts to default installation fonts", None))
#endif // QT_CONFIG(tooltip)
        self.tabs.setTabText(self.tabs.indexOf(self.tab_3), QCoreApplication.translate("Q7OptionsWindow", u"Fonts", None))
        self.__O_showtableindex.setText(QCoreApplication.translate("Q7OptionsWindow", u"Show table index", None))
        self.__O_oneviewpertreenode.setText(QCoreApplication.translate("Q7OptionsWindow", u"One view per tree/node", None))
#if QT_CONFIG(tooltip)
        self.__O_showhelpballoons.setToolTip(QCoreApplication.translate("Q7OptionsWindow", u"Shows the help balloon you are reading right now.", None))
#endif // QT_CONFIG(tooltip)
        self.__O_showhelpballoons.setText(QCoreApplication.translate("Q7OptionsWindow", u"Show tooltips", None))
        self.__O_show1dasplain.setText(QCoreApplication.translate("Q7OptionsWindow", u"Show 1D values as Python plain types", None))
        self.__O_transposearrayforview.setText(QCoreApplication.translate("Q7OptionsWindow", u"Transpose arrays for view/edit", None))
        self.__O_autoexpand.setText(QCoreApplication.translate("Q7OptionsWindow", u"Auto expand tree view ", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("Q7OptionsWindow", u"Show Tree Column Titles", None))
        self.__O_showsidscolumn.setText(QCoreApplication.translate("Q7OptionsWindow", u"SIDS type", None))
        self.__O_showlinkcolumn.setText(QCoreApplication.translate("Q7OptionsWindow", u"Link", None))
        self.__O_showselectcolumn.setText(QCoreApplication.translate("Q7OptionsWindow", u"Mark", None))
        self.__O_showcheckcolumn.setText(QCoreApplication.translate("Q7OptionsWindow", u"Check", None))
        self.__O_showusercolumn.setText(QCoreApplication.translate("Q7OptionsWindow", u"User", None))
        self.__O_showshapecolumn.setText(QCoreApplication.translate("Q7OptionsWindow", u"Shape", None))
        self.__O_showdatatypecolumn.setText(QCoreApplication.translate("Q7OptionsWindow", u"Data Type", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_4), QCoreApplication.translate("Q7OptionsWindow", u"Windows", None))
        self.groupBox.setTitle(QCoreApplication.translate("Q7OptionsWindow", u"Debug", None))
        self.__O_querynoexception.setText(QCoreApplication.translate("Q7OptionsWindow", u"Do not catch any exception", None))
        self.checkBox.setText(QCoreApplication.translate("Q7OptionsWindow", u"Ignore queries operating modification of the tree", None))
        self.checkBox_2.setText(QCoreApplication.translate("Q7OptionsWindow", u"Add query results to previous query results", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab), QCoreApplication.translate("Q7OptionsWindow", u"Queries", None))
        self.label_4.setText(QCoreApplication.translate("Q7OptionsWindow", u"User queries directory", None))
        self.label_5.setText(QCoreApplication.translate("Q7OptionsWindow", u"Snapshot directory", None))
        self.label_6.setText(QCoreApplication.translate("Q7OptionsWindow", u"Selection list directory", None))
        self.label_10.setText(QCoreApplication.translate("Q7OptionsWindow", u"ADF conversion", None))
        self.label_11.setText(QCoreApplication.translate("Q7OptionsWindow", u"Temporary directory", None))
        self.__O_activatemultithreading.setText(QCoreApplication.translate("Q7OptionsWindow", u"Activate Multi-threading", None))
        self.__O_navtrace.setText(QCoreApplication.translate("Q7OptionsWindow", u"CGNS.NAV trace", None))
        self.__O_chlonetrace.setText(QCoreApplication.translate("Q7OptionsWindow", u"CHLone trace", None))
        self.bResetIgnored.setText(QCoreApplication.translate("Q7OptionsWindow", u"Reset ignored messages", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_5), QCoreApplication.translate("Q7OptionsWindow", u"Other", None))
        self.bReset.setText(QCoreApplication.translate("Q7OptionsWindow", u"Reset", None))
        self.bInfo.setText("")
        self.bApply.setText(QCoreApplication.translate("Q7OptionsWindow", u"Apply", None))
        self.bClose.setText(QCoreApplication.translate("Q7OptionsWindow", u"Close", None))
    # retranslateUi

