#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# 
import sys
import re
import numpy

from PySide.QtCore    import *
from PySide.QtGui     import *

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

# -----------------------------------------------------------------
class Q7TableView(QTableView):
    def  __init__(self,parent):
        QTableView.__init__(self,None)
        self._parent=None
    def mousePressEvent(self,event):
       self.lastPos=event.globalPos()
       self.lastButton=event.button()
       QTableView.mousePressEvent(self,event)
       
# -----------------------------------------------------------------
class Q7TableModel(QAbstractTableModel):  
    def __init__(self,node,showparams,parent=None):
        QAbstractTableModel.__init__(self,parent)
        self.node=node
        self.showmode=showparams['mode']
        self.cs=showparams['cs']
        self.ls=showparams['ls']
        self.fmt="%-s"
        tx=self.fmt%'w'*16
        if (self.node.sidsDataType() in [CGK.R4,CGK.R8]):
            self.fmt="% 0.8e"
            tx=self.fmt%9.9999999999999
        if (self.node.sidsDataType() in [CGK.I4,CGK.I8]):
            self.fmt="%8d"
            tx=self.fmt%999999999
        if (self.node.sidsDataType() in [CGK.C1]):
            self.fmt="%1s"
            tx=self.fmt%'w'
        if (self.node.sidsDataType() not in [CGK.MT,CGK.LK]):
            if (self.node.sidsValueFortranOrder()):
                self.flatindex=self.flatindex_F
            else:
                self.flatindex=self.flatindex_C
            self.flatarray=self.node.sidsValue().flat
        else:
            self.flatarray=None
        self.hmin=1
        self.vmin=1
        self.font=QFont("Courier new",10)
        fm=QFontMetrics(self.font)
        self.colwidth=fm.width(tx)
    def setRange(self,minh,minv): 
        self.hmin=minh
        self.vmin=minv
    def headerData(self, section, orientation, role):  
        if ((orientation == Qt.Horizontal) and (role == Qt.DisplayRole)):
            hix=section+self.hmin
            return hix
        if ((orientation == Qt.Vertical) and (role == Qt.DisplayRole)):
            vix=section+self.vmin
            return vix
        return None
    def flatindex_C(self,index):
        return index.row()*self.cs+index.column()
    def flatindex_F(self,index):
        return index.row()+index.column()*self.ls
    def columnCount(self, parent):
        return self.cs
    def rowCount(self, parent):
        return self.ls
    def index(self, row, column, parent):
        return self.createIndex(row, column, 0)  
    def data(self, index, role=Qt.DisplayRole):
        if (role not in [Qt.DisplayRole,Qt.FontRole]): return None
        if (self.flatarray is None): return None
        if (role==Qt.FontRole): return self.font
        return self.fmt%self.flatarray[self.flatindex(index)]
    def flags(self, index):  
        if (not index.isValid()):  return Qt.NoItemFlags  
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable
    def getValue(self,index):
        return self.flatarray[self.flatindex(index)]
    def getEnumeratedValueIfPossible(self,index):
        if (self.node.sidsType()==CGK.Elements_ts):
            if ((index.row()==0) and (index.column()==0)):
                ev=CGK.ElementType_l
                et=[s[:-1]+'t' for s in CGK.cgnsenums]
                eti=et.index(CGK.ElementType_ts)
                evi=self.flatarray[self.flatindex(index)]
                return (et,ev,eti,evi)
        return None

# -----------------------------------------------------------------
class Q7DocEditor(QTextEdit):  
    def __init__(self,parent=None):
      QTextEdit.__init__(self,parent)
    def initText(self,text):
      self.clear()
      self.insertPlainText(text)
      self.moveCursor(QTextCursor.Start,QTextCursor.MoveAnchor)
      self.ensureCursorVisible()

# -----------------------------------------------------------------
class Q7PythonEditor(QTextEdit):  
    def __init__(self,parent=None):
      QTextEdit.__init__(self,parent)
      self.highlighter = Q7PythonEditorHighlighter(self.document())
      fsz=12
      ffm='Courier'
      self.setStyleSheet("font: %dpt \"%s\";"%(fsz,ffm))
    def initText(self,text):
      self.clear()
      self.insertPlainText(text)
      self.moveCursor(QTextCursor.Start,QTextCursor.MoveAnchor)
      self.ensureCursorVisible()

class Q7PythonEditorHighlighter(QSyntaxHighlighter):
    def __init__(self,*args):
        keywords =r"\bimport\b|\bas\b|\bfor\b|\bdef\b|\bpass\b"
        keywords+=r"|\bclass\b|\bfrom\b|\bif\b|\bthen\b|\belse\b"
        keywords+=r"|\belif\b|\btry\b|\bexcept\b|\bfinally\b|\braise\b"        
        keywords+=r"|\bprint\b|\bin\b|\bnot\b|\band\b|\bor\b|\bcontinue\b"
        keywords+=r"|\bwhile\b|\breturn\b|\blambda\b|\bwith\b|\del\b"
        keywords+=r"|\bglobal\b|\byield\b|\bexec\b|\bassert\b|\break\b"
        constants=r"\bNone\b|\bTrue\b|\bFalse\b|\bself\b"
        autovars =r"\bNODE\b|\bNAME\b|\bVALUE\b|\bSIDSTYPE\b|\bCHILDREN\b"
        autovars+=r"|\bTREE\b|\bPATH\b|\bRESULT\b|\bARGS\b|\bPARENT\b"
        numbers=r'[-+]?\d+'+'|[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
        comment=r'\#.*$'
        textstring=r'"[^"]*?"|\'[^\']*?\''
        self.f_keywords=QTextCharFormat() 
        self.f_keywords.setFontWeight(QFont.Bold)
        self.f_keywords.setForeground(QColor('Blue'))
        self.f_comment=QTextCharFormat()
        self.f_comment.setFontWeight(QFont.Light)
        self.f_comment.setForeground(QColor('Red'))
        self.f_constants=QTextCharFormat()
        self.f_constants.setFontWeight(QFont.Light)
        self.f_constants.setForeground(QColor('Navy'))
        self.f_autovars=QTextCharFormat()
        self.f_autovars.setFontWeight(QFont.Bold)
        self.f_autovars.setForeground(QColor('Green'))
        self.f_textstring=QTextCharFormat()
        self.f_textstring.setFontWeight(QFont.Bold)
        self.f_textstring.setForeground(QColor('Coral'))
        self.f_numbers=QTextCharFormat()
        self.f_numbers.setFontWeight(QFont.Light)
        self.f_numbers.setForeground(QColor('Gray'))
        self.r_keywords=re.compile(keywords)
        self.r_numbers=re.compile(numbers)
        self.r_comment=re.compile(comment)
        self.r_constants=re.compile(constants)
        self.r_autovars=re.compile(autovars)
        self.r_textstring=re.compile(textstring)
        self.in_comment=False
        self.in_string=False
        self.r_syntax=[]
        self.r_syntax.append((self.r_keywords,self.f_keywords))
        self.r_syntax.append((self.r_numbers,self.f_numbers))
        self.r_syntax.append((self.r_autovars,self.f_autovars))
        self.r_syntax.append((self.r_constants,self.f_constants))
        self.r_syntax.append((self.r_comment,self.f_comment))
        self.r_syntax.append((self.r_textstring,self.f_textstring))
        QSyntaxHighlighter.__init__(self,*args)
        
    def highlightBlock(self, textblock):
        text=unicode(textblock)
        fdef=self.format(0)
        fmax=len(text)
        for (rex,fmt) in self.r_syntax:
            mit=rex.finditer(text)
            for m in mit:
                self.setFormat(m.span()[0],m.span()[1],fmt)
                self.setFormat(m.span()[1],fmax,fdef)

# -----------------------------------------------------------------
