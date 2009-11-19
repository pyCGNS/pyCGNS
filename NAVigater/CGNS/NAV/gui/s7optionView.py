#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import Tkinter
Tkinter.wantobjects=0 #necessary for tk-8.5 and some buggy tkinter installs
from Tkinter import *
from TkTreectrl import *
import os
import sys
import string
import numpy as Num

from CGNS.pyCGNSconfig import version as __vid__

import s7globals
G___=s7globals.s7G

import s7treeFingerPrint
import s7treeSimple
import s7tableSimple
import s7patternBrowser
import s7operateView
import s7utils
import s7fileDialog
import s7windoz


optionfiletemplate="""# pyS7 - options file
expandRecurse=%(expandRecurse)d
maxRecurse=%(maxRecurse)d
sidsRecurse=%(sidsRecurse)d
lateLoading=%(lateLoading)d
singleView=%(singleView)d
flyCheck=%(flyCheck)d
maxDisplaySize=%(maxDisplaySize)d
showIndex=%(showIndex)d
showColumnTitle=%(showColumnTitle)d
defaultProfile='%(defaultProfile)s'
profilePath=%(profilePath)s
followLinks=%(followLinks)d
saveLinks=%(saveLinks)d
historyFile='%(historyFile)s'
noData=%(noData)d
showSIDS=%(showSIDS)d
helpBallooons=%(helpBallooons)d
# last line
"""
# -----------------------------------------------------------------------------
class wOptionView(s7windoz.wWindoz):

  def optCheck(self,fr,key,st=NORMAL):
    cm=getattr(self,'_'+key)
    tx=getattr(self,'d_'+key)
    vr=getattr(self,'v_'+key)
    bt=Checkbutton(fr,text='%-32s'%tx,variable=vr,justify=RIGHT,
                   relief=FLAT,borderwidth=1,command=cm,
                   font=G___.font['L'],state=st)
    bt.grid(row=fr.row,column=fr.col,sticky=W)
    fr.row+=1
    self.olist[key]=bt
    return bt
    
  def optValue(self,fr,key,large=0):
    cm=getattr(self,'_'+key)
    tx=getattr(self,'d_'+key)
    vr=getattr(self,'v_'+key)
 #   f=Frame(fr,relief=FLAT)
    l=Label(fr,text=tx,justify=LEFT,font=G___.font['L'])
    if (not large):
      e=Entry(fr,textvariable=vr,justify=RIGHT,width=16,font=G___.font['E'],
              background='white')
    else:
      e=Text(fr,height=5,width=60,font=G___.font['E'],background='white')
      e.insert('1.0',vr.get())
    if (not large):
      l.grid(row=fr.row,column=fr.col,sticky=NW)
      e.grid(row=fr.row,column=fr.col+1,sticky=W+E)
      fr.row+=1
    else:
      l.grid(row=fr.row,column=fr.col,sticky=NW)
      fr.row+=1
      e.grid(row=fr.row,column=fr.col,columnspan=2,sticky=W)
      fr.row+=1
    self.olist[key]=e
    return fr
     
  def optVlist(self,fr,txt,ls,vr,cm):
    f=Frame(fr,relief=FLAT)
    l=Label(f,text=txt,justify=LEFT,font=G___.font['L'])
    m=Menubutton(f,textvariable=vr)
    m.menu=Menu(m,tearoff=0)
    m['menu']=m.menu
    for v in ls:
      m.menu.add_radiobutton(label=v,var=vr,value=v)
    l.grid(row=fr.row,column=fr.col,sticky=NW)
    m.grid(row=fr.row,column=fr.col+1,sticky=E)
    f.grid()
    fr.row+=1
    self.olist[key]=m
    return f

  def _expandRecurse(self):
    if (G___.expandRecurse): G___.expandRecurse=0
    else:                    G___.expandRecurse=G___.maxRecurse
    
  def _showIndex(self):
    G___.showIndex=not G___.showIndex
    
  def _maxRecurse(self):
    pass
    
  def _maxDisplaySize(self):
    pass
    
  def _helpBallooons(self):
    G___.helpBallooons=not G___.helpBallooons
    
  def _followLinks(self):
    G___.followLinks=not G___.followLinks
    
  def _saveLinks(self):
    G___.saveLinks=not G___.saveLinks
    
  def _showSIDS(self):
    G___.showSIDS=not G___.showSIDS
    
  def _noData(self):
    G___.noData=not G___.noData
    
  def _sidsRecurse(self):
    G___.sidsRecurse=not G___.sidsRecurse
    
  def _lateLoading(self):
    G___.lateLoading=not G___.lateLoading
    
  def _showColumnTitle(self):
    G___.showColumnTitle=not G___.showColumnTitle
    
  def _singleView(self):
    G___.singleView=not G___.singleView

  def _flyCheck(self):
    G___.flyCheck=not G___.flyCheck

  def _profilePath(self):
    pass
  
  def _defaultProfile(self):
    pass

  def _historyFile(self):
    pass

  def updateVars(self):
    tx=self.olist['profilePath'].get('1.0',END)
    lp=tx.replace('\n',':').split(':')
    rlp=[]
    for l in lp:
      if (l!=''):
        rlp.append(l)
    G___.profilePath=rlp
    G___.maxRecurse=getattr(self,'v_maxRecurse').get()
    G___.maxDisplaySize=getattr(self,'v_maxDisplaySize').get()
    G___.defaultProfile=getattr(self,'v_defaultProfile').get()
    G___.historyFile=getattr(self,'v_historyFile').get()
    if (getattr(self,'v_expandRecurse').get()):   G___.expandRecurse=1
    else:                                         G___.expandRecurse=0    
    if (getattr(self,'v_helpBallooons').get()):   G___.helpBallooons=1
    else:                                         G___.helpBallooons=0    
    
  def setVars(self):
    self.olist={}
    
    self.v_expandRecurse=IntVar()
    self.v_expandRecurse.set(G___.expandRecurse)
    self.d_expandRecurse='Recursive tree display'
    self.v_showIndex=IntVar()
    self.v_showIndex.set(G___.showIndex)
    self.d_showIndex='Show table index'
    self.v_showColumnTitle=IntVar()
    self.v_showColumnTitle.set(G___.showColumnTitle)
    self.d_showColumnTitle='Show columns titles'
    self.v_sidsRecurse=IntVar()
    self.v_sidsRecurse.set(G___.sidsRecurse)
    self.d_sidsRecurse='Recurse SIDS patterns load/copy'
    self.v_lateLoading=IntVar()
    self.v_lateLoading.set(G___.lateLoading)
    self.d_lateLoading='Load node on display required'
    self.v_singleView=IntVar()
    self.v_singleView.set(G___.singleView)
    self.d_singleView='One view per tree/node'
    self.v_helpBallooons=IntVar()
    self.v_helpBallooons.set(G___.helpBallooons)
    self.d_helpBallooons='Show menu help balloons'
    self.v_flyCheck=IntVar()
    self.v_flyCheck.set(G___.flyCheck)
    self.d_flyCheck='Check on the fly'
    self.v_followLinks=IntVar()
    self.v_followLinks.set(G___.followLinks)
    self.d_followLinks='Follow links during file load'
    self.v_saveLinks=IntVar()
    self.v_saveLinks.set(G___.saveLinks)
    self.d_saveLinks='Do NOT follow links during file save'
    self.v_showSIDS=IntVar()
    self.v_showSIDS.set(G___.showSIDS)
    self.d_showSIDS='Show SIDS status column'
    self.v_noData=IntVar()
    self.v_noData.set(G___.noData)
    self.d_noData='Do not load large DataArrays'
    self.v_maxRecurse=IntVar()
    self.v_maxRecurse.set(G___.maxRecurse)
    self.d_maxRecurse='Max tree parse recursion level'
    self.v_maxDisplaySize=IntVar()
    self.v_maxDisplaySize.set(G___.maxDisplaySize)
    self.d_maxDisplaySize='Max length of data displayed in table'
    self.v_defaultProfile=StringVar()
    self.v_defaultProfile.set(G___.defaultProfile)
    self.d_defaultProfile='Default profiles'
    self.v_profilePath=StringVar()
    self.v_profilePath.set(string.join(G___.profilePath,':').replace(':','\n'))
    self.d_profilePath='Profiles path list:'
    self.v_historyFile=StringVar()
    self.v_historyFile.set(G___.historyFile)
    self.d_historyFile='History file name:'
  
  def __init__(self,wcontrol):
    s7windoz.wWindoz.__init__(self,wcontrol,'pyS7: Options view')
    self.setVars()

    self.options = self._wtop
    self.options.title('pyS7: Options')
    self.options.fleft=Frame(self.options,relief=GROOVE,borderwidth=3)
    self.options.fright=Frame(self.options,relief=GROOVE,borderwidth=3)
    self.options.fleft.row=1
    self.options.fleft.col=0
    self.options.fright.row=1
    self.options.fright.col=1
    _left=self.options.fleft
    _right=self.options.fright
    
    self.optCheck(_left,'expandRecurse')
    self.optCheck(_left,'singleView',DISABLED)
    self.optCheck(_left,'showIndex')
    self.optCheck(_left,'showColumnTitle')
    self.optCheck(_left,'helpBallooons')
    self.optCheck(_left,'sidsRecurse')
    self.optCheck(_left,'lateLoading',DISABLED)    
    self.optCheck(_left,'flyCheck',DISABLED)
    self.optCheck(_left,'followLinks')
    self.optCheck(_left,'saveLinks')    
    self.optCheck(_left,'noData')
    self.optCheck(_left,'showSIDS')    
#    self.optVlist(_right,'Follow file links :',
#                  ['Always','Never','Only first level'],
#                  self.opt_sidsrecurse,self.s7_sids_recurse)
    self.optValue(_right,'maxRecurse')
    self.optValue(_right,'maxDisplaySize')
    self.optValue(_right,'historyFile')
    self.optValue(_right,'defaultProfile')
    self.optValue(_right,'profilePath',large=1)

    self.options.fleft.grid(row=1,column=0,sticky=NW)
    self.options.fright.grid(row=1,column=1,sticky=NE,columnspan=3)

    self.options.ok=Button(self.options,text="Ok",font=G___.font['B'],
                           command=self.save,borderwidth=1)
    self.options.ok.grid(row=self.options.fright.row,column=3,sticky=W)
    self.options.nx=Button(self.options,text="Cancel",font=G___.font['B'],
                           command=self.cancel,borderwidth=1)
    self.options.nx.grid(row=self.options.fright.row,column=3,sticky=E)
    
    t=self.options.winfo_toplevel() 
    t.rowconfigure(0,weight=1)
    self.options.grab_set()

  def save(self):
    self.updateVars()
    import os
    path=os.environ['HOME']+'/.s7options.py'
    f=open(path,'w+')
    f.write(optionfiletemplate%G___.__dict__)
    f.close()
    self.options.destroy()
     
  def onexit(self):
    self.leave()

  def leave(self):
    self.updateVars()
    self.options.destroy()

  def cancel(self):
    self.options.destroy()

# -----------------------------------------------------------------------------
# --- last line
