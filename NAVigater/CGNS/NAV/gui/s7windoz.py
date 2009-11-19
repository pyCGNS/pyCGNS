#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $File$
#  $Node$
#  $Last$
#  -------------------------------------------------------------------------

import Tkinter
Tkinter.wantobjects=0 #necessary for tk-8.5 and some buggy tkinter installs
from Tkinter import *
from TkTreectrl import *
import os
import sys
import string
import FileDialog

import s7globals
G___=s7globals.s7G


import s7viewControl
import s7utils
import s7treeFingerPrint
import CGNS.NAV.supervisor.s7parser as s7parser

# --------------------------------------------------------------------
class wWindoz:
  _nodebuffer=None
  
  def __init__(self,wcontrol,title=None):

    wtop=Toplevel(wcontrol)
    wtop.title(title)
    wtop.protocol("WM_DELETE_WINDOW",self.onexit)
    
    self._wtop=wtop
    self._icon=s7utils.wIconFactory()
    self._control=wcontrol
    self._viewid=None
    self._tree=None
    self._paths={}
    self._viewtree=None
    self._parent=None

    self.ondelay=200
    self.offdelay=2000
    self.onxpos=+35
    self.onypos=-35

    self.menufont =G___.font['M']
    self.listfont =G___.font['E']
    self.titlefont=G___.font['H']
    self.labelfont=G___.font['L']

    self.unlockMouse()

  def menu(self,mlist):
    self._tool=Frame(self._wtop)
    self._menuentries=[]
    col=-1
    for (bi,ba,bp,bc,bh) in mlist:
      col+=1
      lb=Label(self._tool,text="",border=0)
      lb.grid(row=0,column=col,padx=bp,pady=2,sticky=ba)
      col+=1
      bx=Button(self._tool,border=0,text=bi,image=self._icon[bi],command=bc)
      self._menuentries.append(bx)
      bx.ballooon=Menu(self._wtop,tearoff=0,border=0,bg='yellow')
      bx.ballooon.add_command(font=self.menufont,
                              label=bh,state=DISABLED)
      bx.bind('<Enter>',self.enterballooon)
      bx.grid(row=0,column=col,pady=2)
    #self._tool.bind('<Leave>',self.forceleaveballooon)      
    self._tool.grid(row=0,sticky=NW)
    t=self._tool.winfo_toplevel() 
    t.rowconfigure(0,weight=0)
    t.columnconfigure(0,weight=0)
    self.currentballooon=None
    self.lastballooon=None
    return col

  def enterballooon(self,e):
    if (not G___.helpBallooons): return
    if (self.currentballooon==e.widget): return
    if (self.currentballooon!=None): self.leaveballooon(self.currentballooon)
    self.currentballooon=e.widget
    self.lastsetupballooon=e.widget.after(self.ondelay,self.setballooon,e)
    
  def setballooon(self,e):
    e.widget.ballooon.tk_popup(e.x_root+self.onxpos,e.y_root+self.onypos,0)
    self.lastballooon=e.widget.after(self.offdelay,self.leaveballooon,e.widget)
    e.widget.ballooon.grab_release()
    
  def forceleaveballooon(self,e):
    self.leaveballooon(e.widget)
      
  def leaveballooon(self,e):
    if (self.lastballooon): e.after_cancel(self.lastballooon)
    for bl in self._menuentries:
      bl.ballooon.grab_release()
      bl.ballooon.unpost()
    
  def title(self,t):
    wtop.title(t)

  def top(self,msg):
    if 0:
      t=os.times()
      print '## pyS7: %8.4f %8.4f'%(t[0],t[1]),
      print '[%s]'%msg

  def lockMouse(self):
    for t in G___.treeStorage:
      for vw in t.viewlist:
        rw=vw.view._wtop.winfo_toplevel()
        rw.configure(cursor="watch")
        rw.update_idletasks()
    r=self._wtop.winfo_toplevel() 
    r.configure(cursor="watch")
    r.update_idletasks()

  def unlockMouse(self):
    for t in G___.treeStorage:
      for vw in t.viewlist:
        rw=vw.view._wtop.winfo_toplevel()
        rw.configure(cursor="arrow")
        rw.update_idletasks()
    r=self._wtop.winfo_toplevel() 
    r.configure(cursor="arrow")
    r.update_idletasks()

  # ---
  def closeWindow(self):
    self._wtop.destroy()

# --------------------------------------------------------------------
