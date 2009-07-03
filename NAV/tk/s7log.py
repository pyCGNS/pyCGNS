# -----------------------------------------------------------------------------
# pyS7 - CGNS/SIDS editor
# ONERA/DSNA - marc.poinot@onera.fr
# pyS7 - $Rev: 70 $ $Date: 2009-01-30 11:49:10 +0100 (Fri, 30 Jan 2009) $
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source
# tree for license information.
#
import Tkinter
Tkinter.wantobjects=0 #necessary for tk-8.5 and some buggy tkinter installs
from Tkinter import *
from TkTreectrl import *
import os
import sys
import s7fileDialog

import s7globals
G___=s7globals.s7G

import s7treeFingerPrint
import s7viewControl
import s7utils
import s7windoz

# -----------------------------------------------------------------------------
class wLog(s7windoz.wWindoz):
  def __init__(self,wtree,wcontrol):

    self.wcontrol=wcontrol
    self.tagcount=0
    self.currenttag=0
    self.tagdict={}
    self.isclear=1
    self.viewtree=wtree._viewtree

    s7windoz.wWindoz.__init__(self,self.wcontrol,'pyS7: Log view')

    mcols=self.menu([('check-bwd',W,2,self.prevtag, 'Previous message'),
                     ('check-fwd',W,2,self.nexttag, 'Next message'),
                     ('check-save',W,20,self.save,'Save log'),
                     ('check-clear',W,2,self.clearLog,'Clear log')])

    self.tagentry=StringVar()
    self.tagentry.set("%d/%d"%(0,0))
    self.log=Frame(self._wtop, borderwidth=2)
    self.log.nx=Entry(self._wtop,textvariable=self.tagentry,width=8,
                      font=G___.font['E'],background='white',justify=RIGHT)
    self.log.nx.grid(row=0,column=0,sticky=E)

    self.T=Text(self._wtop,wrap=NONE,font=G___.font['L'])
    self.scrollx=Scrollbar(self._wtop,orient=HORIZONTAL,command=self.T.xview)
    self.scrolly=Scrollbar(self._wtop,orient=VERTICAL,command=self.T.yview)
    self.scrollx.grid(row=2,column=0,sticky=S+E+W)    
    self.scrolly.grid(row=1,column=1,sticky=E+N+S)
    self.T["xscrollcommand"]=self.scrollx.set
    self.T["yscrollcommand"]=self.scrolly.set
    self.T["state"]=DISABLED

    self.T.tag_config('#INFO',background='white',font=G___.font['E'])
    self.T.tag_config('#FAIL',background='coral')
    self.T.tag_config('#WARNING',background='yellow')    

    self.T.grid(row=1,column=0,sticky=NW+E+S)
    self.log.grid(row=1,sticky=N+E+S+W)
    t=self.log.winfo_toplevel() 
    t.rowconfigure(1,weight=1)
    t.columnconfigure(0,weight=1)

  def nexttag(self):
    self.currenttag+=1
    if (self.currenttag > self.tagcount): self.currenttag=1
    try:
      self.T.see(self.tagdict['#%d'%self.currenttag])
    except KeyError:
      pass
    self.tagentry.set("%d/%d"%(self.currenttag,self.tagcount))    

  def prevtag(self):
    self.currenttag-=1
    if (self.currenttag < 0): self.currenttag=self.tagcount
    try:
      self.T.see(self.tagdict['#%d'%self.currenttag])
    except KeyError:
      pass
    self.tagentry.set("%d/%d"%(self.currenttag,self.tagcount))    

  def save(self):
    file=s7fileDialog.s7filedialog(self._wtop,save=1,pat='*.txt')
    if (not file): return
    if (os.path.exists(file)):
      bakfile=file+'.bak'
      os.rename(file,bakfile)
    if (file):
      f=open(file,'w+')
      f.write(self.T.get("1.0",END))
      f.close()
    
  def clearLog(self):
    clearChecks(self.viewtree)
    self.clear()
    
  def clear(self):
    self.T["state"]=NORMAL    
    self.T.delete("1.0",END)
    self.T["state"]=DISABLED
    self.tagcount=0
    self.currenttag=0
    self.tagentry.set("%d/%d"%(0,0))
    self.isclear=1
    
  def push(self,txt,tag=None):
    self.isclear=0
    self.T["state"]=NORMAL
    if (tag):
      idx=self.T.index(END)
      (l,c)=idx.split('.')
      l=str(int(l)+1)
      c='end'
      if (tag in ['#FAIL','#WARNING']):
        self.tagcount+=1
        self.T.insert('%s.%s'%(l,c),'%.3d%s'%(self.tagcount,txt[3:]),tag)
        self.tagdict['#%d'%self.tagcount]=idx
      else:
        self.T.insert('%s.%s'%(l,c),txt,tag)

    else:
      self.T.insert(END,txt)
    self.T["state"]=DISABLED
    self.tagentry.set("%d/%d"%(0,self.tagcount))

  def onexit(self):
    self.wcontrol.logViewClose()

# --------------------------------------------------------------------
def clearChecks(viewtree):
  for v in viewtree.viewlist:
    v.view.clearViewChecks()
    
# --- last line
