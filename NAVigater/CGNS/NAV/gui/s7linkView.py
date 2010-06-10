#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#
import Tkinter
Tkinter.wantobjects=0 #necessary for tk-8.5 and some buggy tkinter installs
from Tkinter import *
from TkTreectrl import *
import os
import sys
import FileDialog

import s7globals
G___=s7globals.s7G

import s7treeFingerPrint
import s7viewControl
import s7utils
import s7windoz

def sortLinkList(a,b):
  (acurfile,acurnode,atargetfile,atargetnode,alklevel,alkstatus)=a
  (bcurfile,bcurnode,btargetfile,btargetnode,blklevel,blkstatus)=b
  print acurnode, bcurnode
  if (acurnode<bcurnode): return -1
  if (acurnode>bcurnode): return  1
  return 0

# -----------------------------------------------------------------------------
class wLinkView(s7windoz.wWindoz,ScrolledTreectrl):
  def __init__(self,wcontrol,treefingerprint):
    s7windoz.wWindoz.__init__(self,wcontrol,
                              'pyS7: Link view [%s]'%treefingerprint.filename)
    ScrolledTreectrl.__init__(self,self._wtop,relief=GROOVE,border=3)

    self._tree=self.treectrl
    self._tree.configure(yscrollincrement=40)
    self._tree.state_define('marked')
    self._tree.state_define('leaf')
    self._tree.state_define('islink')
    self._tree.state_define('islinkignored')
    self._tree.state_define('islinkbroken')            

    self._viewtree=treefingerprint
    self._viewtree.links.sort(sortLinkList)

    self._viewWindow=wcontrol

    self._files={}

#    self.banner= Frame(self._wtop, borderwidth=2)
#    self.banner.ok=Button(self.banner,text="ok",border=0,justify=RIGHT,
#                             image=self._icon['close-view'],
#                             command=self._wtop.destroy,anchor=E)
#    self.banner.ok.pack(side=RIGHT,anchor=E)
#    self.banner.pack(fill=X,side=TOP)

    ik=self._icon
    it=self._tree
    self.col_graph=self._tree.column_create(font=self.titlefont,
                                            text="File",expand=1)
    self.col_status=self._tree.column_create(font=self.titlefont,
                                             text="L",expand=0,width=20)
    self.col_depth=self._tree.column_create(font=self.titlefont,
                                            text="D",expand=0,width=20)
    self.col_nodeT=self._tree.column_create(font=self.titlefont,
                                            text="Target")
    self.col_node1=self._tree.column_create(font=self.titlefont,
                                            text="Level 1")
    self.col_node2=self._tree.column_create(font=self.titlefont,
                                            text="Level 2")
    self.col_node3=self._tree.column_create(font=self.titlefont,
                                            text="Level 3")
    self.el_t1=it.element_create(type=IMAGE,image=(ik['node-sids-closed'],OPEN,
                                                   ik['node-sids-opened'],''))
    self.el_t5=it.element_create(type=IMAGE,image=ik['node-sids-leaf'])
    self.el_t7=it.element_create(type=IMAGE,image=(\
      ik['link-node'],  'islink',\
      ik['link-ignore'],'islinkignored',\
      ik['link-error'], 'islinkbroken'))
    self.el_font1=G___.font['Tb']
    self.el_font2=G___.font['Ta']
    self.el_text1=self._tree.element_create(type=TEXT,
                                            fill=(G___.color_Ta, SELECTED))
    self._tree.element_configure(self.el_text1,font=self.el_font1)
    self.el_text2=self._tree.element_create(type=TEXT,
                                            fill=(G___.color_Ta, SELECTED))
    self._tree.element_configure(self.el_text2,font=self.el_font2)
    self.el_w1=self._tree.element_create(type=WINDOW)

    self.el_select1=self._tree.element_create(type=RECT,
                                               fill=(G___.color_Tc, SELECTED),
                                               width=400)
    self.st_folder=self._tree.style_create()
    self._tree.style_elements(self.st_folder,self.el_t1,self.el_text1)    
    self._tree.style_layout(self.st_folder,self.el_text1,pady=2,padx=0)
    self._tree.style_layout(self.st_folder,self.el_t1,pady=2)

    self.st_leaf=self._tree.style_create()
    self._tree.style_elements(self.st_leaf,self.el_t5,self.el_text2)    
    self._tree.style_layout(self.st_leaf,self.el_text2,pady=2,padx=0)
    self._tree.style_layout(self.st_leaf,self.el_t5,pady=2)

    self.st_link=self._tree.style_create()
    self._tree.style_elements(self.st_link,self.el_select1,self.el_t7)
    self._tree.style_layout(self.st_link, self.el_t7, pady=2)
    self._tree.style_layout(self.st_link, self.el_select1,
                            union=(self.el_t7),
                            ipadx=2,iexpand=NS)

    self.st_nodename=self._tree.style_create()
    self._tree.style_elements(self.st_nodename,self.el_select1,self.el_text1)
    self._tree.style_layout(self.st_nodename,self.el_text1,pady=2)
    self._tree.style_layout(self.st_nodename,self.el_select1,
                             union=(self.el_text1,),ipadx=2,iexpand=NS)

    self._tree.configure(treecolumn=self.col_graph,
                         background='white',
                         height=300,width=600,
                         showheader=G___.showColumnTitle)

    self._tree.itemstyle_set(ROOT,self.col_graph,self.st_folder)
    self._tree.itemelement_config(ROOT,self.col_graph,self.el_text1,
                                  text=self._viewtree.filename,datatype=STRING)
    self._tree.item_config(ROOT, button=0)
    self._tree.notify_generate('<Expand-before>', item=ROOT)
    self._tree.see(ROOT)

    self.addLinkEntries()

    self.pack(fill=BOTH, expand=1)

  def onexit(self):
    self.closeWindow()
    self._viewWindow._haslinkWindow=None

  def addLinkEntries(self):
    for f in self._viewtree.links:
      (currentfile, currentnode, targetfile, targetnode, lklevel, lkstatus)=f
      if (not self._files.has_key(lklevel)):
        self._files[lklevel]={}
      if (not self._files[lklevel].has_key(targetfile)):
        self._files[lklevel][targetfile]=self.addEntryL1(ROOT,targetfile)
      self.addEntryL2(self._files[lklevel][targetfile],currentnode,targetnode,lkstatus)
        
  def addEntryL1(self,item,filename):
    t=self._tree
    if (os.path.splitext(filename)[0]==self._viewtree.filename):
      filename='{SELF}'
    enew=t.create_item(parent=item,button=True,open=0)[0]
    t.itemstyle_set(enew, self.col_status,self.st_link)
    t.itemstate_set(enew,'!leaf')
    t.itemstate_set(enew,'!islink')
    t.itemstate_set(enew,'!islinkignored')
    t.itemstate_set(enew,'!islinkbroken')    
    t.itemstyle_set(enew,self.col_graph,self.st_folder)
    t.itemelement_config(enew,self.col_graph,self.el_text1,
                         text=filename,datatype=STRING,draw=True)
    return enew

  def addEntryL2(self,item,currentnode,targetnode,lkstatus):
    t=self._tree
    enew=t.create_item(parent=item,button=False,open=0)[0]
    t.itemstate_set(enew,'leaf')
    t.itemstyle_set(enew, self.col_status,self.st_link)
    t.itemstate_set(enew,'!islink')
    t.itemstate_set(enew,'!islinkignored')
    t.itemstate_set(enew,'!islinkbroken')    
    t.itemstyle_set(enew,self.col_graph,self.st_leaf)
    t.itemelement_config(enew,self.col_graph,self.el_text2,
                         text=currentnode,datatype=STRING,draw=True)
    t.itemstyle_set(enew,self.col_nodeT,self.st_nodename)
    t.itemelement_config(enew,self.col_nodeT,self.el_text1,
                         text=targetnode,datatype=STRING)
    return enew
               
def getLinkStatusForThisNode(pth,node,parent,CGNStarget,treefinger):
  rt=0
  for lk in treefinger.links:
    if (lk[1]==pth):             rt=1
    if ((rt==1) and (lk[5]==0)): rt=3
    if ((rt==1) and (not G___.followLinks)): rt=2
  return rt

# --- last line
