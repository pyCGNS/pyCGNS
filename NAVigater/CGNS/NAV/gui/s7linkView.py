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
import s7treeSimple
import s7utils
import s7windoz

def sortLinkList(a,b):
  (acurfile,acurnode,atargetfile,atargetnode,alklevel,alkstatus)=a
  (bcurfile,bcurnode,btargetfile,btargetnode,blklevel,blkstatus)=b
  #print acurnode, bcurnode
  if (acurnode<bcurnode): return -1
  if (acurnode>bcurnode): return  1
  return 0

# -----------------------------------------------------------------------------
class wLinkEdit(s7windoz.wWindoz):
  def __init__(self,wcontrol,tree,fingerprint,node,ppath):

    self.control=wcontrol

    if (self.control.addLinkWindow): return
    self.control.addLinkWindow=True
    
    s7windoz.wWindoz.__init__(self,wcontrol,'CGNS.NAV: Link form')

    self.fingerprint=fingerprint
    self.tree=fingerprint.tree

    fname=fingerprint.filename+fingerprint.fileext
    self.l_width=min(max(len(fingerprint.filedir),len(fname),
                         len(ppath),len(node[0]))+5,80)
    self.targetNode=node
    self.targetPath=ppath
    
    self.v_ldir=StringVar()
    self.v_ldir.set(fingerprint.filedir)
    self.v_lfile=StringVar()
    self.v_lfile.set(fname)
    self.v_lpath=StringVar()
    self.v_lpath.set(ppath)
    self.v_lnode=StringVar()
    self.v_lnode.set(node[0])
    self.v_ddir=StringVar()
    self.v_ddir.set(fingerprint.filedir)
    self.v_dfile=StringVar()
    self.v_dfile.set(fingerprint.keyword)
    self.v_dpath=StringVar()
    self.v_dpath.set(ppath)
    self.v_dnode=StringVar()
    self.v_dnode.set(node[0])
    self.xdir=Label(self._wtop,text='Dir',font=G___.font['L'])
    self.xfile=Label(self._wtop,text='File',font=G___.font['L'])
    self.xpath=Label(self._wtop,text='Path',font=G___.font['L'])
    self.xnode=Label(self._wtop,text='Node',font=G___.font['L'])
    self.labl=Label(self._wtop,text='Local',font=G___.font['L'])
    self.ldir=Entry(self._wtop,textvariable=self.v_ldir,width=self.l_width,
                    font=G___.font['E'],state=DISABLED)
    self.lfile=Entry(self._wtop,textvariable=self.v_lfile,width=self.l_width,
                     font=G___.font['E'],state=DISABLED)
    self.lpath=Entry(self._wtop,textvariable=self.v_lpath,width=self.l_width,
                    font=G___.font['E'],state=DISABLED)
    self.lnode=Entry(self._wtop,textvariable=self.v_lnode,width=self.l_width,
                     font=G___.font['E'],state=DISABLED)
    self.labd=Label(self._wtop,text='Destination',font=G___.font['L'])
    self.ddir=Entry(self._wtop,textvariable=self.v_ddir,width=self.l_width,
                     font=G___.font['E'])
    self.dfile=Entry(self._wtop,textvariable=self.v_dfile,width=self.l_width,
                     font=G___.font['E'])
    self.dpath=Entry(self._wtop,textvariable=self.v_dpath,width=self.l_width,
                     font=G___.font['E'])
    self.dnode=Entry(self._wtop,textvariable=self.v_dnode,width=self.l_width,
                     font=G___.font['E'])

    self.xdir.grid( row=2,column=0,sticky=W)
    self.xfile.grid(row=3,column=0,sticky=W)
    self.xpath.grid(row=4,column=0,sticky=W)
    self.xnode.grid(row=5,column=0,sticky=W)
    self.labl.grid( row=1,column=1,sticky=W)
    self.ldir.grid( row=2,column=1,sticky=W)
    self.lfile.grid(row=3,column=1,sticky=W)
    self.lpath.grid(row=4,column=1,sticky=W)
    self.lnode.grid(row=5,column=1,sticky=W)
    self.labd.grid( row=1,column=2,sticky=W)
    self.ddir.grid( row=2,column=2,sticky=W)
    self.dfile.grid(row=3,column=2,sticky=W)
    self.dpath.grid(row=4,column=2,sticky=W)
    self.dnode.grid(row=5,column=2,sticky=W)
    
    self.b=Button(self._wtop,text="Ok",command=self.checkandsave,
                  font=G___.font['B'])
    self.b.grid(row=6,column=0,sticky=E)
    self.c=Button(self._wtop,text="Cancel",command=self.leave,
                  font=G___.font['B'])
    self.c.grid(row=6,column=1,sticky=W)
    self._wtop.grid()
    t=self._wtop.winfo_toplevel() 
    t.rowconfigure(3,weight=0)
    t.columnconfigure(1,weight=0)

  def checkandsave(self):
    dfile=self.v_dfile.get()
    dnode=self.v_dnode.get()
    dpath=self.v_dpath.get()
    ddir=self.v_ddir.get()
    lnk="[%s/%s]->%s/%s:[%s/%s]"%(self.targetPath,self.targetNode[0],
                                  ddir,dfile,dpath,dnode)
    r=s7utils.addLinkWarning(lnk)
    if (r):
      fg=self.fingerprint
      self.control.normalizeLinkList(fg)
      nnames=os.path.normpath(fg.filedir+'/'+fg.filename+fg.fileext)
      npaths=os.path.normpath(self.targetPath+'/'+self.targetNode[0])
      nnamed=os.path.normpath(ddir+'/'+dfile)
      npathd=os.path.normpath(dpath+'/'+dnode)
      nlink=(nnames,npaths,nnamed,npathd,0,0)
      if (nlink not in fg.links):
        fg.links.append(nlink)
      self.targetNode[1]=None
      self.targetNode[2]=[]
      self.targetNode[3]=''
      s7treeSimple.updateViews(fg,links=True)
    self.leave()
    
  def leave(self):
    self._wtop.destroy()
    self.control.addLinkWindow=None
    
  def onexit(self):
    self.leave()

# -----------------------------------------------------------------------------
class wLinkView(s7windoz.wWindoz,ScrolledTreectrl):
  def __init__(self,wcontrol,treefingerprint):
    s7windoz.wWindoz.__init__(self,wcontrol,
                              'CGNS.NAV: Link view [%s]'%treefingerprint.filename)
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
