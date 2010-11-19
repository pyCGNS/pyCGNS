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
import copy
import numpy as NY

import s7globals
G___=s7globals.s7G

import CGNS.PAT.cgnsutils  as s7parser
from   CGNS.NAV.supervisor import s7check

import s7viewControl
import s7utils
import s7treeFingerPrint
import s7windoz
import s7linkView
import s7operateView
import s7log
    
# --------------------------------------------------------------------
class wEditBarStore:
  pass
class wEditBar(Frame):
  def __init__(self, master):
    Frame.__init__(self, master)
    self.master=master
    self.label=Label(self,bd=0,relief=FLAT)
    self.label.grid()
    self.font=G___.font['E']
    self.typedPanels={}
    self.currentPanel=None
  def getStoreValue(self,panel):
    return self.typedPanels[panel].value.get()
  def setStoreValue(self,panel,node,tree,path):
    self.typedPanels[panel].node=node
    self.typedPanels[panel].tree=tree
    self.typedPanels[panel].path=path
    self.typedPanels[panel].value.set(self.typedPanels[panel].setValue(node))
  def feed(self):
    self.typedPanels['__Accept']=wAccept(self)
    self.typedPanels['__Rename']=wRename(self)
    self.typedPanels['__Retype']=wRetype(self)    
    self.typedPanels['__ChangeTextValue']=wChangeTextValue(self)    
  def set(self, format, *args):
    self.label.config(text=format%args,font=self.font) 
    self.label.update_idletasks()
  def clear(self):
    self.label.config(text="")
    self.label.update_idletasks()
  def show(self,panel,grab=0):
    self.label.grid_remove()
    if (not self.typedPanels.has_key(panel)): self.createPanel(panel)
    self.typedPanels[panel].frame.grid()
    self.currentPanel=panel
    if (grab):
      tp=self.typedPanels[panel]
      tp.previousfocus=self.master.focus_get()
      tp.target.wait_visibility()
      tp.target.grab_set()
      tp.target.focus_set()            
    return self.typedPanels[panel]
  def hide(self):
    self.typedPanels[self.currentPanel].frame.grid_remove()
    self.label.grid()

# ------------------------------    
class wAccept(wEditBarStore):
  def __init__(self,editbar):
    r=self
    r.editbar=editbar
    r.frame=Frame(editbar)
    r.b=Button(r.frame,text="Ok",font=G___.font['B'])
    r.b.grid(row=2,column=0,sticky=W)
    r.c=Button(r.frame,text="Cancel",font=G___.font['B'])
    r.c.grid(row=2,column=1,sticky=W)
  def setCallBack(self,cbktp):
    self.b.config(command=cbktp[0])
    self.c.config(command=cbktp[1])    
class wChangeTextValue(wEditBarStore):
  def __init__(self,editbar):
    r=self
    r.editbar=editbar
    r.frame=Frame(editbar)
    r.value=StringVar()
    r.e=Entry(r.frame,textvariable=r.value,width=48,
              font=r.editbar.font,background='white')
    r.e.grid()
    r.target=r.e
    r.b=Button(r.frame,text="Ok",command=self.validate,font=G___.font['B'])
    r.b.grid()
  def setValue(self,node):
    return s7utils.toBeShown(node[1])
  def validate(self,event=None):
    r=self
    e=r.editbar.typedPanels[r.editbar.currentPanel]
    pnode=e.tree.findParentNodeFromPath(e.path,absolute=1)
    r.oldname=e.path
    r.newvalue=r.editbar.getStoreValue('__ChangeTextValue')
    e.target.grab_release()
    e.node[1]=r.editbar.getStoreValue('__ChangeTextValue')
    updateViews(e.tree._viewtree,[(r.oldname,r.oldname)])
    r.editbar.hide()
    if (r.previousfocus): r.previousfocus.focus_set()
    e.tree.modified()
    
class wRename(wEditBarStore):
  def __init__(r,editbar):
    r.editbar=editbar
    r.frame=Frame(editbar)
    r.value=StringVar()
    r.l=Label(r.frame,text='Name:',font=G___.font['L'])
    r.l.grid(row=2,column=0,sticky=W+S+N)
    r.e=Entry(r.frame,textvariable=r.value,width=48,
              font=r.editbar.font,background='white')
    r.e.grid(row=2,column=1,sticky=W+S+N)
    r.target=r.e
    r.target.bind('<Return>',r.validate)
    r.target.bind('<Escape>',r.close)    
  def setValue(self,node):
    try:
      r=node[0]
    except IndexError:
      return ''
    return node[0]
  def validate(r,event=None):
    e=r.editbar.typedPanels[r.editbar.currentPanel]
    if ('/' in r.editbar.getStoreValue('__Rename')):
      pass
    else:
      pnode=e.tree.findParentNodeFromPath(e.path,absolute=1)
      r.oldname=e.path
      r.newname=string.join(e.path.split('/')[:-1],'/')+'/'\
                +r.editbar.getStoreValue('__Rename')
      if (r.newname in s7parser.childNames(pnode)):
        s7utils.renameError(r.newname)
      else:
        e.node[0]=r.editbar.getStoreValue('__Rename')
      updateViews(e.tree._viewtree,[(r.oldname,r.newname)])
    r.editbar.typedPanels[r.editbar.currentPanel].tree.modified()
    r.close()
  def close(self,event=None):
    self.editbar.hide()
    if (self.previousfocus): self.previousfocus.focus_set()
    self.editbar.typedPanels[self.editbar.currentPanel].target.grab_release()
    
class wRetype(wEditBarStore):
  def __init__(r,editbar):
    r.editbar=editbar
    r.frame=Frame(editbar)
    r.value=StringVar()
    r.l=Label(r.frame,text='Type:',font=G___.font['L'])
    r.l.grid(row=2,column=0,sticky=W+S+N)
    r.e=Entry(r.frame,textvariable=r.value,width=48,font=r.editbar.font,
              background='white')
    r.target=r.e
    r.e.grid(row=2,column=1,sticky=W+S+N)
    r.target.bind('<Return>',r.validate)
    r.target.bind('<Escape>',r.close)    
  def setValue(self,node):
    return node[3]
  def validate(self,event=None):
    e=self.editbar.typedPanels[self.editbar.currentPanel]
    e.node[3]=self.editbar.getStoreValue('__Retype')
    updateViews(e.tree._viewtree,[(e.path,e.path)])
    self.editbar.typedPanels[self.editbar.currentPanel].tree.modified()
    self.close()
  def close(self,event=None):
    self.editbar.hide()
    if (self.previousfocus): self.previousfocus.focus_set()
    self.editbar.typedPanels[self.editbar.currentPanel].target.grab_release()
    
# --------------------------------------------------------------------
class wTreeSimple(s7windoz.wWindoz,ScrolledTreectrl):
  # --------------------------------------------------------------------
  def modified(self):
    if (self._viewtree.nosave): return
    self._control.modified(self._viewtree)
    self._menuentries[0].configure(image=self._icon['save'])
    self._menuentries[0].configure(command=self.cbk_save)
    self.updateMaxLevel()
  def unmodified(self):
    self._control.unmodified(self._viewtree)
    self._menuentries[0].configure(image=self._icon['save-done'])
    self._menuentries[0].configure(command=self.cbk_savedone)
  def cbk_print(self):
    f=s7utils.dumpWindow(self._tree)
  def cbk_savedone(self):
    pass
  def cbk_save(self):
    if (self._control.saveFile(self.CGNStarget,self._viewtree,
                               self._viewtree.saveFileName(),msg=0)):
      self.unmodified()
  def cbk_saveas(self):
    if (self._viewtree.nosave):
      if (not s7utils.noDataSave()): return
    if (self._control.saveFile(self.CGNStarget,self._viewtree)):
      self._viewtree.nosave=0
      self.unmodified()
  def cbk_clearcheck(self):
    if (self._control.logWindow): self._control.logWindow.clear()
    s7log.clearChecks(self._viewtree)
  def cbk_savecheck(self):
    if (self._control.logWindow): self._control.logWindow.save()
  def cbk_prevslist(self):
    it=self.getItemSelection()
    nit=self.getFirstMarkedFrom(1)
    self._tree.selection_modify(select=nit)
    self._tree.see(nit)
    self._edit.set(self._paths[str(nit)])
  def cbk_nextslist(self):
    it=self.getItemSelection()
    nit=self.getFirstMarkedFrom()
    self._tree.selection_modify(select=nit)
    self._tree.see(nit)
    self._edit.set(self._paths[str(nit)])
  def cbk_checkall(self):
    self._tree.selection_modify(deselect=ALL)
    self._tree.selection_modify(select=ROOT)
    self.SubCheck()
  def cbk_flagall(self):
    self._tree.selection_modify(select=ALL)
    for n in self._tree.selection_get():
      item=self._tree.item_id(n)
      self._tree.itemstate_set(item[0],'marked')
    self._tree.selection_modify(deselect=ALL)
  def cbk_unflagall(self):
    self._tree.selection_modify(select=ALL)
    for n in self._tree.selection_get():
      item=self._tree.item_id(n)
      self._tree.itemstate_set(item[0],'!marked')
    self._tree.selection_modify(deselect=ALL)
  def cbk_revertflags(self):
    self._tree.selection_modify(select=ALL)
    for n in self._tree.selection_get():
      item=self._tree.item_id(n)
      if (self._tree.itemstate_get(item,'marked')):
        self._tree.itemstate_set(item[0],'!marked')
      else:
        self._tree.itemstate_set(item[0],'marked')
    self._tree.selection_modify(deselect=ALL)
  def cbk_zoomin(self):
    self.expandOneMoreLevel()
  def cbk_zoomout(self):
    self.collapsedOneMoreLevel()
  def cbk_close(self):
    self.onexit()
  def cbk_savepattern(self):
    if (self._control.saveAsPattern(self.CGNStarget,self._viewtree)):
      self.unmodified()
  def cbk_link(self):
    self.linkView()
  def cbk_query(self):
    pass
  def cbk_check(self):
    pass
  def cbk_slist(self):
    self._tree.selection_modify(select=ALL)
    r=[]
    for n in self._tree.selection_get():
      item=self._tree.item_id(n)
      if (self._tree.itemstate_get(item,'marked')):
        path=self._paths[str(item[0])]
        node=self.findNodeFromPath(path)
        if (self._parent):
          path=string.join(self._parent.split('/')[:-1],'/')+'/'+path
        if (path!='/'): r.append((path,node[0],node[3]))
    self.operateMenu(r)
    self._tree.selection_modify(deselect=ALL)

  def updateMaxLevel(self):
    self.__levelMaxDepth=self._viewtree.maxDepth()
    
  def __init__(self,wcontrol,treefingerprint,
               startnode=None,parentnode=None):

    s7windoz.wWindoz.__init__(self,wcontrol)
    self._viewtree=treefingerprint
    self._subview=0
    self._parent=None
    self._islocked=0

    self.__expandLevel=1
    self.__levelMaxDepth=0
    self.__noData=G___.noData
    
    if (startnode):
      self._subview=1
      if (parentnode):
        ppth=parentnode.split('/')[1:-1]+startnode.split('/')
        self._parent=string.join(parentnode.split('/')[:-1],'/')+\
                                  '/'+startnode
        self.CGNStarget=s7parser.getNodeFromPath(ppth,
                                                treefingerprint.tree)
      else:
        self._parent=startnode
        self.CGNStarget=s7parser.getNodeFromPath(startnode.split('/')[1:],
                                                treefingerprint.tree)
      rootnodename=startnode.split('/')[-1]
    else:
      self.CGNStarget=treefingerprint.tree
      rootnodename='/'

    self._viewid=treefingerprint.addView(self,self._parent,'T')
    self._wtop.title('CGNS.NAV: [%s] View [%.2d]'%(treefingerprint.filename,
                                               self._viewid))
    ScrolledTreectrl.__init__(self,self._wtop,relief=GROOVE,border=3,
                              height=2*G___.wminheight,
                              width=1.5*G___.wminwidth)
    self.treectrl.configure(yscrollincrement=40) # hmmmm, what else ?
    
    self._tree=self.treectrl
    self._tree.state_define('marked')
    self._tree.state_define('fortran')
    self._tree.state_define('checkgood')
    self._tree.state_define('checkfail')    
    self._tree.state_define('checkwarn')    
    self._tree.state_define('leaf')
    self._tree.state_define('islink')
    self._tree.state_define('islinkignored')
    self._tree.state_define('islinkbroken')            

    self._haslinkWindow=None

    self.updateMaxLevel()
    
    ik=self._icon
    it=self._tree

    self._edit=wEditBar(self._wtop)
    self._edit.feed()
    self._edit.grid(row=2,column=0,sticky=N+W+S,columnspan=3)

    self.menu([('save-done',    NW, 2,self.cbk_savedone,'Save current tree'),
               ('tree-save',    NW,20,self.cbk_saveas,'Save tree as'),
               ('pattern-save', NW, 2,self.cbk_savepattern,'Save as pattern'),
               ('level-in',     NW,20,self.cbk_zoomin,'Show one more level'),
               ('level-out',    NW, 2,self.cbk_zoomout,'Hide lowest level'), 
               ('flag-all',     NW,20,self.cbk_flagall,'Flag all'),
               ('flag-none',    NW, 2,self.cbk_unflagall,'Remove all flags'),
               ('flag-revert',  NW, 2,self.cbk_revertflags,'Revert flags'),
               ('operate-list', NW, 2,self.cbk_slist,'Selected list view'),
               ('flag-bwd',     NW,20,self.cbk_prevslist,'Previous selected'),
               ('flag-fwd',     NW, 2,self.cbk_nextslist,'Next selected'),
               ('check-all',    NW,20,self.cbk_checkall,'Check all levels'),
               ('check-save',   NW, 2,self.cbk_savecheck,'Save check log'),
               ('check-clear',  NW, 2,self.cbk_clearcheck,'Clear check flags'),
               ('link-view',    NW,20,self.cbk_link,'Open link view'),
               ('operate-view', NW, 2,self.cbk_query,'Selection query view'),
               ('check-view',   NW, 2,self.cbk_check,'Check view'),
               ('snapshot',     NW,20,self.cbk_print,'Snapshot current view')])

    def popUpMenuOn(event,treefingerprint,node):
      try:
        tktree=event.widget
        id=tktree.identify(event.x,event.y)
        item=tktree.item_id(id[1])
        if (tktree.itemstate_get(item,'leaf')):
            pass # cannot use DISABLED on tk menu < 8.4
        self._tree.selection_clear()
        self._tree.selection_add(item)
        path=self._paths[str(item[0])]
        self._edit.set(path)
        self.popupmenu.tk_popup(event.x_root,event.y_root,0)
      finally:
        self.popupmenu.grab_release()

    def popUpMenuOff(event):
      self.popupmenu.grab_release()
      self.popupmenu.unpost()
        
    def SubView():
      tktree=self._tree
      selection=tktree.selection_get()
      item=tktree.item_id(selection)
      pth=self._paths[str(item[0])]
      if (tktree.itemstate_get(item,'leaf')): return
      s7viewControl.createDataTreeWindow(self._control,
                                         self._viewtree,
                                         G___.expandRecurse,
                                         startnode=pth,
                                         parent=self._parent)
        
    def AddPatternBrother():
      path=self.getSinglePathSelection()
      if (path == '/'):
        s7utils.pasteBrotherError()
        return
      node=self.findParentNodeFromPath(path)
      emptynode=['new node',None,[],'UserDefinedData_t']
      self.addToParentNode(node,emptynode)

    def AddPatternChild():
      path=self.getSinglePathSelection()
      if (path == '/'): node=self.CGNStarget
      elif self._parent:
        node=s7parser.getNodeFromPath(path.split('/'),
                                     [None,None,[self.CGNStarget],None])  
      else:            
        node=s7parser.getNodeFromPath(path.split('/')[1:],self.CGNStarget)
      emptynode=['new node',None,[],'UserDefinedData_t']
      self.addToParentNode(node,emptynode)

    def tcCom(ntype,node,pth):
      node[3]=ntype
      updateViews(self._viewtree,[(pth,pth)])
      self.modified()

    def tdCom(ntype,node,pth):
      print 'Change data type to :',ntype
      updateViews(self._viewtree,[(pth,pth)])
      self.modified()

    def updateRetypeCascade(node,path):
      if (self.RetypeCascade.maxIndex):
        self.RetypeCascade.delete(0,self.RetypeCascade.maxIndex)
      pnode=self.findParentNodeFromPath(path)
      tlist=s7parser.getNodeAllowedChildrenTypes(pnode,node)
      try:
        tlist.remove(node[3])
        tlist=[node[3]]+tlist
      except ValueError:
        pass
      self.RetypeCascade.maxIndex=len(tlist)
      for t in tlist:
        self.RetypeCascade.add_command(label=t,font=G___.font['E'],
                                       command=lambda n=node,t=t,p=path:\
                                       tcCom(t,n,p))

    def updateDataRetypeCascade(node,path):
      if (self.DataRetypeCascade.maxIndex):
        self.DataRetypeCascade.delete(0,self.DataRetypeCascade.maxIndex)
      pnode=self.findParentNodeFromPath(path)
      tlist=s7parser.getNodeAllowedDataTypes(node)
      self.DataRetypeCascade.maxIndex=len(tlist)
      for t in tlist:
        self.DataRetypeCascade.add_command(label=t,font=G___.font['E'],
                                           command=lambda n=node,t=t,p=path:\
                                           tdCom(t,n,p))

    self.RetypeCascade=Menu(self._wtop,tearoff=0)
    self.RetypeCascade.maxIndex=0

    self.DataRetypeCascade=Menu(self._wtop,tearoff=0)
    self.DataRetypeCascade.maxIndex=0

    self.popupmenu = Menu(self._wtop,tearoff=0)
    self.popupmenu.add_command(font=self.menufont,
                               label="Node menu",
                               state=DISABLED)
    self.popupmenu.add_separator()
    self.popupmenu.add_command(font=self.menufont,
                               label="Change name (C-a)",
                               command=self.SubRename)
    self.popupmenu.add_cascade(font=self.menufont,
                               label="Change CGNS type ",
                               menu=self.RetypeCascade)
    self.popupmenu.add_command(font=self.menufont,
                               label="Change CGNS type (C-s)",
                               command=self.SubRetype)
    self.popupmenu.add_cascade(font=self.menufont,
                               label="Change data type",
                               menu=self.DataRetypeCascade)
    self.popupmenu.add_command(font=self.menufont,
                               label="Change value (C-e)",
                               command=self.SubEdit)
    self.popupmenu.add_separator()
    self.popupmenu.add_command(font=self.menufont,
                               label="Add link (C-l)",
                               command=self.SubAddLink)
    self.popupmenu.add_command(font=self.menufont,
                               label="Remove link",
                               command=self.SubDelLink)
    self.popupmenu.add_separator()
    self.popupmenu.add_command(font=self.menufont,
                               label="Copy (C-c)",
                               command=self.SubCopy)
    self.popupmenu.add_command(font=self.menufont,
                               label="Cut (C-x)",
                               command=self.SubCut)
    self.popupmenu.add_command(font=self.menufont,
                               label="Paste as brother (C-v)",    
                               command=self.SubPasteBrother)
    self.popupmenu.add_command(font=self.menufont,
                               label="Paste as child (C-y)",
                               command=self.SubPasteChild)
    self.popupmenu.add_separator()    

    if (not self._subview):
      self.popupmenu.add_command(font=self.menufont,
                                 label="Open tree view (C-w)",   
                                 command=self.SubView)
    self.popupmenu.add_command(font=self.menufont,
                               label="Open table view (C-t)",
                               command=self.SubTable)
    self.popupmenu.add_separator()
    self.popupmenu.add_command(font=self.menufont,
                               label="Add brother (C-b)",          
                               command=self.SubAddPatternBrother)
    self.popupmenu.add_command(font=self.menufont,
                               label="Add child (C-u)",          
                               command=self.SubAddPatternChild)
    self.popupmenu.add_separator()
    self.popupmenu.add_command(font=self.menufont,
                               label="Check (C-z)",
                               command=self.SubCheck)
    self.popupmenu.bind('<FocusOut>',popUpMenuOff)

    self._tree.grid()
    
    # create a column and define it as the widget's treecolumn
    self.col_graph=self._tree.column_create(font=self.titlefont,
                                            text="Graph",expand=1)
    self.col_type=self._tree.column_create(font=self.titlefont,
                                           text="Type",expand=1)

    self.col_status=self._tree.column_create(font=self.titlefont,
                                             text="M",expand=0,width=20)
    self.col_optionsids=self._tree.column_create(font=self.titlefont,
                                                 text="R",expand=0,width=20)
    self.col_linkstatus=self._tree.column_create(font=self.titlefont,
                                                 text="L",expand=0,width=20)
    self.col_check=self._tree.column_create(font=self.titlefont,
                                            text="C",expand=0,width=20)
    self.col_ddata=self._tree.column_create(font=self.titlefont,
                                            text="Shape",expand=1,width=90)
    self.col_tdata=self._tree.column_create(font=self.titlefont,
                                            text="Data",expand=1)
    self.col_xdata=self._tree.column_create(font=self.titlefont,
                                            text="T",expand=1,width=20)

    self.el_t1=it.element_create(type=IMAGE,image=(ik['node-sids-closed'],OPEN,
                                                   ik['node-sids-opened'],''))
    self.el_t3=it.element_create(type=IMAGE,image=ik['data-array-large'])
    self.el_t5=it.element_create(type=IMAGE,image=ik['node-sids-leaf'])
    self.el_t2=it.element_create(type=IMAGE,image=ik['mark-node'])
    self.el_t6=it.element_create(type=IMAGE,image=(\
      ik['subtree-sids-failed'], 'checkfail',\
      ik['subtree-sids-ok'],     'checkgood',\
      ik['subtree-sids-warning'],'checkwarn'))
    self.el_t7=it.element_create(type=IMAGE,image=(\
      ik['link-node'],  'islink',\
      ik['link-ignore'],'islinkignored',\
      ik['link-error'], 'islinkbroken'))

    self.el_option={}
    self.el_option[G___.SIDSmandatory]=\
         it.element_create(type=IMAGE,image=ik['mandatory-sids-node'])

    self._tree.column_move(self.col_linkstatus,self.col_tdata)
    self._tree.column_move(self.col_optionsids,self.col_tdata)
    self._tree.column_move(self.col_type,self.col_tdata)
    self._tree.column_move(self.col_ddata,self.col_tdata)
    self._tree.column_move(self.col_xdata,self.col_tdata)
    self._tree.column_move(self.col_check,self.col_tdata)
    self._tree.column_move(self.col_status,self.col_tdata)

    self._tree.column_configure(self.col_graph,lock=LEFT)

    if (G___.showSIDS):
      self._tree.column_configure(self.col_optionsids,visible=1)
    else:
      self._tree.column_configure(self.col_optionsids,visible=0)      
    self._tree.column_configure(self.col_check,     visible=1)
    
    self._tree.configure(treecolumn=self.col_graph,
                          background='white',
                          showheader=G___.showColumnTitle)

    # --- text 
    self.el_font1=G___.font['Ta']
    self.el_text1=self._tree.element_create(type=TEXT,
                                            fill=(G___.color_Ta, SELECTED))
    self._tree.element_configure(self.el_text1,font=self.el_font1)

    self.el_font4=G___.font['Tb']
    self.el_text4=self._tree.element_create(type=TEXT,
                                            fill=(G___.color_Tb, SELECTED))
    self._tree.element_configure(self.el_text4,font=self.el_font4)

    self.el_font2=G___.font['Tc']
    self.el_text2=self._tree.element_create(type=TEXT)    
    self._tree.element_configure(self.el_text2,font=self.el_font2)

    self.el_font3=G___.font['Td']
    self.el_text3=self._tree.element_create(type=TEXT)    
    self._tree.element_configure(self.el_text3,font=self.el_font3)

    self.el_font5=G___.font['Tf']
    self.el_text5=self._tree.element_create(type=TEXT)  
    self._tree.element_configure(self.el_text5,font=self.el_font5)

    self.el_text6=self._tree.element_create(type=TEXT,
                                            fill=(G___.color_Tm, "!fortran"))  
    self._tree.element_configure(self.el_text6,font=self.el_font5)

    self.el_w1=self._tree.element_create(type=WINDOW)

    self.el_select1=self._tree.element_create(type=RECT,
                                              fill=(G___.color_Tc, SELECTED),
                                              width=400)
    # Styles
    self.st_folder=self._tree.style_create()
    self._tree.style_elements(self.st_folder, self.el_t1,
                               self.el_text4, self.el_text1)    
    self._tree.style_layout(self.st_folder, self.el_text1, pady=2, padx=0)
    self._tree.style_layout(self.st_folder, self.el_text4, pady=2, padx=0)
    self._tree.style_layout(self.st_folder, self.el_t1, pady=2)

    self.st_leaf=self._tree.style_create()
    self._tree.style_elements(self.st_leaf, self.el_t5,
                               self.el_text4, self.el_text1)    
    self._tree.style_layout(self.st_leaf, self.el_text1, pady=2, padx=0)
    self._tree.style_layout(self.st_leaf, self.el_text4, pady=2, padx=0)
    self._tree.style_layout(self.st_leaf, self.el_t5, pady=2)

    self.st_typesids=self._tree.style_create()
    self._tree.style_elements(self.st_typesids, self.el_select1, self.el_text2)
    self._tree.style_layout(self.st_typesids, self.el_text2, pady=2)
    self._tree.style_layout(self.st_typesids, self.el_select1,
                             union=(self.el_text2,), ipadx=2, iexpand=NS)

    self.st_statusicon=self._tree.style_create()
    self._tree.style_elements(self.st_statusicon, self.el_select1, self.el_t2)
    self._tree.style_layout(self.st_statusicon, self.el_t2, pady=2)
    self._tree.style_layout(self.st_statusicon, self.el_select1,
                             union=(self.el_t2,), ipadx=2, iexpand=NS)

    self.st_checkicon=self._tree.style_create()
    self._tree.style_elements(self.st_checkicon, self.el_select1,self.el_t6)
    self._tree.style_layout(self.st_checkicon, self.el_t6, pady=2)
    self._tree.style_layout(self.st_checkicon, self.el_select1,
                            union=(self.el_t6),
                            ipadx=2,iexpand=NS)

    self.st_link=self._tree.style_create()
    self._tree.style_elements(self.st_link, self.el_select1, self.el_t7)
    self._tree.style_layout(self.st_link, self.el_t7, pady=2)
    self._tree.style_layout(self.st_link, self.el_select1,
                            union=(self.el_t7),
                            ipadx=2,iexpand=NS)

    self.st_optionsids=self._tree.style_create()
    self._tree.style_elements(self.st_optionsids,
                               self.el_select1,
                               self.el_option[G___.SIDSmandatory])
#                               self.el_option[G___.USERmandatory],
#                               self.el_option[G___.SIDSoptional])
    self._tree.style_layout(self.st_optionsids,
                             self.el_option[G___.SIDSmandatory], pady=2)
#    self._tree.style_layout(self.st_optionsids,
#                             self.el_option[G___.SIDSoptional], pady=2)
#    self._tree.style_layout(self.st_optionsids,
#                             self.el_option[G___.USERmandatory], pady=2)
    self._tree.style_layout(self.st_optionsids, self.el_select1,
                             union=(self.el_option[G___.SIDSmandatory]),
#                                    self.el_option[G___.USERmandatory],
#                                    self.el_option[G___.SIDSoptional]),
                             ipadx=2, iexpand=NS)

    self.st_dataentry=self._tree.style_create()
    self._tree.style_elements(self.st_dataentry,
                               self.el_select1, self.el_text3, self.el_t3)
    self._tree.style_layout(self.st_dataentry,self.el_text3,pady=2)
    self._tree.style_layout(self.st_dataentry,self.el_t3,pady=2)
    self._tree.style_layout(self.st_dataentry,self.el_select1,
                             union=(self.el_t3,self.el_text3),
                             ipadx=2,iexpand=NS)

    self.st_dataedit=self._tree.style_create()
    self._tree.style_elements(self.st_dataedit,self.el_w1)

    self.st_datatype=self._tree.style_create()
    self._tree.style_elements(self.st_datatype,self.el_select1,self.el_text5)
    self._tree.style_layout(self.st_datatype,self.el_text5,pady=2)
    self._tree.style_layout(self.st_datatype,self.el_select1,
                             union=(self.el_text5,),ipadx=2,iexpand=NS)

    self.st_datadims=self._tree.style_create()
    self._tree.style_elements(self.st_datadims,self.el_select1,self.el_text6)
    self._tree.style_layout(self.st_datadims,self.el_text6,pady=2)
    self._tree.style_layout(self.st_datadims,self.el_select1,
                             union=(self.el_text6,),ipadx=2,iexpand=NS)

    self._tree.set_sensitive( (self.col_graph,self.st_folder,
                                self.el_text1,self.el_text4,self.el_t1),
                               (self.col_type,self.st_typesids,
                                self.el_text2),
                               (self.col_graph,self.st_leaf,self.el_t5,
                                self.el_text1,self.el_text4),
                               (self.col_tdata,self.st_dataentry,
                                self.el_text3,self.el_t3),
                               (self.col_xdata,self.st_datatype,
                                self.el_text5))

    def link_node(event):
      if (not self._control.linkWindow):
        self._control.linkWindow=self.linkView()

    def copy_node(event):
      self.SubCopy()

    def add_node(event):
      self.SubAddPatternBrother()

    def add_node_child(event):
      self.SubAddPatternChild()

    def paste_node(event):
      self.SubPasteBrother()

    def paste_node_child(event):
      self.SubPasteChild()

    def cut_node(event):
      self.SubCut()

    def local_menu(event):
      tktree=event.widget
      id=tktree.identify(event.x,event.y)
      if (len(id) < 4): return
      item=tktree.item_id(id[1])
      self._tree.selection_clear()
      self._tree.selection_add(item)
      path=self._paths[str(item[0])]
      node=self.findNodeFromPath(path)
      self._edit.set(path)
      col=int(id[3])
      if (col==self.col_graph): self.SubRename()
      if (col==self.col_type):
        updateRetypeCascade(node,path)
        self.RetypeCascade.tk_popup(event.x_root,event.y_root,0)
        self.RetypeCascade.grab_release()
      if (col==self.col_xdata):
        updateDataRetypeCascade(node,path)
        self.DataRetypeCascade.tk_popup(event.x_root,event.y_root,0)
        self.DataRetypeCascade.grab_release()
      if (col==self.col_linkstatus):
        print 'Remove/Add link'
          
    def mark_or_edit_node(event):
      tktree=event.widget
      id=tktree.identify(event.x,event.y)
      if (len(id) < 4): return
      if (id[3]=='6'):  edit_node(event) # ### WATCH OUT ! HARD CODED COL ID
      else:             mark_node(event)
      
    def open_table(event):
      self.SubTable()
      
    def open_view(event):
      self.SubView()
      
    def edit_node(event):
      self.SubEdit()
      
    def check_node(event):
      self.SubCheck()
      
    def retype_node(event):
      self.SubRetype()

    def rename_node(event):
      self.SubRename()
      
    def mark_node(event):
      tktree=event.widget
      selection=tktree.selection_get()
      item=tktree.item_id(selection)
      if (tktree.itemstate_get(item,'marked')):
        tktree.itemstate_set(item[0],'!marked')
      else:
        tktree.itemstate_set(item[0],'marked')
      path=self._paths[str(item[0])]
      self._edit.set(path)

    def open_data(event):
      path=self.getSinglePathSelection()
      self._edit.set(path)

    def open_menu(event):
      tktree=event.widget
      id=tktree.identify(event.x,event.y)
      if ((id==None)or (len(id)<1)): return
      item=tktree.item_id(id[1])
      self._tree.selection_clear()
      self._tree.selection_add(item)
      pth=self.getSinglePathSelection()
      node=self.findNodeFromPath(pth)
      if (self._parent):
        pth=string.join(self._parent.split('/')[:-1],'/')+'/'+pth
      last=os.path.basename(pth)
      updateRetypeCascade(node,pth)
      updateDataRetypeCascade(node,pth)
      self.popupmenu.entryconfigure(0,label=last)
      popUpMenuOn(event,self._viewtree,pth)

    def open_node(event):
      tktree=event.widget
      item=event.item
      self.open_call(tktree,item)
      tktree.itemstate_set(item,'!checkgood')
      tktree.itemstate_set(item,'!checkwarn')
      tktree.itemstate_set(item,'!checkfail')

    # bind the callback to <Expand-before> events
    self._tree.bind('<Button-2>', open_table)
    self._tree.bind('<Shift-Button-1>', local_menu)
    self._tree.bind('<Double-Button-1>', mark_or_edit_node)
    self._tree.bind('<Delete>', cut_node)
    self._tree.bind('<space>', mark_node)
    self._tree.bind('<Button-3>', open_menu)
    self._tree.bind('<Control-a>',rename_node)
    self._tree.bind('<Control-s>',retype_node)    
    self._tree.bind('<Control-e>',edit_node)
    self._tree.bind('<Control-t>',open_table)    
    self._tree.bind('<Control-c>',copy_node)
    self._tree.bind('<Control-v>',paste_node)
    self._tree.bind('<Control-y>',paste_node_child)
    self._tree.bind('<Control-b>',add_node)
    self._tree.bind('<Control-u>',add_node_child)
    self._tree.bind('<Control-x>',cut_node)
    self._tree.bind('<Control-z>',check_node)    
    self._tree.bind('<Control-w>',open_view)    
    #self._tree.bind('l',self.linkView)

    self._tree.bind('<BackSpace>',self._control.do_focus_back)    
    self._tree.notify_bind('<Expand-before>',open_node)

#    print self._tree.bind()
    
    # set up the root item (base)
    self._tree.itemstyle_set(ROOT, self.col_graph, self.st_folder)
    self._paths[str(ROOT)]=rootnodename
    self._paths['0']=rootnodename
    self._tree.itemelement_config(ROOT,self.col_graph,self.el_text1,
                                   text=rootnodename,datatype=STRING)
    self._tree.item_config(ROOT, button=0)
    self._tree.notify_generate('<Expand-before>', item=ROOT)
    self._tree.see(ROOT)
    self.grid(row=1,sticky=N+E+W+S)
    t=self.winfo_toplevel() 
    t.rowconfigure(1,weight=1)
    t.rowconfigure(2,weight=0)
    t.columnconfigure(0,weight=1)

  def findNodeFromPath(self,path):
    if (path == '/'):
      node=self.CGNStarget
    elif self._parent:
      node=s7parser.getNodeFromPath(path.split('/'),
                                   [None,None,[self.CGNStarget],None])  
    else:            
      node=s7parser.getNodeFromPath(path.split('/')[1:],self.CGNStarget)
    return node    
    
  def findParentPathFromPath(self,path,absolute=0):
    pnode=self.findParentNodeFromPath(path,absolute)
    return s7parser.getPathFromNode(pnode,self.CGNStarget)
  
  def findParentNodeFromNode(self,node):
    path=s7parser.getPathFromNode(node,self.CGNStarget)
    return self.findParentNodeFromPath(path)
  
  def findParentNodeFromPath(self,path,absolute=0):
    if (not absolute and self._parent):
      path=string.join(self._parent.split('/')[:-1],'/')+'/'+path
    if (len(path.split('/'))>2):
      return s7parser.getNodeFromPath(path.split('/')[1:-1],self._viewtree.tree)
    else:
      # root node is parent
      return self._viewtree.tree
    
  def SubView(self):
    if (not self.getItemSelection()[0]): return
    tktree=self._tree
    selection=tktree.selection_get()
    item=tktree.item_id(selection)
    pth=self._paths[str(item[0])]
    if (tktree.itemstate_get(item,'leaf')): return
    s7viewControl.createDataTreeWindow(self._control,
                                       self._viewtree,
                                       G___.expandRecurse,
                                       startnode=pth,
                                       parent=self._parent)
      
  def SubAddPatternBrother(self):
    path=self.getSinglePathSelection()
    if (path == '/'):
      s7utils.pasteBrotherError()
      return
    node=self.findParentNodeFromPath(path)
    emptynode=['new node',None,[],'UserDefinedData_t']
    self.addToParentNode(node,emptynode)
    self.modified()

  def SubAddPatternChild(self):
    path=self.getSinglePathSelection()
    if (path == '/'): node=self.CGNStarget
    elif self._parent:
      node=s7parser.getNodeFromPath(path.split('/'),
                                   [None,None,[self.CGNStarget],None])  
    else:            
      node=s7parser.getNodeFromPath(path.split('/')[1:],self.CGNStarget)
    emptynode=['new node',None,[],'UserDefinedData_t']
    self.addToParentNode(node,emptynode)
    self.modified()
        
  def SubCheck(self):
    currentpath=self.getSinglePathSelection()
    if (self._parent):
      currentpath=string.join(self._parent.split('/')[:-1],'/')+'/'+currentpath
    if (not self._control.logWindow):
      self._control.logWindow=s7log.wLog(self,self._control)
    else:
      self._control.logWindow._wtop.deiconify()
      if (not self._control.logWindow.isclear): return
    self._control.logWindow.push("Checking now... %s\n"%currentpath)
    rcheck=[]
    ptlist=[]
    if (currentpath != '/'):
      ptlist=[self.findNodeFromPath(currentpath)]
      cpath=''
    else:
      ptlist=self.findNodeFromPath(currentpath)[2]
      cpath='/'
    for nn in ptlist:
      rcheck+=s7check.checkTree(cpath+nn[0],nn,
                                self._parent,self.CGNStarget,
                                1,self._control.logWindow)
    if (self._parent=='/'):
      for c in rcheck:
        c[0]='/%s'%(c[0])
    else:
      if (currentpath!='/'):
        for c in rcheck:
          c[0]='%s/%s'%(string.join(currentpath.split('/')[:-1],'/'),c[0])
    updateViews(self._viewtree,checklist=rcheck)
        
  def SubCopy(self):
    if (not self.getItemSelection()[0]): return
    path=self.getSinglePathSelection()
    if (path == '/'):
      return
    elif self._parent:
      node=s7parser.getNodeFromPath(path.split('/'),
                                   [None,None,[self.CGNStarget],None])  
    else:            
      node=s7parser.getNodeFromPath(path.split('/')[1:],
                                   self.CGNStarget)
    wTreeSimple._nodebuffer=node    
        
  def SubAddLink(self):
    if (not self.getItemSelection()[0]): return
    currentpath=self.getSinglePathSelection()
    node=self.findNodeFromPath(currentpath)
    ppath=self.findParentPathFromPath(currentpath,absolute=1)
    s7linkView.wLinkEdit(self._control,self._tree,self._viewtree,node,ppath)
    
  def SubDelLink(self):
    if (not self.getItemSelection()[0]): return
    currentpath=self.getSinglePathSelection()
    node=self.findNodeFromPath(currentpath)
    if (self._parent):
      currentpath=string.join(self._parent.split('/')[:-1],'/')+'/'+currentpath
    s7linkView.wLinkEdit(self._control,self._tree,self._viewtree,
                         node,currentpath)

  def SubRename(self):
    if (not self.getItemSelection()[0]): return
    currentpath=self.getSinglePathSelection()
    node=self.findNodeFromPath(currentpath)
    if (self._parent):
      currentpath=string.join(self._parent.split('/')[:-1],'/')+'/'+currentpath
    self._edit.setStoreValue('__Rename',node,self,currentpath)
    self._edit.show('__Rename',grab=1)
        
  def SubRetype(self):
    if (not self.getItemSelection()[0]): return
    currentpath=self.getSinglePathSelection()
    node=self.findNodeFromPath(currentpath)
    if (self._parent):
      currentpath=string.join(self._parent.split('/')[:-1],'/')+'/'+currentpath
    self._edit.setStoreValue('__Retype',node,self,currentpath)
    self._edit.show('__Retype',grab=1)
        
  def SubTable(self):
    if (not self.getItemSelection()[0]): return
    path=self.getSinglePathSelection()
    if (self._parent):
        path=string.join(self._parent.split('/')[:-1],'/')+'/'+path
    s7viewControl.createDataTableWindow(self._control,self._viewtree,path)

  def SubEdit(self):
    if (self._islocked): return
    item=self.getItemSelection()
    if (not item[0]): return
    currentpath=self.getSinglePathSelection()
    node=self.findNodeFromPath(currentpath)
    if (not s7utils.canBeShown(node[1])): return
    if (self._parent):
      currentpath=string.join(self._parent.split('/')[:-1],'/')+'/'+currentpath
    vlist=s7check.getEnumerateList(node)
    if (vlist):
      wnew=self.wEditEnumerate(item,node,self,currentpath,vlist)
    else:
      wnew=self.wEditEntry(item,node,self,currentpath)
    self._islocked=1
        
  def SubCut(self):
    if (not self.getItemSelection()[0]): return
    path=self.getSinglePathSelection()
    if (    (path == '/')
         or ((path[0] != '/') and (len(path.split('/'))==1))):
      s7utils.cutError()
      return
    if self._parent:
      path=string.join(self._parent.split('/')[:-1],'/')+'/'+path
    s7parser.removeNodeFromPath(path.split('/')[1:],self._viewtree.tree)
    updateViews(self._viewtree)
    self.modified()

  def addToParentNode(self,parentnode,newnode):
    if (newnode[0] in s7parser.childNames(parentnode)):
      if (G___.generateCopyNames):
        nm=s7utils.generateName(parentnode,newnode)
        nnode=[nm,copy.copy(newnode[1]),copy.deepcopy(newnode[2]),newnode[3]]
      else:
        s7utils.copyNameCollision(newnode[0])
        return
    else:
      nnode=copy.deepcopy(newnode)
    parentnode[2].append(nnode)
    updateViews(self._viewtree)
    
  def SubPasteBrother(self):
    if (wTreeSimple._nodebuffer == None): return
    path=self.getSinglePathSelection()
    if (path == '/'):
      s7utils.pasteBrotherError()
      return
    node=self.findParentNodeFromPath(path)
    self.addToParentNode(node,wTreeSimple._nodebuffer)
    self.modified()
        
  def SubPasteChild(self):
    if (wTreeSimple._nodebuffer == None): return
    currentpath=self.getSinglePathSelection()
    if (currentpath == '/'): node=self.CGNStarget
    elif self._parent:
      node=s7parser.getNodeFromPath(currentpath.split('/'),
                                   [None,None,[self.CGNStarget],None])  
    else:            
      node=s7parser.getNodeFromPath(currentpath.split('/')[1:],self.CGNStarget)
    self.addToParentNode(node,wTreeSimple._nodebuffer)
    self.modified()
  def updateMarkedFromList(self,sl):
    for k in self._paths:
      if (self._paths[k] in sl): self._tree.itemstate_set(k,'marked')
      else:                      self._tree.itemstate_set(k,'!marked')
  def getFirstMarkedFrom(self,back=0):
    sitem=self.getItemSelection()[0]
    self._tree.selection_modify(select=ALL)
    r=[]
    for n in self._tree.selection_get():
      item=self._tree.item_id(n)
      if (self._tree.itemstate_get(item[0],'marked')):r+=[n]
    self._tree.selection_modify(deselect=ALL)
    ord=self.getOrderedListOfItem()
    if (not r): pass
    elif (sitem not in r): sitem=r[0]
    else:
      rc=[]
      for rd in ord:
        if (rd[1] in r): rc+=[rd[1]]
      i=rc.index(sitem)
      if (back):
        i-=1
        if (i<0): i=len(rc)-1
      else:
        i+=1
        if (i==len(rc)): i=0
      sitem=rc[i]
      #print rc, i, rc[i], self._paths[str(rc[i])]
    return sitem
  def getItemSelection(self):
    tktree=self._tree
    selection=tktree.selection_get()
    if (selection == None): selection=ROOT
    item=tktree.item_id(selection)
    return item
  def getSinglePathSelection(self):
    item=self.getItemSelection()
    path=self._paths[str(item[0])]
    return path
  def getSinglePathSelection(self):
    item=self.getItemSelection()
    path=self._paths[str(item[0])]
    return path

  # --------------------------------------------------------------------
  def getOrderedListOfItem(self):
    r=[]
    for k in self._paths:
      if (k!='root'): r+=[(self._paths[k],int(k))]
    r.sort()
    return r
  
  # --------------------------------------------------------------------
  def clearViewChecks(self):
    tktree=self._tree
    for it in tktree.item_id(ALL):
      tktree.itemstate_set(it,'!checkwarn')
      tktree.itemstate_set(it,'!checkgood')
      tktree.itemstate_set(it,'!checkfail')    

  # --------------------------------------------------------------------
  def updateViewAfterRenameOrChangeValue(self,namelist):
    tktree=self._tree
    for it in tktree.item_id(ALL):
      path=self._paths[str(it)]
      if (path == '/'):
        node=self.CGNStarget
      elif self._parent:
        node=s7parser.getNodeFromPath(path.split('/'),
                                     [None,None,[self.CGNStarget],None])
      else:            
        node=s7parser.getNodeFromPath(path.split('/')[1:],self.CGNStarget)
      if (node == -1):
        for nl in namelist:
          opath=nl[0]
          npath=nl[1]
          if (self._parent):
            if (self._parent=='/'):
              opath=nl[0][1:]
              npath=nl[1][1:]
            elif (    (self._parent[0]=='/')
                  and (len(self._parent.split('/'))==2)):
              opath=nl[0][1:]
              npath=nl[1][1:]
            else:
              parentbase=string.join(self._parent.split('/')[:-1],'/')
              opath=string.split(nl[0],parentbase)[1][1:]
              npath=string.split(nl[1],parentbase)[1][1:] 
          if (opath == path):
            ppath=npath
            if (ppath == '/'):
              pnode=self.CGNStarget
            elif self._parent:
              pnode=s7parser.getNodeFromPath(ppath.split('/'),
                                           [None,None,[self.CGNStarget],None])
            else:            
              pnode=s7parser.getNodeFromPath(ppath.split('/')[1:],
                                            self.CGNStarget)
            imode=s7check.getStatusForThisNode(ppath,pnode,self._parent,
                                                self.CGNStarget)
            if (imode[0]):
              tktree.itemelement_config(it,self.col_graph,self.el_text1,
                                        text=pnode[0],
                                        datatype=STRING, draw=True)
              tktree.itemelement_config(it,self.col_graph,self.el_text4,
                                        text="", datatype=STRING,
                                        draw=False)
            else:
              tktree.itemelement_config(it,self.col_graph,self.el_text1,
                                        text="", datatype=STRING,
                                        draw=False)
              tktree.itemelement_config(it,self.col_graph,self.el_text4,
                                        text=""+pnode[0],
                                        datatype=STRING, draw=True)
            if (it):
              tktree.itemelement_config(it, self.col_type, self.el_text2,
                                        text=pnode[3], datatype=STRING)      
              tktree.itemelement_config(it, self.col_xdata, self.el_text5,
                                        text=s7parser.getNodeType(pnode),
                                        datatype=STRING)
            # update all children paths
            # subviews paths are relatives... more difficult
            self._paths[str(it)]=ppath
            clist=tktree.item_children(it)
            bpath=ppath.split('/')
            lpath=len(bpath)
            if (not self._parent): # absolute
              for itc in clist:
                opath=self._paths[str(itc)].split('/')[lpath:]
                npath=string.join(bpath+opath,'/')
                self._paths[str(itc)]=npath
      else:
        for nl in namelist:
          opath=nl[0]
          if (self._parent):
            # change paths to relatives, add base
            if (self._parent=='/'):
              opath=nl[0][1:]
            elif (    (self._parent[0]=='/')
                  and (len(self._parent.split('/'))==2)):
              opath=nl[0][1:]
            else:
              parentbase=string.join(self._parent.split('/')[:-1],'/')
              opath=string.split(nl[0],parentbase)[1][1:]
          if (it and (opath == path)):
            if (s7utils.canBeShown(node[1])) :
              tktree.itemelement_config(it,self.col_tdata,self.el_text3,
                                        text=s7utils.toBeShown(node[1]),
                                        datatype=STRING)
              tktree.itemelement_config(it,self.col_tdata,self.el_t3,
                                        draw=False)
            else :
              tktree.itemelement_config(it,self.col_tdata,self.el_text3,draw=False)
              tktree.itemelement_config(it,self.col_tdata,self.el_t3,draw=True)
            
            tktree.itemelement_config(it, self.col_type, self.el_text2,
                                      text=node[3], datatype=STRING)      
            tktree.itemelement_config(it, self.col_xdata, self.el_text5,
                                      text=s7parser.getNodeType(node),
                                      datatype=STRING)
    
  # --------------------------------------------------------------------
  def updateViewAfterCheck(self,checklist):
    pdict={}
    for c in checklist:
      pdict[c[0]]=c[1]
    tktree=self._tree
    for it in tktree.item_id(ALL):
      path=self._paths[str(it)]
      if (path[0] != '/'):
        if (self._parent=='/'):
          path="/%s"%(path)
        else:
          parentbase=string.join(self._parent.split('/')[:-1],'/')
          path="%s/%s"%(parentbase,path)
      try:
        if (pdict[path] == 0): tktree.itemstate_set(it,'checkgood')
        if (pdict[path] == 1): tktree.itemstate_set(it,'checkwarn')
        if (pdict[path] == 2): tktree.itemstate_set(it,'checkfail')
      except KeyError:
        pass
  
  # --------------------------------------------------------------------
  def updateViewLinks(self):
    tktree=self._tree
    for it in tktree.item_id(ALL):
      path=self._paths[str(it)]
      if (path == '/'):
        node=self.CGNStarget
      elif self._parent:
        node=s7parser.getNodeFromPath(path.split('/'),
                                     [None,None,[self.CGNStarget],None])
      else:            
        node=s7parser.getNodeFromPath(path.split('/')[1:],self.CGNStarget)
      if (node == -1):
          lmode=s7linkView.getLinkStatusForThisNode(path,node,
                                                    self._parent,
                                                    self.CGNStarget,
                                                    self._viewtree)
          if (lmode==1): tktree.itemstate_set(it,'islink')
          if (lmode==2): tktree.itemstate_set(it,'islinkignored')
          if (lmode==3): tktree.itemstate_set(it,'islinkbroken')
  
  # --------------------------------------------------------------------
  def updateViewAfterCutOrAdd(self):
    tktree=self._tree
    for it in tktree.item_id(ALL):
      path=self._paths[str(it)]
      if (path == '/'):
        node=self.CGNStarget
      elif self._parent:
        node=s7parser.getNodeFromPath(path.split('/'),
                                     [None,None,[self.CGNStarget],None])
      else:            
        node=s7parser.getNodeFromPath(path.split('/')[1:],self.CGNStarget)
      if (node != -1):
        for cn in node[2]:
          found=0
          clist=tktree.item_children(it) 
          if (len(clist)):
            for itc in clist:
              if (path == '/'):
                if (self._paths[str(itc)]=="/%s"%(cn[0])):
                  found=1
                  break
              else:
                if (self._paths[str(itc)]=="%s/%s"%(path,cn[0])):
                  found=1
                  break
          # current node has new children
          if ((not found) and (len(clist))):
            if (path != '/'): pth="%s/%s"%(path,cn[0])
            else:             pth="/%s"%(cn[0])
            tktree.item_configure(it,button=True)
            self.getSIDSelement(tktree,pth,cn,it)
          # special case for empty root node
          elif ((not found) and (path=='/')):
            pth="/%s"%(cn[0])
            tktree.item_configure(it,button=True)
            self.getSIDSelement(tktree,pth,cn,it)
          # new child on leaf
          elif (not found):
            if (path != '/'): pth="%s/%s"%(path,cn[0])
            else:             pth="/%s"%(cn[0])
            tktree.item_configure(it,button=True)
            self.getSIDSelement(tktree,pth,cn,it)
      else:
        # current node is no more there
        if (tktree.item_id(it)):
          p=tktree.item_ancestors(it)[0]
          tktree.item_delete(it)
          if (not tktree.item_children(p)):
            tktree.item_configure(p,button=False)
            
  def onexit(self):
    self._control.delTreeView(self._viewid,
                              self._viewtree.filedir,
                              self._viewtree.filename)
    self.closeWindow()
    
  def open_call(self,tktree,item):
    if tktree.item_numchildren(item): return
    try:
        path=self._paths[str(item)]
    except KeyError: return
    if (path == '/'):
      targetNode=self.CGNStarget
    elif self._parent:
      targetNode=s7parser.getNodeFromPath(path.split('/'),
                                         [None,None,[self.CGNStarget],None])  
    else:            
      targetNode=s7parser.getNodeFromPath(path.split('/')[1:],self.CGNStarget)
    for nd in targetNode[2]:
      if (path != '/'): pth=path+"/%s"%nd[0]
      else:             pth=path+"%s"%nd[0]
      self.getSIDSelement(tktree,pth,nd,item)

  def operateMenu(self, selecteditems):
    self._control.operateView(self,selecteditems)
    
  def expandAndFocusByPath(self,path):
    for k in self._paths:
      if (self._paths[k]==path):
        self._tree.see(int(k))
        self._tree.selection_modify(deselect=ALL)
        self._tree.selection_modify(select=int(k))
        break

  def expandOneMoreLevel(self):
    if (self.__expandLevel < self.__levelMaxDepth):
      self.__expandLevel+=1
      self.expandOneMoreLevelAux(0,ROOT)

  def expandOneMoreLevelAux(self,count,itemdesc):
    if (count >= self.__expandLevel): return
    self._tree.item_expand(itemdesc)
    count+=1
    for itemchild in self._tree.item_children(itemdesc):
      self.expandOneMoreLevelAux(count,itemchild)  
      
  def collapsedOneMoreLevel(self):
    if (self.__expandLevel > 1):
      self.__expandLevel-=1
      self.collapseOneMoreLevelAux(0,ROOT)

  def collapseOneMoreLevelAux(self,count,itemdesc):
    if (count > self.__expandLevel): return
    count+=1
    for itemchild in self._tree.item_children(itemdesc):
      self.collapseOneMoreLevelAux(count,itemchild)
    if (count > self.__expandLevel):
      self._tree.item_collapse(itemdesc)

  def linkView(self):
    if (not self._haslinkWindow):
      self._haslinkWindow=s7linkView.wLinkView(self,self._viewtree)
    return self._haslinkWindow
    
  def getSIDSelement(self,tktree,pth,node,item):
    enew = tktree.create_item(parent=item, button=1, open=0)[0]

    imode=s7check.getStatusForThisNode(pth,node,self._parent,self.CGNStarget)
    self._paths[str(enew)]=pth
    
    tktree.itemstate_set(enew,'!leaf')

    tktree.itemstyle_set(enew, self.col_check, self.st_checkicon)
    tktree.itemstate_set(enew,'!checkwarn')
    tktree.itemstate_set(enew,'!checkgood')
    tktree.itemstate_set(enew,'!checkfail')

    lmode=s7linkView.getLinkStatusForThisNode(pth,node,
                                              self._parent,self.CGNStarget,
                                              self._viewtree)

    tktree.itemstyle_set(enew, self.col_linkstatus,self.st_link)
    tktree.itemstate_set(enew,'!islink')
    tktree.itemstate_set(enew,'!islinkignored')
    tktree.itemstate_set(enew,'!islinkbroken')    
    if (lmode==1): tktree.itemstate_set(enew,'islink')
    if (lmode==2): tktree.itemstate_set(enew,'islinkignored')
    if (lmode==3): tktree.itemstate_set(enew,'islinkbroken')

    if (node[2] == []):
      tktree.item_configure(enew,button=False)
      tktree.itemstate_set(enew,'leaf')
      tktree.itemstyle_set(enew, self.col_graph, self.st_leaf)
    else:   
      tktree.itemstyle_set(enew, self.col_graph, self.st_folder)

    if (imode[0]):
      tktree.itemelement_config(enew, self.col_graph, self.el_text1,
                                text=node[0], datatype=STRING, draw=True)
      tktree.itemelement_config(enew, self.col_graph, self.el_text4,
                                text="", datatype=STRING, draw=False)
    else:
      tktree.itemelement_config(enew, self.col_graph, self.el_text1,
                                text="", datatype=STRING, draw=False)
      tktree.itemelement_config(enew, self.col_graph, self.el_text4,
                                text=""+node[0], datatype=STRING, draw=True)

    tktree.itemstyle_set(enew, self.col_type, self.st_typesids)
    tktree.itemelement_config(enew, self.col_type, self.el_text2,
                              text=node[3], datatype=STRING)      

    tktree.itemstyle_set(enew, self.col_xdata, self.st_datatype)
    tktree.itemelement_config(enew, self.col_xdata, self.el_text5,
                              text=s7parser.getNodeType(node),datatype=STRING)

    tktree.itemstyle_set(enew, self.col_ddata, self.st_datadims)
    tktree.itemelement_config(enew, self.col_ddata, self.el_text6,
                              text=s7parser.getNodeShape(node),datatype=STRING)
    if (s7parser.hasFortranFlag(node)):
      tktree.itemstate_set(enew,'fortran')
    else:
      tktree.itemstate_set(enew,'!fortran')

    tktree.itemstyle_set(enew, self.col_tdata, self.st_dataentry)
    if (s7utils.canBeShown(node[1])) :
      tktree.itemelement_config(enew,self.col_tdata,self.el_text3,
                                text=s7utils.toBeShown(node[1]),
                                datatype=STRING)
      tktree.itemelement_config(enew,self.col_tdata,self.el_t3,draw=False)
    else :
      tktree.itemelement_config(enew,self.col_tdata,self.el_text3,draw=False)
      tktree.itemelement_config(enew,self.col_tdata,self.el_t3,draw=True)

    tktree.itemstyle_set(enew, self.col_optionsids, self.st_optionsids)
    for okey in self.el_option:
      if (okey == imode[1]):
        tktree.itemelement_config(enew,self.col_optionsids,
#                                  self.el_option[okey],draw=False)
                                  self.el_option[okey],draw=True)
      else:
        tktree.itemelement_config(enew,self.col_optionsids,
                                  self.el_option[okey],draw=False)      

    tktree.itemstyle_set(enew, self.col_status, self.st_statusicon)
    tktree.element_configure(self.el_t2,draw=(0,'!marked'))
    tktree.itemstate_set(enew,'!marked')
    
  # --------------------------------------------------------------------
  class wEditBase(Frame):
    def __init__(self,item,node,master,path,vlist):
      Frame.__init__(self,master._tree)
      self.value=StringVar()
      self.node=node
      self.wtree=master
      self.path=path
      self.item=item
      self.value.set(self.setValue(node))
      self.font=self.wtree.el_font3
      self.vlist=vlist
    def show(self):
      self.wtree._tree.itemstyle_set(self.item,
                                     self.wtree.col_tdata,
                                     self.wtree.st_dataedit)
      self.wtree._tree.itemelement_config(self.item,
                                          self.wtree.col_tdata,
                                          self.wtree.el_w1,window=self)
      ack=self.wtree._edit.show('__Accept')
      ack.setCallBack((self.validate,self.cancel))
    def cancel(self):
      self.wtree._tree.itemstyle_set(self.item,
                                     self.wtree.col_tdata,
                                     self.wtree.st_dataentry)
      self.wtree._edit.hide()
      self.wtree._tree.itemstyle_set(self.item,
                                     self.wtree.col_tdata,
                                     self.wtree.st_dataentry)
      if (s7utils.canBeShown(self.node[1])) :
        self.wtree._tree.itemelement_config(self.item,
                                            self.wtree.col_tdata,
                                            self.wtree.el_text3,
                                            text=s7utils.toBeShown(self.node[1]),
                                            datatype=STRING)
        self.wtree._tree.itemelement_config(self.item,
                                            self.wtree.col_tdata,
                                            self.wtree.el_t3,
                                            draw=False)
      else:
        self.wtree._tree.itemelement_config(self.item,self.col_tdata,
                                            self.el_text3,draw=False)
        self.wtree._tree.itemelement_config(self.item,self.col_tdata,
                                            self.el_t3,draw=True)
      self.wtree._islocked=0
    def update(self):
      self.m.configure(text=self.value.get())
    def setValue(self,node):
      return s7utils.toBeShown(node[1])
    def validate(self):
      pnode=self.wtree.findParentNodeFromPath(self.path,absolute=1)
      self.oldname=self.path
      self.newvalue=self.value.get()
      # Starts with [ or ( it is a tuple of values
      #        with 0-9 it is a number
      #        with something else it is a string
      oldshape=self.node[1].shape
      newarray=self.node[1]
      if (self.newvalue):
        if (self.newvalue[0] in ['[','(']):
          newarray=NY.array(list(eval(self.newvalue)))
        elif (self.newvalue[0] in list('0123456789')):
          try:
            newarray=NY.array([int(self.newvalue)],dtype='i')
          except ValueError:
            try:
              if (self.node[0]!='CGNSLibraryVersion'):
                newarray=NY.array([float(self.newvalue)],
                                  dtype=NY.Float64,order='F')
              else:
                newarray=NY.array([float(self.newvalue)],
                                  dtype=NY.Float32,order='F')
            except ValueError:
              newarray=NY.array(tuple(self.newvalue),dtype='S1',order='F')
        else:    
            newarray=NY.array(tuple(self.newvalue),dtype='S1',order='F')
        if (newarray.shape==oldshape):
          if (G___.transposeOnViewEdit):
            self.node[1]=NY.array(newarray.T,order='F').reshape(oldshape)
          else:
            self.node[1]=NY.array(newarray,order='F').reshape(oldshape)
        else:
          s7utils.shapeChangeError(oldshape)
      self.wtree._sedit.hide()
      self.wtree._tree.itemstyle_set(self.item,
                                     self.wtree.col_tdata,
                                     self.wtree.st_dataentry)
      updateViews(self.wtree._viewtree,[(self.oldname,self.oldname)])
      self.wtree._islocked=0
      self.wtree.modified()

  # --------------------------------------------------------------------
  class wEditEntry(wEditBase):
    def __init__(self,item,node,master,path):
      wTreeSimple.wEditBase.__init__(self,item,node,master,path,[])
      self.e=Entry(self,textvariable=self.value,
                   font=self.font,background='white',
                   width=len(self.value.get()))
      self.e.grid()
      self.grid()
      self.show()

  # --------------------------------------------------------------------
  class wEditEnumerate(wEditBase):
    def __init__(self,item,node,master,path,vlist):
      wTreeSimple.wEditBase.__init__(self,item,node,master,path,vlist)
      self.m=Menubutton(self,textvariable=self.value,font=G___.font['E'],
                        borderwidth=2,indicatoron=0,relief=RAISED,anchor="c",
                        highlightthickness=2)
      self.m.menu=Menu(self.m,tearoff=0)
      self.m['menu']=self.m.menu
      for v in self.vlist:
        self.m.menu.add_radiobutton(label=v,var=self.value,value=v,
                                    indicatoron=0,command=self.update,
                                    font=G___.font['E'])
      self.m.grid()
      self.grid()
      self.show()
      
# --------------------------------------------------------------------
# Propagates to all relevant views
#
def updateViews(viewtree,namelist=[],checklist=[],links=False):
  for v in viewtree.viewlist:
    if   (checklist): v.view.updateViewAfterCheck(checklist)
    elif (namelist) : v.view.updateViewAfterRenameOrChangeValue(namelist)
    else:             v.view.updateViewAfterCutOrAdd()
    if (links):       v.view.updateViewLinks()
    
# --------------------------------------------------------------------
