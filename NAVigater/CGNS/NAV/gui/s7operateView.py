# -----------------------------------------------------------------------------
# pyS7 - CGNS/SIDS editor
# ONERA/DSNA - marc.poinot@onera.fr
# pyS7 - $Rev: 72 $ $Date: 2009-02-10 15:58:15 +0100 (Tue, 10 Feb 2009) $
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source
# tree for license information.
#
import Tkinter
Tkinter.wantobjects=0 #necessary for tk-8.5 and some buggy tkinter installs
from Tkinter    import *
from TkTreectrl import *
import os
import sys
import s7globals
G___=s7globals.s7G

import s7fileDialog
import s7treeFingerPrint
import s7viewControl
import s7utils
import s7windoz
import s7operateView

from CGNS.NAV.supervisor.s7check import s7Query

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class wQueryView(s7windoz.wWindoz,ScrolledTreectrl):
  def getItemSelection(self):
    tktree=self._qtree
    selection=tktree.selection_get()
    if (selection == None): selection=ROOT
    item=tktree.item_id(selection)
    return item
  # --------------------------------------------------------------------
  class qEntry:
    VAR=2
    VAL=4
    OPT=3
    #
    # Q nodes ---
    # [ it, OR,  Q, Q, Q] simple list of ORed  Qs: Q or Q or Q (default)
    # [ it, AND, Q, Q, Q] simple list of ANDed Qs: Q and Q and Q
    # [ it, NOT, Q]       single NOT Q:            not Q
    #
    # Complex query
    # [ it, OR, Q, [ it, NOT, [ it, OR, Q, [ it, AND, Q , Q, Q]]], Q]
    #
    # Q leaf ---
    # (it, var, operator, value)
    #
    def __init__(self):
      self.Q=[ROOT, s7Query.OR]
    def getEntry(self,item):
      if (item in [ROOT,0]): return self.Q
      rtp=self.getEntryAux(item,self.Q,0)
      rte=(rtp[0],self.getList(rtp[1]))
      return rte
    def getEntryAux(self,item,Q,pitem):
      rt=-1
      pt=pitem
      for cq in Q:
        if (type(cq)==type([])):
            if (cq[0]==item):
              rt=cq
            else:
              (rt,pt)=self.getEntryAux(item,cq,cq[0])
            if (rt!=-1): break
        if (type(cq)==type((1,))):
          if (cq[0]==item):
            rt=cq
            break
      return (rt,pt)
    def getList(self,item):
      if (item in [ROOT,0]): return self.Q
      rtp=self.getListAux(item,self.Q,0)
      return rtp[0]
    def getListAux(self,item,Q,pitem):
      rt=-1
      pt=pitem
      for cq in Q:
        if (type(cq)==type([])):
            if (cq[0]==item):
              rt=cq
            else:
              (rt,pt)=self.getListAux(item,cq,cq[0])
            if (rt!=-1): break
      return (rt,pt)
    def cutAux(self,item,Q):
      for n in range(len(Q)):
        if (Q[n][0]==item):
          del Q[n]
          break
    def cut(self,item):
      if (item in [ROOT,0]):
        self.Q=[ROOT, s7Query.OR]
      self.cutAux(item,self.Q)
    def add(self,iparent,ichild,connector):
      if (type(connector)==type((1,))):
        rl=self.getList(iparent)
        if (rl==-1): return 0
        rl.append((ichild,connector[0],connector[1]))
      else:
        rl=self.getList(iparent)
        if (rl==-1): return 0
        rl.append([ichild,connector])
      return 1
    def getAsQueryAux(self,wqry):
      qry=[]
      for wcq in wqry:
        if (type(wcq)==type((1,))):
          qry.append((wcq[1],wcq[2]))
        elif (type(wcq)==type([])):
          r=[wcq[1]]
          r+=self.getAsQueryAux(wcq[2:])
          qry.append(r)
      return qry
    def getAsQuery(self):
      return self.getAsQueryAux([self.Q])[0]

  # --------------------------------------------------------------------
  class wEditBase(Frame):
    def __init__(self,item,master,qentry,ivar):
      def active_validate(event):
        if (self.we==None): return
        self.we.destroy()
        self.validate()
        self.we=None
      def active_cancel(event):
        if (self.we==None): return
        self.we.destroy()
        self.close()
        self.we=None
      Frame.__init__(self,master._qtree)
      self.value=StringVar()
      self.previousfocus=None
      self.nvar=ivar
      self.value.set(self.nvar)
      self.qw=master
      self.item=item
      self.qentry=qentry
      self.bind('<Escape>',active_cancel)
      self.bind('<Return>',active_validate)
      self.active_cancel=active_cancel
      self.active_validate=active_validate
      master._qtree.bind('<Escape>',active_cancel)
      master._qtree.bind('<Return>',active_validate)
    def validate(self):
      if (self.we==None): return
      self.nvar=self.value.get()
      qe=self.qentry[1]
      qv=self.qentry[0]
      idx=qe.index(qv)
      self.setEntry(idx,qe,qv)
      self.close()
  class wEditEnumerate(wEditBase):
    def __init__(self,item,master,vlist,qentry):
      self.master=master
      self.font=G___.font['Tc']
      wQueryView.wEditBase.__init__(self,item,master,qentry,qentry[0][1])
      self.m=Menubutton(self,textvariable=self.value,font=self.font,
                        borderwidth=2,indicatoron=0,relief=RAISED,anchor="c",
                        highlightthickness=2)
      self.we=self.m
      self.m.menu=Menu(self.m,tearoff=0)
      self.m['menu']=self.m.menu
      for v in vlist:
        self.m.menu.add_radiobutton(label=v,var=self.value,value=v,
                                    indicatoron=0,command=self.update,
                                    font=self.font)
      self.qw.active=self
      self.m.grid()
      self.grid()
      self.show()
    def show(self):
      self.qw._qtree.itemstyle_set(self.item,self.qw.cl_var,self.qw.st_varopt)
      self.qw._qtree.itemelement_config(self.item,self.qw.cl_var,
                                        self.qw.el_varopt,window=self)
    def close(self):
      self.qw._qtree.itemstyle_set(self.item,self.qw.cl_var,self.qw.st_var)
      self.qw._qtree.itemelement_config(self.item,self.qw.cl_var,
                                        self.qw.el_vartext,
                                        text=self.nvar,datatype=STRING) 
      self.qw.active=None
    def setEntry(self,idx,qe,qv):
      qe[idx]=(qv[0],self.nvar,qv[2])
  class wEditEntry(wEditBase):
    def __init__(self,item,master,qentry):
      self.master=master
      self.font=G___.font['E']
      wQueryView.wEditBase.__init__(self,item,master,qentry,qentry[0][2])
      self.e=Entry(self,textvariable=self.value,
                   font=self.font,background='white',width=32)
      self.we=self.e
      def local_validate(event):
        self.active_validate(event)
      self.we.bind('<Return>',local_validate)
      def local_cancel(event):
        self.active_cancel(event)
      self.we.bind('<Escape>',local_cancel)
      self.e.grid()
      self.grid()
      self.show()
    def show(self):
      self.qw._qtree.itemstyle_set(self.item,self.qw.cl_val,self.qw.st_valopt)
      self.qw._qtree.itemelement_config(self.item,self.qw.cl_val,
                                        self.qw.el_valopt,window=self)
      self.previousfocus=self.master.focus_get()
      self.we.wait_visibility()
      self.we.grab_set()
      self.we.focus_set()            
    def close(self):
      if (self.previousfocus): self.previousfocus.focus_set()
      self.qw._qtree.itemstyle_set(self.item,self.qw.cl_val,self.qw.st_val)
      self.qw._qtree.itemelement_config(self.item,self.qw.cl_val,
                                        self.qw.el_valtext,
                                        text=self.nvar,datatype=STRING) 
      self.qw.active=None
    def setEntry(self,idx,qe,qv):
      qe[idx]=(qv[0],qv[1],self.nvar)
  def cbk_probe(self):
    pass
  def cbk_save(self):
    file=s7fileDialog.s7filedialog(self._wtop,save=1,pat='*.py')
    if (not file): return
    if (os.path.exists(file)):
      bakfile=file+'.bak'
      os.rename(file,bakfile)
    if (file):
      name='Name'
      comment='comment'
      query=self.query.getAsQuery()
      date=s7utils.timeTag()
      q=s7Query(query,name,comment)
      q.save(file,date)
  def cbk_load(self):
    filedir=s7fileDialog.s7filedialog(self._wtop,save=0,pat='*.py')
    if (not filedir): return
    (dir,file)=os.path.split(filedir)
    mod=os.path.splitext(file)[0]
    q=s7Query(None)
    q.load(dir,mod)
    self.updateQueryTree(q.Q)
  def cbk_execute(self):
    tree=self._wtree.CGNStarget[2][1] # first base
    qry=self.query.getAsQuery()
    q=s7Query(qry)
    for s in self.woperate.selectedlist:
      node      =self._wtree.findNodeFromPath(s[1])
      parentnode=self._wtree.findParentNodeFromPath(s[1])
      s[0]=q.evalQuery(tree,node,parentnode)
    self.woperate.updateSelection(None)
  def cbk_edit(self):
    pass
  def open_node(self,event):
    tktree=event.widget
    item=event.item
    self.open_call(tktree,item)
  def add_entry(self,item,qent):    
    enew=self._qtree.create_item(parent=item, button=0, open=0)[0]    
    self._qtree.itemstate_set(enew,'leaf')
    self._qtree.itemstyle_set(enew, self.cl_ctr, self.st_ent)
    self._qtree.itemstyle_set(enew, self.cl_val, self.st_val)
    self._qtree.itemstyle_set(enew, self.cl_var, self.st_var)
    self._qtree.itemelement_config(enew, self.cl_val, self.el_valtext,
                                   text=qent[1], datatype=STRING)      
    self._qtree.itemelement_config(enew, self.cl_var, self.el_vartext,
                                   text=qent[0], datatype=STRING)      
    return enew
  def add_group(self,item,op):
    enew=self._qtree.create_item(parent=item, button=1, open=0)[0]    
    self._qtree.itemstate_set(enew,op)
    self._qtree.itemstyle_set(enew,self.cl_ctr,self.st_ctr)
    return enew
  def open_call(self,tktree,item):
    pass
  def updateQueryTree(self,qry,item=ROOT):
    for wcq in qry:
      if (type(wcq)==type((1,))):
        qe=(wcq[0],wcq[1])
        inew=self.add_entry(item,qe)
        self.query.add(item,inew,qe)
      elif (type(wcq)==type([])):
        if (wcq[0] in s7Query.conlist):
          inew=self.add_group(item,wcq[0])    
          self.query.add(item,inew,wcq[0])
          self.updateQueryTree(wcq[1:],inew)

  def __init__(self,woperate,wcontrol,wtree):
    self.vlist=s7Query.taglist
    self.defvalue='.*'
    self.active=None
    self.woperate=woperate

    def open_menu(event):
      tktree=event.widget
      id=tktree.identify(event.x,event.y)
      item=tktree.item_id(id[1])
      self._qtree.selection_clear()
      self._qtree.selection_add(item)
      self.popupmenu.entryconfigure(0,label='')
      popUpMenuOn(event)

    s7windoz.wWindoz.__init__(self,wcontrol,'pyS7: Query view')
    ScrolledTreectrl.__init__(self,self._wtop,relief=GROOVE,border=3,
                              height=G___.wminheight,
                              width=G___.wminwidth/2)
    
    self.treectrl.configure(yscrollincrement=40) # hmmmm, what else ?

    self.menu([('operate-add',    NW, 2,self.cbk_load,'Load query'),
               ('operate-save',   NW, 2,self.cbk_save,'Save query definition'),
               ('operate-probe',  NW, 2,self.cbk_probe,'Probe query'),
               ('operate-execute',NW,20,self.cbk_execute,'Execute query')]) 

    self._qtree=self.treectrl
    self._qtree.state_define('and')
    self._qtree.state_define('or')
    self._qtree.state_define('not')
    self._qtree.state_define('leaf')
    self._qtree.state_define('isroot')

    def popUpMenuOn(event):
      try:
        tktree=event.widget
        id=tktree.identify(event.x,event.y)
        item=tktree.item_id(id[1])
        if (tktree.itemstate_get(item,'leaf')):
            pass # cannot use DISABLED on tk menu < 8.4
        self._qtree.selection_clear()
        self._qtree.selection_add(item)
        self.popupmenu.tk_popup(event.x_root,event.y_root,0)
      finally:
        self.popupmenu.grab_release()

    def popUpMenuOff(event):
      self.popupmenu.grab_release()
      self.popupmenu.unpost()
        
    def kadd_entry(event):
      self.AddEntry()
    
    def kadd_and(event):
      self.AddAnd()
    
    def kadd_or(event):
      self.AddOr()
    
    def kadd_not(event):
      self.AddNot()
    
    def kcut_node(event):
      self.SubCut()
    
    def kedit_var(event):
      self.SubEditVar()

    def kedit_val(event):
      self.SubEditVal()
    
    self._qtree.bind('<Button-3>', open_menu)
#     self._qtree.bind('<Control-c>',copy_node)
    self._qtree.bind('<Control-r>',kedit_var)
    self._qtree.bind('<Control-l>',kedit_val)    
    self._qtree.bind('<Control-x>',kcut_node)
    self._qtree.bind('<Control-e>',kadd_entry)
    self._qtree.bind('<Control-a>',kadd_and)
    self._qtree.bind('<Control-o>',kadd_or)
    self._qtree.bind('<Control-n>',kadd_not)

    self.popupmenu = Menu(self._wtop,tearoff=0)
    self.popupmenu.add_command(font=self.menufont,
                               label="Query entry menu",
                               state=DISABLED)
    self.popupmenu.add_command(font=self.menufont,
                               label="New query entry (C-e)",
                               command=self.AddEntry)
    self.popupmenu.add_separator()
    self.popupmenu.add_command(font=self.menufont,
                               label="Add AND (C-a)",
                               command=self.AddAnd)
    self.popupmenu.add_command(font=self.menufont,
                               label="Add OR  (C-o)",
                               command=self.AddOr)
    self.popupmenu.add_command(font=self.menufont,
                               label="Add NOT (C-n)",
                               command=self.AddNot)
    self.popupmenu.add_separator()
    self.popupmenu.add_command(font=self.menufont,
                               label="Cut (C-x)",
                               command=self.SubCut)
    self.popupmenu.add_command(font=self.menufont,
                               label="Paste (C-v)",    
                               command=self.SubPaste)
    self.popupmenu.add_command(font=self.menufont,
                               label="Copy (C-c)",
                               command=self.SubCopy)
    self.popupmenu.add_separator()    
    self.popupmenu.add_command(font=self.menufont,
                               label="Edit entry variable (C-r)",
                               command=self.SubEditVar)
    self.popupmenu.add_command(font=self.menufont,
                               label="Edit entry value (C-l)",
                               command=self.SubEditVal)
    self.popupmenu.bind('<FocusOut>',popUpMenuOff)

    ik=self._icon

    if (wtree != None):
      self._tree=wtree._tree
      self._view=wtree._viewtree
      self._paths=wtree._paths
      self._wtree=wtree

    tf=self.titlefont
    # --- Graph column
    self.cl_ctr=self._qtree.column_create(font=tf,text=" ",expand=1)
    self.el_ctr=self._qtree.element_create(type=IMAGE,image=(\
      ik['query-and'],s7Query.AND,\
      ik['query-or'], s7Query.OR,\
      ik['query-not'],s7Query.NOT))
    self.el_ctricon=self._qtree.element_create(type=IMAGE,
                                               image=(ik['query-or'],'isroot'))
    self.st_ctr=self._qtree.style_create()
    self.st_ent=self._qtree.style_create()
    self._qtree.style_elements(self.st_ctr,self.el_ctr,self.el_ctricon)    
    self._qtree.style_layout(self.st_ctr,self.el_ctr,pady=2)
    self._qtree.configure(treecolumn=self.cl_ctr,
                          background='white',
                          showheader=G___.showColumnTitle)
    # --- Var column
    self.cl_var=self._qtree.column_create(font=tf,text="Variable",expand=1)
    self.el_varopt=self._qtree.element_create(type=WINDOW)
    self.el_varfont=G___.font['Tc']
    self.el_vartext=self._qtree.element_create(type=TEXT)
    self._qtree.element_configure(self.el_vartext,font=self.el_varfont)    
    self.el_varsel=self._qtree.element_create(type=RECT,
                                              fill=(G___.color_Tc, SELECTED),
                                              width=200)
    self.st_var=self._qtree.style_create()
    self._qtree.style_elements(self.st_var,self.el_varsel,self.el_vartext)
    self.st_varopt=self._qtree.style_create()    
    self._qtree.style_elements(self.st_varopt,self.el_varopt)
    self._qtree.style_layout(self.st_var,self.el_vartext,pady=2)
    self._qtree.style_layout(self.st_var,self.el_varsel,
                             union=(self.el_vartext,),ipadx=2, iexpand=NS)    

    # --- Value column
    self.cl_val=self._qtree.column_create(font=tf,text="Value",expand=1)   
    self.el_valfont=G___.font['E']
    self.el_valtext=self._qtree.element_create(type=TEXT)    
    self._qtree.element_configure(self.el_valtext,font=self.el_valfont)
    self.el_valsel=self._qtree.element_create(type=RECT,
                                             fill=(G___.color_Tc, SELECTED),
                                             width=200)
    self.el_valopt=self._qtree.element_create(type=WINDOW)
    self.st_val=self._qtree.style_create()
    self._qtree.style_elements(self.st_val,self.el_valsel,self.el_valtext)
    self.st_valopt=self._qtree.style_create()    
    self._qtree.style_elements(self.st_valopt,self.el_valopt)
    self._qtree.style_layout(self.st_val,self.el_valtext,pady=2)
    self._qtree.style_layout(self.st_val,self.el_valsel,
                             union=(self.el_valtext,),ipadx=2, iexpand=NS)    
    self._qtree.set_sensitive((self.cl_ctr,self.st_ctr,self.el_ctr),
                              (self.cl_val,self.st_val,self.el_valtext,
                               self.el_valsel),
                              (self.cl_var,self.st_var,self.el_vartext,
                               self.el_varsel))
    self._qtree.column_configure(self.cl_ctr,lock=LEFT)
      
    self._qtree.notify_bind('<Expand-before>',self.open_node)
    self._qtree.itemstyle_set(ROOT,self.cl_ctr,self.st_ctr)
    self._qtree.itemelement_config(ROOT,self.cl_ctr,self.el_ctricon,draw=True)
    self._qtree.item_config(ROOT, button=0)
    self._qtree.itemstate_set(ROOT,'isroot')
    self._qtree.notify_generate('<Expand-before>', item=ROOT)
    self._qtree.see(ROOT)

    self.query=self.qEntry()
            
    self.grid(row=1,sticky=N+E+W+S)
    t=self.winfo_toplevel() 
    t.rowconfigure(1,weight=1)
    t.rowconfigure(2,weight=0)
    t.columnconfigure(0,weight=1)

  def createQueryList(self):
    pass

  def updateQuery(self):
    pass

  def onexit(self):
#    s7utils.queryViewCloseError()
    self._wtop.destroy()

  def SubPaste(self):
    pass
    
  def SubCopy(self):
    pass
    
  def SubEditVar(self):
    if (self.active): self.active.close()
    item=self.getItemSelection()
    if (item==None): return
    if (not self._qtree.itemstate_get(item,'leaf')): return
    qentry=self.query.getEntry(item[0])
    print 'EDIT VAR ',qentry
    self.wEditEnumerate(item,self,self.vlist,qentry)
    print 'Var ',self.query.Q

  def SubEditVal(self):
    if (self.active): self.active.close()
    item=self.getItemSelection()
    if (item==None): return
    if (not self._qtree.itemstate_get(item,'leaf')): return
    qentry=self.query.getEntry(item[0])
    self.wEditEntry(item,self,qentry)    
    
  def SubCut(self):
    item=self.getItemSelection()
    if (item==None): return
    self.query.cut(item[0])
    p=self._qtree.item_ancestors(item)[0]
    self._qtree.item_delete(item)
    
  def AddAnd(self):
    item=self.getItemSelection()
    enew=self.add_group(item[0],s7Query.AND)    
    self.query.add(item[0],enew,s7Query.AND)
   
  def AddOr(self):
    item=self.getItemSelection()
    enew=self.add_group(item[0],s7Query.OR)    
    self.query.add(item[0],enew,s7Query.OR)
    
  def AddNot(self):
    item=self.getItemSelection()
    enew=self.add_group(item[0],s7Query.NOT)    
    self.query.add(item[0],enew,s7Query.NOT)

  def AddEntry(self):
    qe=(self.vlist[0],self.defvalue)
    item=self.getItemSelection()
    if (self._qtree.itemstate_get(item,'leaf')): return
    enew=self.add_entry(item[0],qe)
    self.query.add(item[0],enew,qe)

# -----------------------------------------------------------------------------
class wOperateView(s7windoz.wWindoz,ScrolledMultiListbox):
  def cbk_probe(self):
    pass

  def cbk_save(self):
    pass

  def cbk_update(self):
    r=[]
    for s in self.selectedlist:
      if (s[0]): r+=[s[1]]
    self._wtree.updateMarkedFromList(r)

  def cbk_execute(self):
    self._wqueryview=s7operateView.wQueryView(self,self._wcontrol,self._wtree)

  def __init__(self,wcontrol,wtree,selectedlist):

    self._wqueryview=None
    self.colmax=3
    if (G___.operateListSplit):
      for sl in selectedlist:
        self.colmax=max(self.colmax,len(sl[0].split('/')))

      self.selectedlist=[]
      for sl in selectedlist:
        ll=list(sl[0].split('/')[1:-2])
        cl=len(ll)
        sd=['0']+ll+[""]*(self.colmax-cl-1)+list(sl[1:])
        self.selectedlist.append(sd)
    else:
      self.selectedlist=[]
      for sl in selectedlist:
        self.selectedlist+=[[0]+list(sl)]
    
    s7windoz.wWindoz.__init__(self,wcontrol,
                              'pyS7: Selection List on [%s] view [%.2d]'%\
                              (wtree._viewtree.filename,wtree._viewid))
    ScrolledMultiListbox.__init__(self,self._wtop,relief=GROOVE,border=3,
                                  height=G___.wminheight,
                                  width=G___.wminwidth)
    
    self.listbox.configure(selectmode=MULTIPLE)

    self.menu([('select-save',  NW, 2,self.cbk_save,'Save selection'),
               ('select-update',NW, 2,self.cbk_update,'Update selection'),
               ('operate-view', NW,20,self.cbk_execute,'Query view')])

    if (wtree != None):
      self._tree=wtree._tree
      self._view=wtree._viewtree
      self._paths=wtree._paths
      self._wtree=wtree
      self._wcontrol=wcontrol

    self.createOperateList()
    self.listbox.sort_allowed=range(1,self.colmax)
    self.listbox.sort_init=0
    self.listbox.configure(height=100,width=600)
    self.listbox.grid()

    self.grid(row=1,sticky=N+E+W+S)

    t=self.winfo_toplevel() 
    t.rowconfigure(1,weight=1)
    t.columnconfigure(0,weight=1)

  def go_operate(self):
    if (self._tree == None): return
    for itid in self.opselection:
      item=self._tree.item_id(itid)
      if (item[0]):
        print self._paths[str(item[0])]

  def createOperateList(self):
    colnames=['S']
    for n in range(self.colmax-2):
      colnames+=["Level %d"%n]
    colnames+=['Name','Type']
    self.listbox.config(columns=tuple(colnames),
                        selectbackground='AntiqueWhite',
                        selectforeground='black',
                        selectmode='extended')
    colors = ('white',)
    self.listbox.column_configure(self.listbox.column(0),
                                  font=self.titlefont,
                                  borderwidth=1,
                                  itembackground=colors,
                                  arrow='down',
                                  arrowgravity='right')
    for n in range(self.colmax-2):
      self.listbox.column_configure(self.listbox.column(n+1),
                                    font=self.titlefont,borderwidth=1,
                                    itembackground=colors,
                                    arrow='down',arrowgravity='right')
    self.listbox.column_configure(self.listbox.column(self.colmax-1),
                                  font=self.titlefont,
                                  borderwidth=1,
                                  itembackground=colors,
                                  arrow='down',
                                  arrowgravity='right')
    self.listbox.column_configure(self.listbox.column(self.colmax),
                                  font=self.titlefont,
                                  justify='center',
                                  borderwidth=1,
                                  itembackground=colors,
                                  arrow='down',
                                  arrowgravity='right')
    self.listbox.notify_install('<Header-invoke>')
    self.listbox.bind('<Return>',self.do_focus)
    self.listbox.notify_bind('<Header-invoke>',s7utils.operate_sort_list)
    self.updateSelection(G___.defaultProfile)

  def do_focus(self,e):
    try:
      id=self.listbox.curselection()[0]
    except IndexError: return
    wt=self.listbox.get(id)
    self._wtree.expandAndFocusByPath(wt[0][0])

  def updateSelection(self,prof):
    self.listbox.delete(0,END)
    count=0
    for op in self.selectedlist:
      self.listbox.insert('end',*op)
      it=self.listbox.item('end')
      for c in range(self.colmax+1):
        cld=self.listbox.column(c)
        elt=self.listbox.element('text')
        print 'Sous i (py %.3d)'%count
        print 'Sous j (py %.3d)'%count
        print 'Sous k (py %.3d)'%count
        count+=1
        self.listbox.itemelement_configure(it,cld,elt,font=self.listfont)
        if (op[0]):
          self.listbox.itemelement_configure(it,cld,elt,font=G___.font['F'])

  def onexit(self):
    self._control.operateViewClose()
    if self._wqueryview: self._wqueryview._wtop.destroy()

# --- last line
