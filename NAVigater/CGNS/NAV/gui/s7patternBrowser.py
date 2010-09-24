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
import FileDialog

import s7globals
G___=s7globals.s7G

import s7treeFingerPrint
import s7viewControl
import s7windoz
import s7utils
import s7fileDialog

class wPatternEdit(s7windoz.wWindoz):
  def __init__(self,wcontrol,tree,fingerprint):
    s7windoz.wWindoz.__init__(self,wcontrol,'CGNS.NAV: Pattern Save form')

    self.fingerprint=fingerprint
    self.tree=fingerprint.tree
    
    self.v_prof=StringVar()
    self.v_prof.set(fingerprint.filedir)
    self.v_name=StringVar()
    self.v_name.set(fingerprint.filename)
    self.v_keyw=StringVar()
    self.v_keyw.set(fingerprint.keyword)
    self.v_comm=StringVar()
    self.v_comm.set(fingerprint.comment)
    self.labp=Label(self._wtop,text='Profile:',
                    font=G___.font['L'])
    self.prof=Entry(self._wtop,textvariable=self.v_prof,width=20,
                    font=G___.font['E'])
    self.labn=Label(self._wtop,text='Pattern:',
                    font=G___.font['L'])
    self.name=Entry(self._wtop,textvariable=self.v_name,width=20,
                    font=G___.font['E'])
    self.labk=Label(self._wtop,text='Keyword:',
                    font=G___.font['L'])    
    self.keyw=Entry(self._wtop,textvariable=self.v_keyw,width=10,
                    font=G___.font['E'])
    self.labc=Label(self._wtop,text='Comment:',
                    font=G___.font['L'])    
    self.comm=Entry(self._wtop,textvariable=self.v_comm,width=60,
                    font=G___.font['E'])

    self.v_path=StringVar()
    self.v_path.set(fingerprint.profdir)
    self.lpth=Label(self._wtop,text='Path:',justify=LEFT,
                    font=G___.font['L'])
    self.mpth=Menubutton(self._wtop,textvariable=self.v_path,
                         font=G___.font['E'])
    self.mpth.menu=Menu(self.mpth,tearoff=0,borderwidth=1)
    self.mpth['menu']=self.mpth.menu
    for v in G___.profilePath:
      self.mpth.menu.add_radiobutton(label=v,var=self.v_path,value=v,
                                     indicatoron=0,font=G___.font['E'])

    self.labp.grid(row=1,column=0,sticky=W)
    self.prof.grid(row=1,column=1,sticky=W)
    self.labn.grid(row=1,column=2,sticky=W)
    self.name.grid(row=1,column=3,sticky=W)
    self.labk.grid(row=1,column=4,sticky=E)
    self.keyw.grid(row=1,column=5,sticky=E)
    self.labc.grid(row=2,column=0,sticky=W)
    self.comm.grid(row=2,column=1,columnspan=5,sticky=E+W)
    self.lpth.grid(row=3,column=0,sticky=W)
    self.mpth.grid(row=3,column=1,columnspan=5,sticky=W)
    self.b=Button(self._wtop,text="Ok",command=self.checkandsave,
                  font=G___.font['B'])
    self.b.grid(row=4,column=4,sticky=E)
    self.c=Button(self._wtop,text="Cancel",command=self.leave,
                  font=G___.font['B'])
    self.c.grid(row=4,column=5,sticky=E)
    self._wtop.grid()
    t=self._wtop.winfo_toplevel() 
    t.rowconfigure(3,weight=0)
    t.columnconfigure(1,weight=0)

  def checkandsave(self):
    prof=self.v_prof.get()
    name=self.v_name.get()
    path=self.v_path.get()
    comm=self.v_comm.get()
    keyw=self.v_keyw.get()
    if (not os.path.exists("%s/%s"%(path,prof))):
      s7utils.profileDirNotFound("%s/%s"%(path,prof))
      return
    if (' ' in name):
      s7utils.spaceInFileName(name)
      return
    if (os.path.exists("%s/%s/%s.py"%(path,prof,name))):
      if (not s7utils.patternOverwrite(name)):
        return
    if (not os.path.exists("%s/%s/__init__.py"%(path,prof))):
      if (not s7utils.createProfileDirInit(name)):
        return
      else:
        self.generatePatternDir(path,prof)
    self.generatePatternFile(path,prof,name,comm,keyw)
    self.fingerprint.comment =comm
    self.fingerprint.keyword =keyw
    self.fingerprint.profdir =path
    self.fingerprint.filename=name
    self.fingerprint.filedir =prof
    self._control.patternWindow.reloadPattern(path,prof,name)
    self.leave()

  def generatePatternDir(self,path,prof):
    s ='# CGNS.NAV generated CGNS profile dir - see http://www.cgns.org\n'
    s+='# profile: %s\n'%prof
    from time import localtime, strftime
    t=strftime("%Y-%m-%d.%H:%M:%S", localtime())
    s+='# date   : %s\n'%t
    s+="#\n"
    s+="import os\n"
    s+="import fnmatch\n"
    s+="import sys\n"
    s+="import imp\n"
    s+="\n"
    s+="profdir='%s/%s'\n"%(path,prof)
    s+="profile={}\n"
    s+="sprev=sys.path\n"
    s+="sys.path=[profdir]+sys.path\n"
    s+="for f in os.listdir(profdir):\n"
    s+="  if ((fnmatch.fnmatch(f,'*.py')) and (f != '__init__.py')):\n"
    s+="    n=os.path.splitext(f)[0]\n"
    s+="    m=imp.find_module(n)\n"
    s+="    p=imp.load_module(n,m[0],m[1],m[2]).pattern\n"
    s+="    profile[n]=p\n"
    s+="sys.path=sprev\n"
    s+="# last line\n"    
    fp=open("%s/%s/__init__.py"%(path,prof),'w+')
    fp.write(s)
    fp.close()

  def generatePatternFile(self,path,prof,name,comm,keyw):
    data=s7utils.printNode(self.tree)[:-1]
    from time import localtime, strftime
    t=strftime("%Y-%m-%d.%H:%M:%S", localtime())
    s ='# CGNS.NAV generated CGNS pattern file - see http://www.cgns.org\n'
    s+='# profile: %s\n'%prof
    s+='# pattern: %s\n'%name
    s+='# date   : %s\n'%t
    s+="#\n"
    s+="from numpy import *\n"
    s+='data=%s\n'%data
    s+='status="%s"\n'%keyw
    s+='comment="""%s"""\n'%comm
    s+='pattern=[data,status,comment]\n'
    s+="# last line\n"    
    fp=open("%s/%s/%s.py"%(path,prof,name),'w+')
    fp.write(s)
    fp.close()

  def leave(self):
    self._wtop.destroy()
    
  def onexit(self):
    self.leave()

# ----------------------------------------------------------------------
class wPatternBrowser(s7windoz.wWindoz,ScrolledMultiListbox):

  def onexit(self):
    self._control.patternViewClose()
  def cbk_addpattern(self):
    self.loadPattern()
  def cbk_relpattern(self):
    index = self.listbox.index('active')
    if index > -1:
      self.listbox.see(index)
      pt=self.listbox.get(index)
      pdir=self.patternDict["%s/%s"%(pt[0][0],pt[0][1])][3]
      prof=pt[0][0]
      patn=pt[0][1]      
      self.reloadPattern(pdir,prof,patn)
  def cbk_delpattern(self):
    pass
  def cbk_close(self):
    self._control.patternViewClose()
  def __init__(self, wcontrol):

    self.patternDict={}
    self.patternSet=[]
    
    s7windoz.wWindoz.__init__(self,wcontrol,'CGNS.NAV: Pattern Browser')
    ScrolledMultiListbox.__init__(self,self._wtop,relief=GROOVE,border=3,
                                  height=2*G___.wminheight,
                                  width=1.5*G___.wminwidth)

    self.menu([('pattern-open',    W, 2,self.cbk_addpattern,'Open profile'),
               ('pattern-close',   W, 2,self.cbk_delpattern,'Close profile'),
               ('pattern-reload',  W, 2,self.cbk_delpattern,'Reload profile')])

    self.createPatternList()
    self.listbox.sort_allowed=[0,1,2]
    self.listbox.sort_init=0
    self.listbox.configure(height=500,width=600)
    self.listbox.element_configure(self.listbox._el_text,font=G___.font['E'])
    self.listbox.element_configure(self.listbox._el_select,
                                   fill=(G___.color_Ca, 'selected'))
    self.listbox.bind('<space>',)
    self.listbox.bind('<Double-Button-1>',self.open_pattern)
    self.listbox.grid()
    self.grid(row=1,sticky=N+E+S+W)
    t=self.winfo_toplevel() 
    t.rowconfigure(1,weight=1)
    t.columnconfigure(0,weight=1)

  def open_pattern(self,event):
    index = self.listbox.index('active')
    if index > -1:
      self.listbox.see(index)
      pt=self.listbox.get(index)
      tt=self.patternDict["%s/%s"%(pt[0][0],pt[0][1])][0]
      if (self._control.getSameView(pt[0][0],pt[0][1],'/')):
        return
      treefingerprint=s7treeFingerPrint.wTreeFingerPrint(pt[0][0],pt[0][1],tt)
      treefingerprint.viewtype='P'
      treefingerprint.comment=self.patternDict["%s/%s"%(pt[0][0],pt[0][1])][2]
      treefingerprint.keyword=self.patternDict["%s/%s"%(pt[0][0],pt[0][1])][1] 
      treefingerprint.profdir=self.patternDict["%s/%s"%(pt[0][0],pt[0][1])][3]
      G___.treeStorage.append(treefingerprint)
      wt=s7viewControl.createDataTreeWindow(self._control,
                                            treefingerprint,
                                            G___.expandRecurse)

  # --------------------------------------------------------------------
  def createPatternList(self):
    self.coldim=2
    self.listbox.config(columns=('profile','type','K','Comment'),
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
    self.listbox.column_configure(self.listbox.column(1),
                                  font=self.titlefont,
                                  borderwidth=1,
                                  itembackground=colors,
                                  arrow='down',
                                  arrowgravity='right')
    self.listbox.column_configure(self.listbox.column(2),
                                 font=self.titlefont,
                                  justify='center',
                                  borderwidth=1,
                                  itembackground=colors,
                                  arrow='down',
                                  arrowgravity='right')
    self.listbox.column_configure(self.listbox.column(3),
                                  font=self.titlefont,
                                  justify='center',
                                  borderwidth=1,
                                  itembackground=colors)
    self.listbox.notify_install('<Header-invoke>')
    self.listbox.notify_bind('<Header-invoke>',s7utils.operate_sort_list)
    for pdir in G___.profilePath:
      print "%s/%s/__init__.py"%(pdir,G___.defaultProfile)
      if (os.path.exists("%s/%s/__init__.py"%(pdir,G___.defaultProfile))):
        self.updateProfile(pdir,G___.defaultProfile)
        break
    else:
      s7utils.importProfileWarning(G___.defaultProfile,'No such file!')

  def importProfile(self,dir,profile,force,pattern='__init__'):
    import imp
    if (not force):
      for pdir,pprof in self.patternSet:
        if ((pdir==dir) and (pprof==profile)): return None # should reload here
    sprev=sys.path
    sys.path=["%s/%s"%(dir,profile)]+sys.path
    try:
      m=imp.find_module(pattern)
      pdict=imp.load_module(pattern,m[0],m[1],m[2]).profile
    except TypeError,e: #(ValueError,KeyError,ImportError,AttributeError), e:
      s7utils.importProfileWarning(profile,e)
      sys.path=sprev
      return None
    sys.path=sprev
    return pdict

  def importPattern(self,dir,profile,pattern):
    import imp
    prof={}
    sprev=sys.path
    sys.path=["%s/%s"%(dir,profile)]+sys.path
    try:
      m=imp.find_module(pattern)
      pattn=imp.load_module(pattern,m[0],m[1],m[2]).pattern
    except (ValueError, KeyError, ImportError), e:
      s7utils.importProfileWarning(profile,e)
      sys.path=sprev
      return None
    sys.path=sprev
    prof[pattern]=pattn
    return prof

  def removeFromListbox(self,dir,prof,target=None):
    for idx in  range(self.listbox.size()-1,-1,-1):
      ep=self.listbox.get(idx)
      if ((target == None) or (target == ep[0][1])):
        if (ep[0][0] == prof):
          self.listbox.delete(idx)
          if (target != None): return idx
          pkey='%s/%s'%(prof,ep[0][1])
          del self.patternDict[pkey]
    return -1
      
  def updateProfile(self,dir,prof,index=-1,force=0,target=None):
    if (target): ptdict=self.importPattern(dir,prof,target)
    else:        ptdict=self.importProfile(dir,prof,force)    
    if (ptdict == None): return
    atleastone=0
    for kpt in ptdict:
      pkey='%s/%s'%(prof,kpt)
      if (    (force or (pkey not in self.patternDict))
          and ((target==None) or (target==kpt))):
        atleastone=1
        if (index == -1):
          self.listbox.insert('end',prof,kpt,ptdict[kpt][1],ptdict[kpt][2])
        else:
          self.listbox.insert(index,prof,kpt,ptdict[kpt][1],ptdict[kpt][2])
        self.patternDict[pkey]=[ptdict[kpt][0],
                                ptdict[kpt][1],
                                ptdict[kpt][2],dir]
      else:
        if (((target==None) or (target==kpt))
             and (not s7utils.duplicatedPattern(kpt))):
          break
    if (atleastone and not force): self.patternSet.append((dir,prof))

  def loadPattern(self):
    (dir,profile)=s7fileDialog.s7profiledialog(self._wtop)
    self.updateProfile(dir,profile)
      
  def reloadPattern(self,dir,profile,pattern):
    # oups ! only the 'first level' imports would be updated no way
    # to recurse on other imports
    idx=self.removeFromListbox(dir,profile,target=pattern)
    self.updateProfile(dir,profile,index=idx,force=1,target=pattern)
      
# --- last line
