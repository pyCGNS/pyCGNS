# -----------------------------------------------------------------------------
# pyS7 - CGNS/SIDS editor
# ONERA/DSNA - marc.poinot@onera.fr
# pyS7 - $Rev: 72 $ $Date: 2009-02-10 15:58:15 +0100 (Tue, 10 Feb 2009) $
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source
# tree for license information.


# this file needs to be re-organized
# it was first using FileDialog, then add to much modifs
# and now has its own FileDialog copy...

import Tkinter
Tkinter.wantobjects=0 #necessary for tk-8.5 and some buggy tkinter installs
from Tkinter import *
from TkTreectrl import *
import os
import sys
import numpy as Num

# ------------------------------------------------------------
"""File selection dialog classes. (MODIFIED COPY FROM PYTHON TkINTER LIB)

Classes:

- FileDialog
- LoadFileDialog
- SaveFileDialog

"""

from Tkinter import *
from Dialog import Dialog

import os
import fnmatch

dialogstates = {}

class FileDialog:

    """Standard file selection dialog -- no checks on selected file.

    Usage:

        d = FileDialog(master)
        fname = d.go(dir_or_file, pattern, default, key)
        if fname is None: ...canceled...
        else: ...open file...

    All arguments to go() are optional.

    The 'key' argument specifies a key in the global dictionary
    'dialogstates', which keeps track of the values for the directory
    and pattern arguments, overriding the values passed in (it does
    not keep track of the default argument!).  If no key is specified,
    the dialog keeps no memory of previous state.  Note that memory is
    kept even when the dialog is canceled.  (All this emulates the
    behavior of the Macintosh file selection dialogs.)

    """

    title = "File Selection Dialog"

    def __init__(self, master, title=None):
        if title is None: title = self.title
        self.master = master
        self.directory = None
        self.lockpattern=0
        
        self.top = Toplevel(master)
        self.top.title(title)
        self.top.iconname(title)

        self.botframe = Frame(self.top)
        self.botframe.pack(side=BOTTOM, fill=X)

        self.selection = Entry(self.top)
        self.selection.pack(side=BOTTOM, fill=X)
        self.selection.bind('<Return>', self.ok_event)

        self.filter = Entry(self.top)
        self.filter.pack(side=TOP, fill=X)
        self.filter.bind('<Return>', self.filter_command)

        self.midframe = Frame(self.top)
        self.midframe.pack(expand=YES, fill=BOTH)

        self.filesbar = Scrollbar(self.midframe)
        self.filesbar.pack(side=RIGHT, fill=Y)
        self.files = Listbox(self.midframe, exportselection=0,
                             yscrollcommand=(self.filesbar, 'set'))
        self.files.pack(side=RIGHT, expand=YES, fill=BOTH)
        btags = self.files.bindtags()
        self.files.bindtags(btags[1:] + btags[:1])
        self.files.bind('<ButtonRelease-1>', self.files_select_event)
        self.files.bind('<Double-ButtonRelease-1>', self.files_double_event)
        self.filesbar.config(command=(self.files, 'yview'))

        self.dirsbar = Scrollbar(self.midframe)
        self.dirsbar.pack(side=LEFT, fill=Y)
        self.dirs = Listbox(self.midframe, exportselection=0,
                            yscrollcommand=(self.dirsbar, 'set'))
        self.dirs.pack(side=LEFT, expand=YES, fill=BOTH)
        self.dirsbar.config(command=(self.dirs, 'yview'))
        btags = self.dirs.bindtags()
        self.dirs.bindtags(btags[1:] + btags[:1])
        self.dirs.bind('<ButtonRelease-1>', self.dirs_select_event)
        self.dirs.bind('<Double-ButtonRelease-1>', self.dirs_double_event)

        self.ok_button = Button(self.botframe,
                                 text="OK",
                                 command=self.ok_command)
        self.ok_button.pack(side=LEFT)
        self.filter_button = Button(self.botframe,
                                    text="Filter",
                                    command=self.filter_command)
        self.filter_button.pack(side=LEFT, expand=YES)
        self.cancel_button = Button(self.botframe,
                                    text="Cancel",
                                    command=self.cancel_command)
        self.cancel_button.pack(side=RIGHT)

        self.top.protocol('WM_DELETE_WINDOW', self.cancel_command)
        # XXX Are the following okay for a general audience?
        self.top.bind('<Alt-w>', self.cancel_command)
        self.top.bind('<Alt-W>', self.cancel_command)

    def go(self,dir_or_file=os.curdir,pattern="*",default="",key=None,lock=0):
        self.lockpattern=lock
        if key and dialogstates.has_key(key):
            self.directory, pattern = dialogstates[key]
        else:
            dir_or_file = os.path.expanduser(dir_or_file)
            if os.path.isdir(dir_or_file):
                self.directory = dir_or_file
            else:
                self.directory, default = os.path.split(dir_or_file)
        self.set_filter(self.directory, pattern)
        self.set_selection(default)
        self.filter_command()
        self.selection.focus_set()
        self.top.wait_visibility() # window needs to be visible for the grab
        self.top.grab_set()
        self.how = None
        self.master.mainloop()          # Exited by self.quit(how)
        if key:
            directory, pattern = self.get_filter()
            if self.how:
                directory = os.path.dirname(self.how)
            dialogstates[key] = directory, pattern
        self.top.destroy()
        return self.how

    def quit(self, how=None):
        self.how = how
        self.master.quit()              # Exit mainloop()

    def dirs_double_event(self, event):
        self.filter_command()

    def dirs_select_event(self, event):
        dir, pat = self.get_filter()
        subdir = self.dirs.get('active')
        dir = os.path.normpath(os.path.join(self.directory, subdir))
        self.set_filter(dir, pat)

    def files_double_event(self, event):
        self.ok_command()

    def files_select_event(self, event):
        file = self.files.get('active')
        self.set_selection(file)

    def ok_event(self, event):
        self.ok_command()

    def ok_command(self):
        self.quit(self.get_selection())

    def filter_command(self, event=None):
        dir, pat = self.get_filter()
        try:
            names = os.listdir(dir)
        except os.error:
            self.master.bell()
            return
        self.directory = dir
        self.set_filter(dir, pat)
        names.sort()
        subdirs = [os.pardir]
        matchingfiles = []
        for name in names:
            fullname = os.path.join(dir, name)
            if os.path.isdir(fullname):
                subdirs.append(name)
            elif fnmatch.fnmatch(name, pat):
                matchingfiles.append(name)
        self.dirs.delete(0, END)
        for name in subdirs:
            self.dirs.insert(END, name)
        self.files.delete(0, END)
        for name in matchingfiles:
            self.files.insert(END, name)
        head, tail = os.path.split(self.get_selection())
        if tail == os.curdir: tail = ''
        self.set_selection(tail)

    def get_filter(self):
        filter = self.filter.get()
        filter = os.path.expanduser(filter)
        if filter[-1:] == os.sep or os.path.isdir(filter):
            filter = os.path.join(filter, "*")
        return os.path.split(filter)

    def get_selection(self):
        file = self.selection.get()
        file = os.path.expanduser(file)
        return file

    def cancel_command(self, event=None):
        self.quit()

    def set_filter(self, dir, pat):
        if not os.path.isabs(dir):
            try:
                pwd = os.getcwd()
            except os.error:
                pwd = None
            if pwd:
                dir = os.path.join(pwd, dir)
                dir = os.path.normpath(dir)
        self.filter.delete(0, END)
        self.filter.insert(END, os.path.join(dir or os.curdir, pat or "*"))

    def set_selection(self, file):
        self.selection.delete(0, END)
        self.selection.insert(END, os.path.join(self.directory, file))


class LoadFileDialog(FileDialog):

    """File selection dialog which checks that the file exists."""

    title = "Load File Selection Dialog"

    def ok_command(self):
        file = self.get_selection()
        if not os.path.isfile(file):
            self.master.bell()
        else:
            self.quit(file)


class SaveFileDialog(FileDialog):

    """File selection dialog which checks that the file may be created."""

    title = "Save File Selection Dialog"

    def ok_command(self):
        file = self.get_selection()
        if os.path.exists(file):
            if os.path.isdir(file):
                self.master.bell()
                return
            d = Dialog(self.top,
                       title="Overwrite Existing File Question",
                       text="Overwrite existing file %r?" % (file,),
                       bitmap='questhead',
                       default=1,
                       strings=("Yes", "Cancel"))
            if d.num != 0:
                return
        else:
            head, tail = os.path.split(file)
            if not os.path.isdir(head):
                self.master.bell()
                return
        self.quit(file)

# ------------------------------------------------------------

from S7.version import __vid__

import s7globals
G___=s7globals.s7G

import s7treeFingerPrint
import s7treeSimple
import s7tableSimple
import s7patternBrowser
import s7operateView
import s7utils
import s7windoz
import s7history

s7history.loadHistory()

# -----------------------------------------------------------------------------
class wProfileDialog(s7windoz.wWindoz,ScrolledMultiListbox):
  def __init__(self,wcontrol,title):
    s7windoz.wWindoz.__init__(self,wcontrol,title)
    ScrolledMultiListbox.__init__(self,self._wtop,relief=GROOVE,border=3)
    self.coldim=3
    self.listbox.config(columns=('directory','profile','Comment'),
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
                                  itembackground=colors)
    for pdir in G___.profilePath:
      pcomment=''
      if (os.path.isdir(pdir)):
        dlist=os.listdir(pdir)
        for pname in dlist:
          self.listbox.insert('end',pdir,pname,pcomment)

    self.selecteddir=''
    self.selectedprofile=''

    self.listbox.notify_install('<Header-invoke>')
    self.listbox.notify_bind('<Header-invoke>',s7utils.operate_sort_list)
    self.listbox.sort_allowed=[0,1,2]
    self.listbox.sort_init=0
    self.listbox.element_configure(self.listbox._el_text,font=G___.font['E'])
    self.listbox.element_configure(self.listbox._el_select,
                                   fill=(G___.color_Ca, 'selected'))
    self.listbox.configure(height=100,width=600)
    self.listbox.bind('<Double-Button-1>', self.open_profile)
    self.listbox.bind('<space>', self.open_profile)
    self.listbox.pack(expand=1,fill=BOTH)    
    self.pack(fill=BOTH, expand=1)
    self.grab_set()
    self._wtop.mainloop()

  def get(self):
    return (self.selecteddir,self.selectedprofile)

  def onexit(self):
    self.closeWindow()
  
  def open_profile(self,event):
    index = self.listbox.index('active')
    if index > -1:
      self.listbox.see(index)
      op=self.listbox.get(index)[0]
      self.selecteddir=op[0]
      self.selectedprofile=op[1]
      self.grab_release()
      self._wtop.quit() 
    else:
      pass

# -----------------------------------------------------------------------------
class wFileDialog(FileDialog):
  def popUpMenuOn(self,event):
    self.v_path.set('%s'%self.get_filter()[0])
    self.mpth.tk_popup(event.x_root,event.y_root,0)
    self.mpth.grab_set()
  def popUpMenuOff(self,event):
    if (self.check_filter(self.v_path.get(),self.filetype.get())):
      self.set_filter(self.v_path.get(),self.filetype.get())
    self.mpth.grab_release()
    self.mpth.unpost()
    self.updateselection()
  def showDirHistory(self,event):
    self.popUpMenuOn(event)
  def __init__(self,wparent,title,save,lock,key):
    FileDialog.__init__(self,wparent,title=title)
    self.directoriesHistory=G___.directoriesHistory
    self.v_path=StringVar()
    self.updateDirPopUp(wparent)
    self.filter_button.pack_forget()
    self.filter.configure(font=G___.font['E'],background='white')
    self.filter.bind('<Button-3>',self.showDirHistory)
    self.files.configure(font=G___.font['E'])
    self.selection.configure(font=G___.font['E'],background='white')
    self.files.configure(font=G___.font['E'])
    self.dirs.configure(font=G___.font['E'])
    self.ok_button.configure(font=G___.font['B'])
    self.cancel_button.configure(font=G___.font['B'])    
    self.ok_button.pack_forget()
    self.cancel_button.pack_forget()
    self.cancel_button.pack(side=RIGHT)
    self.ok_button.pack(side=RIGHT)
    self.wparent=wparent
    self.save=save
    self.options=Frame(self.top)
    self.options.pack(side=TOP, fill=X)
    self.options.label=Label(self.options,bd=1,relief=FLAT,anchor=W,
                             text='Select file type:',font=G___.font['L'])
    self.options.label.pack(side=LEFT)
    self.filetype=StringVar()
    # the variable below has been stolen from FileDialog implementation
    # potential bug here...
    pat=''
    if key and dialogstates.has_key(key):
      dir,pat=dialogstates[key]
    if (pat in ['*.py','*.cgns','*.adf','*.hdf','*']):
      self.filetype.set(pat)
    else:
      self.filetype.set('*.cgns')
    self.options.b_mll=Radiobutton(self.options,text='*.cgns',width=7,
                                   selectcolor='white',indicatoron=0,
                                   variable=self.filetype,value='*.cgns',
                                   relief=FLAT,borderwidth=2,
                                   command=self.updateselection,
                                   font=G___.font['B'])
    self.options.b_adf=Radiobutton(self.options,text='*.adf',width=7,
                                   selectcolor='white',indicatoron=0,
                                   variable=self.filetype,value='*.adf',
                                   relief=FLAT,borderwidth=2,
                                   command=self.updateselection,
                                   font=G___.font['B'])
    self.options.b_hdf=Radiobutton(self.options,text='*.hdf',width=7,
                                   selectcolor='white',indicatoron=0,
                                   variable=self.filetype,value='*.hdf',
                                   relief=FLAT,borderwidth=2,
                                   command=self.updateselection,
                                   font=G___.font['B'])
    self.options.b_py=Radiobutton(self.options,text='*.py',width=7,
                                  selectcolor='white',indicatoron=0,
                                  variable=self.filetype,value='*.py',
                                  relief=FLAT,borderwidth=2,
                                  command=self.updateselection,
                                  font=G___.font['B'])
    if (not save):
      self.options.b_all=Radiobutton(self.options,text='*',width=7,
                                     selectcolor='white',indicatoron=0,
                                     variable=self.filetype,value='*',
                                     relief=FLAT,borderwidth=2,
                                     command=self.updateselection,
                                     font=G___.font['B'])
    self.options.b_mll.pack(side=LEFT)
    self.options.b_adf.pack(side=LEFT)
    self.options.b_hdf.pack(side=LEFT)
    self.options.b_py.pack(side=LEFT)
    if (not save):
      self.options.b_all.pack(side=LEFT)
    self.updateselection()
  def updateselection(self):
    if (self.lockpattern): return
    dir,pat=self.get_filter()
    self.set_filter(dir,self.filetype.get())
    self.filter_command()
  def updateDirPopUp(self,wparent):
    self.mpth = Menu(wparent,tearoff=0)
    for v in self.directoriesHistory:
      self.mpth.add_radiobutton(label=v,var=self.v_path,value=v,
                                indicatoron=0,font=G___.font['E'])
    self.mpth.bind('<FocusOut>',self.popUpMenuOff)
    self.mpth.bind('<ButtonRelease-3>',self.popUpMenuOff)    
  def updateDirList(self):
    dir,pat=self.get_filter()
    if dir not in G___.directoriesHistory:
      G___.directoriesHistory+=[dir]
  def check_filter(self,dir,pat):
    if (os.path.exists(dir)): return 1
    elif s7utils.removeNotFoundDir(dir):
      G___.directoriesHistory.remove(dir)
      self.updateDirPopUp(self.wparent)
    else: return 0
  def ok_command(self):
    if (self.save): self.ok_commandsave()
    else:           self.ok_commandload()  
  def ok_commandload(self):
    file = self.get_selection()
    if not os.path.isfile(file):
      self.wparent.bell()
    else:
      self.updateDirList()
      self.quit(file)
  def ok_commandsave(self):
    file = self.get_selection()
    if os.path.exists(file):
      if os.path.isdir(file):
          self.wparent.bell()
          return
      d = Dialog(self.top,
                 title="Overwrite Existing File Question",
                 text="Overwrite existing file %r?" % (file,),
                 bitmap='questhead',
                 default=1,
                 strings=("Yes", "Cancel"))
      if d.num != 0:
        # no overwrite
        return
      else:
        os.unlink(file)
    else:
      head, tail = os.path.split(file)
      if not os.path.isdir(head):
          self.wparent.bell()
          return
    self.updateDirList()
    self.quit(file)

# -----------------------------------------------------------------------------
def s7profiledialog(master,save=0):
  if (save):
    return None
  else:
    fd=wProfileDialog(master,'pyS7: Profile Load Selection')
    (dir,profile)=fd.get()
    fd._wtop.destroy()
  return (dir,profile)

# -----------------------------------------------------------------------------
def s7filedialog(master,save=0,lock=None,pat=None):
  lck=0
  if (save):
    fd=wFileDialog(master,'pyS7: Save File Selection',save,lock,'S7save')
    if (pat==None): pat=fd.filetype.get()
    else:           lck=1
    file=fd.go(key='S7save',pattern=pat,lock=lck)
  else:
    fd=wFileDialog(master,'pyS7: Load File Selection',save,lock,'S7load')
    if (pat==None): pat=fd.filetype.get()
    else:           lck=1
    file=fd.go(key='S7load',pattern=pat,lock=lck)
  return file
