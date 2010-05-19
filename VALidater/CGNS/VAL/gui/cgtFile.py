"""File selection dialog classes for Tk.

This file was expanded from the original FileDialog.py in python's
lib-tk directory in order to support the selection of multiple files
simultaneously, to support filename completion, and to support filtering
of the files with multiple comma-separated filename extensions.

Classes in this file:

  FileDialog - the base class, draws a Tk file selection box
  LoadFileDialog - specialized for loading files
  SaveFileDialog - specialized for saving files

Also see the documentation below for each class, and the short example
at the end of this file.
"""

from Tkinter    import *
from FileDialog import *

import os, time, string
import fnmatch

dialogstates = {}

class cgtFileDialog(Dialog):

    # default title and "OK button" text
    title = "File Selection Dialog"
    oktext = " OK "
    
    def __init__(self, master, title=None, oktext=None,
                 fonts=(('fixed', 10),('arial', 10),('fixed', 12, 'bold'))):
        if title is None: title = self.title
        if oktext is None: oktext = self.oktext
        self.master = master
        self.directory = None
        self.modal = 0
        self.key = None
        self.oldselection = None

        self.fontX = fonts[0]
        self.fontJ = fonts[1]
        self.fontT = fonts[2]

        # the window itself
        self.top = Toplevel(master)
        self.top.title(title)
        self.top.iconname(title)

        # build the interface from the bottom to the top

        # frame for the bottom row of buttons
        self.botframe = Frame(self.top)
        self.botframe.pack(side=BOTTOM, fill=X)

        self.cancel_button = Button(self.botframe,
                                    text="Cancel", font=self.fontJ,
                                    command=self.cancel_command,width=8)
        self.cancel_button.pack(side=LEFT)

        self.selection = Entry(self.botframe, font=self.fontX)
        self.selection.pack(side=LEFT, fill=X, expand=TRUE)
        self.selection.bind('<Return>', self.completion_event)
        self.selection.bind('<KeyPress>', self.markdirty)

        self.ok_button = Button(self.botframe,
                                 text=oktext, font=self.fontJ,
                                 command=self.ok_command,width=8)
        self.ok_button.pack(side=LEFT)

        # frame for the upper row of buttons
        self.topframe = Frame(self.top)
        self.topframe.pack(side=BOTTOM, fill=X)

        self.filterbutton = Button(self.topframe,
                                   text="Filter", font=self.fontJ,
                                   command=self.filter_command,width=8)
        self.filterbutton.pack(side=LEFT)

        self.filter = Entry(self.topframe, font=self.fontX)
        self.filter.pack(side=LEFT, fill=X, expand=TRUE)
        self.filter.bind('<Return>', self.filter_command)

#        self.select_all_button = Button(self.topframe,
#                                        text="Select All", font=self.fontJ,
#                                        command=self.select_all_command,width=8)
#        self.select_all_button.pack(side=LEFT)

        self.check_type_button = Button(self.topframe,
                                        text="Check type", font=self.fontJ,
                                        command=self.check_type_command,width=8)
#        self.select_all_button.pack(side=LEFT)
        self.check_type_button.pack(side=LEFT)
        # frame for the boxes that contain information about the files
        self.midframe = Frame(self.top)
        self.midframe.pack(expand=YES, fill=BOTH)

        # directory list box
        self.dirsframe = Frame(self.midframe)
        self.dirsframe.pack(side=LEFT, expand=YES, fill=BOTH)
        self.dirslabel = Label(self.dirsframe,text="Directories",font=self.fontT)
        self.dirslabel.pack(side=TOP, fill=X)
        self.dirsbar = Scrollbar(self.dirsframe)
        self.dirsbar.pack(side=LEFT, fill=Y)
        self.dirs = Listbox(self.dirsframe, exportselection=0, font=self.fontX,
                            width=18)
        self.dirs.pack(side=LEFT, expand=YES, fill=BOTH)
        self.dirsbar.config(command=(self.dirs, 'yview'))
        self.dirs.config(yscrollcommand=(self.dirsbar, 'set'))
        btags = self.dirs.bindtags()
        self.dirs.bindtags(btags[1:] + btags[:1])
        self.dirs.bind('<ButtonRelease-1>', self.dirs_select_event)

        # the primary field: file list
        self.filesframe = Frame(self.midframe)
        self.filesframe.pack(side=LEFT, expand=YES, fill=BOTH)
        self.fileslabel = Label(self.filesframe,text="Files",font=self.fontT)
        self.fileslabel.pack(side=TOP, fill=X)
        self.files = Listbox(self.filesframe, exportselection=0,
                             selectmode="multiple", font=self.fontX,
                             width=30)
        self.files.pack(side=LEFT, expand=YES, fill=BOTH)
        self.files.config(yscrollcommand=self.scrollfiles)

        # the second field: file sizes
        self.sizesframe = Frame(self.midframe)
        self.sizesframe.pack(side=LEFT, expand=YES, fill=BOTH)
        self.sizeslabel = Label(self.sizesframe,text="Size",font=self.fontT)
        self.sizeslabel.pack(side=TOP, fill=X)
        self.sizes = Listbox(self.sizesframe, exportselection=0,
                             yscrollcommand=self.scrollfiles, takefocus=FALSE,
                             selectmode="multiple", font=self.fontX, width=12)
        self.sizes.pack(side=LEFT, expand=YES, fill=BOTH)

        # the third field: file modified times
        self.datesframe = Frame(self.midframe)
        self.datesframe.pack(side=LEFT, expand=YES, fill=BOTH)
        self.dateslabel = Label(self.datesframe,text="Date",font=self.fontT)
        self.dateslabel.pack(side=TOP, fill=X)
        self.dates = Listbox(self.datesframe, exportselection=0,
                             yscrollcommand=self.scrollfiles, takefocus=FALSE,
                             selectmode="multiple", font=self.fontX, width=21)
        self.dates.pack(side=LEFT, expand=YES, fill=BOTH)

        # bind the same button/keys to each field
        self.fields = [self.files, self.sizes, self.dates]
        for field in self.fields:
            btags = field.bindtags()
            field.bindtags(btags[1:] + btags[:1])
            field.bind('<ButtonPress-1>', self.files_start_event)
            field.bind('<B1-Motion>', self.files_drag_event)
            field.bind('<ButtonRelease-1>', self.files_select_event)
            field.bind('<Double-ButtonRelease-1>', self.files_double_event)

        # and make one scrollbar to control the lot of them
        self.filesbar = Scrollbar(self.datesframe)
        self.filesbar.pack(side=LEFT, fill=Y)
        self.filesbar.config(command=self.scrollfilesbar)

        # other misc tk stuff that was in the original FileDialog.py
        self.top.protocol('WM_DELETE_WINDOW', self.cancel_command)
        # XXX Are the following okay for a general audience?
        self.top.bind('<Alt-w>', self.cancel_command)
        self.top.bind('<Alt-W>', self.cancel_command)

    def go(self, dir_or_file=os.curdir, pattern="*", default="", key=None,
           modal=1):
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
        self.top.grab_set()
        self.how = None
        self.modal = modal
        self.key = key
        if modal:
            self.master.mainloop()          # Exited by self.quit(how)
            if key:
                directory, pattern = self.get_filter()
                if self.how:
                    try:
                        directory = os.path.dirname(self.how)
                    except:
                        try:
                            directory = os.path.dirname(self.how[0])
                        except:
                            pass
                dialogstates[key] = directory, pattern
            self.top.destroy()
            return self.how

    def getfiledir(self):
        return self.directory

    def quit(self, how=None):
        self.how = how
        if self.modal:
            self.master.quit()              # Exit mainloop()
        else:
            self.top.destroy()

    def scrollfiles(self, *args):
        apply(self.filesbar.set,args)
        for field in self.fields:
            apply(field.yview,('moveto',args[0]))

    def scrollfilesbar(self, *args):
        for field in self.fields:
            apply(field.yview,args)

    def dirs_select_event(self, event):
        dir, pat = self.get_filter()        
        subdir = self.dirs.get(ACTIVE)
        if not (os.path.splitdrive(dir)[1] == os.sep and \
                (subdir == os.pardir or subdir == os.pardir+os.sep) or \
                os.name != 'posix' and os.path.ismount(dir)):
            dir = os.path.normpath(os.path.join(self.directory, subdir))
        try:
            names = os.listdir(dir)
        except os.error:
            self.master.bell()
            return
        self.set_filter(dir, pat)
        self.filter_command()

    def files_double_event(self, event):
        self.ok_command()

    def files_start_event(self, event):
        if int(event.state) & 1 == 0:
            for field in self.fields:
                field.selection_anchor('@%i,%i' % (event.x,event.y))
            self.shiftstart = '@%i,%i' % (event.x,event.y)
        else:
            try:
                for field in self.fields:
                    field.selection_anchor(self.shiftstart)
            except AttributeError:
                for field in self.fields:
                    field.selection_anchor('@%i,%i' % (event.x,event.y))
        self.selectmode = 'select'
        if int(event.state) & 4 == 0:
            for field in self.fields:
                field.selection_clear(0, END)
        elif self.files.selection_includes('@%i,%i' % (event.x,event.y)):
            self.selectmode = 'clear'
        self.saveselection = list(self.files.curselection())
        if self.selectmode == 'clear':
            for field in self.fields:
                field.selection_clear(ANCHOR,'@%i,%i' % (event.x,event.y))
        else:
            for field in self.fields:
                field.selection_set(ANCHOR,'@%i,%i' % (event.x,event.y))

    def files_drag_event(self, event):
        for field in self.fields:
            field.selection_clear(0, END)
        for i in self.saveselection:
            for field in self.fields:
                field.selection_set(i,i)
        if self.selectmode == 'clear':
            for field in self.fields:
                field.selection_clear(ANCHOR,'@%i,%i' % (event.x,event.y))
        else:
            for field in self.fields:
                field.selection_set(ANCHOR,'@%i,%i' % (event.x,event.y))

    def files_select_event(self, event):
        file = self.files.get(ACTIVE)
        self.set_selection(file)

#    def select_all_command(self):
#        for field in self.fields:
#            field.selection_set(0, END)

    def check_type_command(self):
        files = self.get_selection_list()
        if len(files) == 0:
            files = [self.get_selection()]
        for cfile in files:
            print "CHECK ",cfile

    def ok_command(self):
        files = self.get_selection_list()
        if len(files) == 0:
            files = [self.get_selection()]
        self.quit(files)

    def fnmatch(self, name, pat):
        if not pat:
            return 1
        for item in string.split(pat,','):
            if fnmatch.fnmatch(name, item):
                return 1
        return 0

    def markdirty(self, event=None):
        self.oldselection = None

    def completion_event(self, event=None):
        dir, pat = self.get_filter()
        dir, comp = os.path.split(self.get_selection())
        try:
            names = os.listdir(dir)
        except os.error:
            self.master.bell()
            return
        self.set_filter(dir, pat)
        names.sort()
        subdirs = [os.pardir]
        matchingfiles = []
        compnames = []
        
        if comp == '':
            for name in names:
                fullname = os.path.join(dir, name)
                if os.path.isdir(fullname):
                    subdirs.append(name)
                    compnames.append(name)
                elif self.fnmatch(name, pat):
                    matchingfiles.append(name)
                    compnames.append(name)
        else:
            n = len(comp)
            for name in names:
                if name[0:n] == comp:
                    if os.path.isdir(os.path.join(dir,name)):
                        subdirs.append(name)
                        compnames.append(name)
                    elif self.fnmatch(name, pat):
                        matchingfiles.append(name)
                        compnames.append(name)
            if len(compnames) != 0:
                comp = os.path.commonprefix(compnames)

        if len(compnames) == 1:
            if os.path.isdir(os.path.join(dir,compnames[0])):
                try:
                    names = os.listdir(os.path.join(dir,compnames[0]))
                    names.sort()
                    dir = os.path.join(dir,compnames[0])
                    comp = ''
                    self.directory = dir
                    subdirs = [os.pardir]
                    matchingfiles = []
                    for name in names:
                        fullname = os.path.join(dir, name)
                        if os.path.isdir(fullname):
                            subdirs.append(name)
                        elif self.fnmatch(name, pat):
                            matchingfiles.append(name)
                except os.error:
                    self.master.bell()

        oldsubdirs = list(self.dirs.get(0,END))
        oldfiles = list(self.files.get(0,END))
                
        self.update_fields(subdirs,matchingfiles)
        self.set_selection(comp)
        self.set_filter(dir, pat)

        for i in xrange(len(oldsubdirs)):
            if oldsubdirs[i][-1] == os.sep:
                oldsubdirs[i] = oldsubdirs[i][0:-1]

        selection = self.get_selection()
        oldselection = self.oldselection
        self.oldselection = selection

        if oldselection == selection and \
           oldsubdirs == subdirs and \
           oldfiles == matchingfiles and \
           (not os.path.isdir(selection)) and \
           self.fnmatch(os.path.split(selection)[-1],pat):
            self.ok_command()
            
    def filter_command(self, event=None):
        dir, pat = self.get_filter()
        if self.key:
            dialogstates[self.key] = dir, pat
        try:
            names = os.listdir(dir)
        except os.error:
            self.master.bell()
            return
        self.set_filter(dir, pat)
        names.sort()
        subdirs = [os.pardir]
        matchingfiles = []
        for name in names:
            fullname = os.path.join(dir, name)
            if os.path.isdir(fullname):
                subdirs.append(name)
            elif self.fnmatch(name, pat):
                matchingfiles.append(name)
        self.update_fields(subdirs,matchingfiles)
        head, tail = os.path.split(self.get_selection())
        if tail == os.curdir: tail = ''
        self.set_selection(tail)

    def update_fields(self, dirs, files):
        self.dirs.delete(0, END)
        for name in dirs:
            self.dirs.insert(END, name+os.sep)
        for field in self.fields:
            field.delete(0, END)
        for name in files:
            fullpath = os.path.join(self.directory,name)
            self.files.insert(END, name)
            try:
                self.sizes.insert(END, "%11d" % (os.path.getsize(fullpath),))
            except:
                self.sizes.insert(END, "%11s" % ("???",))
            try:
                self.dates.insert(END, time.strftime("%d %b %Y %H:%M:%S",
                             time.localtime(os.path.getmtime(fullpath))))
            except:
                self.dates.insert(END, "%s %s %s %s:%s:%s" %
                              ("??", "???", "????", "??", "??", "??"))
                
    def get_filter(self):
        filter = self.filter.get()
        if filter == '':
            filter = "*"
        return (os.path.expanduser(self.directory),filter)

    def get_selection(self):
        file = self.selection.get()
        file = os.path.expanduser(file)
        return file

    def get_selection_list(self):
        filelist = list(self.files.curselection())
        for i in xrange(len(filelist)):
            file = self.files.get(filelist[i])
            file = os.path.join(self.directory, file)
            filelist[i] = os.path.expanduser(file)
        return filelist

    def cancel_command(self, event=None):
        self.quit()

    def set_filter(self, dir, pat):
        try:
            names = os.listdir(dir)
        except os.error:
            self.master.bell()
            return
        if not (os.path.isabs(dir) or os.path.splitdrive(dir)[0]):
            try:
                pwd = os.getcwd()
                if os.name == 'posix':
                    split = string.split(pwd,os.sep)
                    if len(split) > 1 and (split[1] == 'net' or \
                                           split[1] == 'hosts' or \
                                           split[1] == '.automount'):
                        if split[1] == '.automount':
                            testpwd = string.join(split[0:1]+split[4:],os.sep)
                        else:
                            testpwd = string.join(split[0:1]+split[3:],os.sep)
                        try:
                            if os.path.samefile(pwd,testpwd):
                                pwd = testpwd
                        except:
                            pass
            except os.error:
                pwd = None
            if pwd:
                dir = os.path.join(pwd, dir)
                dir = os.path.normpath(dir)
        self.filter.delete(0, END)
        self.filter.insert(END, pat or "*")
        self.directory = dir

    def set_selection(self, file):
        self.selection.delete(0, END)
        self.selection.insert(END, os.path.join(self.directory, file))
        float,self.selection.xview(END)


class LoadFileDialog(FileDialog):

    """File selection dialog which checks that the file exists."""

    title = "Load File Selection Dialog"
    oktext = "Load"

    def ok_command(self):
        file = self.get_selection()
        if not os.path.isfile(file):
            self.master.bell()
        else:
            self.quit(file)


class SaveFileDialog(FileDialog):

    """File selection dialog which checks that the file may be created."""

    title = "Save File Selection Dialog"
    oktext = "Save"

    def ok_command(self):
        file = self.get_selection()
        if os.path.exists(file):
            if os.path.isdir(file):
                self.master.bell()
                return
            d = Dialog(self.top,
                       title="Overwrite Existing File Question",
                       text="Overwrite existing file %s?" % `file`,
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


def testFileListDialog():
    """Simple test program."""
    root = Tk()
    root.withdraw()
    fd = FileDialog(root)
    files = fd.go(key="test",pattern="*.jpg,*.png,*.tif,*.gif")
    root.update()
    print files

if __name__ == '__main__':
    testFileListDialog()

