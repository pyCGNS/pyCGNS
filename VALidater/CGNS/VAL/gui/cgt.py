#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#
from Tkinter import *
from tkMessageBox import *
import ScrolledText 

#from   xml.parsers import pyexpat
#import xml.dom            as     XmlDom

import string
import commands
import os
import re

from CCCCC.gui import __cgtvid__

import CGNS
import cgtTree
import cgtHelp

fnt1=cgtTree.cgtTreeDefaultFont
fnt2=cgtTree.cgtTreeDefaultFontDialog
fnt3=cgtTree.cgtTreeDefaultFontTitle

import cgtFile

def tryFile(fname):
  try:
    f=open(fname,"r+")
    f.close()
  except:
    print "C5: No such file (or not readable): %s"%fname
  return 1

    
# =============================================================================
class BarButton(Menubutton):
  def __init__(self, master=None, fside=LEFT, **cnf):
    apply(Menubutton.__init__, (self, master), cnf)
    self.pack(side=fside)
    self.menu = Menu(self, name='menu', tearoff=0)
    self.menu.add_command(label='Open')
    self['menu'] = self.menu
    self.menu.delete(0) #?
    try:
      self.font=cnf['font']
    except KeyError:
      pass
  # --------------------------------------------------
  def addEntries(self,elist):
    for e in elist:
      if (e):
        if (len(e) == 4):
          self.menu.add_radiobutton(label=e[0],command=e[1],
				    variable=e[2],value=e[3],font=self.font)
        elif (len(e) == 3):
          self.menu.add_checkbutton(label=e[0],variable=e[1],
				    state=e[2],font=self.font)
        else:
          self.menu.add_command(label=e[0],command=e[1],font=self.font)
      else:
        self.menu.add('separator')    

# =============================================================================
class RText(Text):
  def __init__(self, master, font, **cnf):
    apply(Text.__init__, (self, master), cnf)
    
class StatusBar(Frame):
  def __init__(self, master, font):
    Frame.__init__(self, master)
    self.label = RText(self, bd=1, font=font, height=1, width=25)
    self.label.pack(fill=X)
#    self.label.configure(state=DISABLED)
  def blink(self):
    if self.__blink:
      self.__blink=0
      self.clear()
      self.label.insert(END,self.__text)
    else:
      self.__blink=1
      self.clear()
      self.label.insert(END,self.__text)
    self.label.update_idletasks()
  def set(self, format, *args):
    self.__text=format % args
    self.clear()
    self.label.insert(END,self.__text)
    self.__blink=0
    self.label.update_idletasks()
  def clear(self):
    self.label.delete("0.0",END)
    self.label.update_idletasks()
  def ok(self):
    self.clear()
    self.label.insert(END,"Ok")
    self.label.update_idletasks()

class MyDialog:
    def __init__(self, parent, title,txt):
        top = self.top = Toplevel(parent)
        Label(top, text=title).pack()
        self.e = ScrolledText.ScrolledText(top)
        self.e.insert(END, txt)
        self.e.config(state=DISABLED)
        self.e.pack(padx=5)
        b = Button(top, text="Close", command=self.ok)
        b.pack(pady=5)

    def ok(self):
        self.top.destroy()

# =============================================================================
class CGNSXmlTreeEditor(Frame):
  def lockMenu(self,menu):
    pass
  def showHelpWindow(self,title,txt):
    MyDialog(self,title,txt)
  def do_update(self):
    pass
  def do_motion(self, e):
    e.widget.select_clear('0', 'end')
    e.widget.select_set(e.widget.nearest(e.y))
  def do_leave(self, e):
    e.widget.select_clear('0', 'end')
  def helpAbout(self):
    self.showHelpWindow("About CGT",cgtHelp.helpAbout)
  def helpCGNS(self):
    self.showHelpWindow("About CGNS",cgtHelp.helpCGNS)
  def helpKeys(self):
    self.showHelpWindow("Key bindings",cgtHelp.helpKeys)
  def helpMenus(self):
    self.showHelpWindow("Menus", cgtHelp.helpMenus)
  # --------------------------------------------------
  def __loadTree(self,exp):
     self.status.grab_set()
     if (not self.filename):
       self.frame=cgtTree.cgtTree(self.frameT,None,None,
                                  root_label="NO FILE")
     else:
       fn=os.path.split(self.filename)[-1]
       self.frame=cgtTree.cgtTree(self.frameT,
                                  self.schemafilename,
                                  self.filename,
                                  root_label=fn)
     self.frame.top=self # oups !
     if exp: self.frame.expandAllTree()
     self.sbx.configure(command=self.frame.xview)
     self.sby.configure(command=self.frame.yview)
     self.frame.configure(yscrollcommand=self.sby.set)
     self.frame.configure(xscrollcommand=self.sbx.set)
     self.frame.pack(expand=1,fill=BOTH)
     self.frame.statusDisplay=1
     self.status.ok()
     self.status.grab_release()

  # --------------------------------------------------
  def selectXML(self):
    self.__filedialog=cgtFile.cgtFileDialog(self.root,
                                             'CGNS/XML File Selection',
                                             fonts=(fnt1,fnt2,fnt3))
    f=self.__filedialog.go(key="XML",pattern="*.xml")
    if f: rf=f[0]
    else: rf=None
    return rf
  # --------------------------------------------------
  def writeTreeAs(self):
    pass
  # --------------------------------------------------
  def writeTree(self):
    filexml=self.selectXML()
    s=self.frame.xml_dom_object.toxml()
    f=open(filexml,'w+')
    f.write(s)
    f.close()
  # --------------------------------------------------
  def selectEPS(self):
    self.__filedialog=cgtFile.cgtFileDialog(self.root,
                                             'CGNS/XML File Selection',
                                             fonts=(fnt1,fnt2,fnt3))
     
    f=self.__filedialog.go(key="EPS",pattern="*.eps")
    if f: rf=f[0]
    else: rf=None
    return rf
  # --------------------------------------------------
  def printAsEPS(self):
    self.update()
    epsfile=self.selectEPS()
    self.frame.postscript(file=epsfile)
  # --------------------------------------------------
  def printAllAsEPS(self):
    self.update()
    fbx=self.frame.bbox('all')
    #print fbx
    epsfile=self.selectEPS()
    self.frame.postscript(file=epsfile,width=fbx[2],height=fbx[3])
  # --------------------------------------------------
  def getFile(self):
     self.__filedialog=cgtFile.cgtFileDialog(self.root,
                                               'CGNS/XML File Selection',
                                               fonts=(fnt1,fnt2,fnt3))
     f=self.__filedialog.go(key="LOAD",pattern="*.xml,*.cgns,*.adf,*.hdf")
     del self.__filedialog
     if f:
       self.filename=f[0]
       self.status.set("Loading file > %s"%(self.filename))
       self.frame.update()
       if (self.filename):
         self.frame.destroy()
         del self.frame
         x=self.expandall.get()
         self.__loadTree(x)
       self.status.ok()
     return None
     
  # ====================================================================
  def __init__(self,master=None,schema=None,filename=None,
               showatts=0,followlinks=0,showdata=0,showname=1,showtypes=0,
               showlink=0, readonlyfile=1, showcom=0, expandAtOpen=0, clearBuffer=0, **cnf):
      cnf['borderwidth']=5
      apply(Frame.__init__, (self, master), cnf)
      self.pack(expand=1, fill=BOTH)
      self.filename=filename
      self.schemafilename=schema
      self.__filedialog=None
      self.root=master
      #
      # --- Menu bar
      self.bar = Frame(self, name='bar', relief=RAISED,
                       borderwidth=2)
      self.bar.pack(side=TOP,fill=X)
      self.bar.file = BarButton(self.bar, text='File', font=fnt2)
      self.bar.file.addEntries([['Open',self.getFile],
                                #['New',pass],
                                 None,
                                ['Save',self.writeTree],
                                ['Save As',self.writeTreeAs],
                                None,
                                ['Print all tree',self.printAllAsEPS],
                                ['Print current view',self.printAsEPS],
                                ['Export',self.printAsEPS],
                                None,
                                ['Quit',self.quit]])
      self.bar.view = BarButton(self.bar,
                                text='CGNS',
                                font=fnt2)
      self.bar.format = BarButton(self.bar,
                                  text='Editor',font=fnt2)
      self.bar.help = BarButton(self.bar, fside=RIGHT,
                                  text='Help',font=fnt2)
      self.links = IntVar(self)
      self.links.set(followlinks)
      self.showdata = IntVar(self)
      self.showdata.set(showdata)
      self.expandall = IntVar(self)
      self.expandall.set(expandAtOpen)
      self.readonly = IntVar(self)
      self.readonly.set(readonlyfile)
      self.showname = IntVar(self)
      self.showname.set(showname)
      self.showtype = IntVar(self)
      self.showtype.set(showtypes)
      self.showattributes = IntVar(self)
      self.showattributes.set(showatts)
      self.showcomment = IntVar(self)
      self.showcomment.set(showcom)
      self.showlink = IntVar(self)
      self.showlink.set(showlink)
      self.clearBuffer = IntVar(self)
      self.clearBuffer.set(clearBuffer)
      # ---
      self.bar.view.addEntries([
        ['Refresh',self.update],
        None,
        ['Ignore links',self.do_update,self.links,0],
        ['Check links',self.do_update,self.links,1],
        ['Follow links',self.do_update,self.links,2],
#        ['Data as sub-link',self.do_update,self.showdata,1],
        None,
        ['ReadOnly tree',self.readonly,ACTIVE],
        None,
        ['Save options',self.update]])
      # ---
      self.bar.format.addEntries([
        ['No data',self.do_update,self.showdata,0],
        ['Data in separate window',self.do_update,self.showdata,1],
        ['Data in tree window',self.do_update,self.showdata,2],
#        ['Data as sub-link',self.do_update,self.showdata,3],        
        None,
        ['Expand all sub-tree',self.expandall,ACTIVE],
        ['Clear buffer',self.clearBuffer,ACTIVE],
        None,
        ['Comment',self.showcomment,ACTIVE],
        ['Name',self.showname,ACTIVE],
        ['Type',self.showtype,ACTIVE],
        ['Attributes',self.showattributes,ACTIVE]])
#        ['Link flag',self.showlink,ACTIVE]])
      self.bar.help.addEntries([
        ['CGNS Tree Editor - v%s'%(__cgtvid__),
         self.helpAbout],                  
        None,                  
        ['CGT read and generate fonctions',self.helpAbout],
        ['About CGNS',self.helpCGNS],
        None,
        ['Menus',self.helpMenus],
        ['Editor and key bindings',self.helpKeys]])

      # --- Top Frame ----------------------------------------
      self.frameT = Frame(self, relief=RAISED, borderwidth=2)
      self.frameT.pack(expand=1, fill=BOTH)
      self.frameT.grid_rowconfigure(0, weight=1)
      self.frameT.grid_columnconfigure(0, weight=1)
      self.sby=Scrollbar(self.frameT)
      self.sby.grid(row=0, column=1, sticky='ns')
      self.sby.pack(side=RIGHT,fill=BOTH)
      self.sbx=Scrollbar(self.frameT, orient=HORIZONTAL)
      self.sbx.grid(row=1, column=0, sticky='ns')
      self.sbx.pack(side=BOTTOM,fill=BOTH)

      self.status=StatusBar(self.root,font=fnt2)
      self.status.pack(side=BOTTOM, fill=X)
      self.frameT.status=self.status

      if (filename): self.status.set("Loading file : %s"%(self.filename))
      self.__loadTree(expandAtOpen)
      
      self.frameT.pack(expand=1,fill=BOTH)
      self.header = StringVar(self)
      self.do_update()

# --------------------------------------------------------------------
def usage():
  usage="""
cgt v%s - CGNS/XML file editor
usage: cgt [option] <filename>

-a             : Show attributes
-l             : Show link paths
-d             : Show data
-n             : Do NOT show name
-t             : Show type
-c             : Show comment

-s <schema>    : Specify schema name
-x             : Expand all tree
-W             : Read/write file
-L <depth>     : Follow links

"""%__cgtvid__
  print usage
  import sys
  sys.exit(1)
  
def cgt(args):
  import getopt
  filename=None
  schemafilename=None
  showAttributesFlag=0
  showDataFlag=0
  showLinkFlag=0
  showNameFlag=1
  showTypeFlag=0
  showComment=0
  followLinkDepth=0
  readOnlyFlag=1
  exp=0
  clrbf=0
  try:
    (opts,files)=getopt.getopt(args[1:],"bRaldnts:xL:")
    if (files): filename=files[0]
    for opt in opts:
      if (opt[0] == '-W'): readOnlyFlag=0
      if (opt[0] == '-d'): showDataFlag=1
      if (opt[0] == '-a'): showAttributesFlag=1
      if (opt[0] == '-l'): showLinkFlag=1
      if (opt[0] == '-n'): showNameFlag=0
      if (opt[0] == '-t'): showTypeFlag=1
      if (opt[0] == '-c'): showComment=1
      if (opt[0] == '-L'): followLinkDepth=opt[1]
      if (opt[0] == '-s'): schemafilename=opt[1]
      if (opt[0] == '-x'): exp=1
      if (opt[0] == '-b'): clrbf=1
  except getopt.error,e: usage()

  root = Tk()
  if filename:
    if (tryFile(filename)):
      CXTE = CGNSXmlTreeEditor(root,
                               schema=schemafilename,
                               filename=filename,
                               showatts=showAttributesFlag,
                               followlinks=followLinkDepth,
                               showdata=showDataFlag,
                               showname=showNameFlag,
                               showtypes=showTypeFlag,
                               showlink=showLinkFlag,
                               readonlyfile=readOnlyFlag,
                               showcom=showComment,
                               expandAtOpen=exp,
                               clearBuffer=clrbf
                               )
    else:
      return None
  else:
    CXTE = CGNSXmlTreeEditor(root,None)

  CXTE.winfo_toplevel().title('CGNS/XML Editor [v%s]'%__cgtvid__)
  CXTE.winfo_toplevel().minsize(1, 1)
  CXTE.mainloop()
  return None

