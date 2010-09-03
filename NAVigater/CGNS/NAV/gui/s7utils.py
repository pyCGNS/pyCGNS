#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import Tkinter
Tkinter.wantobjects=0 #necessary for tk-8.5 and some buggy tkinter installs
from Tkinter import *
import os
import string

import CGNS.NAV.supervisor.s7error as ERR

import s7globals
G___=s7globals.s7G

import numpy as Num

# --------------------------------------------------------------------
def wIconFactory():
  if (G___.iconStorage != {}): return G___.iconStorage
  for iconKey in G___.iconKeys:
    if (not os.path.exists('%s/%s.gif'%(G___.iconDir,iconKey))):
      raise ERR.S7Exception(10,'%s/%s.gif'%(G___.iconDir,iconKey))
    G___.iconStorage[iconKey]=PhotoImage(file='%s/%s.gif'%
                                         (G___.iconDir,iconKey))
  return G___.iconStorage

def canBeShown(data):
  if (data == None ): return 1
  if (type(data) in [type(""),type(1.2),type(1)]): return 0
  if (     (type(data) == type(Num.ones((1,))))
       and (data.size < G___.maxDisplaySize)): return 1
  return 0

def toBeShown(data):
  if (not canBeShown(data)): return "???1"
  if (data == None ):        return ""
  else:                      return showOneArray(data)

def showOneArray(data):
  return str(data)

def showOneFloat(f):
  return string.rstrip('%8.6f'%f,'0')
  
def showOneDouble(f):
  return string.rstrip('%8.6f'%f,'0')
  
def showOneInteger(f):
  return string.rstrip('%d'%f,'0')
  
def showOneLong(f):
  return string.rstrip('%d'%f,'0')
  
def showFloatArrayAsList(data):
  return __showFloatArrayAsList(data).replace("'","")[:-1]

def __showFloatArrayAsList(ldata):
  if (    (type(ldata)==type([]))
      and (len(ldata)>0)
      and (type(ldata[0])==type([]))):
    sdata='['
    for lldata in ldata:
      sdata+=str(__showFloatArrayAsList(lldata))
    sdata=sdata[:-1]+'],'
  elif (type(ldata[0])==type(1.0)):
    sdata=str(map(lambda x: showOneFloat(x),ldata))+','
  elif (type(ldata[0])==type(1)):
    sdata=str(map(lambda x: '%d'%x,ldata))+','
  else:
    sdata=''
  return sdata

def getFileSize(f):
  import os
  st=os.stat(f)
  return st.st_size

# --------------------------------------------------
def childNames(node):
  cn=[]
  for n in node[2]:
    cn+=[n[0]]
  return cn

# --------------------------------------------------
def generateName(noderoot,nodeappend):
  cn=childNames(noderoot)
  nn=ng=nodeappend[0]
  count=0
  try:
    if ((ng in cn) and ('#' in ng)):
      count=int(ng.split('#')[-1])
      nn=string.join(ng.split('#')[:-1],'#')
  except ValueError:
    nn=ng
  while True:
    if (ng not in cn): return ng
    ng=nn+'#%.3d'%count
    count+=1

# --------------------------------------------------
class msginfo:
  def __init__(self,title,msg,other=None):
    self._result = None
    def wtopIsTrue():
      self._result=1
      wtop.destroy()
    def wtopIsFalse():
      self._result=0
      wtop.destroy()
    wtop=Toplevel()
    wtop.title(title)
    f=Frame(wtop)
    b=Button(f,text=msg,relief=GROOVE,justify=LEFT,
             font=G___.font['L'],command=wtopIsTrue,default=ACTIVE)
    b.grid(sticky=N+E+W)
    if (other):
      c=Button(f,text=other,relief=FLAT,justify=CENTER,fg='gray',
             font=G___.font['L'],command=wtopIsFalse)
      c.grid(sticky=S+E+W)
    f.grid()
    f.bind("<Return>", wtopIsTrue)
    f.bind("<Escape>", wtopIsFalse)
    wtop.protocol("WM_DELETE_WINDOW", wtopIsFalse)
    wtop.wait_visibility()
    wtop.grab_set()
    wtop.wait_window(wtop)
  def result(self):
    return self._result

# --------------------------------------------------
def onexit():
  closeWarning()

def mainCloseWarning():
  return msginfo("CGNS.NAV: Warning",
  "Please use Quit icon (Control Panel) to close CGNS.NAV.").result()
def closeWarning():
  return msginfo("CGNS.NAV: Warning",
                 "Please use Control Panel to close views.").result()
def forceNoRecursionWarning():
  return msginfo("CGNS.NAV: Warning",
                 "CGNS.NAV forces [No Recursion] flag.").result()
def operateWarning():
  return msginfo("CGNS.NAV: Warning",
"""Operate view is already open.
Close operate view to perform actions on another view.""").result()
def fileWarning(ext):
  return msginfo("CNS.NAV: Error",
                 "Don't know how to read [%s] files..."%ext).result()
def badFileError(name):
  return msginfo("CNS.NAV: Error",
                 "Something wrong with file\n%s"%name).result()
def renameError(name):
  return msginfo("CGNS.NAV: Error",
                 "Name [%s] already exists"%name).result()
def queryViewCloseError():
  return msginfo("CGNS.NAV: Error",
                 "Please close first Selection List View").result()
def cutError():
  return msginfo("CGNS.NAV: Error",
                 "Cannot cut root node").result()
def pasteBrotherError():
  return msginfo("CGNS.NAV: Error",
                 "Cannot paste as root node brother").result()
def importCGNSWarning(ext):
  return msginfo("CGNS.NAV: Error",
  "Cannot import pyCGNS module\nThis is required for %s files"%ext).result()

def importProfileWarning(mod,msg):
  return msginfo("CGNS.NAV: Error",
  "Cannot import %s profile\nError message is:\n%s"%(mod,msg)).result()

def spaceInFileName(fnm):
  return msginfo("CGNS.NAV: Error",
                 "Pattern name contains <space> char: '%s'"%fnm).result()

def importCGNSFunction(ext):
  return msginfo("CGNS.NAV: Error",
                 "Cannot import pyCGNS module\nRequired for %s"%ext).result()

def copyNameCollision(name):
  return msginfo("CGNS.NAV: Error",
  "Inserted tree has same name as existing child\n[%s]"%name).result()

def snapShotWarning(file):
  return msginfo("CGNS.NAV: Info","Snapshot into file\n[%s]"%file).result()

def saveFileWarning(file):
  return msginfo("CGNS.NAV: Info",
                 "Tree saved into file\n%s"%file).result()

def profileDirNotFound(prf):
  return msginfo("CGNS.NAV: Error",
                 "Profile dir doesn't exist:\n%s"%prf).result()

def createProfileDirInit(prof):
  return msginfo("CGNS.NAV: Confirm",
                 "No __init__.py in profile dir '%s'.\nDo you want me to create it?"%prof).result()

def removeNotFoundDir(dir):
  return msginfo("CNS.NAV: Confirm",
                 "Directory '%s' not found.\nDo you want to remove it in history list?"%dir,"no, keep directory in history list").result()

def cutRootWarning():
  return msginfo("CGNS.NAV: Confirm",
                 "You selected Root node to Cut! Delete all tree?",
                 "cancel").result()

def bigFileWarning(flag):
  msg="The file you want to load looks quite big!\n"
  if (flag):
    msg+="You have set the [No Data] options, but\n"
  else:
    msg+="Maybe you could set the [No Data] options,\n"
  msg+="do you really want to load the file?"
  return msginfo("CGNS.NAV: Confirm",msg,"Don't load the file").result()

def patternOverwrite(file):
  return msginfo("CGNS.NAV: Confirm",
"""Pattern '%s' already exists! Do you really want to replace it ?"""%file,
"cancel").result()

def duplicatedPattern(pat):
  return msginfo("CGNS.NAV: Confirm",
"""Pattern '%s' already imported!
CGNS.NAV skips this pattern,
do you want to continue to import profile?"""%pat,"cancel").result()

def leaveCheck():
  return msginfo("CGNS.NAV: Confirm","Close all CGNS.NAV views and leave",
                 "cancel").result()

def saveCheck():
  return msginfo("CGNS.NAV: Confirm",
                 "Tree is modified, close without save ?","cancel").result()

def overwriteCheck():
  return msginfo("CGNS.NAV: Confirm",
                 "Tree already exist, overwrite ?","cancel").result()

def noDataSave():
  return msginfo("CGNS.NAV: Confirm",
"""This CGNS tree has been read WITHOUT large DataArrays.
Your save will store data with ZERO values.
Click if you really want to overwrite with ZEROs.
Select [cancel] if you do not want to save file""","cancel").result()

def updateTreeLoad():
  return msginfo("CGNS.NAV: Confirm",
"""You have already read this file,
do you want to force load and replace all these tree views?""",
                 "cancel").result()

# --------------------------------------------------
def printNode(node):
  if (node[0]==None):  s='[%s,%s,['%(node[0],repr(node[1]))
  else:                s='["%s",%s,['%(node[0],repr(node[1]))
  for cnode in node[2]:
    s+=printNode(cnode)
  if (node[2]): s=s[:-1]
  if (node[3]==None):  s+='],%s],'%(node[3])
  else:                s+='],"%s"],'%(node[3])
  return s

# --------------------------------------------------
def asFileString(tree):
  Num.set_printoptions(threshold=sys.maxint)
  from time import localtime, strftime
  t=strftime("%H%M%S", localtime())
  s ='# Saved by pyS7\n'
  s+='# Date: %s\n'%t
  s+='from numpy import *\n'
  s+='data='
  s+=printNode(tree)
  s=s[:-1]
  return s

# --------------------------------------------------
def dumpWindow(frame,file=None):
  import os
  if (not file):
    from time import localtime, strftime
    t=strftime("%H%M%S", localtime())
    file='/tmp/s7-%s-%s.pnm'%(os.getpid(),t)
  os.system('xwd -id %s -nobdrs -silent >%s'%(frame.winfo_id(),file))
  snapShotWarning(file)
  
# --------------------------------------------------
def timeTag():
  from time import gmtime, strftime
  tag=strftime("%Y-%m-%d %H:%M:%S", gmtime())
  return tag

# -----------------------------------------------------------------------------
def operate_sort_list(event):
  listbox=event.widget
  if (not listbox.sort_init):
    listbox.sortorder_flags={}
    for col in listbox.sort_allowed:
      listbox.sortorder_flags[col]='increasing'
    listbox.sort_init=1
  if (event.column not in listbox.sort_allowed): return
  listbox.sort(column=event.column,mode=listbox.sortorder_flags[event.column])
  if listbox.sortorder_flags[event.column] == 'increasing':
    listbox.column_configure(listbox.column(event.column),arrow='up')
    listbox.sortorder_flags[event.column] = 'decreasing'
  else:
    listbox.column_configure(listbox.column(event.column),arrow='down')
    listbox.sortorder_flags[event.column] = 'increasing'

# --------------------------------------------------
