# -----------------------------------------------------------------------------
# pyS7 - CGNS/SIDS editor
# ONERA/DSNA - marc.poinot@onera.fr
# pyS7 - $Rev: 72 $ $Date: 2009-02-10 15:58:15 +0100 (Tue, 10 Feb 2009) $
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source
# tree for license information.
import Tkinter
Tkinter.wantobjects=0 #necessary for tk-8.5 and some buggy tkinter installs
from Tkinter import *
import tkMessageBox
import os
import string

import S7.supervisor.s7error     as E

import s7globals
G___=s7globals.s7G

import numpy as Num

# --------------------------------------------------------------------
def wIconFactory():
  if (G___.iconStorage != {}): return G___.iconStorage
  for iconKey in G___.iconKeys:
    if (not os.path.exists('%s/%s.gif'%(G___.iconDir,iconKey))):
      raise E.S7Exception(10,'%s/%s.gif'%(G___.iconDir,iconKey))
    G___.iconStorage[iconKey]=PhotoImage(file='%s/%s.gif'%
                                         (G___.iconDir,iconKey))
  return G___.iconStorage

def canBeShown(data):
    if ( data == None ):
        return 1
    if ( (type(data) == type("")) and (len(data)<G___.maxDisplaySize) ):
        return 1
    if ( (type(data) == type(1.2))):
        return 1
    if ( (type(data) == type(1))):
        return 1
    if ( (type(data) == type(Num.ones((1,)))) ):
        if (data.size < G___.maxDisplaySize): return 1
    if ((type(data) == type([])) and (len(data))): # oups !
      if (type(data[0]) == type("")):   return 1
      if (type(data[0]) == type(0)):    return 1
      if (type(data[0]) == type(0.0)):  return 1
    return 0

def toBeShown(data):
    if (data == None ): return ""
    if ((type(data) == type("")) and (len(data)<G___.maxDisplaySize) ):
      return str(data)
    if ((type(data) == type(1.2))): return str(data)
    if ((type(data) == type(1))):   return str(data)
    if (     (type(data) == type(Num.ones((1,))))
         and (data.size<G___.maxDisplaySize) ):
      if (data.dtype.char in ['S','c']):
        return data.tostring(order='F')
      if (data.size == 1):
        if (data.dtype == Num.dtype('float32')):
          return showOneFloat(data.flat[0])
        return str(data.flat[0])
      return showFloatArrayAsList(data.tolist())
    if ((type(data) == type([])) and (len(data))): # oups !
      return str(Num.asarray(data))
    return str(data)

def showOneFloat(f):
  return string.rstrip('%8.6f'%f,'0')
  
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
def onexit():
  closeWarning()

def mainCloseWarning():
  tkMessageBox.showwarning("pyS7: Warning",
                           "Please use S7 Control Panel\nQuit icon to close S7.")
def closeWarning():
  tkMessageBox.showwarning("pyS7: Warning",
                           "Please use S7 Control Panel\nto close views.")
def forceNoRecursionWarning():
  tkMessageBox.showwarning("pyS7: Warning",
                           "S7 forces [No Recursion] flag.")
def operateWarning():
  tkMessageBox.showwarning("pyS7: Warning",
                           "Operate view is already open.\nClose operate view to perform actions on another view.")
def fileWarning(ext):
  tkMessageBox.showerror("pyS7: Error",
                         "Don't know how to read [%s] files..."%ext)
def badFileError(name):
  tkMessageBox.showerror("pyS7: Error",
                         "Something wrong with file\n%s"%name)
def renameError(name):
  tkMessageBox.showerror("pyS7: Error",
                         "Name [%s] already exists"%name)
def queryViewCloseError():
  tkMessageBox.showerror("pyS7: Error",
                         "Please close first Selection List View")
def cutError():
  tkMessageBox.showerror("pyS7: Error",
                         "Cannot cut root node")
def pasteBrotherError():
  tkMessageBox.showerror("pyS7: Error",
                         "Cannot paste as root node brother")
def importCGNSWarning(ext):
  tkMessageBox.showerror("pyS7: Error",
                         "Cannot import pyCGNS module\nThis is required for %s files"%ext)

def importProfileWarning(mod,msg):
  tkMessageBox.showerror("pyS7: Error",
                         "Cannot import %s profile\nError message is:\n%s"\
                         %(mod,msg))

def spaceInFileName(fnm):
  tkMessageBox.showerror("pyS7: Error",
                         "Pattern name contains <space> char: '%s'"%fnm)

def importCGNSFunction(ext):
  tkMessageBox.showerror("pyS7: Error",
                         "Cannot import pyCGNS module\nRequired for %s"%ext)

def copyNameCollision(name):
  tkMessageBox.showerror("pyS7: Error",
                         "Copy tree has same name as existing child [%s]"%name)

def snapShotWarning(file):
  tkMessageBox.showinfo("pyS7: Info",
                        "Snapshot into file\n[%s]"%file)

def saveFileWarning(file):
  tkMessageBox.showinfo("pyS7: Info",
                        "Tree saved into file\n[%s]"%file)

def profileDirNotFound(prf):
  tkMessageBox.showerror("pyS7: Error",
                         "Profile dir doesn't exist:\n%s"%prf)

def createProfileDirInit(prof):
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "No __init__.py in profile dir '%s'.Do you want me to create it?"%prof)

def removeNotFoundDir(dir):
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "Directory '%s' not found. Do you want to remove it in history list?"%dir)

def cutRootWarning():
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "You selected Root node to Cut! Delete all tree?")

def bigFileWarning(flag):
  msg="The file you want to load looks quite big! "
  if (flag):
    msg+="You have set the [No Data] options, but "
  else:
    msg+="Maybe you could set the [No Data] options, "
  msg+="do you really want to load the file?"
  return tkMessageBox.askyesno("pyS7: Confirm",msg)

def patternOverwrite(file):
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "Pattern '%s' already exists! Do you really want to replace it ?"%file)

def duplicatedPattern(pat):
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "Pattern '%s' already imported! S7 skips this pattern, do you want to continue to import profile?"%pat)

def leaveCheck():
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "Close all S7 views and leave ?")

def saveCheck():
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "Tree is modified, close without save ?")

def overwriteCheck():
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "Tree already exist, overwrite ?")

def noDataSave():
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "This CGNS tree has been read WITHOUT large DataArrays. Your save will store data with ZERO values. Select OK if you really want to overwrite with ZEROs.")

def updateTreeLoad():
  return tkMessageBox.askokcancel("pyS7: Confirm",
                                  "You have already read this file, do you want to force load and replace all these tree views?")

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
