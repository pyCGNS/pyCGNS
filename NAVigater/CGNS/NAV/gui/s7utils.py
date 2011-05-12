#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
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
import CGNS.APP.parse.print_utils
import CGNS.pyCGNSconfig as Cfg

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


import CGNS.PAT.cgnskeywords as CK

# --------------------------------------------------------------------
def checkShownType(data,datatype):
  if ((data.dtype.char in ['f']) and (datatype==CK.R4)): return 1
  if ((data.dtype.char in ['d']) and (datatype==CK.R8)): return 1
  if ((data.dtype.char in ['i']) and (datatype==CK.I4)): return 1
  if ((data.dtype.char in ['l']) and (datatype==CK.I8)): return 1
  if ((data.dtype.char in ['S','c','s']) and (datatype==CK.C1)): return 1
  return 0

# --------------------------------------------------------------------
def canBeShown(data):
  if (data == None ): return 1
  if (type(data) in [type(""),type(1.2),type(1)]): return 0
  if (     (type(data) == type(Num.ones((1,))))
       and (data.size < G___.maxDisplaySize)): return 1
  return 0

def toBeShown(data):
  if (not canBeShown(data)): return "???"
  if (data == None ):        return ""
  else:                      return showOneArray(data)

def showOneArray(fdata):
  #print fdata, fdata.shape
  if (G___.transposeOnViewEdit):
    data=fdata.T
  else:
    data=fdata
  if ((G___.compactedValue) and (len(data.shape)==1) and (data.shape[0]==1)):
    if (checkShownType(data,CK.R4)): return showOneFloat(data[0])
    if (checkShownType(data,CK.R8)): return showOneDouble(data[0])
    if (checkShownType(data,CK.I4)): return showOneInteger(data[0])
    if (checkShownType(data,CK.I8)): return showOneLong(data[0])
    if (checkShownType(data,CK.C1)): return showOneString(data[0])
  if ((G___.compactedValue) and (len(data.shape)==1) and (data.shape[0]>1)):
    if (checkShownType(data,CK.C1)): return showOneString(data)
  if (checkShownType(data,CK.C1)
      and G___.compactedValue
      and (len(data.shape)>1)):
    sdata=""
    for ndata in data:
      sdata+=showOneString(ndata)
    return sdata
  return str(data.tolist())

def showOneFloat(f):
  return string.rstrip('%8.6f'%f,'0')
  
def showOneDouble(f):
  return string.rstrip('%8.6f'%f,'0')
  
def showOneInteger(f):
  return string.rstrip('%d'%f,'0')
  
def showOneLong(f):
  return string.rstrip('%d'%f,'0')
  
def showOneString(f):
  return '%s'%f.tostring()

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
  def __init__(self,title,msg,other=None,geo=None):
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
    if (geo!=None):
      wtop.geometry('+%s+%s'%geo)
    else:
      wtop.geometry('+30+30')
    wtop.wait_visibility()
    wtop.grab_set()
    wtop.wait_window(wtop)
  def result(self):
    return self._result

# --------------------------------------------------
def onexit():
  closeWarning()

def mainCloseWarning(pos=None):
  return msginfo("CGNS.NAV: Warning",
  "Please use Quit icon (Control Panel) to close CGNS.NAV.",geo=pos).result()
def closeWarning(pos=None):
  return msginfo("CGNS.NAV: Warning",
                 "Please use Control Panel to close views.",geo=pos).result()
def forceNoRecursionWarning(pos=None):
  return msginfo("CGNS.NAV: Warning",
                 "CGNS.NAV forces [No Recursion] flag.",geo=pos).result()
def operateWarning(pos=None):
  return msginfo("CGNS.NAV: Warning",
"""Operate view is already open.
Close operate view to perform actions on another view.""",geo=pos).result()
def fileWarning(extpos=None):
  return msginfo("CNS.NAV: Error",
                 "Don't know how to read [%s] files..."%ext,geo=pos).result()
def badFileError(name,pos=None):
  return msginfo("CNS.NAV: Error",
                 "Something wrong with file\n%s"%name,geo=pos).result()
def renameError(name,pos=None):
  return msginfo("CGNS.NAV: Error",
                 "Name [%s] already exists"%name,geo=pos).result()
def queryViewCloseError(pos=None):
  return msginfo("CGNS.NAV: Error",
                 "Please close first Selection List View",geo=pos).result()
def cutError(pos=None):
  return msginfo("CGNS.NAV: Error",
                 "Cannot cut root node",geo=pos).result()
def pasteBrotherError(pos=None):
  return msginfo("CGNS.NAV: Error",
                 "Cannot paste as root node brother",geo=pos).result()
def shapeChangeError(shape,pos=None):
  return msginfo("CGNS.NAV: Error",
                 "Value rejected: you change the previous shape %s"%shape,
                 geo=pos).result()
def importCGNSWarning(ext,pos=None):
  return msginfo("CGNS.NAV: Error",
  "Cannot import pyCGNS module\nThis is required for %s files"%ext,
                 geo=pos).result()

def importProfileWarning(mod,msg,pos=None):
  return msginfo("CGNS.NAV: Error",
  "Cannot import %s profile\nError message is:\n%s"%(mod,msg),geo=pos).result()

def spaceInFileName(fnm,pos=None):
  return msginfo("CGNS.NAV: Error",
                 "Pattern name contains <space> char: '%s'"%fnm,geo=pos).result()

def importCGNSFunction(ext,pos=None):
  return msginfo("CGNS.NAV: Error",
                 "Cannot import pyCGNS module\nRequired for %s"%ext,geo=pos).result()

def copyNameCollision(name,pos=None):
  return msginfo("CGNS.NAV: Error",
  "Inserted tree has same name as existing child\n[%s]"%name,geo=pos).result()

def aboutInfo(pos=None):
  msg="""

 %s
 (built %s on %s)

 CGNS.MAP [v%s]
 CGNS.PAT [v%s]
 CGNS.WRA [v%s]
 CGNS.APP [v%s]
 CGNS.NAV [v%s]

pyCGNS web site is http://www.python-science.org/projets/pyCGNS
 
  """%(Cfg.__doc__,Cfg.DATE,Cfg.PLATFORM,
  Cfg.MAP_VERSION,Cfg.PAT_VERSION,Cfg.WRA_VERSION,
  Cfg.APP_VERSION,Cfg.NAV_VERSION)
  return msginfo("CGNS.NAV: About",msg,geo=pos).result()

def snapShotWarning(file,pos=None):
  return msginfo("CGNS.NAV: Info","Snapshot into file\n[%s]"%file,geo=pos).result()

def saveFileWarning(file,pos=None):
  return msginfo("CGNS.NAV: Info",
                 "Tree saved into file\n%s"%file,geo=pos).result()

def profileDirNotFound(prf,pos=None):
  return msginfo("CGNS.NAV: Error",
                 "Profile dir doesn't exist:\n%s"%prf,geo=pos).result()

def createProfileDirInit(prof,pos=None):
  return msginfo("CGNS.NAV: Confirm",
                 "No __init__.py in profile dir '%s'.\nDo you want me to create it?"%prof,geo=pos).result()

def removeNotFoundDir(dir,pos=None):
  return msginfo("CNS.NAV: Confirm",
                 "Directory '%s' not found.\nDo you want to remove it in history list?"%dir,"no, keep directory in history list",geo=pos).result()

def cutRootWarning(pos=None):
  return msginfo("CGNS.NAV: Confirm",
                 "You selected Root node to Cut! Delete all tree?",
                 "cancel",geo=pos).result()

def addLinkWarning(lnk,pos=None):
  return msginfo("CGNS.NAV: Confirm",
                 "You remove existing subtree and replace it by the link:\n%s"%lnk,
                 "cancel",geo=pos).result()

def bigFileWarning(flag,pos=None):
  msg="The file you want to load looks quite big!\n"
  if (flag):
    msg+="You have set the [No Data] options, but\n"
  else:
    msg+="Maybe you could set the [No Data] options,\n"
  msg+="do you really want to load the file?"
  return msginfo("CGNS.NAV: Confirm",msg,"Don't load the file",geo=pos).result()

def patternOverwrite(file,pos=None):
  return msginfo("CGNS.NAV: Confirm",
"""Pattern '%s' already exists! Do you really want to replace it ?"""%file,
"cancel",geo=pos).result()

def fileOverwrite(file,pos=None):
  return msginfo("CGNS.NAV: Confirm",
"""File '%s' already exists! Do you really want to overwrite it ?"""%file,
"cancel",geo=pos).result()

def duplicatedPattern(pat,pos=None):
  return msginfo("CGNS.NAV: Confirm",
"""Pattern '%s' already imported!
CGNS.NAV skips this pattern,
do you want to continue to import profile?"""%pat,"cancel",geo=pos).result()

def leaveCheck(pos=None):
  return msginfo("CGNS.NAV: Confirm","Close all CGNS.NAV views and leave",
                 "cancel",geo=pos).result()

def saveCheck(pos=None):
  return msginfo("CGNS.NAV: Confirm",
                 "Tree is modified, close without save ?","cancel",geo=pos).result()

def overwriteCheck(pos=None):
  return msginfo("CGNS.NAV: Confirm",
                 "Tree already exist, overwrite ?","cancel",geo=pos).result()

def noDataSave(pos=None):
  return msginfo("CGNS.NAV: Confirm",
"""This CGNS tree has been read WITHOUT large DataArrays.
Your save will store data with ZERO values.
Click if you really want to overwrite with ZEROs.
Select [cancel] if you do not want to save file""","cancel",geo=pos).result()

def updateTreeLoad(pos=None):
  return msginfo("CGNS.NAV: Confirm",
"""You have already read this file,
do you want to force load and replace all these tree views?""",
                 "cancel",geo=pos).result()

# --------------------------------------------------
def asFileString(tree):
  return CGNS.APP.parse.print_utils.asString(tree)

# --------------------------------------------------
def dumpWindow(frame,file=None):
  import os
  if (not file):
    from time import localtime, strftime
    t=strftime("%H%M%S", localtime())
    file='/tmp/CGNSNAV-%s-%s.pnm'%(os.getpid(),t)
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

# -----------------------------------------------------------------------------
def forceNumpyAndFortranFlags(tree,path=''):
  path+='/'+tree[0]
  for child in tree[2]:
    print "parse ",path,"  :",
    if ((child[1] != None) and (not Num.isfortran(child[1]))):
      print "force Fortran on "
      child[1]=Num.array(child[1],order='F')
    else:
      print
    forceNumpyAndFortranFlags(child,path)

# --------------------------------------------------
