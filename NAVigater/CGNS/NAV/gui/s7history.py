# -----------------------------------------------------------------------------
# pyS7 - CGNS/SIDS editor
# ONERA/DSNA - marc.poinot@onera.fr
# pyS7 - $Rev: 70 $ $Date: 2009-01-30 11:49:10 +0100 (Fri, 30 Jan 2009) $
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source
# tree for license information.
#

import shutil
import os
import time
import sys
from time import gmtime, strftime

import s7globals
G___=s7globals.s7G

# ---
def printHistory(filelist,dirlist):
  s="""# S7 - File history file - Generated %s
filesHistory={
# unused 
%s
}
directoriesHistory=%s
# last line
"""
  f="""'%(key)%:\
['%(name)s','%(dir)s','%(type)s',%(size)d,%(status)d,%(kwds)s,'%(modif)s'],"""
  fs=""
  for fl in filelist:
    fs+=f%fl
  gdate=strftime("%Y-%m-%d %H:%M:%S", gmtime())
  st=s%(gdate,fs,dirlist)
  return st

def saveHistory():
  fn=G___.historyFile
  opath=os.environ['HOME']
  fh=open(opath+'/'+fn,'w+')
  fh.write(printHistory(G___.filesHistory,G___.directoriesHistory))
  fh.close()
    
def readHistory():
  fn=G___.historyFile
  opath=os.environ['HOME']
  dpath='/tmp/s7history:%d.%s'%(os.getpid(),time.time())
  os.mkdir(dpath)
  try:
    shutil.copyfile(opath+'/'+fn,dpath+'/s7historyData.py')
  except IOError:
    shutil.rmtree(dpath)    
    return ([],[])
  sprev=sys.path
  sys.path=[dpath]+sys.path
  import s7historyData
  shutil.rmtree(dpath)
  sys.path=sprev
  (fh,dh)=(None,None)
  try:
    fh=s7historyData.filesHistory
    dh=s7historyData.directoriesHistory    
  except (NameError, ValueError):
    pass
  return (fh,dh)

def loadHistory():
  (fh,dh)=readHistory()
  G___.directoriesHistory=dh
  
# --- last line
