#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#
import CGNS
from CGNS.midlevel import *
from CGNS.wrap import *

from numarray import *
from string  import *

import CCCCC.utils.stuff as stuff

(mCGNSPYTHON,
 mXML,
 mPATH,
 mLARGE,
 mNOLARGE,
 mFOLLOW,
 mNOFOLLOW,
 mCGNSXML,
 mCGNSPATH)=range(9)

# ------------------------------------------------------------
class ShowBase:
  def __init__(self,ar,out):
    self._arrays=ar
    self._out=out
    
  # l is a list, l[0] is parent, l[1:] is children list
  def recprint(self,l,s,r,sep='/'):
    if (type(l) == type([])):
      ns="%s%s%s"%(s,sep,l[0])
      if (len(l) > 1):
        for ll in l[1:]:
          self.recprint(ll,ns,r)
      else:
        ns="%s%s%s"%(s,sep,l[0])
    else:
      ns="%s%s%s"%(s,sep,l)
    r.append(ns)

  def closeshow(self,db,nodeinfo,decay):
    pass
  
  def openshow(self,db,nodeinfo,decay):
    pass
  
  def printArray(self,s,a,decay):
    # --- HARD CODED SIZE HERE
    maxsize=10
    if not s: return
    if (a.typecode() == 'c'): # string
      for n in split(a.tostring(),'\012'):
        self._out.write(decay*' '+n+'\012')
    elif (self._arrays):
     if (len(s.flat) > maxsize):
       r=reshape(a,(s,))
       for v in r[:maxsize]:
         self._out.write(v)
       self._out.write('\012')
     else:
       for v in a:
         self._out.write(v)
       self._out.write('\012')
       
# ------------------------------------------------------------
class ShowXML(ShowBase):
  def __init__(self,ar,out):
    ShowBase.__init__(self,ar,out)

  def closeshow(self,db,nodeinfo,decay):
    self._out.write(decay*' '+"</NODE>\n")

  def openshow(self,db,nodeinfo,decay):
    self._out.write(decay*' ')
    self._out.write("<NODE name='%(name)s' label='%(label)s'>\n"%nodeinfo)
    self._out.write(decay*' ')
    self._out.write(" <DATA type='%(datatype)s' />\n"%nodeinfo)
    if nodeinfo['dimensions']: size=1
    else: size=0
    sdim=""
    for d in nodeinfo['dimensions']:
      sdim+=decay*' '+"   %d"%d
      size*=d
    self._out.write(decay*' ')
    self._out.write(" <DIMENSIONS total='%d'>\n"%size)
    if sdim:
       self._out.write(sdim)
       self._out.write('\012')
    self._out.write(decay*' '+" </DIMENSIONS>\n")
    if (size):
        ar=db.read_all_data(nodeinfo['id'])
    else: ar=None
    self._out.write(decay*' '+" <VALUES>\n")
    self.printArray(size,ar,decay)
    self._out.write(decay*' '+" </VALUES>\n")
  
# ------------------------------------------------------------
# This class uses a seperate module, in order to be re-used for
# some validation tools, such as Sift.
#
import CCCCC.parser.match as cgx
#
class ShowCGNSXML(ShowBase):
  def __init__(self,ar,out):
    ShowBase.__init__(self,ar,out)
    self._translator=cgx.NodeTranslator(ar,out)

  def footer(self):
    self._translator.footer()
    
  def header(self,filename):
    try:
      d=pyADF(filename,adf.READ_ONLY,adf.NATIVE) # pyCGNS FORCES LINKS PARSE
    except adf.error,e:
      stuff.error(12,"(Using %s) %s"%(filename,e))
    libversion=d.read_all_data(d.get_node_id(d.root(),'CGNSLibraryVersion'))
    d.database_close()
    from time import *
    date=strftime('%d/%m/%Y %H:%M',localtime(time()))
    self._translator.header(filename,date,libversion[0])

  def closeshow(self,db,nodeinfo,decay):
    self._translator.close(db,nodeinfo,decay)

  def openshow(self,db,nodeinfo,decay):
    self._translator.open(db,nodeinfo,decay)

  
# ------------------------------------------------------------
class ShowTXT(ShowBase):
  def __init__(self,ar,out):
    ShowBase.__init__(self,ar,out)

  def closeshow(self,db,nodeinfo,decay):
    pass
    
  def openshow(self,db,nodeinfo,decay):
    self._out.write((decay-1)*' ')
    self._out.write("*[%(name)s(%(label)s,%(datatype)s)]"%nodeinfo)
    if nodeinfo['dimensions']: size=1
    else: size=0
    sdim=""
    for d in nodeinfo['dimensions']:
      sdim+=decay*' '+"   %d"%d
      size*=d
    self._out.write(decay*' ')
    self._out.write("[dim:"+str(nodeinfo['dimensions'])+"]")
    if (size):
      ar=db.read_all_data(nodeinfo['id'])
    else: ar=None
    self.printArray(size,ar,decay)

# ------------------------------------------------------------
class ShowPython(ShowBase):
  def __init__(self,out):
    ShowBase.__init__(self,-1,out)
    #self._translator=cgx.NodeTranslator(-1,out)

  def closeshow(self,db,nodeinfo,decay):
    print "FOOTER"
    self._translator.close(db,nodeinfo,decay)

  def openshow(self,db,nodeinfo,decay):
    print "HEADER"
    self._translator.open(db,nodeinfo,decay)

  def footer(self):
    pass
    
  def header(self,filename):
    pass
    
# ------------------------------------------------------------
class ParseTree:
  
  def __init__(self,mode,out,arraySize=500):
    self._output=mode[0]
    self._followlink=0
    self._parrays=0
    if (mode[2]==mFOLLOW): self._followlink=1
    if (mode[1]==mLARGE):  self._parrays=arraySize
    # factory for ouput driver
    if   (self._output==mXML) :       self._show=ShowXML(self._parrays,out)
    elif (self._output==mCGNSXML):    self._show=ShowCGNSXML(self._parrays,out)
    elif (self._output==mCGNSPYTHON): self._show=ShowPython(out)
    elif (self._output==mCGNSPATH):   self._show=ShowBase(self._parrays,out)
    elif (self._output==mPATH):       self._show=ShowBase(self._parrays,out)
    else:                             self._show=ShowTXT(self._parrays,out)
    
  def sortByLabelAndName(self,a,b):
    #print "<%s,%s>"%(a,b)
    lad=self._db.nodeAsDict(self._db.get_node_id(self._cid,a))
    lbd=self._db.nodeAsDict(self._db.get_node_id(self._cid,b))
    la=lad['label']
    lb=lbd['label']
    if (la == lb):
      #same label
      lan=lad['name']
      lbn=lbd['name']
      if (lan < lbn): return -1
      if (lan > lbn): return  1      
      #print "(%s,%s)"%(la,lb)
      return 0
    if (la <  lb):
      #print "(%s,%s)"%(la,lb)
      return -1
    if (la >  lb):
      #print "(%s,%s)"%(lb,la)
      return 1

  def parse(self,filename):
    # 1- get the root node id
    # 2- get its children list [(name, id),*]
    # 3- get the second element of first tuple (id)
    if (self._output==mCGNSPYTHON) :
      print "HERE PARSE ***"
    if (self._output==mCGNSXML) :
      self._show.header(filename)
    try:
      a=pyADF(filename,adf.READ_ONLY,adf.NATIVE)
    except adf.error, e:
      stuff.error(10,"(Using %s) %s"%(filename,e))
    r=self.parseNode(a,a.root(),1)
    if (self._output==mCGNSXML) :
      self._show.footer()
    a.database_close()
    l=[]
    file=filename.split('/')[-1]
    if (self._output==mPATH):
      self._show.recprint(r,"",l)
    if (self._output==mCGNSPATH):
      for rr in r[1:]:
        self._show.recprint(rr,file,l,':/')
    l.sort()
    return l
  
  def parseNode(self,db,nodeid,decay):
    # tree struct in list is
    # [parent [son-list]]
    # l[0] always is the parent and l[1:] is the son list
    self._db=db
    self._cid=nodeid
    nodeinfo=db.nodeAsDict(nodeid)
    clist=list(nodeinfo['children'])
    clist.sort(self.sortByLabelAndName)
    l=[nodeinfo['name']]
    if not clist and (nodeinfo['datatype']=='LK') and (self._followlink):
      # open the destination db and continue to parse
      # print "OPEN ",nodeinfo['file']
      try:
        ldb=pyADF(nodeinfo['file'],adf.READ_ONLY,adf.NATIVE)
      except adf.error,e:
        stuff.error(11,"(Using %s) %s"%(nodeinfo['file'],e))
      ll=self.parseNode(ldb,ldb.get_node_id(ldb.root(),nodeinfo['path']),decay)
      # now, we have a problem. Cause recursion has been done on dest node link
      # but we would like to see the source node name.
      # [ source-name, [ dest-name, [children-of-dest-name]* ] ]
      # we have to change it to
      # [ source-name, [children-of-dest-name]* ]
      l=l+ll[1:]
      del ldb
    else:
      self._show.openshow(db,nodeinfo,decay)
      #self._show._translator.addEnv(nodeinfo)
      #print "+++ENV open : ",self._show._translator.getEnv()
      decay+=1
      for n in clist:
        l.append(self.parseNode(db,db.get_node_id(nodeid,n),decay))
      decay-=1
      #self._show._translator.clearEnv()
      #print "+++ENV close: ",self._show._translator.getEnv()
      self._show.closeshow(db,nodeinfo,decay)
    return l
  
# ----------------------------------------
# last line
