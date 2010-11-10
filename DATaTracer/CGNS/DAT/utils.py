#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#
import CGNS.DAT.exceptions as dxEX
#
DAXtraceFlag=None
# ----------------------------------------------------------------------
def ptrace(m,level=0):
  global DAXtraceFlag
  if ((DAXtraceFlag == 2) or ((level == 1) and (DAXtraceFlag == 1))):
    print "#T#", m
# ----------------------------------------------------------------------
def pdebug(m):
  print "#D#", m
# ----------------------------------------------------------------------
def perror(m,forceexit=0):
  from sys import exit
  if (forceexit):
    print "#F#", m
    exit(1)
  else:
    print "#E#", m
# ----------------------------------------------------------------------
def date(vtime=0,fmt="%Y-%m-%d %H:%M:%S"):
  from time import strftime, gmtime
  if vtime: return strftime(fmt,gmtime(vtime))
  return strftime(fmt,gmtime())
# ----------------------------------------------------------------------
def getCGNSBase(filename):
  import CGNS
  f=CGNS.pyCGNS(filename,CGNS.MODE_READ)
  b=f.baseread(1)
  f.close()
  return b
# ----------------------------------------------------------------------
def getStat(filename):
  import os
  import pwd
  import grp
  if (filename):
    rs=os.stat(filename)
    # mode, uid, gid, size, mtime
    rt=(("%o"%rs[0])[2:],
        pwd.getpwuid(rs[4])[0],
        grp.getgrgid(rs[5])[0],
        rs[6],
        date(vtime=rs[8]))
  else:
    rt=(0,"unknown","unknown",0,"00/00/0000 00:00:00")
  return rt  
# ----------------------------------------------------------------------
def checksum(filename,adf=None):
  import DAX.cksum
  import os
  import md5
  import CGNS
  ck=0
  if (adf):
      cks=md5.new()
      ck=DAX.cksum.checksumtree(adf,cks).hexdigest()
  if (os.path.isfile(filename)):
      f=CGNS.pyADF(filename,CGNS.READ_ONLY,CGNS.NATIVE)
      cks=md5.new()
      ck=DAX.cksum.checksumtree(f,cks).hexdigest()
      f.database_close()
  return ck
# ----------------------------------------------------------------------
import string
DAXID_PREFIX=['D','F','E','X']
DAXID_IDBODY=string.ascii_letters+string.digits+'+-'
#
def checkDAXID(id):
  if ( (type(id) == type(""))
  and  (len(id) > 5)
  and  (len(id) < 64)       
  and  (id[0] in DAXID_PREFIX)
  and  (id[1] in string.digits)
  and  (id[2] in string.digits)
  and  (id[3] == '=' ) ):
    for cid in id[4:]:
      if (cid not in DAXID_IDBODY):
        raise dxEX.DAXBadIDString(id)
  else:
    raise dxEX.DAXBadIDString(id)
# ----------------------------------------------------------------------
def xn(filename):
  id=filename.split('/')[-1][:-5]
  return id
# ----------------------------------------------------------------------
def fn(id):
  return "%s.cgns"%id
# ----------------------------------------------------------------------
def dn(pth,id):
  if pth: return "%s/%s.cgns"%(pth,id)
  return "%s.cgns"%(id)
# ----------------------------------------------------------------------
def tn(pth,id,cp):
  if (cp): return "%s/%s.dax.tar.%s"%(pth,id,cp)
  else:    return "%s/%s.dax.tar"%(pth,id)
# ----------------------------------------------------------------------
def checkPath(path):
  import os.path
  if (not os.path.isdir(path)):
    raise dxEX.DAXNoSuchPath(path)
# ----------------------------------------------------------------------
def checkExport(path,id,cp):
  import os.path
  if (not os.path.isfile(tn(path,id,cp))):
    raise dxEX.DAXNoSuchCGNSFile(tn(path,id,cp))
# ----------------------------------------------------------------------
def checkFile(id,path):
  import os.path
  if (not os.path.isfile(dn(path,id))):
    raise dxEX.DAXNoSuchCGNSFile(dn(path,id))
# ----------------------------------------------------------------------
def transAsPossible(s):
  import string
  import numarray as N
  x=type(N.array("x",N.UInt8))
  if (type(s) == x):
    return s.tostring()
  if (s == None): return s
  try:
    v=eval(s)
  except NameError:
    return s
  except TypeError:
    v=s
  try:
    return string.atoi(v)
  except ValueError:
    pass
  except TypeError:
    pass
  try:
    return string.atof(v)
  except ValueError:
    pass
  except TypeError:
    pass
  try:
    n=N.array(v)
    if (len(n) == 1):
        return n[0]
    return str(n.tolist())
  except ValueError:
    pass
  return v
# ----------------------------------------------------------------------
def getAsArray(v):
  import numarray as N
  if (type(v) == type("")):  return N.array(v,N.UInt8)
  if (type(v) == type(1)):   return N.array(v,N.Int32)
  if (type(v) == type(1.2)): return N.array(v,N.Float64)
#  if (type(v) == type(Numeric.ones(1))):
#      return str(v.tolist())
  return v
# ----------------------------------------------------------------------
def asReportString(v):
  import numarray as N
  if (type(v) == type("")):  return v
  if (type(v) == type(1)):   return str(v)
  if (type(v) == type(1.2)): return "%g"%v
  if (type(v) == type(N.ones(1))):
      return str(v.tolist())
  return v
# ----------------------------------------------------------------------
#
# last line
