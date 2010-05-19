# ------------------------------------------------------------
# pyDAX - Exceptions
# ------------------------------------------------------------
# $Id: exceptions.py 35 2003-08-22 15:21:06Z mpoinot $
#
class DAXException(Exception):
  def __init__(self,message):
    self._message=message
  def __str__(self):
    return self._message
  def p(self,n,t):
    return "#DAX Error %.2d# %s"%(n,t)
#
class DAXDiskFull(DAXException):
  def __init__(self,m):
    self.m=m
  def __str__(self):
    return self.p(6,"Disk Full: %s"%self.m)
#
class DAXNoSuchOwner(DAXException):
  def __init__(self,owner):
    self.owner=owner
  def __str__(self):
    return self.p(26,"No such owner: %s"%self.owner)
#
class DAXNoSuchAttribute(DAXException):
  def __init__(self,att):
    self.att=att
  def __str__(self):
    return self.p(14,"Attribute error: %s"%self.att)
#
class DAXBadConnectionString(DAXException):
  def __init__(self,cx):
    self._cx=str(cx)
  def __str__(self):
    return self.p(1,"Bad <database-connection> argument: %s"%self._cx)
#
class DAXBadIDString(DAXException):
  def __init__(self,id):
    self._id=str(id)
  def __str__(self):
    return self.p(2,"Bad <file-identifier> argument: [%s]"%self._id)
#
class DAXNoSuchCGNSFile(DAXException):
  def __init__(self,pth):
    self._pth=str(pth)
  def __str__(self):
    return self.p(13,"No such File: [%s]"%self._pth)
#
class DAXIncorrectCGNSFile(DAXException):
  def __init__(self,pth):
    self._pth=str(pth)
  def __str__(self):
    return self.p(10,"Incorrect CGNS file [%s]"%self._pth)
#
class DAXIdentifierAlreadyExist(DAXException):
  def __init__(self,id):
    self._id=str(id)
  def __str__(self):
    return self.p(8,"Identifier already exists in DAX: [%s]"%self._id)
#
class DAXNoSuchIdentifier(DAXException):
  def __init__(self,id):
    self._id=str(id)
  def __str__(self):
    return self.p(16,"Identifier not found in DAX: [%s]"%self._id)
#
class DAXNoSuchPath(DAXException):
  def __init__(self,pth):
    self._pth=str(pth)
  def __str__(self):
    return self.p(17,"No such path: [%s]"%self._pth)
#
class DAXCGNSFileAlreadyExist(DAXException):
  def __init__(self,pth):
    self._pth=str(pth)
  def __str__(self):
    return self.p(18,"The file already exists: [%s]"%self._pth)
#
class DAXPreparedQueryNotFound(DAXException):
  def __init__(self,pth):
    self._pth=str(pth)
  def __str__(self):
    return self.p(19,"No such prepared query: [%s]"%self._pth)
#
class DAXIncorrectValues(DAXException):
  def __init__(self,s):
    self._s=s
  def __str__(self):
    return self.p(11,"Data rejected: %s"%self._s)
#
class DAXQueryFailed(DAXException):
  def __init__(self,n,q,v):
    self._qname=str(n)
    self._qstmt=str(q)
    self._vals=str(v)
  def __str__(self):
    return self.p(20,"Query '%s' failed\nSQL: %s\nArgs: %s"%
                  (self._qname,self._qstmt,self._vals))
#
class DAXRemovalRefused(DAXException):
  def __init__(self,s):
    self._s=s
  def __str__(self):
    return self.p(25,"Deletion rejected: %s"%self._s)
#
class DAXUpdateRefused(DAXException):
  def __init__(self,s):
    self._s=s
  def __str__(self):
    return self.p(23,"Update rejected: %s"%self._s)
#
class DAXCheckinRefused(DAXException):
  def __init__(self,s):
    self._s=s
  def __str__(self):
    return self.p(3,"Checkin rejected: %s"%self._s)
#
class DAXSystemError(DAXException):
  def __init__(self,s):
    self._s=s
  def __str__(self):
    return self.p(3,"System Error: %s"%self._s)
#
