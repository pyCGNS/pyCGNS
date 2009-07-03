# -----------------------------------------------------------------------------
# pyS7 - CGNS/SIDS editor
# ONERA/DSNA - marc.poinot@onera.fr
# pyS7 - $Rev: 70 $ $Date: 2009-01-30 11:49:10 +0100 (Fri, 30 Jan 2009) $
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source
# tree for license information.

TAG=''

# --------------------------------------------------------------------
table={
    0 : "No error",
    1 : "No such file or directory [%s]",
    2 : "File already exists [%s]",
    8 : "The Python tree has not a correct root node",    
    9 : "The argument is not a Python/CGNS node",
   10 : "Bad installation, cannot find [%s]",
  999 : "Unknow error code",
}

# --------------------------------------------------------------------
def perr(id, *tp):
  try:
    msg=TAG+" ERROR [%.3d]- %s"%(id,table[id])
  except TypeError,KeyError:
    msg=TAG+" ERROR [%.3d]- %s"%(id,table[999])
  if tp: return msg%tp
  return msg

# --------------------------------------------------------------------
class S7Exception(Exception):
  def __init__(self,value,msg=""):
    self.value = value
    self.msg=msg
  def __str__(self):
    if self.msg: return perr(self.value,self.msg)
    else:        return perr(self.value)

# --------------------------------------------------------------------
