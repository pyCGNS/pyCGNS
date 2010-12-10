# ----------------------------------------------------------------------------
cdef extern from "cgnslib.h":

  cdef extern int cg_open(char *name, char *mode, int *db)
  cdef extern int cg_close(int db)
  cdef extern int cg_version(int db,float *v)

# ----------------------------------------------------------------------------
class MLLException(Exception):
  def __init__(self,msg):
    self.msg=msg
  def __str__(self):
    return self.msg

# ----------------------------------------------------------------------------
cdef class File:

  """The *File* is the support class for all CGNS/MLL calls.
  """

  cdef readonly int db

  def __init__(self,char *name="", char *mode=""):
    cdef int ret

    ret=cg_open(name,mode,&self.db)
    if (self.db == -1): raise MLLException("Cannot open file [%s]"%name)
    if (not ret):       raise MLLException("CGNS/MLL error [%d]"%ret)

  def cg_version(self):
    cdef float v

    ret=cg_version(self.db,&v)
    return v

  def __del__(self):
    self.cg_close()
    
  def mll_close(self):
    return cg_close(self.db)

# ----------------------------------------------------------------------------
