#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release: v4.0.1 $
#  -------------------------------------------------------------------------

import os.path
import CGNS.PAT.cgnskeywords as CK
import numpy

cimport cgnslib
cimport numpy 

# ------------------------------------------------------------
cdef cg_open_(char *filename, int mode):
  cdef int fid
  fname=os.path.normpath(filename)
  fid=-1
  err=cgnslib.cg_open(filename,mode,&fid)
  return (fid,err)

cdef enum:
  MAXNAMELENGTH = 33
  
# ====================================================================
cdef class pyCGNS(object):
  """pyCGNS doc string"""
  cdef public object _mode
  cdef public object _root
  cdef public object _error
  cdef public object _filename
  
  def __init__(self,filename,mode):
    self._filename=filename
    self._mode=mode
    (self._root,self._error)=cg_open_(filename,mode)

  # ---------------------------------------------------------------------------
  cpdef close(self):
    cgnslib.cg_close(self._root)
    return CK.CG_OK
  # ---------------------------------------------------------------------------
  cpdef float version(self):
    cdef float v
    cgnslib.cg_version(self._root,&v)
    return v
  # ---------------------------------------------------------------------------
  cpdef nbases(self):
    cdef int n
    self._error=cgnslib.cg_nbases(self._root,&n)
    if (n!=0): return xrange(1,n+1)
    return []
  # ---------------------------------------------------------------------------
  cpdef base_read(self, int B):
    """base_read doc string"""
    cdef char basename[MAXNAMELENGTH]
    cdef int  cdim
    cdef int  pdim
    self._error=cgnslib.cg_base_read(self._root,B,basename,&cdim,&pdim)
    return (B, basename, cdim, pdim)
  # ---------------------------------------------------------------------------
  cpdef base_id(self, int B):
    cdef double ibid
    self._error=cgnslib.cg_base_id(self._root,B,&ibid)
    return ibid
  # ---------------------------------------------------------------------------
  cpdef base_write(self, char *basename, int cdim, int pdim):
    cdef int bid
    self._error=cgnslib.cg_base_write(self._root,basename,cdim,pdim,&bid)
    return bid
  # ---------------------------------------------------------------------------
  cpdef zones(self, int B):
    """
    Returns the zone indices::

      for Z in db.zones(B):
         print db.zone_read(B,Z)[2]

    - Args:
     * `B`: the parent base id (:py:func:`bases` and :py:func:`nbases`).

    - Return:
     * An `xrange` from 1 to <number of zones> or an empty list if there is
       no zone at all.
       
    - Remarks:
     * See also :py:func:`nzones`
    """
    cdef int n
    self._error=cgnslib.cg_nzones(self._root,B,&n)
    if (n!=0): return xrange(1,n+1)
    return []
  # ---------------------------------------------------------------------------
  cpdef nzones(self, int B):
    """
    Returns the number of zones::

      for Z in range(1,db.nzones(B)+1):
         print db.zone_read(B,Z)[2]

    - Args:
     * `B`: the parent base id (:py:func:`bases` and :py:func:`nbases`).

    - Return:
     * The number of zones as an integer

    - Remarks:
     * See also :py:func:`zones`
    """
    cdef int n
    self._error=cgnslib.cg_nzones(self._root,B,&n)
    return n
  # ---------------------------------------------------------------------------
  cpdef zone_read(self, int B, int Z):
    cdef char zonename[MAXNAMELENGTH]
    cdef int *zsize
    (bid,bname,cdim,pdim)=self.base_read(B)
    azsize=numpy.ones((cdim*pdim),dtype=numpy.int32)
    zsize=<int *>numpy.PyArray_DATA(azsize)
    self._error=cgnslib.cg_zone_read(self._root,B,Z,zonename,zsize)
    return (B,Z,zonename,azsize)

# ====================================================================
