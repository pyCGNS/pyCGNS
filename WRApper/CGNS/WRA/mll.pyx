#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRApper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
"""
CGNS.WRA.mll is a CGNS/MLL Python wrapper. It defines a `pyCGNS` class
which performs the `cg_open` and holds the opened file descriptor for
subsequent calls.

CGNS.WRA.mll is using a Cython wrapping, it replaces the CGNS.WRA._mll which
was an old hand-coded wrapper, difficult to maintain.

To get the actual list of CGNS/MLL functions wrapped::
  python -c 'import CGNS.WRA.mll;CGNS.WRA.mll.getCGNSMLLAPIlist()'

"""
import os.path
import CGNS.PAT.cgnskeywords as CK
import numpy as PNY

cimport cgnslib
cimport numpy as CNY

# -----------------------------------------------------------------------------
# CGNS/MLL API v3.2.01

CG_MODE_READ  = MODE_READ   = 0
CG_MODE_WRITE = MODE_WRITE  = 1
CG_MODE_MODIFY= MODE_MODIFY = 2
CG_MODE_CLOSED= MODE_CLOSED = 3

CG_OK             = 0
CG_ERROR          = 1
CG_NODE_NOT_FOUND = 2
CG_INCORRECT_PATH = 3
CG_NO_INDEX_DIM   = 4

CG_Null           = Null              = 0
CG_UserDefined    = UserDefined       = 1

CG_FILE_NONE      = 0
CG_FILE_ADF       = 1
CG_FILE_HDF5      = 2
CG_FILE_ADF2      = 3
CG_FILE_PHDF5     = 4

CG_MAX_GOTO_DEPTH = 20

CG_CONFIG_ERROR           =   1
CG_CONFIG_COMPRESS        =   2
CG_CONFIG_SET_PATH        =   3
CG_CONFIG_ADD_PATH        =   4
CG_CONFIG_FILE_TYPE       =   5
CG_CONFIG_HDF5_COMPRESS   = 201

def getCGNSMLLAPIlist():
  import CGNS.WRA.mll as M
  lf=dir(M.pyCGNS)
  for f in lf:
    if ((f[0]!='_') and (f not in ['error'])):
      print 'cg_'+f
  
# ------------------------------------------------------------
cdef cg_open_(char *filename, int mode):
  """
  cg_open_ is reserved for internal use by pyCGNS class, you should not
  call it.
  """
  cdef int fid
  fname=os.path.normpath(filename)
  fid=-1
  err=cgnslib.cg_open(filename,mode,&fid)
  return (fid,err)

cdef enum:
  MAXNAMELENGTH = 33
  MAXGOTODEPTH = 20

cdef asnumpydtype(cgnslib.DataType_t dtype):
    if (dtype==CK.RealSingle):  return PNY.float32
    if (dtype==CK.RealDouble):  return PNY.float64
    if (dtype==CK.Integer):     return PNY.int32
    if (dtype==CK.LongInteger): return PNY.int64
    if (dtype==CK.Character):   return PNY.uint8
    return None
        
cdef fromnumpydtype(dtype):
    if (dtype.char=='f'):  return CK.RealSingle
    if (dtype.char=='d'):  return CK.RealDouble
    if (dtype.char=='i'):  return CK.Integer
    if (dtype.char=='l'):  return CK.LongInteger
    if (dtype.char=='S'):  return CK.Character
    return None

class CGNSException(Exception):
    def __init__(self,code,msg):
        self.msg=msg
        self.code=code
    def __str__(self):
        return "pyCGNS Error: [%.4d] %s"%(self.code,self.msg)
    
# ====================================================================
cdef class pyCGNS(object):
  """
  A `pyCGNS` object is a CGNS/MLL file handler. You can call any
  CGNS/MLL function on this object, a `cg_<function>` in the C API
  is named `<function>` in this Python API. Functions are returning
  values instead of error code, you have to check each function
  mapping to known what are the inputs/outputs.  You would also find
  extra functions providing the user with a better Python interface.

  A `pyCGNS` object creation is a call to `cg_open`::

    db=CGNS.WRA.mll.pyCGNS('5blocks.cgns',CGNS.WRA.mll.MODE_READ)
    for b in db.bases():
      for z in db.zones(b):
         print db.zone_read(b,z)

  - Args:
   * `filename`: the target file path
   * `mode`:     an integer setting the open mode, an enumerate you should
   use through the Python module enumerate (CGNS.WRA.mll.MODE_READ,
   CGNS.WRA.mll.MODE_WRITE,CGNS.WRA.mll.MODE_MODIFY)

  - Return:
   * A pyCGNS class instance, the object is the CGNS/MLL file descriptor
   on which you can call CGNS/MLL functions.
       
  """
  cdef public object _mode
  cdef public object _root
  cdef public object _error
  cdef public object _filename

  def __init__(self,filename,mode):
    self._filename=filename
    self._mode=mode
    (self._root,self._error)=cg_open_(filename,mode)

  def error(self):
    return (self._error, cgnslib.cg_get_error())

  @property
  def _ok(self):
    if (self._error==0): return True
    return False

  # ---------------------------------------------------------------------------
  cdef asCharPtrInList(self,char **ar,nstring,dstring):
    ns=[]
    for n in xrange(nstring):
      ns.append(' '*dstring)
      ar[n]=<char *>ns[n]
    return ns

  # ---------------------------------------------------------------------------
  cdef fromCharPtrInList(self,char **ar,lstring):
    rlablist=[]
    for n in xrange(len(lstring)):
        if (len(ar[n].strip())>0): 
            rlab=ar[n].strip()
            rlablist.append(rlab)
    return rlablist

  # ---------------------------------------------------------------------------
  cpdef close(self):
    """
    Closes the CGNS/MLL file descriptor.

    - Remarks:
    * The pyCGNS object is still there, however you cannot call any CGNS/MLL
    function with it. The `__del__` of the Python object calls the `close`.
    """
    cgnslib.cg_close(self._root)
    return CG_OK

  # ---------------------------------------------------------------------------
  cpdef float version(self):
    """
    Returns the CGNS/MLL library version::

      v=db.version()

    - Return:
     * The version number as a float, for example 2.400 is CGNS/MLL version 2.4

    - Remarks:
     * The Python print may display a float value without rounding, this may
     change with your Python version and the way you actually print the value::
     
      print '%g'%(db.version())
     
    """
    cdef float v
    cgnslib.cg_version(self._root,&v)
    return v

  # ---------------------------------------------------------------------------
  cpdef gopath(self,char *path):
    self._error=cgnslib.cg_gopath(self._root,path)

  # ---------------------------------------------------------------------------
  cpdef where(self):
    cdef int depth
    cdef CNY.ndarray num
    cdef int *numptr
    cdef int B
    cdef int fn
    cdef char *lab[MAXGOTODEPTH]
    num=PNY.ones((MAXGOTODEPTH,),dtype=PNY.int32)
    numptr=<int *>num.data
    lablist=self.asCharPtrInList(lab,MAXGOTODEPTH,32)
    cgnslib.cg_where(&fn,&B,&depth,lab,numptr)
    lablist=self.fromCharPtrInList(lab,lablist)
    rlab=[]
    for i in range(depth):
      rlab.append((lablist[i],num[i]))
    return (B, rlab)
    
  # ---------------------------------------------------------------------------
  cpdef nbases(self):
    """
    Returns the number of bases::

      for B in range(1,db.nbases()+1):
         print db.base_read(B)

    - Return:
     * The number of bases as an integer

    - Remarks:
     * See also :py:func:`bases`
    """
    cdef int n
    self._error=cgnslib.cg_nbases(self._root,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef bases(self):
    """
    Returns the bases indices::

      for B in db.bases():
         print db.base_read(B)

    - Return:
     * An `xrange` from 1 to <number of bases> or an empty list if there is
       no base at all.
       
    - Remarks:
     * See also :py:func:`nbases`
    """
    cdef int n
    self._error=cgnslib.cg_nbases(self._root,&n)
    if (n!=0): return xrange(1,n+1)
    return []

  # ---------------------------------------------------------------------------
  cpdef base_read(self, int B):
    """
    Returns a tuple with information about the base.

    - Args:
     * `B`: the base id 

    - Return:
     * argument base id (`int`)
     * base name (`string`)
     * cell dimensions (`int`)
     * physical dimensions (`int`)
    """
    cdef char basename[MAXNAMELENGTH]
    cdef int  cdim
    cdef int  pdim
    self._error=cgnslib.cg_base_read(self._root,B,basename,&cdim,&pdim)
    return (B, basename, cdim, pdim)

  # ---------------------------------------------------------------------------
  cpdef base_id(self, int B):
    """
    Returns the base internal id.

    - Args:
     * `B`: the CGNS/MLL base id (`int`) (this is **not** the internal id)

    - Return:
     * The base internal id (`int`)

    - Remarks:
     * This id can be used with the CGNS/ADF or CGNS/HDF5 API. If you don't
     know what this id is... then you should not use it.
    """
    cdef double id=-1
    self._error=cgnslib.cg_base_id(self._root,B,&id)
    return id

  # ---------------------------------------------------------------------------
  cpdef base_write(self, char *basename, int cdim, int pdim):
    """
    Creates a new base::

      bid=db.base_write('Base-001',3,3)

    - Args:
     * `basename`: name of the new base (`string` should not exceed 32 chars)
     * `celldim`: cell dimensions of the base (`int`)
     * `physicaldim`: physical dimension of the base (`int`)

    - Return:
     * New base id (`int`)
    """
    cdef int bid
    self._error=cgnslib.cg_base_write(self._root,basename,cdim,pdim,&bid)
    return bid

  # ---------------------------------------------------------------------------
  cpdef zones(self, int B):
    """
    Returns all the zones indices of a base::

      for Z in db.zones(B):
         print db.zone_read(B,Z)[2]

    - Args:
     * `B`: the parent base id (`int`) (:py:func:`bases` 
       and :py:func:`nbases`).

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
    Returns the number of zones in a base::

      for Z in range(1,db.nzones(B)+1):
         print db.zone_read(B,Z)[2]

    - Args:
     * `B`: Parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).

    - Return:
     * Number of zones as an integer (`int`)

    - Remarks:
     * See also :py:func:`zones`
    """
    cdef int n
    self._error=cgnslib.cg_nzones(self._root,B,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef zone_read(self, int B, int Z):
    """
    Returns a tuple with information about the zone.

    - Args:
     * `B`: base id (`int`)
     * `Z`: zone id (`int`)

    - Return:
     * argument base id (`int`)
     * argument zone id (`int`)     
     * zone name (`string`)
     * zone size (`numpy.ndarray`)

     - Remarks:
     * returned array of zone size is a 1D array, you have to read it taking
     into account the cell and physical dimensions of the parent base,
     see :py:func:`base_read`
   """
    cdef char zonename[MAXNAMELENGTH]
    cdef int *zsize
    cdef CNY.ndarray[dtype=CNY.int32_t,ndim=1] azsize
    (bid,bname,cdim,pdim)=self.base_read(B)
    azsize=PNY.ones((cdim*pdim),dtype=PNY.int32)
    zsize=<int *>azsize.data
    self._error=cgnslib.cg_zone_read(self._root,B,Z,zonename,zsize)
    return (B,Z,zonename,azsize)

  # ---------------------------------------------------------------------------
  cpdef zone_type(self, int B, int Z):
    """
    Returns the CGNS type of a zones.

    - Args:
     * `B`: the parent base id (`int`) (:py:func:`bases` 
       and :py:func:`nbases`).
     * `Z`: the parent zone id (`int`) (:py:func:`zones` 
       and :py:func:`nzones`).

    - Return:
     * The ZoneType_t of the zone as an integer (`int`). 
    """
    cdef cgnslib.ZoneType_t ztype
    ztype=cgnslib.ZoneTypeNull
    self._error=cgnslib.cg_zone_type(self._root,B,Z,&ztype)
    return ztype

  # ---------------------------------------------------------------------------
  cpdef zone_id(self, int B, int Z):
    """
    Returns the zone internal id.

    - Args:
     * `B`: the CGNS/MLL base id (`int`)
     * `Z`: the CGNS/MLL zone id (`int`) (this is **not** the internal id)

    - Return:
     * The zone internal id (`int`)

    - Remarks:
     * This id can be used with the CGNS/ADF or CGNS/HDF5 API. If you don't
     know what this id is... then you should not use it.
    """
    cdef double id=-1
    self._error=cgnslib.cg_zone_id(self._root,B,Z,&id)
    return id

  # ---------------------------------------------------------------------------
  cpdef zone_write(self, int B, char *zonename, ozsize,  
                   cgnslib.ZoneType_t ztype):
    """
    Creates a new zone::

      zsize=numpy.array((i,j,k,i-1,j-1,k-1,0,0,0),dtype=numpy.int32)
      zid=db.zone_write(B,'Zone-001',zsize,CGNS.PAT.cgnskeywords.Structured)

    - Args:
     * `B`: parent base id (`int` within :py:func:`nbases` ids)
     * `zonename`: name of the new zone (`string` should not exceed 32 chars)
     * `zsize`: numpy array of `int`
     * `zonetype`: type of the zone `int`

    - Return:
     * New zone id

    - Remarks:
     * No zone size check
     * Zone size can be 1D
     * Zone type is an integer that should be one of 
       the `CGNS.PAT.cgnskeywords.ZoneType_` keys
     * Zone size depends on base dimensions and zone type (see `CGNS/SIDS 6.3)

    """
    cdef CNY.ndarray[dtype=CNY.int32_t,ndim=1] zsize
    cdef int  zid=-1
    cdef int *ptrzs
    zsize=PNY.require(ozsize.flatten(),dtype=PNY.int32)
    ptrzs=<int *>zsize.data
    self._error=cgnslib.cg_zone_write(self._root,B,zonename,ptrzs,ztype,&zid)
    return zid

  # -----------------------------------------------------------------------    
  cpdef nfamilies(self, int B):
    """
    Returns the number of families in a base::

      for F in range(1,db.nfamilies(B)+1):
         print db.family_read(B)

    - Args:
     * `B`: the parent base id (`int`) (:py:func:`bases` 
       and :py:func:`nbases`).

    - Return:
     * The number of families as an integer (`int`)

    - Remarks:
     * See also :py:func:`families`
    """
    cdef int n
    self._error=cgnslib.cg_nfamilies(self._root,B,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef families(self, int B):
    """
    Returns all the families indices of a base::

      for F in db.families(B):
         print db.family_read(B)

    - Args:
     * `B`: parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).

    - Return:
     * An `xrange` from 1 to <number of families> or an empty list if there is
       no family at all.
       
    - Remarks:
     * See also :py:func:`nfamilies`
    """
    cdef int n=0
    self._error=cgnslib.cg_nfamilies(self._root,B,&n)
    if (n!=0): return xrange(1,n+1)
    return []

  # ---------------------------------------------------------------------------
  cpdef family_read(self, int B, int F):
    """
    Returns a tuple with information about the family.

    - Args:
     * `B`: base id (`int`)
     * `F`: family id (`int`)

    - Return:
     * argument base id (`int`)
     * argument family id (`int`)     
     * family name (`string`)
     * number of `FamilyBC_t` children nodes (`int`)
     * number of `GeometryReference_t` children nodes (`int`)

     - Remarks:
     * returned numbers of children for each `FamilyBC_t`
     and `GeometryReference_t` have to be :py:func:`fambc_read` and
     :py:func:`geo_read`. You have to parse each child a compare with
     some parameter of yours to find the one you are looking for.
    """
    cdef char  family_name[MAXNAMELENGTH]
    cdef int   nboco
    cdef int   ngeo
    self._error=cgnslib.cg_family_read(self._root,B,F,family_name,&nboco,&ngeo)
    return (B,F,family_name,nboco,ngeo)

  # ---------------------------------------------------------------------------
  cpdef family_write(self, int B, char *familyname):
    """
    Creates a new family::

      bid=db.family_write(B,'LeftWing')

    - Args:
     * `B`: the parent base id (`int`) (:py:func:`bases` 
       and :py:func:`nbases`).
     * `familyname`: name of the new family (`string` <= 32 chars)

    - Return:
     * New family id
    """
    cdef int fid
    self._error=cgnslib.cg_family_write(self._root,B,familyname,&fid)
    return fid
  # ---------------------------------------------------------------------------
  cpdef fambc_read(self, int B, int F, int BC):
    """
    Returns a tuple with information about the family BC.

    - Args:
     * `B`: base id (`int`)
     * `F`: family id (`int`)
     * `BC`: BC family id (`int`)

    - Return:
     * argument base id (`int`) 
     * argument family id (`int`)     
     * argument BC family id (`int`)     
     * BC family name (`string`)
     * BC type (`int`)

     - Remarks:
     * The BC type is one of the keys of `CGNS.PAT.cgnskeywords.BCType_`
    """
    cdef char fambc_name[MAXNAMELENGTH]
    cdef cgnslib.BCType_t bocotype
    self._error=cgnslib.cg_fambc_read(self._root,B,F,BC,fambc_name,&bocotype)
    return (B,F,BC,fambc_name,bocotype)

  # ---------------------------------------------------------------------------
  cpdef fambc_write(self, int B, int F,
                    char *fambcname, cgnslib.BCType_t bocotype):
    """
    Creates a new BC family::

      fbcid=db.fambc_write(B,F,CGNS.PAT.cgnskeywords.FamilyBC_s)

    - Args:
     * `B`: parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).
     * `F`: parent family id (`int`) (:py:func:`families` and 
       :py:func:`nfamilies`).
     * `fambcname`: name of the new BC family (`string` <= 32 chars)
     * `bocotype`: type of the actual BC for all BCs refering to the 
       parent family name of `F` (`int`)

    - Return:
     * New BCfamily id (`int`)

    - Remarks:
     * A `BCFamily` takes place as a child of a Family in a Base, once
     created you can create or change some BCs with a type of `FamilySpecified`
     and with a `FamilyName` equals to this `BCFamily` parent Family.
     * a `FamilyBC_t`node name usually is `FamilyBC`
    """
    cdef int fbcid
    self._error=cgnslib.cg_fambc_write(self._root,B,F,fambcname,
                                       bocotype,&fbcid)
    return fbcid

  # ---------------------------------------------------------------------------
  cpdef geo_read(self, int B, int F, int G):
    """
    Returns a tuple with information about the Geometry reference.

    - Args:
     * `B`: base id (`int`)
     * `F`: family id (`int`)
     * `G`: geometry reference id (`int`)

    - Return:
     * argument base id (`int`)
     * argument family id (`int`)     
     * argument geometry reference id (`int`)     
     * geometry reference name (`string`)
     * geometry reference file (`string`)
     * geometry reference CAD name (`string`)
     * geometry reference number of parts (`int`)     

     - Remarks:
     * use :py:func:`family_read` to get the geometry reference id
    """
    cdef char *filename
    cdef int   n
    cdef char  geoname[MAXNAMELENGTH]
    cdef char  cadname[MAXNAMELENGTH]
    self._error=cgnslib.cg_geo_read(self._root,B,F,G,geoname,&filename,
                                    cadname,&n)
    sfilename=str(filename)
    #cgnslib.cg_free(<void *>filename)
    return (B,F,G,geoname,sfilename,cadname,n)

  # ---------------------------------------------------------------------------
  cpdef geo_write(self, int B, int F, char *geoname, char *filename,
                  char *cadname):
    """
    Creates a new Geometry reference.

    - Args:
     * `B`: parent base id (`int`) (:py:func:`bases` 
        and :py:func:`nbases`).
     * `F`: parent family id (`int`) (:py:func:`families` 
       and :py:func:`nfamilies`).
     * `geoname`: name of the new geometry reference (`string` <= 32 chars)
     * `filename`: path to geometry reference file (`string`)
     * `cadname`: name of the geometry reference CAD (`string`)

    - Return:
     * New Geometry reference id (`int`)

    - Remarks:
     * The cad should be an enumerate as described in SIDS section 12.7
    """
    cdef int gid
    self._error=cgnslib.cg_geo_write(self._root,B,F,geoname,filename,cadname,
                                     &gid)
    return gid

  # ---------------------------------------------------------------------------
  cpdef part_read(self, int B, int F, int G, int P):
    """
    Returns a tuple with information about a Geometry reference part.

    - Args:
     * `B`: base id (`int`)
     * `F`: family id (`int`)
     * `G`: geometry reference id (`int`)
     * `P`: geometry reference part id  (`int`)

    - Return:
     * argument base id (`int`)
     * argument family id (`int`)     
     * argument geometry reference id (`int`)
     * argument geometry reference part id (`int`)     
     * geometry reference part name (`string`)
    """
    cdef char partname[MAXNAMELENGTH]
    self._error=cgnslib.cg_part_read(self._root,B,F,G,P,partname)
    return (B,F,G,P,partname)

  # ---------------------------------------------------------------------------
  cpdef part_write(self, int B, int F, int G, char *partname):
    """
   Creates a new Geometry reference part.

    - Args:
     * `B`: parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).
     * `F`: parent family id (`int`) (:py:func:`families` 
        and :py:func:`nfamilies`).
     * `G`: geometry reference id (`int`)
     * `partname`: name of the new geometry reference part (`string`)
       (`string` should not exceed 32 chars)

    - Return:
     * New Geometry reference part id
    """
    cdef int pid
    self._error=cgnslib.cg_part_write(self._root,B,F,G,partname,&pid)
    return pid

  # ---------------------------------------------------------------------------
  cpdef ngrids(self, int B, int Z):
    """
    Returns the number of grids in a zone::

      for G in range(1,db.ngrids(B,Z)+1):
         print db.grid_read(B,Z,G)

    - Args:
     * `B`: parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).
     * `Z`: parent zone id (`int`) (:py:func:`zones` and :py:func:`nzones`).

    - Return:
     * The number of grids as an integer (`int`)

    - Remarks:
     * See also :py:func:`grids`
    """
    cdef int n=0
    self._error=cgnslib.cg_ngrids(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef grids(self, int B, int Z):
    """
    Returns the number of grids indices of a zone::

      for G in db.grids(B,Z):
         print db.grid_read(B,Z,G)

    - Args:
     * `B`: parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).
     * `Z`: parent zone id (`int`) (:py:func:`zones` and :py:func:`nzones`).

    - Return:
     * An `xrange` from 1 to <number of grids> or an empty list if there is
       no grid at all.

    - Remarks:
     * See also :py:func:`ngrids`
    """
    cdef int n=0
    self._error=cgnslib.cg_ngrids(self._root,B,Z,&n)
    if (n!=0): return xrange(1,n+1)
    return []

  # ---------------------------------------------------------------------------
  cpdef grid_read(self, int B, int Z, int G):
    """
    Returns a tuple with information about the grid.

    - Args:
     * `B`: base id (`int`)
     * `Z`: zone id (`int`)
     * `G`: grid id (`int`)

    - Return:
     * argument base id (`int`)
     * argument zone id (`int`)     
     * argument grid id (`int`)     
     * grid name (`string`)
    """
    cdef char gridname[MAXNAMELENGTH]
    self._error=cgnslib.cg_grid_read(self._root,B,Z,G,gridname)
    return (B,Z,G,gridname)

  # ---------------------------------------------------------------------------
  cpdef grid_write(self, int B, int Z, char *gridname):
    """
    Creates a new grid.

    - Args:
     * `B`: base id (`int`)
     * `Z`: zone id (`int`)
     * `gridname`: name of the new grid (`string` <= 32 chars) (`string`)

    - Return:
     * grid id (`int`)

    - Remarks:
     * The `GridCoordinates` name is reserved for the default grid name.
     You should have one `GridCoordinates` grid per zone if your zone is not
     empty. See also :py:func:`coord_write` which creates `GridCoordinates`
     or uses it if present.
    """
    cdef int gid
    self._error=cgnslib.cg_grid_write(self._root,B,Z,gridname,&gid)
    return gid

  # ---------------------------------------------------------------------------
  cpdef ncoords(self, int B, int Z):
    """
    Returns the number of coordinates array in the GridCoordinates node.

    - Args:
     * `B`: parent base id (`int`)
     * `Z`: parent zone id (`int`)

    - Return:
     * The number of coordinates arrays as an integer (`int`)

    - Remarks:
     * See also :py:func:`coords`
    """
    cdef int n=0
    self._error=cgnslib.cg_ncoords(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef coords(self, int B, int Z):
    """
    Returns the number of coordinates array indices of a zone.

    - Args:
     * `B`: parent base id (`int`)
     * `Z`: parent zone id (`int`)

    - Return:
     * An `xrange` from 1 to <number of nodes> or an empty list if there is
       no coordinates at all.

    - Remarks:
     * See also :py:func:`ncoords`
    """
    cdef int n=0
    self._error=cgnslib.cg_ncoords(self._root,B,Z,&n)
    if (n!=0): return xrange(1,n+1)
    return []

  # ---------------------------------------------------------------------------
  cpdef coord_info(self, int B, int Z, int C):
    """
    Returns a tuple with information about the coordinates.

    - Args:
     * `B`: base id (`int`)
     * `Z`: zone id (`int`)
     * `C`: coordinates id (`int`) (:py:func:`coords` and :py:func:`ncoords`)

    - Return:
     * argument base id (`int`)
     * argument zone id (`int`)     
     * argument coordinates id (`int`)
     * coordinates array data type (`int`)
     * coordinate name (`string`)

    - Remarks:
     * With a X,Y,Z coordinate system, you should look for X (one coordinate
     id), X (another coordinate id) and Z (another coordinate id). That makes
     three calls of `coord_info`.
     * The coordinate array datatype is from `CGNS.PAT.cgnskeywords.DataType_`
    """
    cdef cgnslib.DataType_t dtype
    cdef char coordname[MAXNAMELENGTH]
    self._error=cgnslib.cg_coord_info(self._root, B, Z, C,&dtype,coordname)
    return (B,Z,C,dtype,coordname)

  # ---------------------------------------------------------------------------
  cpdef coord_read(self, int B, int Z,
                   char *coordname, cgnslib.DataType_t dtype):
    """
    Returns a tuple with actual coordinates array.

    - Args:
     * `B`: base id (`int`)
     * `Z`: zone id (`int`)
     * `coordname`: coordinate array name to read (`string`)
     * `dtype`: datatype of the array (`int`)

    - Return:
     * argument base id (`int`)
     * argument zone id (`int`)     
     * argument coordinates name (`string`)
     * argument coordinates array data type (`int`)
     * min indices (`numpy.ndarray`)
     * max indices (`numpy.ndarray`)
     * coordinates (`numpy.ndarray`)

    - Remarks:
     * The datatype forces a cast if it is not the original type of the array
     * The coordinate array datatype is from `CGNS.PAT.cgnskeywords.DataType_`
     * The dtype can be a numpy dtype as far as it can be translated
    """
    (bid,bname,cdim,pdim)=self.base_read(B)
    (bid,zid,zname,zsize)=self.zone_read(B,Z)
    rmin=PNY.ones((cdim),dtype=PNY.int32)
    rmax=PNY.ones((cdim),dtype=PNY.int32)
    rminptr=<int *>CNY.PyArray_DATA(rmin)
    rmaxptr=<int *>CNY.PyArray_DATA(rmax)
    cdtype=asnumpydtype(dtype)
    if (cdtype == None):
        ndtype=fromnumpydtype(dtype)
        if (ndtype == None):
            raise CGNSException(10,"No such data type: %s"%str(ndtype))
        dtype=ndtype
        cdtype=asnumpydtype(dtype)
    coords=PNY.ones(zsize,dtype=cdtype)
    coordsptr=<void *>CNY.PyArray_DATA(coords)
    self._error=cgnslib.cg_coord_read(self._root, B, Z,
                                      coordname,dtype,
                                      rminptr,rmaxptr,coordsptr)
    return (B,Z,coordname,dtype,rmin,rmax,coords)

  # ---------------------------------------------------------------------------
  cpdef coord_id(self, int B, int Z, int C):
    """
    Returns the base internal id.

    - Args:
     * `B`: the CGNS/MLL base id (`int`)
     * `Z`: the CGNS/MLL zone id (`int`)
     * `C`: the CGNS/MLL coordinates id (`int`)

    - Return:
     * The coordinates internal id (`int`)

    - Remarks:
     * This id can be used with the CGNS/ADF or CGNS/HDF5 API. If you don't
     know what this id is... then you should not use it.
    """
    cdef double id=-1
    self._error=cgnslib.cg_coord_id(self._root,B,Z,C,&id)
    return id

  # ---------------------------------------------------------------------------
  cpdef coord_write(self, int B, int Z, cgnslib.DataType_t dtype,
                   char *coordname, coords):
    """
    Creates a new coordinates.

    - Args:
     * `B`: base id (`int`)
     * `Z`: zone id (`int`)
     * `dtype`: data type of the array contents (`int`)
     * `coordname`: name of the new coordinates (`string` <= 32 chars)
     * `coords`: array of actual coordinates (`numpy.ndarray`)
     
    - Return:
     * coordinate array id (`int`)

    - Remarks:
     * Creates by default the `GridCoordinates` node
     * the coords array is a `numpy` with correct data type with respect
     to the `CGNS.PAT.cgnskeywords.DataType_` argument.
     * The dtype can be a numpy dtype as far as it can be translated
    """
    cdef int cid
    cdtype=asnumpydtype(dtype)
    if (cdtype == None):
        ndtype=fromnumpydtype(dtype)
        if (ndtype == None):
            raise CGNSException(10,"No such data type: %s"%str(ndtype))
        dtype=ndtype
    coordsptr=<void *>CNY.PyArray_DATA(coords)
    self._error=cgnslib.cg_coord_write(self._root, B, Z,
                                       dtype, coordname,coordsptr,&cid)
    return cid
  # ---------------------------------------------------------------------------
  cpdef coord_partial_write(self, int B, int Z, cgnslib.DataType_t dtype,
                            char *coordname, rmin, rmax, coords):
    """
    Modify coordinates.

    - Args:
     * `B`: base id (`int`)
     * `Z`: zone id (`int`)
     * `dtype`: data type of the array contents (`int`)
     * `coordname`: name of the new coordinates (`string` <= 32 chars)
     * `rmin`: min range of data to write  (`numpy.ndarray`)
     * `rmax`: max range of data to write  (`numpy.ndarray`)
     * `coords`: array of actual coordinates (`numpy.ndarray`)
     
    - Return:
     * Creates by default the `GridCoordinates` node
     * the coords array is a `numpy` with correct data type with respect
     to the `CGNS.PAT.cgnskeywords.DataType_` argument.
    """
    cdef int cid
    rminptr=<int *>CNY.PyArray_DATA(rmin)
    rmaxptr=<int *>CNY.PyArray_DATA(rmax)
    coordsptr=<float *>CNY.PyArray_DATA(coords)
    cdtype=asnumpydtype(dtype)
    if (cdtype == None):
        ndtype=fromnumpydtype(dtype)
        if (ndtype == None):
            raise CGNSException(10,"No such data type: %s"%str(ndtype))
        dtype=ndtype
    self._error=cgnslib.cg_coord_partial_write(self._root, B, Z, dtype,
                                               coordname, rminptr, rmaxptr, 
                                               coordsptr,&cid)
    return cid
  # ------------------------------------------------------------
  cpdef nsols(self, int B, int Z):
    """
    Returns the number of flow solutions.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Return:
    *  number of flow solutions for zone `Z`(`int`)
    
    """
    cdef int n=0
    self._error=cgnslib.cg_nsols(self._root,B,Z,&n)
    return n

  # ------------------------------------------------------------
  cpdef sol_write(self, int B, int Z, char * solname,
                  cgnslib.GridLocation_t location ):
    """
    Creates a new flow solution node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `solname` : name of the flow solution (`string`)
    * `location` : grid location where the solution is recorded (`int`)
      The admissible locations are Vertex, CellCenter, IFaceCenter, JFaceCenter
      and KFaceCenter.

    - Return:
    * flow solution id (`int`) 

    """
    cdef int S=-1
    self._error=cgnslib.cg_sol_write(self._root,B,Z,solname,location,&S)
    return S

  # --------------------------------------------------------------
  cpdef sol_info(self,int B, int Z, int S):
    """
    Returns a tuple with contains the name of the flow solution and
    the grid location of the solution.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : flow solution id which  is comprised between 1 and the total number
      of flow solutions (`int`)

    - Return:
    * name of the flow solution (`string`)
    * grid location of the solution (`int`)
    
    """
    cdef char * solname = " "
    cdef cgnslib.GridLocation_t location
    self._error=cgnslib.cg_sol_info(self._root,B,Z,S,solname,&location)
    return (solname,location)    

  # ---------------------------------------------------------------
  cpdef sol_id(self,int B, int Z, int S):
    cdef double id=-1
    self._error=cgnslib.cg_sol_id(self._root,B, Z, S, &id)
    return id

  # ----------------------------------------------------------------
  cpdef sol_size(self, int B, int Z, int S):
    """
    Returns a tuple with information about the flow solution
    data.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : flow solution id (`int`)

    - Return:
    * number of dimensions defining the solution data (`int`)
      If a point set has been defined, this will be 1, otherwise this will be
      the current zone index dimension
    * array of data_dim dimensions for the solution data (`numpy.ndarray`)
    
    """
    cdef int data_dim
    dim_vals=PNY.ones((CK.ADF_MAX_DIMENSIONS,),dtype=PNY.int32)
    dim_valsptr=<int *>CNY.PyArray_DATA(dim_vals)
    self._error=cgnslib.cg_sol_size(self._root,B,Z,S,&data_dim,dim_valsptr)
    return (data_dim,dim_vals[0:data_dim])

  # ----------------------------------------------------------------
  cpdef nsections(self, int B, int Z):
    """
    Returns the number of element sections.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Return:
    * number of element sections (`int`)

    """
    cdef int n=0
    self._error=cgnslib.cg_nsections(self._root,B,Z,&n)
    return n

  # -----------------------------------------------------------------
  cpdef section_read(self, int B, int Z, int S):
    """
    Returns a tuple with information about the element section
    data.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section id (`int`)

    - Return:
    * name of the `Elements_t` node (`string`)
    * type of element (`int`)
    * index of first element in the section (`int`)
    * index of last element in the section (`int`)
    * index of last boundary element in the section (`int`)
      If the elements are unsorted, this index is set to 0.
    * flag indicating if the parent data are defined (`int`)
      If the parent data exist, `parent_flag` is set to 1

    """
    cdef char * SectionName
    SectionName=' '
    cdef cgnslib.ElementType_t Element_type
    cdef cgnslib.cgsize_t start
    cdef cgnslib.cgsize_t end
    cdef int nbndry
    cdef int parent_flag
    self._error=cgnslib.cg_section_read(self._root,B,Z,S,SectionName,
                                        &Element_type,
                                        &start,&end,&nbndry,&parent_flag)
    return (SectionName,Element_type,start,end,nbndry,parent_flag)

  # ---------------------------------------------------------------------
  cpdef npe(self, cgnslib.ElementType_t type):
    """
    Returns the number of nodes of an element.

    - Args:
    * `type` : type of element (`int`)

    - Return:
    * number of nodes for an element of type `type` (`int`)
    
    """
    cpdef int n=0
    self._error=cgnslib.cg_npe(type,&n)
    return n

  # ----------------------------------------------------------------------
  cpdef ElementDataSize(self, int B, int Z, int S):
    """
    Returns the number of element connectivity data values.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section id (`int`)

    - Return:
    * number of element connectivity data values (`int`)
    
    """
    cdef cgnslib.cgsize_t ElementDataSize
    self._error=cgnslib.cg_ElementDataSize(self._root,B,Z,S,&ElementDataSize)
    return ElementDataSize

  # ---------------------------------------------------------------------
  cpdef ElementPartialSize(self, int B, int Z, int S, cgnslib.cgsize_t start,
                           cgnslib.cgsize_t end):
    """
    Returns the number of element connectivity data values in a range.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section id (`int`)
    * `start` : min range of element connectivity data to read (`int`)
    * `end`: max range of element connectivity data to read (`int`)

    - Return:
    * number of element connectivity data values contained in range (`int`)
    
    """
    cdef cgnslib.cgsize_t ElementDataSize
    self._error=cgnslib.cg_ElementPartialSize(self._root,B,Z,S,start,end,
                                              &ElementDataSize)
    return ElementDataSize

  # -------------------------------------------------------------------------
  cpdef elements_read(self, int B, int Z, int S):
    """
    Returns a tuple with the element connectivity data and the parent data.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section id (`int`)

    - Return:
    * element connectivity data (`numpy.ndarray`)
    * For boundary of interface elements, the `ParentData` array 
      contains information on the
     cells and cell faces sharing the element. (`numpy.ndarray`)
    
    """
    data_size=self.ElementDataSize(B,Z,S)
    elements=PNY.ones((data_size),dtype=PNY.int32)
    parent_data=PNY.ones((data_size),dtype=PNY.int32)
    elementsptr=<int *>CNY.PyArray_DATA(elements)
    parent_dataptr=<int *>CNY.PyArray_DATA(parent_data)
    self._error=cgnslib.cg_elements_read(self._root,B,Z,S,elementsptr,
                                         parent_dataptr)
    return (elements,parent_data)
  
  # ---------------------------------------------------------------------
  cpdef elements_partial_read(self,int B,int Z, int S, cgnslib.cgsize_t start,
                              cgnslib.cgsize_t end):
    """
    Returns a tuple with the element connectivity data and the parent data
    for a given range.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section id (`int`)
    * `start` : min range of element connectivity data to read (`int`)
    * `end`: max range of element connectivity data to read (`int`)
    

    - Return:
    * element connectivity data (`numpy.ndarray`)
    * For boundary of interface elements, the `ParentData` array contains 
      information on the cells and cell faces sharing the element. 
      (`numpy.ndarray`)
    
    """
    elt=self.section_read(B,Z,S)[1]
    elt_type=self.npe(elt)
    size=(end-start+1)*elt_type
    elements=PNY.ones((size),dtype=PNY.int32)
    parent_data=PNY.ones((size),dtype=PNY.int32)
    elementsptr=<int *>CNY.PyArray_DATA(elements)
    parent_dataptr=<int *>CNY.PyArray_DATA(parent_data)
    self._error=cgnslib.cg_elements_partial_read(self._root,B,Z,S,
                                                 start,end,elementsptr,
                                                 parent_dataptr)
    return (elements,parent_data)

  # -------------------------------------------------------------------------  
  cpdef section_write(self,int B, int Z, char * SectionName, 
                      cgnslib.ElementType_t type,
                      cgnslib.cgsize_t start, cgnslib.cgsize_t end, int nbndry,
                      elements):

    """
    Creates a new element section.
    

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `SectionName` : name of the element section (`string`)
    * `type` : type of element (`int`)
    * `start` : min range of element connectivity data to write (`int`)
    * `end`: max range of element connectivity data to write (`int`)
    * `nbndry` : index of last boundary element (`int`)
      If the elements are unsorted, this index is set to 0.
    * `elements` : element connectivity data (`numpy.ndarray`)
    

    - Return:
    * element section id (`int`) 
    
    """
    cdef int S=-1
    elements=PNY.int32(elements)
    elementsptr=<int *>CNY.PyArray_DATA(elements)
    self._error=cgnslib.cg_section_write(self._root,B,Z,
                                         SectionName,type,start,end,
                                         nbndry,elementsptr,&S)
    return S

  # ---------------------------------------------------------------------------
  cpdef parent_data_write(self, int B, int Z, int S, parent_data):
    """
    Writes parent info for an element section.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section id which is comprised between 1 and the 
      total number of element sections(`int`)
    * `parent_data` : For boundary of interface elements, 
      the `ParentData` array contains
      information on the cells and cell faces sharing the element. 
      (`numpy.ndarray`)
    
    """
    parent_data=PNY.int32(parent_data)
    parent_dataptr=<int *>CNY.PyArray_DATA(parent_data)
    self._error=cgnslib.cg_parent_data_write(self._root,B,Z,S,parent_dataptr)

  # ---------------------------------------------------------------------------
  cpdef section_partial_write(self, int B, int Z, char * SectionName,
                              cgnslib.ElementType_t type, 
                              cgnslib.cgsize_t start,
                              cgnslib.cgsize_t end, int nbndry):
    """
    Writes subset of element data.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `SectionName` : name of the element section (`string`)
    * `type` : type of element (`int`)
    * `start` : index of first element in the section to write (`int`)
    * `end` : index of last element in the section to write (`int`)
    * `nbndry` : index of last boundary element in the section (`int`)
      If the elements are unsorted, this index is set to 0.
    
    - Return:
    * element section index
    
    """
    cdef int S=-1
    self._error=cgnslib.cg_section_partial_write(self._root,B,Z,
                                                 SectionName,type,
                                                 start,end,nbndry,&S)
    return S

  # ---------------------------------------------------------------------------
  cpdef elements_partial_write(self, int B, int Z, int S, 
                               cgnslib.cgsize_t start,
                               cgnslib.cgsize_t end, elements):
    """
    Writes element data for an element section.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section id (`int`)
    * `start` : index of first element in the section to write (`int`)
    * `end` : index of last element in the section to write (`int`)
    * `elements` : element conncetivity data (`numpy.ndarray`)

    """
    elements=PNY.int32(elements)
    elementsptr=<int *>CNY.PyArray_DATA(elements)
    self._error=cgnslib.cg_elements_partial_write(self._root,B,Z,S,start,end,
                                                  elementsptr)

  # ---------------------------------------------------------------------------
  cpdef parent_data_partial_write(self, int B, int Z, int S, 
                                  cgnslib.cgsize_t start,
                                  cgnslib.cgsize_t end, parent_data):
    """
    Writes subset of parent info for an element section.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section index number (`int`)
    * `start` : index of first element in the section (`int`) 
    * `end` : index of last element in the section (`int`)   
    * `parent_data` : For boundary of interface elements, the `ParentData` 
       array contains cells and cell faces sharing the element. 
       (`numpy.ndarray`)

    - Return:
    * None
    
    """
    parent_data=PNY.int32(parent_data)
    parent_dataptr=<int *>CNY.PyArray_DATA(parent_data)
    self._error=cgnslib.cg_parent_data_partial_write(self._root,B,Z,S,
                                                     start,end,
                                                     parent_dataptr)

  # ---------------------------------------------------------------------------
  cpdef nbocos(self, int B, int Z):
      """
      Gets number of boundary conditions.
      
      - Args:
      * `B` : base id (`int`)
      * `Z` : zone id (`int`)

      - Return:
      * number of boundary conditions in zone `Z` (`int`)

      """
      cdef int n=0
      self._error=cgnslib.cg_nbocos(self._root,B,Z,&n)
      return n

  # ---------------------------------------------------------------------------
  cpdef boco_info(self, int B, int Z, int BC):
    """
    Gets info from a given boundary condition. Returns info in a tuple.

    - Args:
    * `B` : base id (`int`)(`numpy.ndarray`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)

    - Return:
    * name of the boundary condition (`string`)
    * type of boundary condition defined (`int`)
    * extent of the bc (`int`). The extent may be defined using a range of
      points or elements using `PointRange`using, or using a discrete 
      list of all points or
      elements at which the boundary condition is applied using `PointList`.
    * number of points or elements defining the bc region (`int`)
      For a `ptset_type` of `PointRange`, the number is always 2. 
      For a `ptset_type` of `PointList`,
      the number is equal to the number of points or elements in the list.
    * index vector indicating the computational coordinate direction of bc
      patch normal (`numpy.ndarray`)
    * flag indicating if the normals are defined in `NormalList`(`int`)
      Returns 1 if they are defined, 0 if they are not.
    * data type used in the definition of the normals (`int`)
      Admissible data types are `RealSingle` and `RealDouble`.
    * number of bc datasets for the current boundary condition (`int`)
      
    """
    cdef char * boconame= " "
    cdef cgnslib.BCType_t bocotype
    cdef cgnslib.PointSetType_t ptset_type
    cdef cgnslib.cgsize_t npnts
    NormalIndex=PNY.zeros((3,),dtype=PNY.int32)
    NormalIndexptr=<int *>CNY.PyArray_DATA(NormalIndex)
    cdef cgnslib.cgsize_t NormalListFlag
    cdef cgnslib.DataType_t NormalDataType
    cdef int ndataset
    self._error=cgnslib.cg_boco_info(self._root,B,Z,BC,boconame,
                                     &bocotype,&ptset_type,
                                     &npnts,NormalIndexptr,&NormalListFlag,
                                     &NormalDataType,
                                     &ndataset)
    return (boconame,bocotype,ptset_type,npnts,NormalIndex,
            NormalListFlag,NormalDataType,
            ndataset)

  # ---------------------------------------------------------------------------
  cpdef boco_id(self, int B, int Z, int BC):
    cdef double id=-1
    self._error=cgnslib.cg_boco_id(self._root,B,Z,BC,&id)
    return id

  # ---------------------------------------------------------------------------
  cpdef boco_write(self, int B, int Z, char * boconame, 
                   cgnslib.BCType_t bocotype,
                   cgnslib.PointSetType_t ptset_type, 
                   cgnslib.cgsize_t npnts, pnts):
    """
    Creates a new boundary condition.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `boconame` : name of the boundary condition (`string`)
    * `bocotype` : type of the boundary condition (`int`)
    * `ptset_type : extent of the boundary condition (`int`)
    * `npnts` : number of points or elements defining the boundary 
       condition region (`int`)
    * `pnts` : array of point or element indices defining the boundary 
      condition region (`numpy.ndarray`)

    - Return:
    * boundary condition id (`int`)

    """
    cdef int BC=-1
    array=PNY.int32(pnts)
    arrayptr=<int *>CNY.PyArray_DATA(array)
    self._error=cgnslib.cg_boco_write(self._root,B,Z,boconame,
                                      bocotype,ptset_type,npnts,arrayptr,
                                      &BC)
    return BC

  # ---------------------------------------------------------------------------
  cpdef boco_normal_write(self, int B, int Z, int BC, NormalIndex,
                          int NormalListFlag, NormalDataType=None, 
                          NormalList=None):
    """
    Writes the normals of a given `BC` boundary condition.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)
    * `NormalIndex`:index vector indicating the computational 
      coordinate direction of the
      boundary condition patch normal (`numpy.ndarray`)
    * `NormalListFlag`: flag indicating if the normals are defined 
      in `NormalList`(`int`) .
      The flag is equal to 1 if they are defined, 0 if they are not. 
      If the flag is forced to 0,
      'NormalDataType' and 'NormalList' are not taken into account. 
      In this case, these arguments are not required.
    * `NormalDataType`: data type of the normals (`int`).
      Admissible data types are `RealSingle` and `RealDouble`. 
    * `NormalList`: list of vectors normal to the boundary condition 
      patch pointing into the interior of the zone (`numpy.ndarray`)

    - Returns:
    * None
    
    """
    nix=PNY.int32(NormalIndex)
    nixptr=<int *>CNY.PyArray_DATA(NormalIndex)
    if (NormalListFlag==1):
      ndt=<cgnslib.DataType_t>NormalDataType
      if (ndt==CK.RealDouble):
        nl=PNY.float64(NormalList)
        nlptrD=<double *>CNY.PyArray_DATA(nl)
        self._error=cgnslib.cg_boco_normal_write(self._root,B,Z,BC,nixptr,
                                                 NormalListFlag,ndt,nlptrD)
      else:
        nl=PNY.float32(NormalList)
        nlptrS=<float *>CNY.PyArray_DATA(nl)
        self._error=cgnslib.cg_boco_normal_write(self._root,B,Z,BC,nixptr,
                                                 NormalListFlag,ndt,nlptrS)
    else:
      nl=PNY.ones((1,))
      nlptrS=<float *>CNY.PyArray_DATA(nl)
      NormalDataType=3
      ndt=<cgnslib.DataType_t>NormalDataType
      self._error=cgnslib.cg_boco_normal_write(self._root,B,Z,BC,nixptr,
                                                 NormalListFlag,ndt,nlptrS)

  # ---------------------------------------------------------------------------
  cpdef boco_read(self, int B, int Z, int BC):
    """
    Reads boundary condition data and normals.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)

    - Returns:
    * array of point or element indices defining the boundary condition 
      region (`numpy.ndarray`)
    * list of vectors normal to the boundary condition patch pointing into 
      the interior of the zone (`numpy.ndarray`)
    
    """
    cdef int NormalList
    cdef double * nlptrD
    cdef float * nlptrS
    npnts=self.boco_info(B,Z,BC)[3]
    nls=self.boco_info(B,Z,BC)[5]
    datatype=self.boco_info(B,Z,BC)[6]
    ztype=self.zone_type(B,Z)
    pdim=self.base_read(B)[3]
    if (ztype==CK.Unstructured): dim=1
    elif (ztype==CK.Structured): dim=self.base_read(B)[2]
    else: return None
    pnts=PNY.zeros((npnts,dim),dtype=PNY.int32)
    pntsptr=<int *>CNY.PyArray_DATA(pnts)
    if (nls==0):
      nl=PNY.zeros((3,))
    else:
      if (datatype==CK.RealDouble):
        nl=PNY.ones((nls/pdim,pdim),dtype=PNY.float64)
        nlptrD=<double *>CNY.PyArray_DATA(nl)
        self._error=cgnslib.cg_boco_read(self._root,B,Z,BC,pntsptr,nlptrD)
      else:
        nl=PNY.ones((nls/pdim,pdim),dtype=PNY.float32)
        nlptrS=<float *>CNY.PyArray_DATA(nl)
        self._error=cgnslib.cg_boco_read(self._root,B,Z,BC,pntsptr,nlptrS)
    return (pnts,nl)

  # ---------------------------------------------------------------------------
  cpdef boco_gridlocation_read(self, int B, int Z, int BC):
    """
    Returns the location of a given boundary condition.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)

    - Return:
    * grid location used in the definition of the point set (`int`)

    """
    cdef cgnslib.GridLocation_t location
    self._error=cgnslib.cg_boco_gridlocation_read(self._root,B,Z,BC,&location)
    return location

  # --------------------------------------------------------------------------
  cpdef boco_gridlocation_write(self, int B, int Z, int BC,
                                cgnslib.GridLocation_t location):
    """
    Writes the boundary condition location.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)
    * `location` : grid location of the point set (`int`)

    - Return:
    * None

    """
    self._error=cgnslib.cg_boco_gridlocation_write(self._root,B,Z,BC,location)

  # --------------------------------------------------------------------------
  cpdef dataset_write(self, int B, int Z, int BC, char * name, 
                      cgnslib.BCType_t BCType):
    """
    Writes the dataset set of a given boundary condition.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)
    * `BCType` : boundary condition type (`int`)

    - Return:
    * dataset id (`int`)    

    """
    cdef int DSet=-1
    self._error=cgnslib.cg_dataset_write(self._root,B,Z,BC,name,BCType,&DSet)
    return DSet

  # ---------------------------------------------------------------------------
  cpdef dataset_read(self, int B, int Z, int BC, int DS):
    """
    Returns a tuple with information about a boundary condition dataset.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)
    * dataset id (`int`)

    - Return:
    * name of the dataset (`string`)
    * boundary condition type (`int`)
    * flag indicating if the dataset contains Dirichlet data (`int`)
    * flag indicating if the dataset contains Neumann data (`int`)

    """
    cdef char * name = ""
    cdef cgnslib.BCType_t bct
    cdef int dflag = 0
    cdef int nflag = 0
    self._error=cgnslib.cg_dataset_read(self._root,B,Z,BC,DS,name,
                                        &bct,&dflag,&nflag)
    return (name,bct,dflag,nflag)

  # ---------------------------------------------------------------------------
  cpdef bcdataset_info(self):
    cdef int n
    self._error=cgnslib.cg_bcdataset_info(&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef bcdataset_read(self, int n):
    cdef char name[MAXNAMELENGTH]
    cdef cgnslib.BCType_t bct
    cdef int fdir
    cdef int fneu
    self._error=cgnslib.cg_bcdataset_read(n,name,&bct,&fdir,&fneu)
    return (n,name,bct,fdir,fneu)

  # ---------------------------------------------------------------------------
  cpdef convergence_read(self):
    cdef char *sptr
    cdef int it
    self._error=cgnslib.cg_convergence_read(&it,&sptr)
    s=str(sptr)
    #cgnslib.cg_free(<void *>sptr)
    return (it, s)

  # ---------------------------------------------------------------------------
  cpdef state_read(self):
    cdef char *sptr
    self._error=cgnslib.cg_state_read(&sptr)
    s=str(sptr)
    #cgnslib.cg_free(<void *>sptr)
    return s

  # ---------------------------------------------------------------------------
  cpdef bcdataset_write(self, char * name, cgnslib.BCType_t bct,
                        cgnslib.BCDataType_t bcdt):
    """
    Create a new BCDataSet::

      db.bcdataset_write("Dataset-1",BCTypeNull,Dirichlet)
    """
    self._error=cgnslib.cg_bcdataset_write(name,bct,bcdt)
    return None

  # --------------------------------------------------------------------------
  cpdef narrays(self):
    """
    Returns the number of data arrays under the current node. 
    pathname by using the `gopath` function.

    - Args:
    * None

    - Returns:
    * number of data arrays contained in a given node (`int`)
    
    """
    cdef int n=0
    self._error=cgnslib.cg_narrays(&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef array_info(self, int A):
    """
    Returns a tuple with information about a given array. 
    You need to access the parent
    node of the requested array. You can use the `gopath` function to do it.

    - Args:
    * `A` : data array id between 1 and the number of arrays
      under the current node (`int`)

    - Return:
    * name of the data array (`string`)
    * type of data held in the `DataArray_t` node (`int`)
    * number of dimensions (`int`)
    * number of data elements in each dimension (`int`)
    
    """
    cdef char * aname = " "
    cdef cgnslib.DataType_t dt
    cdef int dd = -1
    cdef cgnslib.cgsize_t * dvptr
    dv = PNY.ones((CK.ADF_MAX_DIMENSIONS,),dtype=PNY.int32)
    dvptr = <cgnslib.cgsize_t *>CNY.PyArray_DATA(dv)
    self._error=cgnslib.cg_array_info(A,aname,&dt,&dd,dvptr)
    return (aname,dt,dd,dv[0:dd])

 # ----------------------------------------------------------------------------
  cpdef array_read(self, int A):
    """
    Reads a data array contained in a given node. Use 'goto'-like

    - Args:
    * `A` : data array id which is comprised between 1 and the number of arrays
      under the current node (`int`)

    - Return:
    * data array (`numpy.ndarray`)

    """
    cdef int * dataptrI
    cdef float * dataptrF
    cdef double * dataptrD
    cdef char * dataptrC
    
    dt=self.array_info(A)[1]
    dv=self.array_info(A)[3]
    
    if (dt==CK.Integer):
      data=PNY.ones(dv,dtype=PNY.int32)
      dataptrI=<int *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read(A,dataptrI)
    if (dt==CK.LongInteger):
      data=PNY.ones(dv,dtype=PNY.int64)
      dataptrI=<int *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read(A,dataptrI)
    if (dt==CK.RealSingle):
      data=PNY.ones(dv,dtype=PNY.float32)
      dataptrF=<float *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read(A,dataptrF)
    if (dt==CK.RealDouble):
      data=PNY.ones(dv,dtype=PNY.float64)
      dataptrD=<double *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read(A,dataptrD)
    if (dt==CK.Character):
      data=PNY.array((""))
      dataptrC=<char *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read(A,dataptrC)
    return data
  
  # ---------------------------------------------------------------------------
  cpdef array_read_as(self, int A, cgnslib.DataType_t type):
    """
    Reads a data array as a certain type. Should set the target node 
    with `goto`-like function.

    - Args:
    * `A` : data array id which is comprised between 1 and the number of arrays
      under the current node (`int`)
    * `type` : requested type of data held in the array (`int`)

    - Return:
    * data array (`numpy.ndarray`)

    - Remarks:
    * The data array is returned only if its data type corresponds to 
      the required data type.
      Otherwise, nothing is returned.

    """
    cdef int * dataptrI
    cdef float * dataptrF
    cdef double * dataptrD
    cdef char * dataptrC
    
    dv=self.array_info(A)[3]

    if (type==CK.Integer):
      data=PNY.ones(dv,dtype=PNY.int32)
      dataptrI=<int *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read_as(A,type,dataptrI)
    if (type==CK.LongInteger):
      data=PNY.ones(dv,dtype=PNY.int64)
      dataptrI=<int *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read_as(A,type,dataptrI)
    if (type==CK.RealSingle):
      data=PNY.ones(dv,dtype=PNY.float32)
      dataptrF=<float *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read_as(A,type,dataptrF)
    if (type==CK.RealDouble):
      data=PNY.ones(dv,dtype=PNY.float64)
      dataptrD=<double *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read_as(A,type,dataptrD)
    if (type==CK.Character):
      data=PNY.array((""))
      dataptrC=<char *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_read_as(A,type,dataptrC)

    return data
              
  # ---------------------------------------------------------------------------
  cpdef array_write(self, char *aname, adata):
    """
    Creates a new data array.

    - Args:
    * `aname` : name of the data array (`string`)
    * `adata` : data array ('numpy.ndarray`)

    - Return:
    * None

    - Remarks:
    * the datatype is guessed from the numpy.ndarray data type

    """
    cdef cgnslib.cgsize_t dv[12]
    cdef cgnslib.DataType_t dt
    cdef int dd
    cdef void *ptr

    dd=adata.ndim
    ptr=<void*>adata.data
    for n in range(dd):
      dv[n]=adata.shape[n]
    dt=fromnumpydtype(adata.dtype)
    self._error=cgnslib.cg_array_write(aname,dt,dd,dv,ptr)
    return None

  # ---------------------------------------------------------------------------
  cpdef int nfields(self, int B, int Z, int S):
    cdef int n
    self._error=cgnslib.cg_nfields(self._root,B,Z,S,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef field_info(self, int B, int Z, int S, int F):
    cdef cgnslib.DataType_t dtype
    cdef char fieldname[MAXNAMELENGTH]
    self._error=cgnslib.cg_field_info(self._root, B, Z, S, F, &dtype,fieldname)
    return (B,Z,S,F,dtype,fieldname)

  # ---------------------------------------------------------------------------
  cpdef field_read(self, int B, int Z, int S, char *fieldname, 
                   cgnslib.DataType_t dtype):
    cdef int cdim
    cdef int ddim
    cdef int tsize=1
    cgnslib.cg_cell_dim(self._root,B,&cdim)
    dim_vals=PNY.ones((CK.ADF_MAX_DIMENSIONS,),dtype=PNY.int32)
    dim_valsPtr=<int *>CNY.PyArray_DATA(dim_vals)
    cgnslib.cg_sol_size(self._root,B,Z,S,&ddim,dim_valsPtr)
    for ix in xrange(ddim):
       tsize*=dim_vals[ix]
    rmin=PNY.ones((cdim),dtype=PNY.int32)
    rmax=PNY.ones((cdim),dtype=PNY.int32)
    rminPtr=<int *>CNY.PyArray_DATA(rmin)
    rmaxPtr=<int *>CNY.PyArray_DATA(rmax)
    cdtype=asnumpydtype(dtype)
    if (cdtype == None):
        ndtype=fromnumpydtype(dtype)
        if (ndtype == None):
            raise CGNSException(10,"No such data type: %s"%str(ndtype))
        dtype=ndtype
    field=PNY.ones(tsize,dtype=cdtype)
    fieldPtr=<void *>CNY.PyArray_DATA(field)
    self._error=cgnslib.cg_field_read(self._root,B,Z,S,fieldname,dtype,
                                      rminPtr,rmaxPtr,fieldPtr)
    return (B,Z,S,fieldname,dtype,rmin,rmax,field)

  # ---------------------------------------------------------------------------
  cpdef int field_write(self, int B,int Z,int S,
                        cgnslib.DataType_t dtype, char *fieldname,
                        Field_ptr):
    cdef int nfield
    cdef void *fptr
    cdtype=asnumpydtype(dtype)
    if (cdtype == None):
        ndtype=fromnumpydtype(dtype)
        if (ndtype == None):
            raise CGNSException(10,"No such data type: %s"%str(ndtype))
        dtype=ndtype
    fptr=<void *>CNY.PyArray_DATA(Field_ptr)
    self._error=cgnslib.cg_field_write(self._root,B,Z,S,dtype,fieldname,
                                       fptr,&nfield)
    return nfield

  # ---------------------------------------------------------------------------
  cpdef int field_partial_write(self, int B, int Z, int S,
                                cgnslib.DataType_t dtype, char *fieldname,
                                rmin, rmax, field_ptr):
    cdef int nfield
    cdef void *fptr
    cdef cgnslib.cgsize_t *rminptr
    cdef cgnslib.cgsize_t *rmaxptr
    cdtype=asnumpydtype(dtype)
    if (cdtype == None):
        ndtype=fromnumpydtype(dtype)
        if (ndtype == None):
            raise CGNSException(10,"No such data type: %s"%str(ndtype))
        dtype=ndtype
    rminptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(rmin)
    rmaxptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(rmax)
    fptr=<void *>CNY.PyArray_DATA(field_ptr)
    self._error=cgnslib.cg_field_partial_write(self._root,B,Z,S,dtype,
                                               fieldname,rminptr,rmaxptr,
                                               fptr,&nfield)
    return nfield

  # ---------------------------------------------------------------------------
  cpdef sol_ptset_info(self, int B, int Z, int S):
    cdef cgnslib.PointSetType_t pst 
    cdef cgnslib.cgsize_t       npnts
    self._error=cgnslib.cg_sol_ptset_info(self._root,B,Z,S,&pst,&npnts)
    return (B,Z,S,pst,npnts)

  # ---------------------------------------------------------------------------
  cpdef sol_ptset_read(self, int B, int Z, int S):
    cdef cgnslib.PointSetType_t pst 
    cdef cgnslib.cgsize_t       npnts
    cdef CNY.ndarray[dtype=CNY.int32_t,ndim=1] pnts
    cdef cgnslib.cgsize_t *pntsptr
    self._error=cgnslib.cg_sol_ptset_info(self._root,B,Z,S,&pst,&npnts)
    if (self._ok):
      pnts=PNY.zeros((npnts,),dtype=PNY.int32)
      pntsptr=<cgnslib.cgsize_t *>pnts.data
      self._error=cgnslib.cg_sol_ptset_read(self._root,B,Z,S,pntsptr)
    else:
      pnts=None
    return (B,Z,S,pnts)

  # ---------------------------------------------------------------------------
  cpdef int sol_ptset_write(self, int B, int Z, char *solname,
                            cgnslib.GridLocation_t location, 
                            cgnslib.PointSetType_t ptset_type, 
                            cgnslib.cgsize_t npnts,
                            CNY.ndarray pnts):
    cdef int S
    cdef cgnslib.cgsize_t * pntsptr
    pntsptr=<cgnslib.cgsize_t *>pnts.data
    self._error=cgnslib.cg_sol_ptset_write(self._root,B,Z,solname,location,
                                           ptset_type,npnts,pntsptr,&S)
    return S

  # ---------------------------------------------------------------------------
  cpdef int nsubregs(self, int B, int Z):
    cdef int n=0
    self._error=cgnslib.cg_nsubregs(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef int subreg_ptset_write(self, int B, int Z, char *regname,
                               int dimension, cgnslib.GridLocation_t location,
                               cgnslib.PointSetType_t ptset_type,
                               cgnslib.cgsize_t npnts,
                               CNY.ndarray pnts):
    cdef int S
    cdef cgnslib.cgsize_t *pntsptr
    pntsptr=<cgnslib.cgsize_t *>pnts.data
    self._error=cgnslib.cg_subreg_ptset_write(self._root,B,Z,regname,dimension,
                                              location,ptset_type,
                                              npnts,pntsptr,&S)
    return S

  # ---------------------------------------------------------------------------
  cpdef subreg_info(self, int B, int Z, int S):
    cdef char regname[MAXNAMELENGTH]
    cdef int dim,bcname_len,gcname_len
    cdef cgnslib.cgsize_t npnts
    cdef cgnslib.PointSetType_t ptset_type
    cdef cgnslib.GridLocation_t loc
    self._error=cgnslib.cg_subreg_info(self._root,B,Z,S,regname,
                                       &dim,&loc,&ptset_type,&npnts,
                                       &bcname_len,&gcname_len)
    return (B,Z,S,regname,dim,loc,ptset_type,npnts,bcname_len,gcname_len)

  # ---------------------------------------------------------------------------
  cpdef subreg_ptset_read(self, int B, int Z, int S):
    cdef char regname[MAXNAMELENGTH]
    cdef int dim,bcname_len,gcname_len
    cdef cgnslib.cgsize_t npnts
    cdef cgnslib.PointSetType_t ptset_type
    cdef cgnslib.GridLocation_t loc
    cdef CNY.ndarray[dtype=CNY.int32_t,ndim=1] pnts
    cdef cgnslib.cgsize_t *pntsptr
    self._error=cgnslib.cg_subreg_info(self._root,B,Z,S,regname,
                                       &dim,&loc,&ptset_type,&npnts,
                                       &bcname_len,&gcname_len)
    if (self._ok):
      pnts=PNY.zeros((npnts,),dtype=PNY.int32)
      pntsptr=<cgnslib.cgsize_t *>pnts.data
      self._error=cgnslib.cg_subreg_ptset_read(self._root,B,Z,S,pntsptr)
    else:
      pnts=None
    return pnts

  # ---------------------------------------------------------------------------
  cpdef subreg_bcname_read(self, int B, int Z, int S):
    cdef char bcname[MAXNAMELENGTH]
    self._error=cgnslib.cg_subreg_bcname_read(self._root,B,Z,S,bcname)
    return bcname

  # ---------------------------------------------------------------------------
  cpdef subreg_gcname_read(self, int B, int Z, int S):
    cdef char gcname[MAXNAMELENGTH]
    self._error=cgnslib.cg_subreg_gcname_read(self._root,B,Z,S,gcname)
    return gcname

  # ---------------------------------------------------------------------------
  cpdef int subreg_bcname_write(self, int B, int Z, char *regname,
                                int dim, char *bcname):
    cdef int S
    self._error=cgnslib.cg_subreg_bcname_write(self._root,B,Z,regname,dim,
                                               bcname,&S)
    return S

  # ---------------------------------------------------------------------------
  cpdef int subreg_gcname_write(self, int B, int Z, char *regname,
                                int dim, char *gcname):
    cdef int S
    self._error=cgnslib.cg_subreg_gcname_write(self._root,B,Z,regname,dim,
                                               gcname,&S)
    return S

  # ---------------------------------------------------------------------------
  cpdef int nholes(self, int B, int Z):
    cdef int n=0
    self._error=cgnslib.cg_nholes(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef hole_info(self, int B, int Z, int I):
    cdef int np,nps
    cdef char name[MAXNAMELENGTH]
    cdef cgnslib.GridLocation_t gloc
    cdef cgnslib.PointSetType_t ptst
    self._error=cgnslib.cg_hole_info(self._root,B,Z,I,name,&gloc,&ptst,&nps,&np)
    return (B,Z,I,name,gloc,ptst,nps,np)

  # ---------------------------------------------------------------------------
  cpdef hole_read(self, int B, int Z, int I):
    cdef int np,nps
    cdef char name[MAXNAMELENGTH]
    cdef cgnslib.GridLocation_t gloc
    cdef cgnslib.PointSetType_t ptst
    cdef cgnslib.cgsize_t *pntsptr
    self._error=cgnslib.cg_hole_info(self._root,B,Z,I,name,
                                     &gloc,&ptst,&nps,&np)
    pnts=PNY.zeros((np,),dtype=PNY.int32) # should check for 64
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(pnts)
    self._error=cgnslib.cg_hole_read(self._root,B,Z,I,pntsptr)
    return pnts

  # ---------------------------------------------------------------------------
  cpdef field_id(self, int B, int Z, int S, int F):
    cdef double id=-1
    self._error=cgnslib.cg_field_id(self._root,B,Z,S,F,&id)
    return id

  # ---------------------------------------------------------------------------
  cpdef hole_id(self, int B, int Z, int I):
    cdef double id=-1
    self._error=cgnslib.cg_hole_id(self._root,B,Z,I,&id)
    return id

  # ---------------------------------------------------------------------------
  cpdef int hole_write(self, int B, int Z, char * holename,
                       cgnslib.GridLocation_t location, 
                       cgnslib.PointSetType_t ptset_type,
                       int nptsets, cgnslib.cgsize_t npnts,
                       CNY.ndarray pnts):
    cdef int H
    cdef cgnslib.cgsize_t *pntsptr
    pntsptr=<cgnslib.cgsize_t *>pnts.data
    self._error=cgnslib.cg_hole_write(self._root,B,Z,holename,location,
                                      ptset_type,nptsets,
                                      npnts,pntsptr,&H)
    return H

  # ---------------------------------------------------------------------------
  cpdef int n_rigid_motions(self, int B, int Z):
    cdef int n
    self._error=cgnslib.cg_n_rigid_motions(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef rigid_motion_read(self, int B, int Z, int R):
    cdef char name[MAXNAMELENGTH]
    cdef cgnslib.RigidGridMotionType_t t
    self._error=cgnslib.cg_rigid_motion_read(self._root,B,Z,R,name,&t)
    return (B,Z,name,t)

  # ---------------------------------------------------------------------------
  cpdef rigid_motion_write(self, int B, int Z,
                           char *name, cgnslib.RigidGridMotionType_t t):
    cdef int r
    self._error=cgnslib.cg_rigid_motion_write(self._root,B,Z,name,t,&r)
    return r

  # ---------------------------------------------------------------------------
  cpdef int arbitrary_motion_write(self, int B, int Z,
                                   char *name, 
                                   cgnslib.ArbitraryGridMotionType_t t):
    cdef int a
    self._error=cgnslib.cg_arbitrary_motion_write(self._root,B,Z,name,t,&a)
    return a

  # ---------------------------------------------------------------------------
  cpdef int n_arbitrary_motions(self, int B, int Z):
    cdef int n=0
    self._error=cgnslib.cg_n_arbitrary_motions(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef arbitrary_motion_read(self, int B, int Z, int A):
    cdef char aname[MAXNAMELENGTH]
    cdef cgnslib.ArbitraryGridMotionType_t atype
    self._error=cgnslib.cg_arbitrary_motion_read(self._root,B,Z,A,aname,&atype)
    return (aname, atype)

  # ---------------------------------------------------------------------------
  cpdef int simulation_type_read(self, int B):
    cdef cgnslib.SimulationType_t t
    self._error=cgnslib.cg_simulation_type_read(self._root,B,&t)
    return t

  # ---------------------------------------------------------------------------
  cpdef simulation_type_write(self, int B,
                                  cgnslib.SimulationType_t stype):
    self._error=cgnslib.cg_simulation_type_write(self._root,B,stype)
    return None

  # ---------------------------------------------------------------------------
  cpdef biter_read(self, int B):
    cdef int it
    cdef char name[MAXNAMELENGTH]
    self._error=cgnslib.cg_biter_read(self._root,B,name,&it)
    return (B,name,it)

  # ---------------------------------------------------------------------------
  cpdef biter_write(self, int B, char *name, int steps):
    self._error=cgnslib.cg_biter_write(self._root,B,name,steps)

  # ---------------------------------------------------------------------------
  cpdef ziter_read(self, int B, int Z):
    cdef char name[MAXNAMELENGTH]
    self._error=cgnslib.cg_ziter_read(self._root,B,Z,name)
    return (B,Z,name)

  # ---------------------------------------------------------------------------
  cpdef ziter_write(self, int B, int Z, char *name):
    self._error=cgnslib.cg_ziter_write(self._root,B,Z,name)
    return None

  # ---------------------------------------------------------------------------
  cpdef gravity_read(self, int B):
    cdef int dv
    cgnslib.cg_cell_dim(self._root,B,&dv)
    gravity_vector=PNY.ones(dv,dtype=PNY.float32)
    gravity_vectorPtr=<float *>CNY.PyArray_DATA(gravity_vector)
    self._error=cgnslib.cg_gravity_read(self._root,B,gravity_vectorPtr)
    return (B,gravity_vector)

  # ---------------------------------------------------------------------------
  cpdef gravity_write(self, int B, CNY.ndarray gravityvector):
    cdef float *gvector
    gvector=<float *>gravityvector.data
    self._error=cgnslib.cg_gravity_write(self._root,B,gvector)
    return None

  # ---------------------------------------------------------------------------
  cpdef axisym_read(self, int B):
    cdef int dv
    cgnslib.cg_cell_dim(self._root,B,&dv)
    axis=PNY.ones(dv,dtype=PNY.float32)
    axisPtr=<float *>CNY.PyArray_DATA(axis)
    ref_point=PNY.ones(dv,dtype=PNY.float32)
    ref_pointPtr=<float *>CNY.PyArray_DATA(ref_point)
    self._error=cgnslib.cg_axisym_read(self._root,B,ref_pointPtr,axisPtr)
    return (B,ref_point,axis)

  # ---------------------------------------------------------------------------
  cpdef axisym_write(self, int B, CNY.ndarray refpoint, CNY.ndarray axis):
    cdef float *refpointptr,*axisptr
    refpointptr=<float *>refpoint.data
    axisptr=<float *>axis.data
    self._error=cgnslib.cg_axisym_write(self._root,B,refpointptr,axisptr)
    return None

  # ---------------------------------------------------------------------------
  cpdef rotating_read(self):
    cdef int dv,B
    cdef float *rotrateptr,*rotcenterptr
    cdef CNY.ndarray rotrate
    cdef CNY.ndarray rotcenter
    B=self.where()[0]
    cgnslib.cg_cell_dim(self._root,B,&dv)
    rotrate=PNY.ones(dv,dtype=PNY.float32)
    rotrateptr=<float *>rotrate.data
    rotcenter=PNY.ones(dv,dtype=PNY.float32)
    rotcenterptr=<float *>rotcenter.data
    self._error=cgnslib.cg_rotating_read(rotrateptr,rotcenterptr)
    return (rotrate,rotcenter)

  # ---------------------------------------------------------------------------
  cpdef rotating_write(self, CNY.ndarray rotrate, CNY.ndarray rotcenter):
    cdef float *rotrateptr,*rotcenterptr
    rotrateptr=<float *>rotrate.data
    rotcenterptr=<float *>rotcenter.data
    self._error=cgnslib.cg_rotating_write(rotrateptr,rotcenterptr)
    return None

  # ---------------------------------------------------------------------------
  cpdef int bc_wallfunction_read(self, int B, int Z, int BC):
    cdef cgnslib.WallFunctionType_t wallfctype
    self._error=cgnslib.cg_bc_wallfunction_read(self._root,B,Z,BC,&wallfctype)
    return wallfctype

  # ---------------------------------------------------------------------------
  cpdef bc_wallfunction_write(self, int B, int Z, int BC,
                              cgnslib.WallFunctionType_t wallfctype):
    self._error=cgnslib.cg_bc_wallfunction_write(self._root,B,Z,BC,wallfctype)
    return None

  # ---------------------------------------------------------------------------
  cpdef bc_area_read(self, int B, int Z, int BC):
    cdef cgnslib.AreaType_t areatype
    cdef char regname[MAXNAMELENGTH]
    cdef float areasurf
    self._error=cgnslib.cg_bc_area_read(self._root,B,Z,BC,&areatype,
                                        &areasurf,regname)
    return (areatype,areasurf,regname)

  # ---------------------------------------------------------------------------
  cpdef bc_area_write(self, int B, int Z, int BC,
                          cgnslib.AreaType_t areatype, float areasurf, 
                          char *regname):
    self._error=cgnslib.cg_bc_area_write(self._root,B,Z,BC,areatype,
                                        areasurf,regname)
    return None

  # ---------------------------------------------------------------------------
  cpdef conn_periodic_read(self, int B, int Z, int I):
    cdef int dv
    cdef CNY.ndarray rotcenter,rotangle,translation
    cdef float *rotcenterptr,*rotangleptr,*translationptr
    cgnslib.cg_cell_dim(self._root,B,&dv)
    rotcenter=PNY.ones(dv,dtype=PNY.float32)
    rotangle=PNY.ones(dv,dtype=PNY.float32)
    translation=PNY.ones(dv,dtype=PNY.float32)
    rotcenterptr=<float *>rotcenter.data
    rotangleptr=<float *>rotangle.data
    translationptr=<float *>translation.data
    self._error=cgnslib.cg_conn_periodic_read(self._root,B,Z,I,
                                              rotcenterptr,rotangleptr,
                                              translationptr)
    return (B,Z,I,rotcenter,rotangle,translation)

  # ---------------------------------------------------------------------------
  cpdef conn_periodic_write(self, int B, int Z, int I,
                                CNY.ndarray rotcenter, CNY.ndarray rotangle,
                                CNY.ndarray translation):
    cdef float *rotcenterptr,*rotangleptr,*translationptr
    rotcenterptr=<float *>rotcenter.data
    rotangleptr=<float *>rotangle.data
    translationptr=<float *>translation.data
    self._error=cgnslib.cg_conn_periodic_write(self._root,B,Z,I,
                                               rotcenterptr,rotangleptr,
                                               translationptr)
    return None

  # ---------------------------------------------------------------------------
  cpdef conn_1to1_periodic_read(self, int B, int Z, int I):
    cdef int dv
    cdef CNY.ndarray rotcenter,rotangle,translation
    cdef float *rotcenterptr,*rotangleptr,*translationptr
    cgnslib.cg_cell_dim(self._root,B,&dv)
    rotcenter=PNY.ones(dv,dtype=PNY.float32)
    rotangle=PNY.ones(dv,dtype=PNY.float32)
    translation=PNY.ones(dv,dtype=PNY.float32)
    rotcenterptr=<float *>rotcenter.data
    rotangleptr=<float *>rotangle.data
    translationptr=<float *>translation.data
    self._error=cgnslib.cg_1to1_periodic_read(self._root,B,Z,I,
                                              rotcenterptr,rotangleptr,
                                              translationptr)
    return (B,Z,I,rotcenter,rotangle,translation)

  # ---------------------------------------------------------------------------
  cpdef conn_1to1_periodic_write(self, int B, int Z, int I,
                                 CNY.ndarray rotcenter, CNY.ndarray rotangle,
                                 CNY.ndarray translation):
    cdef float *rotcenterptr,*rotangleptr,*translationptr
    rotcenterptr=<float *>rotcenter.data
    rotangleptr=<float *>rotangle.data
    translationptr=<float *>translation.data
    self._error=cgnslib.cg_1to1_periodic_write(self._root,B,Z,I,
                                               rotcenterptr,rotangleptr,
                                               translationptr)
    return None

  # ---------------------------------------------------------------------------
  cpdef int conn_average_read(self, int B, int Z, int I):
    cdef cgnslib.AverageInterfaceType_t averagetype
    self._error=cgnslib.cg_conn_average_read(self._root,B,Z,I,&averagetype)
    return averagetype

  # ---------------------------------------------------------------------------
  cpdef conn_average_write(self, int B, int Z, int I,
                           cgnslib.AverageInterfaceType_t averagetype):
    self._error=cgnslib.cg_conn_average_write(self._root,B,Z,I,averagetype)
    return None

  # ---------------------------------------------------------------------------
  cpdef conn_1to1_average_write(self, int B, int Z, int I,
                                cgnslib.AverageInterfaceType_t averagetype):
    self._error=cgnslib.cg_1to1_average_write(self._root,B,Z,I,averagetype)
    return None

  cpdef int conn_1to1_average_read(self, int B, int Z, int I):
    cdef cgnslib.AverageInterfaceType_t averagetype
    self._error=cgnslib.cg_1to1_average_read(self._root,B,Z,I,&averagetype)
    return averagetype

  # ---------------------------------------------------------------------------
  cpdef rind_read(self):
    cdef CNY.ndarray rind
    cdef int *rindptr,cdim
    W=self.where()
    B=W[0]
    Z=W[1][0][1]
    ztype=self.zone_type(B,Z)
    if (ztype==CK.Unstructured): cdim=1
    elif (ztype==CK.Structured): cdim=self.base_read(B)[2]  
    else: return -1
    rind=PNY.ones((cdim*2,),dtype=PNY.int32)
    rindptr=<int *>rind.data
    self._error=cgnslib.cg_rind_read(rindptr)
    return rind

  # ---------------------------------------------------------------------------
  cpdef rind_write(self, CNY.ndarray rind):
    self._error=cgnslib.cg_rind_write(<int *>rind.data)
    return None

  # ---------------------------------------------------------------------------
  cpdef ptset_write(self, cgnslib.PointSetType_t ptset_type, CNY.ndarray pnts):
    cdef int npnts
    npnts=pnts.size
    self._error=cgnslib.cg_ptset_write(ptset_type,npnts,<int *>pnts.data)
    return None


  # ---------------------------------------------------------------------------
  cpdef int ptset_read(self):
    cdef cgnslib.PointSetType_t pst
    cdef cgnslib.cgsize_t npnts
    cdef cgnslib.cgsize_t *pntsptr
    cdef CNY.ndarray pnts
    self._error=cgnslib.cg_ptset_info(&pst,&npnts)
    pnts=PNY.ones((npnts,),dtype=PNY.int32)
    pntsptr=<cgnslib.cgsize_t *>pnts.data
    self._error=cgnslib.cg_ptset_read(pntsptr)
    return pnts

  # ---------------------------------------------------------------------------
  cpdef delete_node(self, char *path):
    self._error=cgnslib.cg_delete_node(path)
    return None

  # ---------------------------------------------------------------------------
  cpdef cell_dim(self, int B):
    cdef int cdim
    self._error=cgnslib.cg_cell_dim(self._root,B,&cdim)
    return cdim

  # ---------------------------------------------------------------------------
  cpdef index_dim(self, int B, int Z):
    cdef int idim
    self._error=cgnslib.cg_index_dim(self._root,B,Z,&idim)
    return idim

  # ---------------------------------------------------------------------------
  cpdef int is_link(self):
    cdef int plen
    self._error=cgnslib.cg_is_link(&plen)
    return plen

  # ---------------------------------------------------------------------------
  cpdef link_read(self, filename, link_path):
    cdef char *lfileptr=NULL
    cdef char *lpathptr=NULL
    lf=str(lfileptr)
    lp=str(lpathptr)
    cgnslib.cg_link_read(&lfileptr,&lpathptr)
    #cgnslib.cg_free(<void *>lfileptr)
    #cgnslib.cg_free(<void *>lpathptr)
    return (lf,lp)

  # ---------------------------------------------------------------------------
  cpdef link_write(self, char * nodename, char * filename,
                   char * name_in_file):
    self._error=cgnslib.cg_link_write(nodename,filename,name_in_file)

  # ---------------------------------------------------------------------------
  cpdef bcdata_write(self, int B, int Z, int BC, int Dset,
                     cgnslib.BCDataType_t bct):
    self._error=cgnslib.cg_bcdata_write(self._root,B,Z,BC,Dset,bct)

  # ---------------------------------------------------------------------------
  cpdef nuser_data(self):
    """
    Counts the number of `UserDefinedData_t` nodes contained in 
    the current node. You can access
    the current node by using the `gopath` function.

    - Args:
    * None

    - Return:
    * number of `UserDefinedData_t` nodes contained in the current node (`int`)

    """
    cdef int n=0
    self._error=cgnslib.cg_nuser_data(&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef user_data_write(self, char * usn):
    """
    Creates a new `UserDefinedData_t` node. You can set the position of
    the node in the `CGNS tree` with the `gopath` function.

    - Args:
    * `usn` : name of the created node (`string`)

    - Return:
    * None

    """
    self._error=cgnslib.cg_user_data_write(usn)
    return None

  # ----------------------------------------------------------------------------
  cpdef user_data_read(self, int Index):
    """
    Returns the name of a given `UserDefinedData_t` node. You can access 
    the node by using the `gopath` function.

    - Args:
    * `Index` : user-defined data id which is necessarily comprised 
      between 1 and the total number of `UserDefinedData_t` nodes under 
      the current node (`int`)

    - Return:
    * name of the required `UserDefinedData_t` node (`string`)

    """
    cdef char * usn = " "
    self._error=cgnslib.cg_user_data_read(Index,usn)
    return usn

  # ---------------------------------------------------------------------------
  cpdef discrete_write(self, int B, int Z, char * name):
    """
    Creates a new `DiscreteData_t` node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `name` : name of the created node (`string`)

    - Return:
    * discreted data id (`int`)
    
    """
    cdef int D = -1
    self._error=cgnslib.cg_discrete_write(self._root,B,Z,name,&D)
    return D

  # ---------------------------------------------------------------------------
  cpdef ndiscrete(self, int B, int Z):
    """
    Returns the number of `DiscreteData_t` nodes in a given zone.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Return:
    * number of `DiscreteData_t`nodes contained in the zone (`int`)
    
    """
    cdef int n=0
    self._error=cgnslib.cg_ndiscrete(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef discrete_read(self, int B, int Z, int D):
    """
    Returns the name of a given `DiscreteData_t` node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised 
      between 1 and the total number of discrete data nodes 
      under the zone (`int`)

    - Return:
    * name of discrete data node (`string`)
    
    """
    cdef char * name = " "
    self._error=cgnslib.cg_discrete_read(self._root,B,Z,D,name)
    return name

  # ---------------------------------------------------------------------------
  cpdef discrete_size(self, int B, int Z, int D):
    """
    Returns the dimensions of a `DiscreteData_t` node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised 
      between 1 and the total number
      of discrete data nodes under the zone (`int`)

    - Return:
    * number of dimensions defining the discrete data (`int`). 
      If a point set has been defined,
      this is 1, otherwise this is the current zone index dimension.
    * array of dimensions ('numpy.ndarray`)
      
    """
    cdef int dd = -1
    cdef cgnslib.cgsize_t * dvptr
    dv=PNY.ones((CK.ADF_MAX_DIMENSIONS,),dtype=PNY.int32)    
    dvptr = <cgnslib.cgsize_t *>CNY.PyArray_DATA(dv)
    self._error=cgnslib.cg_discrete_size(self._root,B,Z,D,&dd,dvptr)
    return (dd,dv[0:dd])

  # ---------------------------------------------------------------------------
  cpdef discrete_ptset_info(self, int B, int Z, int D):
    """
    Returns a tuple with information about a given discrete data node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised 
      between 1 and the total number
      of discrete data nodes under the zone (`int`)

    - Return:
    * type of point set defining the interface ('int`). 
      It can be `PointRange` or `PointList`.
    * number of points defining the interface (`int`)

    """
    cdef cgnslib.PointSetType_t pst 
    cdef cgnslib.cgsize_t npnts 
    self._error=cgnslib.cg_discrete_ptset_info(self._root,B,Z,D,&pst,&npnts)
    return (pst,npnts)

  # ---------------------------------------------------------------------------
  cpdef discrete_ptset_read(self, int B, int Z, int D):
    """
    Reads a point set of a given discrete data node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised 
      between 1 and the total number
      of discrete data nodes under the zone (`int`)

    - Return:
    * array of points defining the interface ('numpy.ndarray`)

    """
    cdef cgnslib.cgsize_t * pntsptr
    npnts=self.discrete_ptset_info(B,Z,D)[1]    
    ztype=self.zone_type(B,Z)
    pdim=self.base_read(B)[3]
    if (ztype==CK.Unstructured):
      dim=1
    elif (ztype==CK.Structured):
      dim=self.base_read(B)[2]  
    pnts=PNY.zeros((npnts,dim),dtype=PNY.int32)
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(pnts)
    self._error=cgnslib.cg_discrete_ptset_read(self._root,B,Z,D,pntsptr)
    return pnts

  # ---------------------------------------------------------------------------
  cpdef discrete_ptset_write(self, int B, int Z, char * name, 
                             cgnslib.GridLocation_t location,
                             cgnslib.PointSetType_t pst, 
                             cgnslib.cgsize_t npnts, pts):
    """
    Creates a new point set `DiscreteData_t` node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised 
      between 1 and the total number
      of discrete data nodes under the zone (`int`)

    - Return:
    * array of points defining the interface ('numpy.ndarray`)

    """
    cdef cgnslib.cgsize_t * pntsptr
    cdef int D = -1
    pnts=PNY.int32(pts)
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(pnts)
    self._error=cgnslib.cg_discrete_ptset_write(self._root,B,Z,name,
                                                location,pst,npnts,
                                                pntsptr,&D)
    return D

  # ---------------------------------------------------------------------------
  cpdef nzconns(self, int B, int Z):
    """
    Returns the number of `ZoneGridConnectivity_t` nodes.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Returns:
    * number of `ZoneGridConnectivity_t` nodes (`int`)

    """
    cdef int n=0
    self._error=cgnslib.cg_nzconns(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef zconn_read(self, int B, int Z, int C):
    """
    Returns the name of the `ZoneGridConnectivity_t` node.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `C` : zone grid connectivity id (`int`)

    - Returns:
    * name of the `ZoneGridConnectivity_t` node (`string`)

    """
    cdef char * name = " "
    self._error=cgnslib.cg_zconn_read(self._root,B,Z,C,name)
    return name

  # ---------------------------------------------------------------------------
  cpdef zconn_write(self, int B, int Z, char * name):
    """
    Creates a new `ZoneGridConnectivity_t` node.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `name` : name of the `ZoneGridConnectivity_t` node (`string`)

    - Returns:
    * zone grid connectivity id (`int`)

    """
    cdef int ZC = -1
    self._error=cgnslib.cg_zconn_write(self._root,B,Z,name,&ZC)
    return ZC

  # ---------------------------------------------------------------------------
  cpdef zconn_get(self, int B, int Z):
    """
    Gets the current `ZoneGridConnectivity_t` node.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Returns:
    * zone grid connectivity id (`int`)

    """
    cdef int ZC = -1
    self._error=cgnslib.cg_zconn_get(self._root,B,Z,&ZC)
    return ZC

  # ---------------------------------------------------------------------------
  cpdef zconn_set(self, int B, int Z, int ZC):
    """
    Gets the current `ZoneGridConnectivity_t` node.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `ZC` : zone grid connectivity id (`int`)

    - Returns:
    * None

    """
    self._error=cgnslib.cg_zconn_set(self._root,B,Z,ZC)

  # ---------------------------------------------------------------------------
  cpdef n1to1(self, int B, int Z):
    """
    Returns the number of 1-to-1 interfaces in a zone.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Returns:
    * number of 1-to-1 interfaces contained in a `GridConnectivity1to1_t` 
      node (`int`)

    - Remarks:
    * 1-to-1 interfaces that may be stored under `GridConnectivity_t 
      nodes are not taken into account.

    """      
    cdef int n=0
    self._error=cgnslib.cg_n1to1(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef conn_1to1_read(self, int B, int Z, int I):
    """
    Returns a tuple with information about a 1-to-1 connectivity data.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `I` : interface id (`int`)

    - Returns:
    * name of the interface (`string`)
    * name of the zone interfacing with the current zone (`string`)
    * range of points for the current zone (`numpy.ndarray`)
    * range of points for the donor zone (`numpy.ndarray`)

    """
    cdef char * name = " "
    cdef char * dname = " "
    cdef cgnslib.cgsize_t * arangeptr
    cdef cgnslib.cgsize_t * drangeptr
    cdef int * trptr
    arange=PNY.ones((2,3),dtype=PNY.int32)
    arangeptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(arange)
    drange=PNY.ones((2,3),dtype=PNY.int32)
    drangeptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(drange)
    tr=PNY.ones((3,),dtype=PNY.int32)
    trptr=<int *>CNY.PyArray_DATA(tr)
    self._error=cgnslib.cg_1to1_read(self._root,B,Z,I,name,dname,
                                     arangeptr,drangeptr,trptr)
    return (name,dname,arange,drange,tr)

  # ---------------------------------------------------------------------------
  cpdef conn_1to1_id(self, int B, int Z, int I):
    cdef double id=-1
    self._error=cgnslib.cg_1to1_id(self._root,B,Z,I,&id)
    return id

  # --------------------------------------------------------------------------
  cpdef conn_1to1_write(self, int B, int Z, char * cname, char * dname, 
                        crange, drange, tr):
    """
    Creates a new 1-to-1 connectivity node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `cname` : name of the interface (`string`)
    * `dname` : name of the zone interfacing with the current zone (`string`)
    * `crange` : range of points for the current zone (`numpy.ndarray`)
    * `drange` : range of points for the donor zone (`numpy.ndarray`)
    * `tr` : notation for the transformation matrix defining the relative 
      orientation of the two zones (`numpy.ndarray`)

    - Returns:
    * interface id (`int`)

    """
    cdef cgnslib.cgsize_t * crangeptr
    cdef cgnslib.cgsize_t * drangeptr
    cdef int * trptr
    cdef int I = -1
    carray=PNY.int32(crange)
    darray=PNY.int32(drange)
    tarray=PNY.int32(tr)
    crangeptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(carray)
    drangeptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(darray)
    trptr=<int *>CNY.PyArray_DATA(tarray)
    self._error=cgnslib.cg_1to1_write(self._root,B,Z,cname,dname,
                                      crangeptr,drangeptr,trptr,&I)
    return I

  # ---------------------------------------------------------------------------
  cpdef n1to1_global(self, int B):
    """
    Counts the number of 1-to-1 interfaces in a base.

    - Args:
    * `B` : base id (`int`)

    - Return:
    * number of 1-to-1 interfaces in the database (`int`)

    """
    cdef int n=0
    self._error=cgnslib.cg_n1to1_global(self._root,B,&n)
    return n

  # --------------------------------------------------------------------------
  cpdef conn_1to1_read_global(self, int B):
    raise CGNSException(99,"conn_1to1_read_global not implemented")

  # -----------------------------------------------------------------------    
  cpdef nconns(self, int B, int Z):
    """
    Returns the number of generalized connectivity data.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Returns:
    * number of interfaces (`int`)

    """
    cdef int n=0
    self._error=cgnslib.cg_nconns(self._root,B,Z,&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef conn_info(self, int B, int Z, int I):
    """
    Generalized connectivity data node info.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `I` : interface id (`int`)

    - Returns:
    * name of the interface (`string`)
    * grid location used in the definition of the point set (`int`)
    * type of the interface. The admissible types are `Overset`, `Abutting` 
      and `Abutting1to1`. (`int`) 
    * type of point set defining the interface in the zone. The admissible 
      types are `PointRange` or `PointList`. (`int`)
    * number of points defining the interface in the zone (`int`)
    * name of the zone interfacing with the zone (`string`)
    * type of the donor zone. The admissible types are `Structured` 
      and `Unstructured`. (`int`)
    * type of point set defining the interface in the donor zone. 
      The admissible types are `PointListDonor`
      and `CellListDonor`. (`int`)
    * data type in which the donor points are stored in the file
    * number of points or cells in the zone. This number is the same 
      as the number of points or cells contained in the donor zone. (`int`)

    """
    cdef char * name = " " 
    cdef cgnslib.GridLocation_t location
    cdef cgnslib.GridConnectivityType_t gtype
    cdef cgnslib.PointSetType_t pst
    cdef cgnslib.cgsize_t npnts
    cdef char * dname = " "
    cdef cgnslib.ZoneType_t dzt
    cdef cgnslib.PointSetType_t dpst
    cdef cgnslib.DataType_t ddt
    cdef cgnslib.cgsize_t ndd
    self._error=cgnslib.cg_conn_info(self._root,B,Z,I,name,
                                     &location,&gtype,&pst,&npnts,dname,
                                     &dzt,&dpst,&ddt,&ndd)
    return (name,location,gtype,pst,npnts,dname,dzt,dpst,ddt,ndd)

  # ---------------------------------------------------------------------------
  cpdef conn_read(self, int B, int Z, int I):
    """
    Generalized connectivity data node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `I` : interface id (`int`)

    - Returns:
    * array of points defining the interface in the zone (`numpy.ndarray`)
    * data type in which the donor points are stored in the file (`int`)
    * array of donor points or cells (`numpy.ndarray`)

    """
    cdef cgnslib.DataType_t ddt
    cdef cgnslib.cgsize_t * pntsptr
    cdef cgnslib.cgsize_t * ddptr
    dim=self.base_read(B)[2]
    np=self.conn_info(B,Z,I)[4]
    ndd=self.conn_info(B,Z,I)[9]
    pnts=PNY.ones((np,dim),dtype=PNY.int32)
    dd=PNY.ones((ndd,dim),dtype=PNY.int32)
    ddt=fromnumpydtype(PNY.int32)
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(pnts)
    ddptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(dd)
    self._error=cgnslib.cg_conn_read(self._root,B,Z,I,pntsptr,ddt,ddptr)
    return (pnts,ddt,dd)
    
  # ---------------------------------------------------------------------------
  cpdef conn_write(self, int B, int Z, char * cname, 
                   cgnslib.GridLocation_t loc,
                   cgnslib.GridConnectivityType_t gct, 
                   cgnslib.PointSetType_t pst,
                   cgnslib.cgsize_t npnts, pnts, char * dname, 
                   cgnslib.ZoneType_t dzt,
                   cgnslib.PointSetType_t dpst, cgnslib.DataType_t ddt, 
                   cgnslib.cgsize_t ndd, dd):
    """
    Creates a new generalized connectivity data node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `cname` : name of the interface (`string`)
    * `loc` : grid location used for the point set (`int`)
    * `gct` : type of interface. The admissible types are `Overset`, 
      `Abutting` and `Abutting1to1`. (`int`)
    * `pst` : type of point set defining the interface in the current zone. 
       The admissible types are `PointRange` and `PointList`. (`int`)
    * `npnts` : number of points defining the interface in the current zone.
      For a type of point set as `PointRange`, `npnts` is always two. 
      For a type of point set as `PointList` , `npnts` is equal
      to the number of points defined in the `PointList`. (`int`)
    * `pnts` : array of points defining the interface in the 
       zone (`numpy.ndarray`)
    * `dname` : name of the donnor zone (`string`)
    * `dzt` : type of the donnor zone. The admissible types are `Structured` 
       and `Unstructured`. (`int`)
    * `dpst` : type of point set of the donnor zone (`int`)
    * `ddt` : data type of the donor points (`int`)
    * `ndd` : number of points or cells in the current zone (`int`)
    * `dd` : array of donor points or cells whose dimension corresponds 
       to the number `ndd`(`numpy.ndarray`) 

    - Returns:
    * interface id (`int`)

    """
    cdef cgnslib.cgsize_t * dataptr
    cdef cgnslib.cgsize_t * ddataptr
    cdef int I = -1
    carray=PNY.int32(pnts)
    darray=PNY.int32(dd)
    dataptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(carray)
    ddataptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(darray)
    self._error=cgnslib.cg_conn_write(self._root,B,Z,cname,loc,gct,
                                      pst,npnts,dataptr,dname,
                                      dzt,dpst,ddt,ndd,ddataptr,&I)
    return I

  # ---------------------------------------------------------------------------
  cpdef conn_id(self, int B, int Z, int I):
    cdef double id=-1
    self._error=cgnslib.cg_conn_id(self._root,B,Z,I,&id)
    return id

  # ---------------------------------------------------------------------------
  cpdef conn_read_short(self, int B, int Z, int I):
    """
    Reads generalized connectivity data without donor information.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `I` : interface id (`int`)

    - Return:
    * array of points as the interface in the current zone (`numpy.ndarray`)

    """
    cdef cgnslib.cgsize_t * pntsptr
    dim=self.base_read(B)[2]
    np=self.conn_info(B,Z,I)[4]
    pnts=PNY.ones((np,dim),dtype=PNY.int32)
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(pnts)
    self._error=cgnslib.cg_conn_read_short(self._root,B,Z,I,pntsptr)
    return pnts

  # ---------------------------------------------------------------------------
  cpdef conn_write_short(self, int B, int Z, char * name, 
                         cgnslib.GridLocation_t location,
                         cgnslib.GridConnectivityType_t gct, 
                         cgnslib.PointSetType_t pst,
                         cgnslib.cgsize_t npnts, pnts, char * dname):
    """
    Creates a new generalized connectivity data node without donor information.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `name` : name of the interface (`string`)
    * `location` : grid location used for the point set (`int`)
    * `gct` : type of interface. The admissible types are `Overset`, 
      `Abutting` and `Abutting1to1`.  (`int`)
    * `pst` : type of point set defining the interface in the current zone. 
      The admissible types are `PointRange` and `PointList`. (`int`)
    * `npnts` : number of points defining the interface in the current zone.
      For a type of point set as `PointRange`, `npnts` is always two. 
      For a type of point set as `PointList` , `npnts` is equal
      to the number of points defined in the `PointList`. (`int`)
    * `pnts` : array of points as the interface in the zone (`numpy.ndarray`)
    * `dname` : name of the donnor zone (`string`)   

    """
    cdef int I = -1
    cdef cgnslib.cgsize_t * dataptr
    carray=PNY.int32(pnts)
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(carray)
    self._error=cgnslib.cg_conn_write_short(self._root,B,Z,name,location,
                                            gct,pst,npnts,pntsptr,dname,&I)
    return I

  # ---------------------------------------------------------------------------
  cpdef convergence_write(self, int iter, char *ndef):
    """
    Creates a convergence history node.

    - Args:
    * `iter` : number of iterations recorded (`int`)
    * `ndef` : description of the convergence information (`string`)

    - Return:
    * None

    """
    cdef bytes sdef = ndef    
    self._error=cgnslib.cg_convergence_write(iter,sdef)

  # ---------------------------------------------------------------------------
  cpdef state_write(self, char *sdes):
    """
    Creates a reference state node.

    - Args:
    * `sdes` : description of the reference state (`string`)

    - Return:
    * None

    """
    self._error=cgnslib.cg_state_write(sdes)

  # ---------------------------------------------------------------------------
  cpdef equationset_write(self, int eqdim):
    """
    Creates a convergence history node.

    - Args:
    * `eqdim` : dimensionality of the governing equations (`int`)
      Dimensionality is the number of spatial variables describing the flow 
      
    - Return:
    * None

    """
    self._error=cgnslib.cg_equationset_write(2)

  # ---------------------------------------------------------------------------
  cpdef equationset_read(self):
    """
    Returns a tuple with information about the flow equation node.

    - Args:
    * None
      
    - Return:
    * dimensionality of the governing equations (`int`)
    * governing equations flag (`int`) 1 if definition available
    * same for gas model
    * same for viscosity model
    * same for thermal conductivity model
    * same for turbulence closure
    * same for turbulence model (`int`)
      
    """
    cdef int eqdim = -1
    cdef int geq = -1
    cdef int gasflag = -1
    cdef int visflag = -1
    cdef int thermflag = -1
    cdef int turbcflag = -1
    cdef int turbmflag = -1

    self._error=cgnslib.cg_equationset_read(&eqdim,&geq,&gasflag,
                                            &visflag,&thermflag,
                                            &turbcflag,&turbmflag)
    return (eqdim,geq,gasflag,visflag,thermflag,turbcflag,turbmflag)

  # ---------------------------------------------------------------------------
  cpdef equationset_chemistry_read(self):
    """
    Returns a tuple with information about the chemistry equation node.

    - Args:
    * None
      
    - Return:
    * thermal relaxation model flag (`int`) 1 if definition available
    * same for chemical kinetics model

    """
    cdef int trflag = -1
    cdef int ckflag = -1

    self._error=cgnslib.cg_equationset_chemistry_read(&trflag,&ckflag)
    return (trflag,ckflag)

  # --------------------------------------------------------------------------
  cpdef equationset_elecmagn_read(self):
    """
    Returns a tuple with information about the electromagnetic equation node.

    - Args:
    * None
      
    - Return:
    * electric field model flag (`int`) 1 if definition available
    * same for magnetic field model
    * same for conductivity model

    """
    cdef int eflag = -1
    cdef int mflag = -1
    cdef int cflag = -1

    self._error=cgnslib.cg_equationset_elecmagn_read(&eflag,&mflag,&cflag)
    return (eflag,mflag,cflag)

  # ---------------------------------------------------------------------------
  cpdef governing_read(self):
    """
    Returns type of the governing equations.

    - Args:
    * None
      
    - Return:
    * type of governing equations (`int`)

    """
    cdef cgnslib.GoverningEquationsType_t etype
    
    self._error=cgnslib.cg_governing_read(&etype)
    return etype

  # ---------------------------------------------------------------------------
  cpdef governing_write(self, cgnslib.GoverningEquationsType_t etype):
    """
    Creates type of the governing equations.

    - Args:
    * `etype`: type of governing equations (`int`)
      
    - Return:
    * None

    """
    self._error=cgnslib.cg_governing_write(etype)

  # ------------------------------------------------------------------------
  cpdef diffusion_write(self, int dmodel):
    """
    Creates a diffusion model node.

    - Args:
    * `dmodel` : flags defining which diffusion terms are included in the 
      governing equations (`int`)
      Only suitable for the Navier-Stokes equations with structured grids.
      
    - Return:
    * None

    """
    self._error=cgnslib.cg_diffusion_write(&dmodel)

  # ---------------------------------------------------------------------------
  cpdef diffusion_read(self):
    """
    Reads a diffusion model node.

    - Args:
    * None

    - Return:
    * flags defining diffusion terms in the governing equations (`int`)
    
    """
    cdef int dmodel = -1
    self._error=cgnslib.cg_diffusion_read(&dmodel)
    return dmodel

  # ---------------------------------------------------------------------------
  cpdef model_write(self, char * label, cgnslib.ModelType_t mtype):
    """
    Writes auxiliary model types.

    - Args:
    * `label` : CGNS label of the defined model (`string`)
      The admissible types are:
      `GasModel_t`
      `ViscosityModel_t`
      `ThermalConductivityModel_t`
      `TurbulenceClosure_t`
      `TurbulenceModel_t`
      `ThermalRelaxationModel_t`
      `ChemicalKineticsModel_t`
      `EMElectricFieldModel_t`
      `EMMagneticFieldModel_t`
      `EMConductivityModel_t`
    * model type allowed for the label selected (`int`)

    - Return:
    * None

    """
    self._error=cgnslib.cg_model_write(label,mtype)

  # ---------------------------------------------------------------------------
  cpdef model_read(self, char * label ):
    """
    Reads auxiliary model types.

    - Args:
    * `label` : CGNS label of the defined model (`string`)
      The admissible types are:
      `GasModel_t`
      `ViscosityModel_t`
      `ThermalConductivityModel_t`
      `TurbulenceClosure_t`
      `TurbulenceModel_t`
      `ThermalRelaxationModel_t`
      `ChemicalKineticsModel_t`
      `EMElectricFieldModel_t`
      `EMMagneticFieldModel_t`
      `EMConductivityModel_t`

    - Return:
    * model type allowed for the label selected (`int`)
    
    """
    cdef cgnslib.ModelType_t mtype
    self._error=cgnslib.cg_model_read(label,&mtype)
    return mtype

  # ---------------------------------------------------------------------------
  cpdef nintegrals(self):
    """
    Returns the number of integral data nodes.

    - Args:
    * None

    - Return:
    * number of integral data nodes contained in the current node (`int`)
    
    """
    cdef int n=0
    self._error=cgnslib.cg_nintegrals(&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef integral_write(self, char * name):
    """
    Creates a new integral data node.

    - Args:
    * `name` : name of the integral data node (`string`)

    - Return:
    * number of integral data nodes contained in the current node (`int`)
    
    """
    self._error=cgnslib.cg_integral_write(name)

  # ---------------------------------------------------------------------------
  cpdef integral_read(self, int idx):
    """
    Returns the name of a integral data node.

    - Args:
    * `idx` : integral data id (`int`)

    - Return:
    * name of the integral data node (`string`)
    
    """
    cdef char * name = " "
    self._error=cgnslib.cg_integral_read(idx,name)
    return name

  # ---------------------------------------------------------------------------
  cpdef descriptor_read(self, char *dname):
    cdef char *sptr
    self._error=cgnslib.cg_descriptor_read(self._root,dname,&sptr)
    s=str(sptr)
    #cgnslib.cg_free(<void *>sptr)
    return s
  
  # ---------------------------------------------------------------------------
  cpdef descriptor_write(self, char * dname, char * dtext):
    """
    Writes descriptive text.

    - Args:
    * `dname` : name of the descriptor node (`string`)
    * `dtext` : description contained in the descriptor node (`string`)

    - Return:
    * None
    
    """
    self._error=cgnslib.cg_descriptor_write(dname,dtext)

  # ---------------------------------------------------------------------------
  cpdef ndescriptors(self):
    """
    Returns the number of descriptor nodes contained in the current node.

    - Args:
    * None

    - Return:
    * number of descriptor nodes under the current node (`int`)
    
    """
    cdef int n=0
    self._error=cgnslib.cg_ndescriptors(&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef units_write(self, cgnslib.MassUnits_t m,cgnslib.LengthUnits_t l, 
                    cgnslib.TimeUnits_t tps,
                    cgnslib.TemperatureUnits_t t, cgnslib.AngleUnits_t a):
    """
    Writes the first five dimensional units in a new node.

    - Args:
    * `m` : mass units (`int`)
      Values are CG_Null, CG_UserDefined, Kilogram, Gram, Slug, and PoundMass. 
    * `l` : length units (`int`)
      Values are CG_Null, CG_UserDefined, Meter, Centimeter, Millimeter, 
      Foot, and Inch. 
    * `tps`: mass units (`int`)
      Values are CG_Null, CG_UserDefined, and Second. 
    * `t`: mass units (`int`)
      Values are CG_Null, CG_UserDefined, Kelvin, Celsius, Rankine, 
      and Fahrenheit. 
    * `a`: mass units (`int`)
      Values are CG_Null, CG_UserDefined, Degree, and Radian. 

    - Return:
    * None
    
    """
    
    self._error=cgnslib.cg_units_write(m,l,tps,t,a)

  # ---------------------------------------------------------------------------
  cpdef nunits(self):
    """
    Returns the number of dimensional units.

    - Args:
    * None
    
    - Return:
    * number of units used in the file (`int`)
    
    """
    cdef int n=0
    self._error=cgnslib.cg_nunits(&n)
    return n

  # ---------------------------------------------------------------------------
  cpdef units_read(self):
    """
    Reads the first five dimensional units.

    - Args:
    * None

    - Return:
    * `m` : mass units (`int`)
    * `l` : length units (`int`)
    * `tps`: mass units (`int`)
    * `t`: mass units (`int`)
    * `a`: mass units (`int`)
    
    """
    cdef cgnslib.MassUnits_t m
    cdef cgnslib.LengthUnits_t l
    cdef cgnslib.TimeUnits_t tps
    cdef cgnslib.TemperatureUnits_t t
    cdef cgnslib.AngleUnits_t a
    
    self._error=cgnslib.cg_units_read(&m,&l,&tps,&t,&a)
    return (m,l,tps,t,a)

  # ---------------------------------------------------------------------------
  cpdef unitsfull_write(self, cgnslib.MassUnits_t m,
                        cgnslib.LengthUnits_t l, cgnslib.TimeUnits_t tps,
                        cgnslib.TemperatureUnits_t t, cgnslib.AngleUnits_t a, 
                        cgnslib.ElectricCurrentUnits_t c,
                        cgnslib.SubstanceAmountUnits_t sa, 
                        cgnslib.LuminousIntensityUnits_t i):
    """
    Writes all eight dimensional units.

    - Args:
    * `m` : mass units (`int`)
      Value in CG_Null, CG_UserDefined, Kilogram, Gram, Slug and PoundMass. 
    * `l` : length units (`int`)
      Value in CG_Null, CG_UserDefined, Meter, Centimeter, Millimeter,
      Foot and Inch. 
    * `tps`: mass units (`int`)
      Value in CG_Null, CG_UserDefined and Second. 
    * `t`: mass units (`int`)
      Value in CG_Null, CG_UserDefined, Kelvin, Celsius,
      Rankine and Fahrenheit. 
    * `a`: mass units (`int`)
      Value in CG_Null, CG_UserDefined, Degree and Radian.
    * `c`: electric current units (`int`)
      Value in CG_Null, CG_UserDefined, Ampere, Abampere, Statampere,
      Edison and auCurrent. 
    * `sa`: admissible value units (`int`)
      Value in CG_Null, CG_UserDefined, Mole, Entities,
      StandardCubicFoot and StandardCubicMeter.
    * `i`: luminous intensity units (`int`)
      Value in CG_Null, CG_UserDefined, Candela, Candle, Carcel,
      Hefner and Violle. 
    
    - Return:
    * None
    
    """
    self._error=cgnslib.cg_unitsfull_write(m,l,tps,t,a,c,sa,i)

  # ---------------------------------------------------------------------------
  cpdef unitsfull_read(self):
    """
    Returns all eight dimensional units.

    - Args:
    * None

    - Return:
    * `m` : mass units (`int`)
    * `l` : length units (`int`)
    * `tps`: mass units (`int`)
    * `t`: mass units (`int`)
    * `a`: mass units (`int`)
    * `c`: electric current units (`int`)
    * `sa`: admissible value units (`int`)
    * `i`: luminous intensity units (`int`)
    
    """
    cdef cgnslib.MassUnits_t m
    cdef cgnslib.LengthUnits_t l
    cdef cgnslib.TimeUnits_t tps
    cdef cgnslib.TemperatureUnits_t t
    cdef cgnslib.AngleUnits_t a
    cdef cgnslib.ElectricCurrentUnits_t c
    cdef cgnslib.SubstanceAmountUnits_t sa
    cdef cgnslib.LuminousIntensityUnits_t i
    
    self._error=cgnslib.cg_unitsfull_read(&m,&l,&tps,&t,&a,&c,&sa,&i)
    return (m,l,tps,t,a,c,sa,i)

  # ---------------------------------------------------------------------------
  cpdef exponents_write(self, cgnslib.DataType_t dt, e):
    """
    Writes the first five dimensional exponents in a new node.

    - Args:
    * `dt` : data type in which the exponents are recorded (`int`)
      Data types for the exponents are RealSingle and RealDouble. 
    * `e` : exponents for the dimensional units are written in that order: 
      mass, length, time, temperature and angle (`numpy.ndarray`)

    - Return:
    * None
    
    """
    cdef float * sexptr
    cdef double * dexptr
    if (e.shape!=(5,)):
      raise CGNSException(98,"exponents_write requires 5 exponent values")
    if (dt==CK.RealSingle):
      exp=PNY.float32(e)
      sexptr=<float *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_exponents_write(dt,sexptr)
    elif (dt==CK.RealDouble):
      exp=PNY.float64(e)
      dexptr=<double *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_exponents_write(dt,dexptr)
    else:
      raise CGNSException(98,"exponents_write requires RealSingle/RealDouble")

  # ---------------------------------------------------------------------------
  cpdef exponents_info(self):
    """
    Returns the exponent data type.

    - Args:
    * None

    - Return:
    * data type of the exponents (`int`)
    
    """
    cdef cgnslib.DataType_t dtype
    self._error=cgnslib.cg_exponents_info(&dtype)
    return dtype
  
  # ---------------------------------------------------------------------------
  cpdef nexponents(self):
    """
    Returns the number of dimensional exponents.

    - Args:
    * None

    - Return:
    * number of exponents used in the file (`int`)
    
    """
    cdef int n=0
    self._error=cgnslib.cg_nexponents(&n)
    return n
  
  # ---------------------------------------------------------------------------
  cpdef exponents_read(self):
    """
    Reads the first five dimensional exponents.

    - Args:
    * None

    - Return:
    * exponents for the dimensional units are written in that order: mass, 
      length, time, temperature and angle (`numpy.ndarray`)
    
    """
    cdef float * sexptr
    cdef double * dexptr
    e=PNY.ones((5,))
    dt=self.exponents_info()
    if (dt==CK.RealSingle):
      exp=PNY.float32(e)
      sexptr=<float *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_exponents_read(sexptr)
    else:
      exp=PNY.float64(e)
      dexptr=<double *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_exponents_read(dexptr)
    return exp

  # ---------------------------------------------------------------------------
  cpdef expfull_write(self, cgnslib.DataType_t dt, e):
    """
    Writes all height dimensional exponents.

    - Args:
    * `dt` : data type in which the exponents are recorded (`int`)
      Data types for the exponents are RealSingle and RealDouble. 
    * `e` : exponents for the dimensional units are written in that order: 
      mass, length, time,
      temperature, angle, electric current, substance amount, 
      and luminous intensity (`numpy.ndarray`)

    - Return:
    * None
    
    """
    cdef float * sexptr
    cdef double * dexptr
    if (e.shape!=(8,)):
      raise CGNSException(98,"expfull_write requires 8 exponent values")
    if (dt==CK.RealSingle):
      exp=PNY.float32(e)
      sexptr=<float *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_expfull_write(dt,sexptr)
    elif (dt==CK.RealDouble):
      exp=PNY.float64(e)
      dexptr=<double *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_expfull_write(dt,dexptr)
    else:
      raise CGNSException(98,"expfull_write requires RealSingle/RealDouble")

  # ---------------------------------------------------------------------------
  cpdef expfull_read(self):
    """
    Reads all eight dimensional exponents.

    - Args:
    * None

    - Return:
    * exponents for the dimensional units are written in that order: mass, 
      length, time, temperature, angle, electric current, substance amount, 
      and luminous intensity (`numpy.ndarray`)
    
    """
    cdef float * sexptr
    cdef double * dexptr
    e=PNY.ones((8,))
    dt=self.exponents_info()
    if (dt==CK.RealSingle):
      exp=PNY.float32(e)
      sexptr=<float *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_expfull_read(sexptr)
    else:
      exp=PNY.float64(e)
      dexptr=<double *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_expfull_read(dexptr)
    return exp

  # ---------------------------------------------------------------------------
  cpdef conversion_write(self, cgnslib.DataType_t dt, fact):
    """
    Writes the conversion factors in a new node.

    - Args:
    * `dt` : data type in which the exponents are recorded (`int`)
      The admissible data types for conversion factors are RealSingle 
      and RealDouble. 
    * `fact` : two-element array which contains the scaling and the 
      offset factors (`numpy.ndarray`)

    - Return:
    * None
    
    """
    cdef float * sfactptr
    cdef double * dfactptr
    if (dt==CK.RealSingle):
      cfact=PNY.float32(fact)
      sfactptr=<float *>CNY.PyArray_DATA(cfact)
      self._error=cgnslib.cg_conversion_write(dt,sfactptr)
    elif (dt==CK.RealDouble):
      cfact=PNY.float64(fact)
      dfactptr=<double *>CNY.PyArray_DATA(cfact)
      self._error=cgnslib.cg_conversion_write(dt,dfactptr)
    else:
      raise CGNSException(98,"conversion_write requires RealSingle/RealDouble")

  # ---------------------------------------------------------------------------
  cpdef conversion_info(self):
    """
    Returns the conversion factors data type.

    - Args:
    * None

    - Return:
    * data type of the conversion factors (`int`)
    
    """
    cdef cgnslib.DataType_t dt
    self._error=cgnslib.cg_conversion_info(&dt)
    return dt

  # ---------------------------------------------------------------------------
  cpdef conversion_read(self):
    """
     Returns the conversion factors.

    - Args:
    * None

    - Return:
    * two-element array with scaling and offset factors (`numpy.ndarray`)
    
    """
    cdef float * sfactptr
    cdef double * dfactptr
    fact=PNY.ones((2,))
    dt=self.conversion_info()
    if (dt==CK.RealSingle):
      cfact=PNY.float32(fact)
      sfactptr=<float *>CNY.PyArray_DATA(cfact)
      self._error=cgnslib.cg_conversion_read(sfactptr)
    else:
      cfact=PNY.float64(fact)
      dfactptr=<double *>CNY.PyArray_DATA(cfact)
      self._error=cgnslib.cg_conversion_read(dfactptr)
    return cfact

  # ---------------------------------------------------------------------------
  cpdef dataclass_write(self, cgnslib.DataClass_t dclass):
    """
    Writes the data class in a new node.

    - Args:
    * `dclass` : data class for the nodes at this level (`int`)
      The admissible data classes are Dimensional, NormalizedByDimensional, 
      NormalizedByUnknownDimensional,
      NondimensionalParameter and DimensionlessConstant.

    - Return:
    * None
    
    """
    
    self._error=cgnslib.cg_dataclass_write(dclass)

  # ---------------------------------------------------------------------------
  cpdef dataclass_read(self):
    """
    Returns the data class.

    - Args:
    * None

    - Return:
    * `dclass` : data class for the nodes at this level (`int`)
    
    """
    cdef cgnslib.DataClass_t dclass
    self._error=cgnslib.cg_dataclass_read(&dclass)
    return dclass

  # ---------------------------------------------------------------------------
  cpdef gridlocation_write(self, cgnslib.GridLocation_t gloc):
    """
    Writes the grid location in a new node.

    - Args:
    * `gloc` : location in the grid (`int`)
      The admissible locations are CG_Null, CG_UserDefined, Vertex, 
      CellCenter, FaceCenter, IFaceCenter, JFaceCenter, KFaceCenter, EdgeCenter
      
    - Return:
    * None
    
    """
    self._error=cgnslib.cg_gridlocation_write(gloc)

  # ---------------------------------------------------------------------------
  cpdef gridlocation_read(self):
    """
    Reads the grid location.

    - Args:
    * None
      
    - Return:
    * location in the grid (`int`)
    
    """
    cdef cgnslib.GridLocation_t gloc
    self._error=cgnslib.cg_gridlocation_read(&gloc)
    return gloc

  # ---------------------------------------------------------------------------
  cpdef ordinal_write(self, int ord):
    """
    Writes the ordinal value in a new node.

    - Args:
    * `ord` : any integer value  (`int`)
      
    - Return:
    * None
    
    """
    self._error=cgnslib.cg_ordinal_write(ord)

  # ---------------------------------------------------------------------------
  cpdef ordinal_read(self):
    """
    Reads the ordinal value.

    - Args:
    * None 
      
    - Return:
    * any integer value  (`int`)
    
    """
    cdef int o
    self._error=cgnslib.cg_ordinal_read(&o)
    return o

  # ---------------------------------------------------------------------------
  cpdef ptset_info(self):
    """
    Returns a tuple with information about the point set.
  
    - Args:
    * None
      
    - Return:
    * point set type  (`int`)
      The types are PointRange for a range of points or cells, 
      list of discrete points or cells.
    * number of points or cells in the point set (`int`)
      For a point set type of PointRange, the number is always two.
      For a point set type of PointList, this is the number of points or 
      cells in the list. 

    """
    cdef cgnslib.PointSetType_t pst
    cdef cgnslib.cgsize_t npnts
    self._error=cgnslib.cg_ptset_info(&pst,&npnts)
    return (pst,npnts)

  # ---------------------------------------------------------------------------
  cpdef famname_write(self, char * name):
    self._error=cgnslib.cg_famname_write(name)

  # ---------------------------------------------------------------------------
  cpdef famname_read(self):
    cdef char name[MAXNAMELENGTH]
    self._error=cgnslib.cg_famname_read(name)
    return name

  # ---------------------------------------------------------------------------
  cpdef multifam_write(self, char * name, char * fam):
      self._error=cgnslib.cg_multifam_write(name,fam)
    
  # ---------------------------------------------------------------------------
  cpdef multifam_read(self, int F):
      cdef char name[MAXNAMELENGTH]
      cdef char fam[MAXNAMELENGTH]
      self._error=cgnslib.cg_multifam_read(F,name,fam)
      return (name,fam)
    
  # ---------------------------------------------------------------------------
  cpdef nmultifam(self):
    cdef int nfam
    self._error=cgnslib.cg_nmultifam(&nfam)
    return nfam
    
  # ---------------------------------------------------------------------------
  cpdef nfamily_names(self, int B, int F):
    cdef int n=0
    self._error=cgnslib.cg_nfamily_names(self._root,B,F,&n)
    return n
  
  # ---------------------------------------------------------------------------
  cpdef family_name_write(self, int B, int F, char *name, char *fam):
    self._error=cgnslib.cg_family_name_write(self._root,B,F,name,fam)
    return None
  
  # ---------------------------------------------------------------------------
  cpdef family_name_read(self, int B, int F, int N):
    cdef char name[MAXNAMELENGTH]
    cdef char fam[MAXNAMELENGTH]
    self._error=cgnslib.cg_family_name_read(self._root,B,F,N,name,fam)
    return (name,fam)
  
  # ---------------------------------------------------------------------------
  cpdef goto(self, int B, lpath):
    cdef int i0,i1,i2,i3,i4,i5,i6,i7,i8,i9
    cdef char *n0,*n1,*n2,*n3,*n4,*n5,*n6,*n7,*n8,*n9
    cdef char *end
    depth=len(lpath)
    end="end"
    if (depth==0):
      self._error=cgnslib.cg_goto(self._root,B,end)
      return
    n0=lpath[0][0]
    i0=lpath[0][1]
    if (depth==1):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,end)
      return
    n1=lpath[1][0]
    i1=lpath[1][1]
    if (depth==2):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,n1,i1,end)
      return
    n2=lpath[2][0]
    i2=lpath[2][1]
    if (depth==3):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,n1,i1,n2,i2,end)
      return
    n3=lpath[3][0]
    i3=lpath[3][1]
    if (depth==4):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,n1,i1,n2,i2,n3,i3,end)
      return
    n4=lpath[4][0]
    i4=lpath[4][1]
    if (depth==5):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,n1,i1,n2,i2,n3,i3,
                                  n4,i4,end)
      return
    n5=lpath[5][0]
    i5=lpath[5][1]
    if (depth==6):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,n1,i1,n2,i2,n3,i3,
                                  n4,i4,n5,i5,end)
      return
    n6=lpath[6][0]
    i6=lpath[6][1]
    if (depth==7):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,n1,i1,n2,i2,n3,i3,
                                  n4,i4,n5,i5,n6,i6,end)
      return
    n7=lpath[7][0]
    i7=lpath[7][1]
    if (depth==4):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,n1,i1,n2,i2,n3,i3,
                                  n4,i4,n5,i5,n6,i6,n7,i7,end)
      return
    n8=lpath[8][0]
    i8=lpath[8][1]
    if (depth==9):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,n1,i1,n2,i2,n3,i3,
                                  n4,i4,n5,i5,n6,i6,n7,i7,n8,i8,end)
      return
    n9=lpath[9][0]
    i9=lpath[9][1]
    if (depth==10):
      self._error=cgnslib.cg_goto(self._root,B,n0,i0,n1,i1,n2,i2,n3,i3,
                                  n4,i4,n5,i5,n6,i6,n7,i7,n8,i8,n9,i9,end)
      return
    raise CGNSException(92,"goto depth larger than expected")

  # ---------------------------------------------------------------------------
  cpdef golist(self, a, b, c, d):
    raise CGNSException(99,"cg_golist not implemented: use goto")

  # ---------------------------------------------------------------------------
  cpdef gorel(self, args):
    raise CGNSException(99,"cg_gorel not implemented: use goto")

  # ---------------------------------------------------------------------------
  cpdef ArbitraryGridMotionTypeName(self, cgnslib.ArbitraryGridMotionType_t v):
    return cgnslib.cg_ArbitraryGridMotionTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef AreaTypeName(self, cgnslib.AreaType_t v):
    return cgnslib.AreaTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef AverageInterfaceTypeName(self, cgnslib.AverageInterfaceType_t v):
    return cgnslib.AverageInterfaceTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef BCDataTypeName(self, cgnslib.BCDataType_t v):
    return cgnslib.BCDataTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef BCTypeName(self, cgnslib.BCType_t v):
    return cgnslib.BCTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef DataClassName(self, cgnslib.DataClass_t v):
    return cgnslib.DataClassName(v)
  
  # ---------------------------------------------------------------------------
  cpdef DataTypeName(self, cgnslib.DataType_t v):
    return cgnslib.DataTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef ElectricCurrentUnitsName(self, cgnslib.ElectricCurrentUnits_t v):
    return cgnslib.ElectricCurrentUnitsName(v)
  
  # ---------------------------------------------------------------------------
  cpdef ElementTypeName(self, cgnslib.ElementType_t v):
    return cgnslib.ElementTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef GoverningEquationsTypeName(self, cgnslib.GoverningEquationsType_t v):
    return cgnslib.GoverningEquationsTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef GridConnectivityTypeName(self, cgnslib.GridConnectivityType_t v):
    return cgnslib.GridConnectivityTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef GridLocationName(self, cgnslib.GridLocation_t v):
    return cgnslib.GridLocationName(v)
  
  # ---------------------------------------------------------------------------
  cpdef LengthUnitsName(self, cgnslib.LengthUnits_t v):
    return cgnslib.LengthUnitsName(v)
  
  # ---------------------------------------------------------------------------
  cpdef LuminousIntensityUnitsName(self, cgnslib.LuminousIntensityUnits_t v):
    return cgnslib.LuminousIntensityUnitsName(v)
  
  # ---------------------------------------------------------------------------
  cpdef MassUnitsName(self, cgnslib.MassUnits_t v):
    return cgnslib.MassUnitsName(v)
  
  # ---------------------------------------------------------------------------
  cpdef ModelTypeName(self, cgnslib.ModelType_t v):
    return cgnslib.ModelTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef PointSetTypeName(self, cgnslib.PointSetType_t v):
    return cgnslib.PointSetTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef RigidGridMotionTypeName(self, cgnslib.RigidGridMotionType_t v):
    return cgnslib.RigidGridMotionTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef SimulationTypeName(self, cgnslib.SimulationType_t v):
    return cgnslib.SimulationTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef SubstanceAmountUnitsName(self, cgnslib.SubstanceAmountUnits_t v):
    return cgnslib.SubstanceAmountUnitsName(v)
  
  # ---------------------------------------------------------------------------
  cpdef TemperatureUnitsName(self, cgnslib.TemperatureUnits_t v):
    return cgnslib.TemperatureUnitsName(v)
  
  # ---------------------------------------------------------------------------
  cpdef TimeUnitsName(self, cgnslib.TimeUnits_t v):
    return cgnslib.TimeUnitsName(v)
  
  # ---------------------------------------------------------------------------
  cpdef WallFunctionTypeName(self, cgnslib.WallFunctionType_t v):
    return cgnslib.WallFunctionTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef ZoneTypeName(self, cgnslib.ZoneType_t v):
    return cgnslib.ZoneTypeName(v)
  
  # ---------------------------------------------------------------------------
  cpdef get_error(self):
    return cgnslib.get_error()

  # ---------------------------------------------------------------------------
  cpdef add_path(self, char *path):
    self._error=cgnslib.cg_add_path(path)
    return None

  # ---------------------------------------------------------------------------
  cpdef set_path(self, char *path):
    self._error=cgnslib.cg_set_path(path)
    return None

  # ---------------------------------------------------------------------------
  cpdef configure(self, int what, value):
    cdef int   ival
    cdef char *sval
    if (what in [CG_CONFIG_ERROR]):
      raise CGNSException(99,"cg_configure cannot set Python error handler")
    elif (what in [CG_CONFIG_SET_PATH, CG_CONFIG_ADD_PATH]):
      sval=value
      ptr=<void *>sval
    else:
      ival=value
      ptr=<void *>ival
    self._error=cgnslib.cg_configure(what,ptr)
    return None
  
  # ---------------------------------------------------------------------------
  cpdef int precision(self):
    cdef int p
    cgnslib.cg_get_compress(&p)
    return p

  cpdef free(self,args):
    raise CGNSException(99,"cg_free not implemented on Python objects")
  
  cpdef error_exit(self):
    raise CGNSException(99,"cg_error_exit not implemented")
  
  cpdef error_print(self):
    cgnslib.cg_error_print()
    return None

  cpdef int get_filetype(self):
    cdef int ftype
    self._error=cgnslib.cg_get_file_type(self._root,&ftype)
    return ftype

  cpdef int get_cgio(self):
    cdef int cgio
    self._error=cgnslib.cg_get_cgio(self._root,&cgio)
    return cgio

  cpdef root_id(self):
    cdef double r
    self._error=cgnslib.cg_root_id(self._root, &r)
    return r

  cpdef save_as(self, char *filename, int file_type,int follow_links):
    self._error=cgnslib.cg_save_as(self._root,filename,file_type,follow_links)
    return None
  
# ---------------------------------------------------------------------------
# config functions to be called without a pyCGNS object
#
cpdef cg_set_compress(int compress):
  cgnslib.cg_set_compress(compress)
  return None

cpdef int cg_get_compress():
  cdef int compress
  cgnslib.cg_get_compress(&compress)
  return compress

cpdef cg_set_filetype(int filetype):
  cgnslib.cg_set_filetype(filetype)
  return None

cpdef int cg_is_cgns(char *filename):
  cdef int file_type
  cgnslib.cg_is_cgns(filename,&file_type)
  return file_type

# ---
