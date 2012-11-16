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
"""
import os.path
import CGNS.PAT.cgnskeywords as CK
import numpy as PNY

cimport cgnslib
cimport numpy as CNY

MODE_READ   = 0
MODE_WRITE  = 1
MODE_CLOSED = 2
MODE_MODIFY = 3

CG_OK             = 0
CG_ERROR          = 1
CG_NODE_NOT_FOUND = 2
CG_INCORRECT_PATH = 3
CG_NO_INDEX_DIM   = 4

Null              = 0
UserDefined       = 1

CG_FILE_NONE      = 0
CG_FILE_ADF       = 1
CG_FILE_HDF5      = 2
CG_FILE_XML       = 3    

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

cdef asnumpydtype(cgnslib.DataType_t dtype):
    if (dtype==CK.RealSingle):  return PNY.float32
    if (dtype==CK.RealDouble):  return PNY.float64
    if (dtype==CK.Integer):     return PNY.int32
    if (dtype==CK.LongInteger): return PNY.int64
    if (dtype==CK.Character):   return PNY.uint8
    return None
        
cdef fromnumpydtype(dtype):
    if (dtype==PNY.float32):  return CK.RealSingle
    if (dtype==PNY.float64):  return CK.RealDouble
    if (dtype==PNY.int32):    return CK.Integer
    if (dtype==PNY.int64):    return CK.LongInteger
    if (dtype==PNY.uint8):    return CK.Character
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

  # ---------------------------------------------------------------------------
  cpdef close(self):
    """
    Closes the CGNS/MLL file descriptor.

    - Remarks:
    * The Python object is still there, however you cannot call any CGNS/MLL
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
    cgnslib.cg_gopath(self._root,path)
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
    cdef double ibid
    self._error=cgnslib.cg_base_id(self._root,B,&ibid)
    return ibid
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
     * `B`: the parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).

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
     * `B`: the parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).
     * `Z`: the parent zone id (`int`) (:py:func:`zones` and :py:func:`nzones`).

    - Return:
     * The ZoneType_t of the zone as an integer (`int`). This is an enumerate value.
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
    
    cdef double izid
    self._error=cgnslib.cg_zone_id(self._root,B,Z,&izid)
    return izid
  # ---------------------------------------------------------------------------
  cpdef zone_write(self, int B, char *zonename, ozsize,  cgnslib.ZoneType_t ztype):
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
     * Zone type is an integer that should be one of the `CGNS.PAT.cgnskeywords.ZoneType_` keys
     * Zone size depends on base dimensions and zone type (see `CGNS/SIDS 6.3 <http://www.grc.nasa.gov/WWW/cgns/CGNS_docs_current/sids/cgnsbase.html#Zone>`_)
    """
    
    cdef CNY.ndarray[dtype=CNY.int32_t,ndim=1] zsize
    cdef int  zid=-1
    cdef int *ptrzs
    zsize=PNY.require(ozsize.flatten(),dtype=PNY.int32)
    ptrzs=<int *>zsize.data
    self._error=cgnslib.cg_zone_write(self._root,B,zonename,ptrzs,ztype,&zid)
    return zid
  # ---------------------------------------------------------------------------    
  cpdef nfamilies(self, int B):
    """
    Returns the number of families in a base::

      for F in range(1,db.nfamilies(B)+1):
         print db.family_read(B)

    - Args:
     * `B`: the parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).

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
    cdef int n
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
     * `B`: the parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).
     * `familyname`: name of the new family (`string` should not exceed 32 chars)

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
     * `F`: parent family id (`int`) (:py:func:`families` and :py:func:`nfamilies`).
     * `fambcname`: name of the new BC family (`string` should not exceed 32 chars)
     * `bocotype`: type of the actual BC for all BCs refering to the parent family name of `F`
       (`int`)

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
    return (B,F,G,geoname,filename,cadname,n)
  # ---------------------------------------------------------------------------
  cpdef geo_write(self, int B, int F, char *geoname, char *filename,
                  char *cadname):
    """
    Creates a new Geometry reference.

    - Args:
     * `B`: parent base id (`int`) (:py:func:`bases` and :py:func:`nbases`).
     * `F`: parent family id (`int`) (:py:func:`families` and :py:func:`nfamilies`).
     * `geoname`: name of the new geometry reference (`string` should not exceed 32 chars)
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
     * `F`: parent family id (`int`) (:py:func:`families` and :py:func:`nfamilies`).
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
    cdef int n
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
    cdef int n
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
     * `gridname`: name of the new grid (`string` should not exceed 32 chars) (`string`)

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
    cdef int n
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
    cdef int n
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
    cdef double icid
    self._error=cgnslib.cg_coord_id(self._root,B,Z,C,&icid)
    return icid
  # ---------------------------------------------------------------------------
  cpdef coord_write(self, int B, int Z, cgnslib.DataType_t dtype,
                   char *coordname, coords):
    """
    Creates a new coordinates.

    - Args:
     * `B`: base id (`int`)
     * `Z`: zone id (`int`)
     * `dtype`: data type of the array contents (`int`)
     * `coordname`: name of the new coordinates (`string` should not exceed 32 chars)
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
     * `coordname`: name of the new coordinates (`string` should not exceed 32 chars)
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
    
    cdef int ns=0
    self._error=cgnslib.cg_nsols(self._root,B,Z,&ns)
    return ns

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
    cdef double sid
    self._error=cgnslib.cg_sol_id(self._root,B, Z, S, &sid)
    return sid

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
    
    cdef int nsections
    self._error=cgnslib.cg_nsections(self._root,B,Z,&nsections)
    return nsections

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
      If the parent data exist, `parent_flag` is set to 1; otherwise it is set to 0.

    """
    
    cdef char * SectionName
    SectionName=' '
    cdef cgnslib.ElementType_t Element_type
    cdef cgnslib.cgsize_t start
    cdef cgnslib.cgsize_t end
    cdef int nbndry
    cdef int parent_flag
    self._error=cgnslib.cg_section_read(self._root,B,Z,S,SectionName,&Element_type,
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
    cpdef int npe
    self._error=cgnslib.cg_npe(type,&npe)
    return npe

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
    * number of element connectivity data values contained in the wanted range (`int`)
    
    """
    cdef cgnslib.cgsize_t ElementDataSize
    self._error=cgnslib.cg_ElementPartialSize(self._root,B,Z,S,start,end,&ElementDataSize)
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
    * For boundary of interface elements, the `ParentData` array contains information on the
    cells and cell faces sharing the element. (`numpy.ndarray`)
    
    """
    data_size=self.ElementDataSize(B,Z,S)
    elements=PNY.ones((data_size),dtype=PNY.int32)
    parent_data=PNY.ones((data_size),dtype=PNY.int32)
    elementsptr=<int *>CNY.PyArray_DATA(elements)
    parent_dataptr=<int *>CNY.PyArray_DATA(parent_data)
    self._error=cgnslib.cg_elements_read(self._root,B,Z,S,elementsptr,parent_dataptr)
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
    * For boundary of interface elements, the `ParentData` array contains information on the
    cells and cell faces sharing the element. (`numpy.ndarray`)
    
    """
    
    elt=self.section_read(B,Z,S)[1]
    elt_type=self.npe(elt)
    size=(end-start+1)*elt_type
    elements=PNY.ones((size),dtype=PNY.int32)
    parent_data=PNY.ones((size),dtype=PNY.int32)
    elementsptr=<int *>CNY.PyArray_DATA(elements)
    parent_dataptr=<int *>CNY.PyArray_DATA(parent_data)
    self._error=cgnslib.cg_elements_partial_read(self._root,B,Z,S,start,end,elementsptr,parent_dataptr)
    return (elements,parent_data)

  # --------------------------------------------------------------------------  
  cpdef section_write(self,int B, int Z, char * SectionName, cgnslib.ElementType_t type,
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
    self._error=cgnslib.cg_section_write(self._root,B,Z,SectionName,type,start,end,
                                         nbndry,elementsptr,&S)
    return S

  # ---------------------------------------------------------------------------
  cpdef parent_data_write(self, int B, int Z, int S, parent_data):
    
    """
    Writes parent info for an element section.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section id which is comprised between 1 and the total number
      of element sections(`int`)
    * `parent_data` : For boundary of interface elements, the `ParentData` array contains
    information on the cells and cell faces sharing the element. (`numpy.ndarray`)
    
    """
    parent_data=PNY.int32(parent_data)
    parent_dataptr=<int *>CNY.PyArray_DATA(parent_data)
    self._error=cgnslib.cg_parent_data_write(self._root,B,Z,S,parent_dataptr)

  # ----------------------------------------------------------------------------
  cpdef section_partial_write(self, int B, int Z, char * SectionName,
                              cgnslib.ElementType_t type, cgnslib.cgsize_t start,
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
    self._error=cgnslib.cg_section_partial_write(self._root,B,Z,SectionName,type,
                                                 start,end,nbndry,&S)
    return S

  # -----------------------------------------------------------------------------
  cpdef elements_partial_write(self, int B, int Z, int S, cgnslib.cgsize_t start,
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

  # ------------------------------------------------------------------------------
  cpdef parent_data_partial_write(self, int B, int Z, int S, cgnslib.cgsize_t start,
                                  cgnslib.cgsize_t end, parent_data):

    """
    Writes subset of parent info for an element section.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `S` : element section index number (`int`)
    * `start` : index of first element in the section (`int`) 
    * `end` : index of last element in the section (`int`)   
    * `parent_data` : For boundary of interface elements, the `ParentData` array contains
       information on the cells and cell faces sharing the element. (`numpy.ndarray`)

    - Return:
    * None
    
    """
    parent_data=PNY.int32(parent_data)
    parent_dataptr=<int *>CNY.PyArray_DATA(parent_data)
    self._error=cgnslib.cg_parent_data_partial_write(self._root,B,Z,S,start,end,
                                                     parent_dataptr)

  # ------------------------------------------------------------------------------
  cpdef nbocos(self, int B, int Z):

      """
      Gets number of boundary conditions.
      
      - Args:
      * `B` : base id (`int`)
      * `Z` : zone id (`int`)

      - Return:
      * number of boundary conditions in zone `Z` (`int`)

      """
      
      cdef int nbbdcd
      self._error=cgnslib.cg_nbocos(self._root,B,Z,&nbbdcd)
      return nbbdcd

  # -----------------------------------------------------------------------------------
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
    * extent of the boundary condition (`int`). The extent may be defined using a range of
      points or elements using `PointRange`using, or using a discrete list of all points or
      elements at which the boundary condition is applied using `PointList`.
    * number of points or elements defining the boundary condition region (`int`)
      For a `ptset_type` of `PointRange`, the number is always 2. For a `ptset_type` of `PointList`,
      the number is equal to the number of points or elements in the list.
    * index vector indicating the computational coordinate direction of the boundary condition
      patch normal (`numpy.ndarray`)
    * flag indicating if the normals are defined in `NormalList`(`int`)
      Returns 1 if they are defined, 0 if they are not.
    * data type used in the definition of the normals (`int`)
      Admissible data types are `RealSingle` and `RealDouble`.
    * number of boundary condition datasets for the current boundary condition (`int`)
      

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
    self._error=cgnslib.cg_boco_info(self._root,B,Z,BC,boconame,&bocotype,&ptset_type,
                                     &npnts,NormalIndexptr,&NormalListFlag,&NormalDataType,
                                     &ndataset)
    return (boconame,bocotype,ptset_type,npnts,NormalIndex,NormalListFlag,NormalDataType,
                                     ndataset)

  # ----------------------------------------------------------------------------------
  cpdef boco_id(self, int B, int Z, int BC):
    cdef double boco_id
    self._error=cgnslib.cg_boco_id(self._root,B,Z,BC,&boco_id)
    return boco_id

  # ----------------------------------------------------------------------------------
  cpdef boco_write(self, int B, int Z, char * boconame, cgnslib.BCType_t bocotype,
                   cgnslib.PointSetType_t ptset_type, cgnslib.cgsize_t npnts, pnts):

    """
    Creates a new boundary condition.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `boconame` : name of the boundary condition (`string`)
    * `bocotype` : type of the boundary condition (`int`)
    * `ptset_type : extent of the boundary condition (`int`)
    * `npnts` : number of points or elements defining the boundary condition region (`int`)
    * `pnts` : array of point or element indices defining the boundary condition region
      (`numpy.ndarray`)

    - Return:
    * boundary condition id (`int`)

    """

    cdef int BC=-1
    array=PNY.int32(pnts)
    arrayptr=<int *>CNY.PyArray_DATA(array)
    self._error=cgnslib.cg_boco_write(self._root,B,Z,boconame,bocotype,ptset_type,npnts,arrayptr,
                                      &BC)
    return BC

  # --------------------------------------------------------------------------------------
  cpdef boco_normal_write(self, int B, int Z, int BC, NormalIndex,
                          int NormalListFlag, NormalDataType=None, NormalList=None):

    """
    Writes the normals of a given `BC` boundary condition.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)
    * `NormalIndex`:index vector indicating the computational coordinate direction of the
      boundary condition patch normal (`numpy.ndarray`)
    * `NormalListFlag`: flag indicating if the normals are defined in `NormalList`(`int`) .
      The flag is equal to 1 if they are defined, 0 if they are not. If the flag is forced to 0,
      'NormalDataType' and 'NormalList' are not taken into account. In this case, these arguments
      are not required.
    * `NormalDataType`: data type used in the definition of the normals (`int`).
      Admissible data types are `RealSingle` and `RealDouble`. 
    * `NormalList`: list of vectors normal to the boundary condition patch pointing into the
      interior of the zone (`numpy.ndarray`)

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

  # ----------------------------------------------------------------------------------
  cpdef boco_read(self, int B, int Z, int BC):

    """
    Reads boundary condition data and normals.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)

    - Returns:
    * array of point or element indices defining the boundary condition region (`numpy.ndarray`)
    * list of vectors normal to the boundary condition patch pointing into the interior
      of the zone (`numpy.ndarray`)
    
    """
    cdef int NormalList
    cdef double * nlptrD
    cdef float * nlptrS
    npnts=self.boco_info(B,Z,BC)[3]
    nls=self.boco_info(B,Z,BC)[5]
    datatype=self.boco_info(B,Z,BC)[6]
    ztype=self.zone_type(B,Z)
    pdim=self.base_read(B)[3]
    if (ztype==CK.Unstructured):
      dim=1
    elif (ztype==CK.Structured):
      dim=self.base_read(B)[2]  
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

  # ------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------
  cpdef boco_gridlocation_write(self, int B, int Z, int BC,
                                cgnslib.GridLocation_t location):

    """
    Writes the boundary condition location.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `BC`: boundary condition id (`int`)
    * `location` : grid location used in the definition of the point set (`int`)

    - Return:
    * None

    """
    
    self._error=cgnslib.cg_boco_gridlocation_write(self._root,B,Z,BC,location)

  # ------------------------------------------------------------------------------------
  cpdef dataset_write(self, int B, int Z, int BC, char * name, cgnslib.BCType_t BCType):

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

  # ------------------------------------------------------------------------------------
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
    self._error=cgnslib.cg_dataset_read(self._root,B,Z,BC,DS,name,&bct,&dflag,&nflag)
    return (name,bct,dflag,nflag)

  # -------------------------------------------------------------------------------------
##   cpdef bcdataset_write(self, char * name, cgnslib.BCType_t bct,
##                            cgnslib.BCDataType_t bcdt):
##     self._error=cgnslib.cg_bcdataset_write(name,bct,bcdt)
##     print cgnslib.cg_get_error(),self._error

  # --------------------------------------------------------------------------------------
  cpdef narrays(self):

    """
    Returns the number of data arrays under the current node. You can access a node via a
    pathname by using the `gopath` function.

    - Args:
    * None

    - Returns:
    * number of data arrays contained in a given node (`int`)
    
    """
    cdef int narrays = -1
    self._error=cgnslib.cg_narrays(&narrays)
    return narrays
  # --------------------------------------------------------------------------------------
  cpdef array_info(self, int A):

    """
    Returns a tuple with information about a given array. You need to access the parent
    node of the requested array. You can use the `gopath` function to do it.

    - Args:
    * `A` : data array id which is necessarily comprised between 1 and the number of arrays
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

 # ---------------------------------------------------------------------------------------
  cpdef array_read(self, int A):

    """
    Reads a data array contained in a given node. You can set the current node by using the
    `gopath` function.

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
  
  # -----------------------------------------------------------------------------------------
  cpdef array_read_as(self, int A, cgnslib.DataType_t type):

    """
    Reads a data array as a certain type. You can set the node which contains the requested array
    by using the `gopath` function.

    - Args:
    * `A` : data array id which is comprised between 1 and the number of arrays
      under the current node (`int`)
    * `type` : requested type of data held in the array (`int`)

    - Return:
    * data array (`numpy.ndarray`)

    - Remarks:
    * The data array is returned only if its data type corresponds to the required data type.
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
              
  # -----------------------------------------------------------------------------------------
  cpdef array_write(self, char * aname, cgnslib.DataType_t dt, int dd,
                    dimv, adata):
    """
    Creates a new data array.

    - Args:
    * `aname` : name of the data array (`string`)
    * `dt` : type of data held in the array (`int`)
    * `dd` : number of dimensions of the data array (`int`)
    * `dimv` : number of data elements in each dimension (`numpy.ndarray`)
    * `adata` : data array ('numpy.ndarray`)

    - Return:
    * None

    """

    cdef int * dataptrI
    cdef float * dataptrF
    cdef double * dataptrD
    cdef char * dataptrC
    cdef cgnslib.cgsize_t * dv

    div=PNY.int32(dimv)
    dv=<cgnslib.cgsize_t *>CNY.PyArray_DATA(div)
        
    if (dt==CK.Integer):
      data=PNY.int32(adata)
      dataptrI=<int *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_write(aname,dt,dd,dv,dataptrI)
    if (dt==CK.LongInteger):
      data=PNY.int64(adata)
      dataptrI=<int *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_write(aname,dt,dd,dv,dataptrI)
    if (dt==CK.RealSingle):
      data=PNY.float32(adata)
      dataptrF=<float *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_write(aname,dt,dd,dv,dataptrF)
    if (dt==CK.RealDouble):
      data=PNY.float64(adata)
      dataptrD=<double *>CNY.PyArray_DATA(data)
      self._error=cgnslib.cg_array_write(aname,dt,dd,dv,dataptrD)
    if (dt==CK.Character):
      dataptrC=<char *>CNY.PyArray_DATA(adata)
      self._error=cgnslib.cg_array_write(aname,dt,dd,dv,dataptrC)

  # -----------------------------------------------------------------------------------------
  cpdef nuser_data(self):

    """
    Counts the number of `UserDefinedData_t` nodes contained in the current node. You can access
    the current node by using the `gopath` function.

    - Args:
    * None

    - Return:
    * number of `UserDefinedData_t` nodes contained in the current node (`int`)

    """
    
    cdef int nuser=-1
    self._error=cgnslib.cg_nuser_data(&nuser)
    return nuser

  # -----------------------------------------------------------------------------------------
  cpdef user_data_write(self, char * usn):

    """
    Creates a new `UserDefinedData_t` node. You can set the position of the node in the `CGNS tree`
    by using the `gopath` function.

    - Args:
    * `usn` : name of the created node (`string`)

    - Return:
    * None

    """
    
    self._error=cgnslib.cg_user_data_write(usn)

  # -----------------------------------------------------------------------------------------
  cpdef user_data_read(self, int Index):

    """
    Returns the name of a given `UserDefinedData_t` node. You can access the node by using
    the `gopath` function.

    - Args:
    * `Index` : user-defined data id which is necessarily comprised between 1 and the total
      number of `UserDefinedData_t` nodes under the current node (`int`)

    - Return:
    * name of the required `UserDefinedData_t` node (`string`)

    """
    
    cdef char * usn = " "
    self._error=cgnslib.cg_user_data_read(Index,usn)
    return usn

  # -----------------------------------------------------------------------------------------
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

  # ---------------------------------------------------------------------------------------
  cpdef ndiscrete(self, int B, int Z):

    """
    Returns the number of `DiscreteData_t` nodes in a given zone.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Return:
    * number of `DiscreteData_t`nodes contained in the zone (`int`)
    
    """
    
    cdef int ndiscrete = -1
    self._error=cgnslib.cg_ndiscrete(self._root,B,Z,&ndiscrete)
    return ndiscrete

  # ---------------------------------------------------------------------------------------
  cpdef discrete_read(self, int B, int Z, int D):

    """
    Returns the name of a given `DiscreteData_t` node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised between 1 and the total number
      of discrete data nodes under the zone (`int`)

    - Return:
    * name of discrete data node (`string`)
    
    """
    
    cdef char * name = " "
    self._error=cgnslib.cg_discrete_read(self._root,B,Z,D,name)
    return name

  # ---------------------------------------------------------------------------------------
  cpdef discrete_size(self, int B, int Z, int D):

    """
    Returns the dimensions of a `DiscreteData_t` node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised between 1 and the total number
      of discrete data nodes under the zone (`int`)

    - Return:
    * number of dimensions defining the discrete data (`int`). If a point set has been defined,
      this is 1, otherwise this is the current zone index dimension.
    * array of dimensions ('numpy.ndarray`)
      
    """
    
    cdef int dd = -1
    cdef cgnslib.cgsize_t * dvptr
    dv=PNY.ones((CK.ADF_MAX_DIMENSIONS,),dtype=PNY.int32)    
    dvptr = <cgnslib.cgsize_t *>CNY.PyArray_DATA(dv)
    self._error=cgnslib.cg_discrete_size(self._root,B,Z,D,&dd,dvptr)
    return (dd,dv[0:dd])

  # ---------------------------------------------------------------------------------------
  cpdef discrete_ptset_info(self, int B, int Z, int D):

    """
    Returns a tuple with information about a given discrete data node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised between 1 and the total number
      of discrete data nodes under the zone (`int`)

    - Return:
    * type of point set defining the interface ('int`). It can be `PointRange` or `PointList`.
    * number of points defining the interface (`int`)

    """
    
    cdef cgnslib.PointSetType_t pst 
    cdef cgnslib.cgsize_t npnts 
    self._error=cgnslib.cg_discrete_ptset_info(self._root,B,Z,D,&pst,&npnts)
    return (pst,npnts)

  # ---------------------------------------------------------------------------------------
  cpdef discrete_ptset_read(self, int B, int Z, int D):
    
    """
    Reads a point set of a given discrete data node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised between 1 and the total number
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

  # ---------------------------------------------------------------------------------------
  cpdef discrete_ptset_write(self, int B, int Z, char * name, cgnslib.GridLocation_t location,
                             cgnslib.PointSetType_t pst, cgnslib.cgsize_t npnts, pts):

    """
    Creates a new point set `DiscreteData_t` node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `D` : discrete data id which is necessarily comprised between 1 and the total number
      of discrete data nodes under the zone (`int`)

    - Return:
    * array of points defining the interface ('numpy.ndarray`)

    """
    
    cdef cgnslib.cgsize_t * pntsptr
    cdef int D = -1
    pnts=PNY.int32(pts)
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(pnts)
    self._error=cgnslib.cg_discrete_ptset_write(self._root,B,Z,name,location,pst,npnts,
                                                pntsptr,&D)
    return D

  # ---------------------------------------------------------------------------------------
  cpdef nzconns(self, int B, int Z):

    """
    Returns the number of `ZoneGridConnectivity_t` nodes.
    
    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Returns:
    * number of `ZoneGridConnectivity_t` nodes (`int`)

    """
    cdef int nzc
    self._error=cgnslib.cg_nzconns(self._root,B,Z,&nzc)
    return nzc

  # ---------------------------------------------------------------------------------------
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

  # ---------------------------------------------------------------------------------------
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

  # ----------------------------------------------------------------------------------------
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

  # ---------------------------------------------------------------------------------------
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

  # ---------------------------------------------------------------------------------------
  cpdef n1to1(self, int B, int Z):

    """
    Returns the number of 1-to-1 interfaces in a zone.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)

    - Returns:
    * number of 1-to-1 interfaces contained in a `GridConnectivity1to1_t` node (`int`)

    - Remarks:
    * 1-to-1 interfaces that may be stored under `GridConnectivity_t nodes are not taken
      into account.

    """      
    cdef int n1to1 = -1
    self._error=cgnslib.cg_n1to1(self._root,B,Z,&n1to1)
    return n1to1

  # ----------------------------------------------------------------------------------------
  cpdef _1to1_read(self, int B, int Z, int I):

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
    self._error=cgnslib.cg_1to1_read(self._root,B,Z,I,name,dname,arangeptr,drangeptr,trptr)
    return (name,dname,arange,drange,tr)

  # -----------------------------------------------------------------------------------------
  cpdef _1to1_id(self, int B, int Z, int I):
    cdef double id1to1 = -1
    self._error=cgnslib.cg_1to1_id(self._root,B,Z,I,&id1to1)
    return id1to1

  # ------------------------------------------------------------------------------------------
  cpdef _1to1_write(self, int B, int Z, char * cname, char * dname, crange, drange, tr):

    """
    Creates a new 1-to-1 connectivity node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `cname` : name of the interface (`string`)
    * `dname` : name of the zone interfacing with the current zone (`string`)
    * `crange` : range of points for the current zone (`numpy.ndarray`)
    * `drange` : range of points for the donor zone (`numpy.ndarray`)
    * `tr` : notation for the transformation matrix defining the relative orientation
      of the two zones (`numpy.ndarray`)

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
    self._error=cgnslib.cg_1to1_write(self._root,B,Z,cname,dname,crangeptr,drangeptr,trptr,&I)
    return I

  # ------------------------------------------------------------------------------------------
  cpdef n1to1_global(self, int B):

    """
    Counts the number of 1-to-1 interfaces in a base.

    - Args:
    * `B` : base id (`int`)

    - Return:
    * number of 1-to-1 interfaces in the database (`int`)

    """
    
    cdef int n = -1
    self._error=cgnslib.cg_n1to1_global(self._root,B,&n)
    return n

  # --------------------------------------------------------------------------
  cpdef _1to1_read_global(self, int B):

    cdef CNY.ndarray[dtype=CNY.uint8_t,ndim=2] tcnames
    cdef CNY.ndarray[dtype=CNY.uint8_t,ndim=2] tznames
    cdef CNY.ndarray[dtype=CNY.uint8_t,ndim=2] tdnames
    cdef CNY.ndarray[dtype=CNY.int32_t,ndim=3] tcrange
    cdef CNY.ndarray[dtype=CNY.int32_t,ndim=3] tdrange
    cdef CNY.ndarray[dtype=CNY.int32_t,ndim=2] ttrange
    cdef CNY.uint8_t *cnameptr
    cdef CNY.uint8_t *znameptr
    cdef CNY.uint8_t *dnameptr
    cdef cgnslib.cgsize_t *crangeptr
    cdef cgnslib.cgsize_t *drangeptr 
    cdef int *trangeptr
    
    cxnum=self.n1to1_global(B)
    tcnames=PNY.array((32,cxnum),dtype=PNY.uint8_t)
    tznames=PNY.array((32,cxnum),dtype=PNY.uint8_t)
    tdnames=PNY.array((32,cxnum),dtype=PNY.uint8_t)
    
    cnameptr=<CNY.uint8_t* >tcnames.data
    znameptr=<CNY.uint8_t* >tznames.data
    dnameptr=<CNY.uint8_t* >tdnames.data
    
    tcrange=PNY.ones((cxnum,2,3),dtype=PNY.int32)
    tdrange=PNY.ones((cxnum,2,3),dtype=PNY.int32)
    ttrange=PNY.ones((cxnum,3),dtype=PNY.int32)
    
    crangeptr=<cgnslib.cgsize_t *>tcrange.data
    drangeptr=<cgnslib.cgsize_t *>tdrange.data
    trangeptr=<int *>ttrange.data
    
    self._error=cgnslib.cg_1to1_read_global(self._root,B,
                                            <char** >cnameptr,
                                            <char** >znameptr,
                                            <char** >dnameptr,
                                            <cgnslib.cgsize_t**>crangeptr,
                                            <cgnslib.cgsize_t** >drangeptr,
                                            <int** >trangeptr)

    
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
    
    cdef int nc = -1
    self._error=cgnslib.cg_nconns(self._root,B,Z,&nc)
    return nc

  # ---------------------------------------------------------------------------------------------
  cpdef conn_info(self, int B, int Z, int I):

    """
    Returns a tuple with information about a generalized connectivity data node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `I` : interface id (`int`)

    - Returns:
    * name of the interface (`string`)
    * grid location used in the definition of the point set (`int`)
    * type of the interface. The admissible types are `Overset`, `Abutting` and `Abutting1to1`. (`int`) 
    * type of point set defining the interface in the zone. The admissible types are `PointRange` or
      `PointList`. (`int`)
    * number of points defining the interface in the zone (`int`)
    * name of the zone interfacing with the zone (`string`)
    * type of the donor zone. The admissible types are `Structured` and `Unstructured`. (`int`)
    * type of point set defining the interface in the donor zone. The admissible types are `PointListDonor`
      and `CellListDonor`. (`int`)
    * data type in which the donor points are stored in the file
    * number of points or cells in the zone. This number is the same as the number of points or cells
      contained in the donor zone. (`int`)

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
    self._error=cgnslib.cg_conn_info(self._root,B,Z,I,name,&location,&gtype,&pst,&npnts,dname,
                                     &dzt,&dpst,&ddt,&ndd)
    return (name,location,gtype,pst,npnts,dname,dzt,dpst,ddt,ndd)

  # ---------------------------------------------------------------------------------------------
  cpdef conn_read(self, int B, int Z, int I):

    """
    Returns a tuple with information about a generalized connectivity data node.

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
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(pnts)
    ddptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(dd)
    self._error=cgnslib.cg_conn_read(self._root,B,Z,I,pntsptr,ddt,ddptr)
    return (pnts,ddt,dd)
    
  # ---------------------------------------------------------------------------------------------
  cpdef conn_write(self, int B, int Z, char * cname, cgnslib.GridLocation_t loc,
                   cgnslib.GridConnectivityType_t gct, cgnslib.PointSetType_t pst,
                   cgnslib.cgsize_t npnts, pnts, char * dname, cgnslib.ZoneType_t dzt,
                   cgnslib.PointSetType_t dpst, cgnslib.DataType_t ddt, cgnslib.cgsize_t ndd,
                   dd):
    """
    Creates a new generalized connectivity data node.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `cname` : name of the interface (`string`)
    * `loc` : grid location used for the point set (`int`)
    * `gct` : type of interface. The admissible types are `Overset`, `Abutting` and `Abutting1to1`.
      (`int`)
    * `pst` : type of point set defining the interface in the current zone. The admissible types are
      `PointRange` and `PointList`. (`int`)
    * `npnts` : number of points defining the interface in the current zone.For a type of point set
      as `PointRange`, `npnts` is always two. For a type of point set as `PointList` , `npnts` is equal
      to the number of points defined in the `PointList`. (`int`)
    * `pnts` : array of points defining the interface in the zone (`numpy.ndarray`)
    * `dname` : name of the donnor zone (`string`)
    * `dzt` : type of the donnor zone. The admissible types are `Structured` and `Unstructured`. (`int`)
    * `dpst` : type of point set of the donnor zone (`int`)
    * `ddt` : data type of the donor points (`int`)
    * `ndd` : number of points or cells in the current zone (`int`)
    * `dd` : array of donor points or cells whose dimension corresponds to the number `ndd`(`numpy.ndarray`) 

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
    self._error=cgnslib.cg_conn_write(self._root,B,Z,cname,loc,gct,pst,npnts,dataptr,dname,
                                      dzt,dpst,ddt,ndd,ddataptr,&I)
    return I

  # ---------------------------------------------------------------------------------------------
  cpdef conn_id(self, int B, int Z, int I):
    cdef double cid
    self._error=cgnslib.cg_conn_id(self._root,B,Z,I,&cid)
    return cid

  # ---------------------------------------------------------------------------------------------
  cpdef conn_read_short(self, int B, int Z, int I):

    """
    Reads generalized connectivity data without donor information.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `I` : interface id (`int`)

    - Return:
    * array of points defining the interface in the current zone (`numpy.ndarray`)

    """
    
    cdef cgnslib.cgsize_t * pntsptr
    dim=self.base_read(B)[2]
    np=self.conn_info(B,Z,I)[4]
    pnts=PNY.ones((np,dim),dtype=PNY.int32)
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(pnts)
    self._error=cgnslib.cg_conn_read_short(self._root,B,Z,I,pntsptr)
    return pnts

  # ---------------------------------------------------------------------------------------------
  cpdef conn_write_short(self, int B, int Z, char * name, cgnslib.GridLocation_t location,
                         cgnslib.GridConnectivityType_t gct, cgnslib.PointSetType_t pst,
                         cgnslib.cgsize_t npnts, pnts, char * dname):

    """
    Creates a new generalized connectivity data node without donor information.

    - Args:
    * `B` : base id (`int`)
    * `Z` : zone id (`int`)
    * `name` : name of the interface (`string`)
    * `location` : grid location used for the point set (`int`)
    * `gct` : type of interface. The admissible types are `Overset`, `Abutting` and `Abutting1to1`.
      (`int`)
    * `pst` : type of point set defining the interface in the current zone. The admissible types are
      `PointRange` and `PointList`. (`int`)
    * `npnts` : number of points defining the interface in the current zone.For a type of point set
      as `PointRange`, `npnts` is always two. For a type of point set as `PointList` , `npnts` is equal
      to the number of points defined in the `PointList`. (`int`)
    * `pnts` : array of points defining the interface in the zone (`numpy.ndarray`)
    * `dname` : name of the donnor zone (`string`)   

    """
    
    cdef int I = -1
    cdef cgnslib.cgsize_t * dataptr
    carray=PNY.int32(pnts)
    pntsptr=<cgnslib.cgsize_t *>CNY.PyArray_DATA(carray)
    self._error=cgnslib.cg_conn_write_short(self._root,B,Z,name,location,gct,pst,npnts,pntsptr,dname,&I)
    return I

  # --------------------------------------------------------------------------------------------
  cpdef convergence_write(self, int iter, char * ndef):

    """
    Creates a convergence history node.

    - Args:
    * `iter` : number of iterations for which convergence information is recorded (`int`)
    * `ndef` : description of the convergence information (`string`)

    - Return:
    * None

    """
    
    self._error=cgnslib.cg_convergence_write(iter,ndef)

  # --------------------------------------------------------------------------------------------
  cpdef state_write(self, char * sdes):

    """
    Creates a reference state node.

    - Args:
    * `sdes` : description of the reference state (`string`)

    - Return:
    * None

    """
    
    self._error=cgnslib.cg_state_write(sdes)

  # --------------------------------------------------------------------------------------------
  cpdef equationset_write(self, int eqdim):

    """
    Creates a convergence history node.

    - Args:
    * `eqdim` : dimensionality of the governing equations (`int`)
      Dimensionality is the number of spatial variables describing the flow 
      
    - Return:
    * None

    """
    
    self._error=cgnslib.cg_equationset_write(eqdim)

  # --------------------------------------------------------------------------------------------
  cpdef equationset_read(self):

    """
    Returns a tuple with information about the flow equation node.

    - Args:
    * None
      
    - Return:
    * dimensionality of the governing equations (`int`)
    * flag indicating if the node contains the definition of the governing equations (`int`)
      0 if it doesn't, 1 if it does. 
    * flag indicating if the node contains the definition of a gas model (`int`)
      0 if it doesn't, 1 if it does.
    * flag indicating if the node contains the definition of a viscosity model (`int`)
      0 if it doesn't, 1 if it does.
    * flag indicating if the node contains the definition of a thermal conductivity model (`int`)
      0 if it doesn't, 1 if it does.       
    * flag indicating if the node contains the definition of the turbulence closure (`int`)
      0 if it doesn't, 1 if it does.
    * flag indicating if the node contains the definition of a turbulence model (`int`)
      0 if it doesn't, 1 if it does.
      
    """
    cdef int eqdim = -1
    cdef int geq = -1
    cdef int gasflag = -1
    cdef int visflag = -1
    cdef int thermflag = -1
    cdef int turbcflag = -1
    cdef int turbmflag = -1

    self._error=cgnslib.cg_equationset_read(&eqdim,&geq,&gasflag,&visflag,&thermflag,&turbcflag,&turbmflag)
    return (eqdim,geq,gasflag,visflag,thermflag,turbcflag,turbmflag)

  # --------------------------------------------------------------------------------------------
  cpdef equationset_chemistry_read(self):
    
    """
    Returns a tuple with information about the chemistry equation node.

    - Args:
    * None
      
    - Return:
    * flag indicating if the node contains the definition of a thermal relaxation model (`int`)
      0 if it doesn't, 1 if it does. 
    * flag indicating if the node contains the definition of a chemical kinetics model (`int`)
      0 if it doesn't, 1 if it does.

    """
    
    cdef int trflag = -1
    cdef int ckflag = -1

    self._error=cgnslib.cg_equationset_chemistry_read(&trflag,&ckflag)
    return (trflag,ckflag)

  # --------------------------------------------------------------------------------------------
  cpdef equationset_elecmagn_read(self):

    """
    Returns a tuple with information about the electromagnetic equation node.

    - Args:
    * None
      
    - Return:
    * flag indicating if the node contains the definition of an electric field model for
      electromagnetic flows (`int`). 0 if it doesn't, 1 if it does. 
    * flag indicating if the node contains the definition a magnetic field model for
      electromagnetic flows(`int`). 0 if it doesn't, 1 if it does.
      * flag indicating if the node contains the definition of a conductivity model for
        electromagnetic flows (`int`). 0 if it doesn't, 1 if it does.

    """
    cdef int eflag = -1
    cdef int mflag = -1
    cdef int cflag = -1

    self._error=cgnslib.cg_equationset_elecmagn_read(&eflag,&mflag,&cflag)
    return (eflag,mflag,cflag)

  # --------------------------------------------------------------------------------------------
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

  # -------------------------------------------------------------------------------------------
  cpdef governing_write(self, cgnslib.GoverningEquationsType_t etype):
    
    """
    Creates type of the governing equations.

    - Args:
    * `etype`: type of governing equations (`int`)
      
    - Return:
    * None

    """
    
    self._error=cgnslib.cg_governing_write(etype)

   # ------------------------------------------------------------------------------------------
  cpdef diffusion_write(self, int dmodel):

    """
    Creates a diffusion model node.

    - Args:
    * `dmodel` : flags defining which diffusion terms are included in the governing equations (`int`)
      This is only suitable for the Navier-Stokes equations with structured grids.
      
    - Return:
    * None

    """
    
    self._error=cgnslib.cg_diffusion_write(&dmodel)

  # ------------------------------------------------------------------------------------------
  cpdef diffusion_read(self):

    """
    Reads a diffusion model node.

    - Args:
    * None

    - Return:
    * flags defining which diffusion terms are included in the governing equations (`int`)
    
    """
    
    cdef int dmodel = -1
    self._error=cgnslib.cg_diffusion_read(&dmodel)
    return dmodel

  # ------------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------------
  cpdef nintegrals(self):

    """
    Returns the number of integral data nodes.

    - Args:
    * None

    - Return:
    * number of integral data nodes contained in the current node (`int`)
    
    """
    
    cdef int nint = -1
    self._error=cgnslib.cg_nintegrals(&nint)
    return nint

  # ------------------------------------------------------------------------------------------
  cpdef integral_write(self, char * name):

    """
    Creates a new integral data node.

    - Args:
    * `name` : name of the integral data node (`string`)

    - Return:
    * number of integral data nodes contained in the current node (`int`)
    
    """
    
    self._error=cgnslib.cg_integral_write(name)

  # ------------------------------------------------------------------------------------------
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
  
  # ------------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------------
  cpdef ndescriptors(self):

    """
    Returns the number of descriptor nodes contained in the current node.

    - Args:
    * None

    - Return:
    * number of descriptor nodes under the current node (`int`)
    
    """
    
    cdef int ndes = -1
    self._error=cgnslib.cg_ndescriptors(&ndes)
    return ndes

  # ------------------------------------------------------------------------------------------
  cpdef units_write(self, cgnslib.MassUnits_t m,cgnslib.LengthUnits_t l, cgnslib.TimeUnits_t tps,
                    cgnslib.TemperatureUnits_t t, cgnslib.AngleUnits_t a):
    
    """
    Writes the first five dimensional units in a new node.

    - Args:
    * `m` : mass units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Kilogram, Gram, Slug, and PoundMass. 
    * `l` : length units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Meter, Centimeter, Millimeter, Foot, and Inch. 
    * `tps`: mass units (`int`)
      The admissible values are CG_Null, CG_UserDefined, and Second. 
    * `t`: mass units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Kelvin, Celsius, Rankine, and Fahrenheit. 
    * `a`: mass units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Degree, and Radian. 

    - Return:
    * None
    
    """
    
    self._error=cgnslib.cg_units_write(m,l,tps,t,a)

  # ------------------------------------------------------------------------------------------
  cpdef nunits(self):

    """
    Returns the number of dimensional units.

    - Args:
    * None
    
    - Return:
    * number of units used in the file (`int`)
    
    """
    
    cdef int n = -1
    self._error=cgnslib.cg_nunits(&n)
    return n

  # ------------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------------
  cpdef unitsfull_write(self, cgnslib.MassUnits_t m,cgnslib.LengthUnits_t l, cgnslib.TimeUnits_t tps,
                        cgnslib.TemperatureUnits_t t, cgnslib.AngleUnits_t a, cgnslib.ElectricCurrentUnits_t c,
                        cgnslib.SubstanceAmountUnits_t sa, cgnslib.LuminousIntensityUnits_t i):

    """
    Writes all eight dimensional units.

    - Args:
    * `m` : mass units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Kilogram, Gram, Slug, and PoundMass. 
    * `l` : length units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Meter, Centimeter, Millimeter, Foot, and Inch. 
    * `tps`: mass units (`int`)
      The admissible values are CG_Null, CG_UserDefined, and Second. 
    * `t`: mass units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Kelvin, Celsius, Rankine, and Fahrenheit. 
    * `a`: mass units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Degree, and Radian.
    * `c`: electric current units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Ampere, Abampere, Statampere, Edison, and auCurrent. 
    * `sa`: admissible value units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Mole, Entities, StandardCubicFoot, and StandardCubicMeter.
    * `i`: luminous intensity units (`int`)
      The admissible values are CG_Null, CG_UserDefined, Candela, Candle, Carcel, Hefner, and Violle. 
    
    - Return:
    * None
    
    """
    
    self._error=cgnslib.cg_unitsfull_write(m,l,tps,t,a,c,sa,i)

  # ------------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------------
  cpdef exponents_write(self, cgnslib.DataType_t dt, e):

    """
    Writes the first five dimensional exponents in a new node.

    - Args:
    * `dt` : data type in which the exponents are recorded (`int`)
      The admissible data types for the exponents are RealSingle and RealDouble. 
    * `e` : exponents for the dimensional units are written in that order: mass, length, time,
      temperature and angle (`numpy.ndarray`)

    - Return:
    * None
    
    """
    
    cdef float * sexptr
    cdef double * dexptr
    if (e.shape!=(5,)):
      print "Bad 2nd arg size: should be 5"
      return 
    if (dt==CK.RealSingle):
      exp=PNY.float32(e)
      sexptr=<float *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_exponents_write(dt,sexptr)
    elif (dt==CK.RealDouble):
      exp=PNY.float64(e)
      dexptr=<double *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_exponents_write(dt,dexptr)
    else:
      print "First arg should be CG_RealDouble or CG_RealSingle"
      return            

  # ------------------------------------------------------------------------------------------
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
  
  # ------------------------------------------------------------------------------------------
  cpdef nexponents(self):

    """
    Returns the number of dimensional exponents.

    - Args:
    * None

    - Return:
    * number of exponents used in the file (`int`)
    
    """
    
    cdef int nexp = -1
    self._error=cgnslib.cg_nexponents(&nexp)
    return nexp
  
  # ------------------------------------------------------------------------------------------
  cpdef exponents_read(self):

    """
    Reads the first five dimensional exponents.

    - Args:
    * None

    - Return:
    * exponents for the dimensional units are written in that order: mass, length, time,
      temperature and angle (`numpy.ndarray`)
    
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

  # ------------------------------------------------------------------------------------------
  cpdef expfull_write(self, cgnslib.DataType_t dt, e):

    """
    Writes all height dimensional exponents.

    - Args:
    * `dt` : data type in which the exponents are recorded (`int`)
      The admissible data types for the exponents are RealSingle and RealDouble. 
    * `e` : exponents for the dimensional units are written in that order: mass, length, time,
      temperature, angle, electric current, substance amount, and luminous intensity (`numpy.ndarray`)

    - Return:
    * None
    
    """
    
    cdef float * sexptr
    cdef double * dexptr
    if (e.shape!=(8,)):
      print "Bad 2nd arg size: should be 8"
      return 
    if (dt==CK.RealSingle):
      exp=PNY.float32(e)
      sexptr=<float *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_expfull_write(dt,sexptr)
    elif (dt==CK.RealDouble):
      exp=PNY.float64(e)
      dexptr=<double *>CNY.PyArray_DATA(exp)
      self._error=cgnslib.cg_expfull_write(dt,dexptr)
    else:
      print "First arg should be CG_RealDouble or CG_RealSingle"
      return

  # ------------------------------------------------------------------------------------------
  cpdef expfull_read(self):

    """
    Reads all eight dimensional exponents.

    - Args:
    * None

    - Return:
    * exponents for the dimensional units are written in that order: mass, length, time,
      temperature, angle, electric current, substance amount, and luminous intensity (`numpy.ndarray`)
    
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

  # ------------------------------------------------------------------------------------------
  cpdef conversion_write(self, cgnslib.DataType_t dt, fact):

    """
    Writes the conversion factors in a new node.

    - Args:
    * `dt` : data type in which the exponents are recorded (`int`)
      The admissible data types for conversion factors are RealSingle and RealDouble. 
    * `fact` : two-element array which contains the scaling and the offset factors (`numpy.ndarray`)

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
      print "First arg should be CG_RealDouble or CG_RealSingle"
      return

  # ------------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------------
  cpdef conversion_read(self):

    """
     Returns the conversion factors.

    - Args:
    * None

    - Return:
    * two-element array which contains the scaling and the offset factors (`numpy.ndarray`)
    
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

  # ------------------------------------------------------------------------------------------
  cpdef dataclass_write(self, cgnslib.DataClass_t dclass):

    """
    Writes the data class in a new node.

    - Args:
    * `dclass` : data class for the nodes at this level (`int`)
      The admissible data classes are Dimensional, NormalizedByDimensional, NormalizedByUnknownDimensional,
      NondimensionalParameter and DimensionlessConstant.

    - Return:
    * None
    
    """
    
    self._error=cgnslib.cg_dataclass_write(dclass)

  # ------------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------------
  cpdef gridlocation_write(self, cgnslib.GridLocation_t gloc):

    """
    Writes the grid location in a new node.

    - Args:
    * `gloc` : location in the grid (`int`)
      The admissible locations are CG_Null, CG_UserDefined, Vertex, CellCenter, FaceCenter,
      IFaceCenter, JFaceCenter, KFaceCenter, and EdgeCenter.
      
    - Return:
    * None
    
    """
    
    self._error=cgnslib.cg_gridlocation_write(gloc)

  # ------------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------------
  cpdef ordinal_write(self, int ord):

    """
    Writes the ordinal value in a new node.

    - Args:
    * `ord` : any integer value  (`int`)
      
    - Return:
    * None
    
    """
    
    self._error=cgnslib.cg_ordinal_write(ord)

  # ------------------------------------------------------------------------------------------
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

  # ------------------------------------------------------------------------------------------
  cpdef ptset_info(self):

    """
    Returns a tuple with information about the point set.
  
    - Args:
    * None
      
    - Return:
    * point set type  (`int`)
      The admissible types are PointRange for a range of points or cells, and PointList for a
      list of discrete points or cells.
    * number of points or cells in the point set (`int`)
      For a point set type of PointRange, the number is always two.
      For a point set type of PointList, this is the number of points or cells in the list. 

    """
    
    cdef cgnslib.PointSetType_t pst
    cdef cgnslib.cgsize_t npnts
    self._error=cgnslib.cg_ptset_info(&pst,&npnts)
    return (pst,npnts)

  # ------------------------------------------------------------------------------------------
  cpdef famname_write(self, char * name):
    self._error=cgnslib.cg_famname_write(name)

  # ------------------------------------------------------------------------------------------
  cpdef famname_read(self):
    cdef char * name = " "
    self._error=cgnslib.cg_famname_read(name)
    return name

  # ------------------------------------------------------------------------------------------
  cpdef multifam_write(self, char * name, char * fam):
      self._error=cgnslib.cg_multifam_write(name,fam)
    
##   cpdef family_name_write(self, int B, int F, char * name, char * family):
##     self._error=cgnslib.cg_family_name_write(self._root,B,F,name,family)
##     print cgnslib.cg_get_error()

  # ------------------------------------------------------------------------------------------
##   cpdef nfamily_names(self, int B, int F):
##     cdef int n = -1
##     self._error=cgnslib.cg_nfamily_names(self._root,B,F,&n)
##     return n
  
## # =================================================================================

    

