#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release: v4.0.1 $
#  -------------------------------------------------------------------------
"""
CGNS.WRA.mll is a CGNS/MLL Python wrapper. It defines a `pyCGNS` class
which performs the `cg_open` and holds the opened file descriptor for
subsequent calls.

CGNS.WRA.mll is using a Cython wrapping, it replaces the CGNS.WRA._mll which
was an old self-coded wrapper, difficult to maintain.
"""
import os.path
import CGNS.PAT.cgnskeywords as CK
import numpy

cimport cgnslib
cimport numpy 

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
    if (dtype==CK.RealSingle):  return numpy.float32
    if (dtype==CK.RealDouble):  return numpy.float64
    if (dtype==CK.Integer):     return numpy.int32
    if (dtype==CK.LongInteger): return numpy.int64
    if (dtype==CK.Character):   return numpy.uint8
    return None
        
cdef fromnumpydtype(dtype):
    if (dtype==numpy.float32):  return CK.RealSingle
    if (dtype==numpy.float64):  return CK.RealDouble
    if (dtype==numpy.int32):    return CK.Integer
    if (dtype==numpy.int64):    return CK.LongInteger
    if (dtype==numpy.uint8):    return CK.Character
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
    return CK.CG_OK
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
     * `B`: the CGNS/MLL base id (this is **not** the internal id)

    - Return:
     * The base internal id

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
     * New base id
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
    Returns the number of zones in a base::

      for Z in range(1,db.nzones(B)+1):
         print db.zone_read(B,Z)[2]

    - Args:
     * `B`: Parent base id (:py:func:`bases` and :py:func:`nbases`).

    - Return:
     * Number of zones as an integer

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
     * `B`: base id 
     * `Z`: zone id 

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
    (bid,bname,cdim,pdim)=self.base_read(B)
    azsize=numpy.ones((cdim*pdim),dtype=numpy.int32)
    zsize=<int *>numpy.PyArray_DATA(azsize)
    self._error=cgnslib.cg_zone_read(self._root,B,Z,zonename,zsize)
    return (B,Z,zonename,azsize)
  # ---------------------------------------------------------------------------
  cpdef zone_type(self, int B, int Z):
    """
    Returns the CGNS type of a zones.

    - Args:
     * `B`: the parent base id (:py:func:`bases` and :py:func:`nbases`).
     * `Z`: the parent zone id (:py:func:`zones` and :py:func:`nzones`).

    - Return:
     * The ZoneType_t of the zone as an integer. This is an enumerate value.
    """
    cdef cgnslib.ZoneType_t ztype
    self._error=cgnslib.cg_zone_type(self._root,B,Z,&ztype)
    return ztype
  # ---------------------------------------------------------------------------
  cpdef zone_id(self, int B, int Z):
    """
    Returns the zone internal id.

    - Args:
     * `B`: the CGNS/MLL base id
     * `Z`: the CGNS/MLL zone id (this is **not** the internal id)

    - Return:
     * The zone internal id

    - Remarks:
     * This id can be used with the CGNS/ADF or CGNS/HDF5 API. If you don't
     know what this id is... then you should not use it.
    """
    cdef double izid
    self._error=cgnslib.cg_zone_id(self._root,B,Z,&izid)
    return izid
  # ---------------------------------------------------------------------------
  cpdef zone_write(self, int B, char *zonename, zsize,
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
     * Zone type is an integer that should be one of the `CGNS.PAT.cgnskeywords.ZoneType_` keys
     * Zone size depends on base dimensions and zone type (see `CGNS/SIDS 6.3 <http://www.grc.nasa.gov/WWW/cgns/CGNS_docs_current/sids/cgnsbase.html#Zone>`_)
    """
    cdef int  zid
    cdef int *ptrzs
    ptrzs=<int *>numpy.PyArray_DATA(zsize)
    self._error=cgnslib.cg_zone_write(self._root,B,zonename,ptrzs,ztype,&zid)
    return zid
  # ---------------------------------------------------------------------------
  cpdef nfamilies(self, int B):
    """
    Returns the number of families in a base::

      for F in range(1,db.nfamilies(B)+1):
         print db.family_read(B)

    - Args:
     * `B`: the parent base id (:py:func:`bases` and :py:func:`nbases`).

    - Return:
     * The number of families as an integer

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
     * `B`: parent base id (:py:func:`bases` and :py:func:`nbases`).

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
     * `B`: base id 
     * `F`: family id 

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
     * `B`: the parent base id (:py:func:`bases` and :py:func:`nbases`).
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
     * `B`: base id 
     * `F`: family id
     * `BC`: BC family id 

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
     * `B`: parent base id (:py:func:`bases` and :py:func:`nbases`).
     * `F`: parent family id (:py:func:`families` and :py:func:`nfamilies`).
     * `fambcname`: name of the new BC family (`string` should not exceed 32 chars)
     * `bocotype`: type of the actual BC for all BCs refering to the parent family name of `F`

    - Return:
     * New BCfamily id

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
     * `B`: base id 
     * `F`: family id
     * `G`: geometry reference id 

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
     * `B`: parent base id (:py:func:`bases` and :py:func:`nbases`).
     * `F`: parent family id (:py:func:`families` and :py:func:`nfamilies`).
     * `geoname`: name of the new geometry reference (`string` should not exceed 32 chars)
     * `filename`: path to geometry reference file
     * `cadname`: name of the geometry reference CAD

    - Return:
     * New Geometry reference id

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
     * `B`: base id 
     * `F`: family id
     * `G`: geometry reference id
     * `P`: geometry reference part id 

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
     * `B`: parent base id (:py:func:`bases` and :py:func:`nbases`).
     * `F`: parent family id (:py:func:`families` and :py:func:`nfamilies`).
     * `G`: geometry reference id
     * `partname`: name of the new geometry reference part (`string` should not exceed 32 chars)

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
     * `B`: parent base id (:py:func:`bases` and :py:func:`nbases`).
     * `Z`: parent zone id (:py:func:`zones` and :py:func:`nzones`).

    - Return:
     * The number of grids as an integer

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
     * `B`: parent base id (:py:func:`bases` and :py:func:`nbases`).
     * `Z`: parent zone id (:py:func:`zones` and :py:func:`nzones`).

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
     * `B`: base id 
     * `Z`: zone id
     * `G`: grid id      

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
     * `B`: base id 
     * `Z`: zone id
     * `gridname`: name of the new grid (`string` should not exceed 32 chars)

    - Return:
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
     * `B`: parent base id
     * `Z`: parent zone id

    - Return:
     * The number of coordinates arrays as an integer

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
     * `B`: parent base id
     * `Z`: parent zone id

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
     * `B`: base id 
     * `Z`: zone id
     * `C`: coordinates id (:py:func:`coords` and :py:func:`ncoords`)

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
     * `B`: base id 
     * `Z`: zone id
     * `coordname`: coordinate array name to read
     * `dtype`: datatype of the array 

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
    (bid,zid,zname,zsize)=self.zone_read(B,Z)
    rmin=numpy.ones((cdim),dtype=numpy.int32)
    rmax=numpy.ones((cdim),dtype=numpy.int32)
    rminptr=<int *>numpy.PyArray_DATA(rmin)
    rmaxptr=<int *>numpy.PyArray_DATA(rmax)
    cdtype=asnumpydtype(dtype)
    if (cdtype == None):
        ndtype=fromnumpydtype(dtype)
        if (ndtype == None):
            raise CGNSException(10,"No such data type: %s"%str(ndtype))
        dtype=ndtype
        cdtype=asnumpydtype(dtype)
    coords=numpy.ones(zsize,dtype=cdtype)
    coordsptr=<void *>numpy.PyArray_DATA(coords)
    self._error=cgnslib.cg_coord_read(self._root, B, Z,
                                      coordname,dtype,
                                      rminptr,rmaxptr,coordsptr)
    return (B,Z,coordname,dtype,rmin,rmax,coords)
  # ---------------------------------------------------------------------------
  cpdef coord_id(self, int B, int Z, int C):
    """
    Returns the base internal id.

    - Args:
     * `B`: the CGNS/MLL base id
     * `Z`: the CGNS/MLL zone id
     * `C`: the CGNS/MLL coordinates id

    - Return:
     * The coordinates internal id

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
     * `B`: base id 
     * `Z`: zone id
     * `dtype`: data type of the array contents (`int`)
     * `coordname`: name of the new coordinates (`string` should not exceed 32 chars)
     * `coords`: array of actual coordinates (`numpy.ndarray`)
     
    - Return:
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
    coordsptr=<void *>numpy.PyArray_DATA(coords)
    self._error=cgnslib.cg_coord_write(self._root, B, Z,
                                       dtype, coordname,coordsptr,&cid)
    return cid
  # ---------------------------------------------------------------------------
  cpdef coord_partial_write(self, int B, int Z, cgnslib.DataType_t dtype,
                            char *coordname, rmin, rmax, coords):
    """
    Modify coordinates.

    - Args:
     * `B`: base id 
     * `Z`: zone id
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
    rminptr=<int *>numpy.PyArray_DATA(rmin)
    rmaxptr=<int *>numpy.PyArray_DATA(rmax)
    coordsptr=<float *>numpy.PyArray_DATA(coords)
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
    
# ====================================================================
