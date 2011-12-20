#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release: v4.0.1 $
#  -------------------------------------------------------------------------

cdef extern from "cgnslib.h":
  
  ctypedef int cgsize_t

  ctypedef enum DataType_t:
      DataTypeNull=0
      DataTypeUserDefined=1
      Integer=2
      RealSingle=3
      RealDouble=4
      Character=5
      LongInteger=6
      
  ctypedef enum ZoneType_t:
      ZoneTypeNull=0
      ZoneTypeUserDefined=1
      Structured=2
      Unstructured=3
      
  ctypedef enum BCType_t:
      BCTypeNull=0
      BCTypeUserDefined=1
      BCAxisymmetricWedge=2
      BCDegenerateLine=3
      BCDegeneratePoint=4
      BCDirichlet=5
      BCExtrapolate=6
      BCFarfield=7
      BCGeneral=8
      BCInflow=9
      BCInflowSubsonic=10
      BCInflowSupersonic=11
      BCNeumann=12
      BCOutflow=13
      BCOutflowSubsonic=14
      BCOutflowSupersonic=15
      BCSymmetryPlane=16
      BCSymmetryPolar=17
      BCTunnelInflow=18
      BCTunnelOutflow=19
      BCWall=20
      BCWallInviscid=21
      BCWallViscous=22
      BCWallViscousHeatFlux=23
      BCWallViscousIsothermal=24
      FamilySpecified=25

  ctypedef enum ElementType_t:
      ElementTypeNull=0
      ElementTypeUserDefined=1
      NODE=2
      BAR_2=3
      BAR_3=4
      TRI_3=5
      TRI_6=6
      QUAD_4=7
      QUAD_8=8
      QUAD_9=9
      TETRA_4=10
      TETRA_10=11
      PYRA_5=12
      PYRA_14=13
      PENTA_6=14
      PENTA_15=15
      PENTA_18=16
      HEXA_8=17
      HEXA_20=18
      HEXA_27=19
      MIXED=20
      PYRA_13=21
      NGON_n=22
      NFACE_n=23
   
  int MODE_READ
  int MODE_WRITE
  int MODE_CLOSED
  int MODE_MODIFY
  
  int cg_open(char *name, int mode, int *db)
  int cg_version(int db,float *v)
  int cg_close(int db)

  int cg_nbases(int db,int *nbases)
  int cg_base_read(int file_number, int B, char *basename,
                   int *cell_dim, int *phys_dim)
  int cg_base_id(int fn, int B, double *base_id)
  int cg_base_write(int file_number, char * basename,
                    int cell_dim, int phys_dim, int *B)

  int cg_nzones(int fn, int B, int *nzones)
  int cg_zone_read(int fn, int B, int Z, char *zonename, int *size)
  int cg_zone_type(int file_number, int B, int Z, ZoneType_t *type)
  int cg_zone_id(int fn, int B, int Z, double *zone_id)
  int cg_zone_write(int fn, int B, char * zonename, 
                    int * size, ZoneType_t type, int *Z)

  int cg_nfamilies(int file_number, int B, int *nfamilies)
  int cg_family_read(int file_number, int B, int F,
                     char *family_name, int *nboco, int *ngeos)
  int cg_family_write(int file_number, int B,
                      char * family_name, int *F)

  int cg_fambc_read(int file_number, int B, int F, int BC,
                    char *fambc_name, BCType_t *bocotype)
  int cg_fambc_write(int file_number, int B, int F,
                     char * fambc_name, BCType_t bocotype, int *BC)

  int cg_geo_read(int file_number, int B, int F, int G, char *geo_name,
                  char **geo_file, char *CAD_name, int *npart)
  int cg_geo_write(int file_number, int B, int F, char * geo_name,
                   char * filename, char * CADname, int *G)

  int cg_part_read(int file_number, int B, int F, int G, int P,
                   char *part_name)
  int cg_part_write(int file_number, int B, int F, int G,
                    char * part_name, int *P)
 
  int cg_ngrids(int file_number, int B, int Z, int *ngrids)
  int cg_grid_read(int file_number, int B, int Z, int G, char *gridname)
  int cg_grid_write(int file_number, int B, int Z,
                    char * gridname, int *G)

  int cg_ncoords(int fn, int B, int Z, int *ncoords)
  int cg_coord_info(int fn, int B, int Z, int C,
                    DataType_t *type, char *coordname)
  int cg_coord_read(int fn, int B, int Z,  char * coordname,
                    DataType_t type,  cgsize_t * rmin,
                    cgsize_t * rmax, void *coord)
  int cg_coord_id(int fn, int B, int Z, int C, double *coord_id)
  int cg_coord_write(int fn, int B, int Z,
                     DataType_t type,  char * coordname,
                     void * coord_ptr, int *C)
  int cg_coord_partial_write(int fn, int B, int Z,
                             DataType_t type,  char * coordname,
                             cgsize_t *rmin,  cgsize_t *rmax,
                             void * coord_ptr, int *C)


  # ---------------------------------------------------------------------
  # Above is wrapped
  # =====================================================================
  # Below is to be wrapped soon
  # ---------------------------------------------------------------------

  int cg_nsections(int file_number, int B, int Z, int *nsections)
  int cg_section_read(int file_number, int B, int Z, int S,
                      char *SectionName, ElementType_t *type,
                      cgsize_t *start, cgsize_t *end,
                      int *nbndry, int *parent_flag)
  int cg_elements_read(int file_number, int B, int Z, int S,
                       cgsize_t *elements, cgsize_t *parent_data)
  int cg_section_write(int file_number, int B, int Z,
                       char * SectionName, ElementType_t type,
                       cgsize_t start, cgsize_t end, int nbndry,
                       cgsize_t * elements, int *S)
  int cg_parent_data_write(int file_number, int B, int Z, int S,
                           cgsize_t * parent_data)
  int cg_npe( ElementType_t type, int *npe)
  int cg_ElementDataSize(int file_number, int B, int Z, int S,
                         cgsize_t *ElementDataSize)
  int cg_section_partial_write(int file_number, int B, int Z,
                               char * SectionName, ElementType_t type,
                               cgsize_t start, cgsize_t end, int nbndry, int *S)
  int cg_elements_partial_write(int fn, int B, int Z, int S,
                                cgsize_t start, cgsize_t end,
                                cgsize_t *elements)
  int cg_parent_data_partial_write(int fn, int B, int Z, int S,
                                   cgsize_t start, cgsize_t end,
                                   cgsize_t *ParentData)
  int cg_elements_partial_read(int file_number, int B, int Z, int S,
                               cgsize_t start, cgsize_t end,
                               cgsize_t *elements, cgsize_t *parent_data)
  int cg_ElementPartialSize(int file_number, int B, int Z, int S,
                            cgsize_t start, cgsize_t end,
                            cgsize_t *ElementDataSize)

# --- last line
