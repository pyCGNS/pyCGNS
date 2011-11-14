#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release: v4.0.1 $
#  -------------------------------------------------------------------------

cdef extern from "cgnslib.h":

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
    
  int MODE_READ
  int MODE_WRITE
  int MODE_CLOSED
  int MODE_MODIFY
  
  int cg_open(char *name, int mode, int *db)
  int cg_version(int db,float *v)
  int cg_close(int db)

  int cg_nbases(int db,int *nbases)
  int cg_base_read(int file_number, int B, char *basename,\
                   int *cell_dim, int *phys_dim)
  int cg_base_id(int fn, int B, double *base_id)
  int cg_base_write(int file_number, char * basename,\
                    int cell_dim, int phys_dim, int *B)

  int cg_nzones(int fn, int B, int *nzones)
  int cg_zone_read(int fn, int B, int Z, char *zonename, int *size)
  int cg_zone_type(int file_number, int B, int Z, ZoneType_t *type)
  int cg_zone_id(int fn, int B, int Z, double *zone_id)
  int cg_zone_write(int fn, int B, char * zonename, \
                    int * size, ZoneType_t type, int *Z)

  int cg_nfamilies(int file_number, int B, int *nfamilies)
  int cg_family_read(int file_number, int B, int F,\
                     char *family_name, int *nboco, int *ngeos)
  int cg_family_write(int file_number, int B,\
                      char * family_name, int *F)

  int cg_fambc_read(int file_number, int B, int F, int BC,\
                    char *fambc_name, BCType_t *bocotype)
  int cg_fambc_write(int file_number, int B, int F,\
                     char * fambc_name, BCType_t bocotype, int *BC)
