cdef extern from "cgnslib.h":

  cdef extern enum ZoneType:
      ZoneTypeNull=0
      ZoneTypeUserDefined=1
      Structured=2
      Unstructured=3
      
  ctypedef ZoneType ZoneType_t
  
  cdef extern int MODE_READ
  cdef extern int MODE_WRITE
  cdef extern int MODE_CLOSED
  cdef extern int MODE_MODIFY
  
  cdef extern int cg_open(char *name, int mode, int *db)
  cdef extern int cg_version(int db,float *v)
  cdef extern int cg_close(int db)

  cdef extern int cg_nbases(int db,int *nbases)
  cdef extern int cg_base_read(int file_number, int B, char *basename,\
                               int *cell_dim, int *phys_dim)
  cdef extern int cg_base_id(int fn, int B, double *base_id)
  cdef extern int cg_base_write(int file_number, char * basename,\
                                int cell_dim, int phys_dim, int *B)

  cdef extern int cg_nzones(int fn, int B, int *nzones)
  cdef extern int cg_zone_read(int fn, int B, int Z, char *zonename, int *size)
  cdef extern int cg_zone_type(int file_number, int B, int Z, ZoneType_t *type)
  cdef extern int cg_zone_id(int fn, int B, int Z, double *zone_id)
  cdef extern int cg_zone_write(int fn, int B, char * zonename, \
                                int * size, ZoneType_t type, int *Z)
