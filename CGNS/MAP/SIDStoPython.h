/* ======================================================================
 * CHLone - CGNS HDF5 LIBRARY only node edition
 * See license.txt in the root directory of this source release
 * ====================================================================== */
#ifndef __SIDSTOPYTHON__H__
#define __SIDSTOPYTHON__H__

/* required for cython, should stay in the implementation */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "bytesobject.h"

#include "CHLone_config.h"
#include "sha256.h"
#include "l3.h"

#ifdef _WINDOWS
#undef _DEBUG
#endif

/* ------------------------------------------------------------------------- */
#define SIDSTOPYTHON_MAJOR 2
#define SIDSTOPYTHON_MINOR 0

/* ------------------------------------------------------------------------- */
#define S2P_SNONE          0x0000
#define S2P_SALL           0xFFFF
#define S2P_SNODATA        0x0001
#define S2P_SPARTIAL       0x0002
#define S2P_SCONTIGUOUS    0x0004
#define S2P_SINTERLACED    0x0008
#define S2P_SSKIP          0x0010

/* ------------------------------------------------------------------------- */
#define S2P_FNONE          0x00000000 /* USED */
#define S2P_FALL           0xFFFFFFFF /* USED */
#define S2P_FTRACE         0x00000001 /* USED */
#define S2P_FFOLLOWLINKS   0x00000002 /* USED */
#define S2P_FNODATA        0x00000004 /* USED */
#define S2P_FKEEPLIST      0x00000008 /* USED */
#define S2P_FCOMPRESS      0x00000010 /* USED */
#define S2P_FREVERSEDIMS   0x00000020 /* USED */
#define S2P_FOWNDATA       0x00000040 /* USED */
#define S2P_FUPDATE        0x00000080 /* USED */
#define S2P_FDELETEMISSING 0x00000100 /* USED */
#define S2P_FALTERNATESIDS 0x00000200 /* USED */
#define S2P_FUPDATEONLY    0x00000400 /* USED */
#define S2P_FFORTRANFLAG   0x00000800 /* USED */
#define S2P_FREADONLY      0x00001000 /* RESERVED */
#define S2P_FNEW           0x00002000 /* USED */
#define S2P_FPROPAGATE     0x00004000 /* USED */
#define S2P_FDEBUG         0x00008000 /* USED */
#define S2P_FLINKOVERRIDE  0x00010000 /* USED */
#define S2P_FCONTIGUOUS    0x00020000 /* RESERVED */
#define S2P_FINTERLACED    0x00040000 /* RESERVED */
#define S2P_FCHECKSUM      0x00080000 /* USED */
#define S2P_FLAG23         0x00100000 /* RESERVED */
#define S2P_FLAG24         0x00200000 /* RESERVED */
#define S2P_FLAG25         0x00400000 /* RESERVED */
#define S2P_FLAG26         0x00800000 /* RESERVED */
#define S2P_FLAG27         0x01000000 /* RESERVED */
#define S2P_FLAG28         0x02000000 /* RESERVED */
#define S2P_FLAG29         0x04000000 /* RESERVED */
#define S2P_FLAG30         0x08000000 /* RESERVED */
#define S2P_FLAG31         0x10000000 /* RESERVED */
#define S2P_FLAG32         0x20000000 /* RESERVED */
#define S2P_FLAG33         0x40000000 /* RESERVED */
#define S2P_FLAG34         0x80000000 /* RESERVED */

#define S2P_FDEFAULT  (S2P_FNONE|S2P_FFOLLOWLINKS|S2P_FDELETEMISSING|S2P_FOWNDATA|S2P_FREVERSEDIMS|S2P_FFORTRANFLAG|S2P_FOWNDATA|S2P_FLINKOVERRIDE)

#define S2P_EFILEUNKWOWN 100
#define S2P_ENOTHDF5FILE 101
#define S2P_EFAILOLDOPEN 102
#define S2P_EFAILUPDOPEN 103
#define S2P_EFAILNEWOPEN 104
#define S2P_EFAILLNKOPEN 105
#define S2P_EBADSTRUCTOB 200
#define S2P_ECANNOTCREAT 201
#define S2P_EDUPLICATEUP 202
#define S2P_EBADTREEROOT 300
#define S2P_EMAXLINKSTCK 400
#define S2P_EMAXCTGINDEX 1024

#define S2P_LKOK         0x0000
#define S2P_LKFAIL       0x0001
#define S2P_LKBADSYNTAX  0x0002
#define S2P_LKNOFILE     0x0004
#define S2P_LKFILENOREAD 0x0008
#define S2P_LKNONODE     0x0010
#define S2P_LKLOOP       0x0020
#define S2P_LKIGNORED    0x0040 
#define S2P_LKUPDATED    0x0080 

#define CG_CGNSLibraryVersion_n		     "CGNSLibraryVersion"
#define CG_CGNSLibraryVersion_ts	     "CGNSLibraryVersion_t"
#define CG_CGNSTree_n			     "CGNSTree"
#define CG_CGNSTree_ts			     "CGNSTree_t"

/* ------------------------------------------------------------------------- */
typedef struct s2p_ent_t
{
  char   	   *filename;
  char   	   *dirname;
  L3_Cursor_t 	   *l3db;
  struct s2p_ent_t *next;
} s2p_ent_t;

/* ------------------------------------------------------------------------- */
typedef struct s2p_lnk_t
{
  char 		   *dst_dirname;
  char 		   *dst_filename;
  char 		   *dst_nodename;
  char 		   *src_dirname;
  char 		   *src_filename;
  char 		   *src_nodename;
  PyObject         *dst_object;
  int               status;
  struct s2p_lnk_t *next;
} s2p_lnk_t;

/* ------------------------------------------------------------------------- */
typedef struct s2p_pth_t
{
  char *path;
  int   state;
  int   dtype;
  int   dims[L3C_MAX_DIMS];
  struct s2p_pth_t *next;
} s2p_pth_t;

/* should not have more than MAX link depth entries */
#define S2P_MAX_LINK_STACK L3C_MAX_LINK_DEPTH+1024

/* ------------------------------------------------------------------------- */
typedef struct s2p_ctx_t
{
  s2p_pth_t *pth;
  s2p_lnk_t *lnk;
  s2p_ent_t *hdf_dbs;                     /* open file entries */
  s2p_ent_t *hdf_stk[S2P_MAX_LINK_STACK]; /* file entries stack */
  int        hdf_idx;                     /* current file entry */
  int        dpt;
  int        mxs;
  PyObject  *lnk_obj;/* original lnk object */
  PyObject  *upd_pth;/* dict of path/objects to update */
  PyObject  *upd_pth_lk;/* dict of path/objects to update wrt linked-to file */
  PyObject  *flt_dct;/* object dict of filters */
  void      *ctg_obj[S2P_EMAXCTGINDEX];/* ptrs to pending contiguous objects */
  void      *ctg_siz[S2P_EMAXCTGINDEX];/* max rank for contiguous objects */
  PyObject  *skp_pth;/* list of paths to ignore */
  PyObject  *skp_pth_lk;/* list of paths to ignore wrt linked-to file */
  PyObject  *err;
  PyObject **pol_oid;/* parsed object list adresses for loop detection */
  int        pol_max;/* max allocated of parsed object list */
  int        pol_cur;/* current index of last used parsed object list */
  unsigned char checksum[32];/* on the fly checksum buffer */
  sha256_t  *sha256;/* on the fly checksum private structure */
  PyGILState_STATE gstate;
  long       flg;
  char      *lsp;
  int        platform;/* unix (0) windows (1) */
} s2p_ctx_t;

#define S2P_PLATFORM_UNIX    0
#define S2P_PLATFORM_WINDOWS 1

/* ------------------------------------------------------------------------- */
PyObject* s2p_loadAsHDF(char      *dirname,
			char      *filename,
			int        flags,
			int        depth,
			int        maxdata,
			char      *path,
			char      *searchpath,
			PyObject  *update,
			PyObject  *filter,
            PyObject  *skip,
			PyObject  *except);
/* ------------------------------------------------------------------------- */
PyObject* s2p_saveAsHDF(char      *dirname,
    	                char      *filename,
    	                PyObject  *tree,
    	                PyObject  *links,
    	                int        flags,
    	                int        depth,
			char      *searchpath,
			PyObject  *update,
			PyObject  *filter,
			PyObject  *skip,
			PyObject  *lkupdate,
			PyObject  *lkskip,
			PyObject  *except);
/* ------------------------------------------------------------------------- */
int s2p_probe(char *filename,char *path);
int s2p_garbage(PyObject *tree);
/* ------------------------------------------------------------------------- */

#endif

