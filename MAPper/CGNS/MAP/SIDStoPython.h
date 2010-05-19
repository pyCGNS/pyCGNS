/* ------------------------------------------------------------------------- */
/* pyCGNS.MAP - CFD General Notation System - SIDS-to-Python MAPping         */
/* See license.txt file in the root directory of this Python module source   */
/* ------------------------------------------------------------------------- */
#ifndef __SIDSTOPYTHON__H__
#define __SIDSTOPYTHON__H__

#include "Python.h"
#include "CHLone/l3.h"
#include "CHLone/SIDS.h"

/* ------------------------------------------------------------------------- */
#define S2P_FNONE          0x0000
#define S2P_FALL           0xFFFF
#define S2P_FTRACE         0x0001  
#define S2P_FFOLLOWLINKS   0x0002
#define S2P_FNODATA        0x0004
#define S2P_FMERGELINKS    0x0008
#define S2P_FCOMPRESS      0x0010
#define S2P_FNOTRANSPOSE   0x0020
#define S2P_FOWNDATA       0x0040
#define S2P_FUPDATE        0x0080
#define S2P_FDELETEMISSING 0x0100

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
  char    	   *targetfilename;
  char    	   *targetnodename;
  char    	   *localnodename;
  struct s2p_lnk_t *next;
} s2p_lnk_t;

/* ------------------------------------------------------------------------- */
typedef struct s2p_ctx_t
{
  s2p_lnk_t *lnk;
  s2p_ent_t *dbs;
  int        thh;
  int        dpt;
  long       flg;
  int       *_c_int;
  long      *_c_long;
  float     *_c_float;
  double    *_c_double;
  char      *_c_char;
} s2p_ctx_t;

/* ------------------------------------------------------------------------- */
PyObject* s2p_loadAsHDF(char *filename,
			int flags,
			int threshold,
			int depth,
			char *path);
/* ------------------------------------------------------------------------- */
int 	  s2p_saveAsHDF(char      *filename,
    	                PyObject  *tree,
    	                PyObject  *links,
    	                int        flags,
    	                int        threshold,
    	                int        depth,
    	                char*      path);
/* ------------------------------------------------------------------------- */

#endif

