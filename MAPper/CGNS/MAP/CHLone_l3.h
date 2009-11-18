/* 
#  -------------------------------------------------------------------------
#  pyCGNS.MAP - Python package for CFD General Notation System - MAPper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $File$
#  $Node$
#  $Last$
#  ------------------------------------------------------------------------- 
*/

/* L3 : LOW LEVEL LIBRARY - single node API */
 
#ifndef __CHLONE_L3_H__
#define __CHLONE_L3_H__

#include "CHLone_config.h"

#define L3_S_NAME    	   "name"
#define L3_S_LABEL   	   "label"
#define L3_S_DTYPE   	   "type"
#define L3_S_FLAGS   	   "flags"
#define L3_S_VERSION 	   " hdf5version"
#define L3_S_FORMAT  	   " format"
#define L3_S_DATA    	   " data"
#define L3_S_FILE    	   " file"
#define L3_S_PATH    	   " path"
#define L3_S_LINK    	   " link"  

#define L3_S_ROOTNODENAME  "HDF5 MotherNode"
#define L3_S_ROOTNODETYPE  "Root Node of HDF5 File"

#define L3_MAX_DIMS        12
#define L3_MAX_VALUE_SIZE  32
#define L3_MAX_LINK_DEPTH  100
#define L3_MAX_SIZE_BUFFER 4096
#define L3_MAX_ATTRIB_SIZE L3_MAX_VALUE_SIZE
#define L3_MAX_NAME        L3_MAX_VALUE_SIZE
#define L3_MAX_LABEL       L3_MAX_VALUE_SIZE
#define L3_MAX_DTYPE       2
#define L3_MAX_VERSION     L3_MAX_VALUE_SIZE
#define L3_MAX_FORMAT      20
#define L3_MAX_FILE        1024
#define L3_MAX_PATH        4096   
#define L3_MAX_MUTEX       256

#define L3_OPEN_NEW 	   0
#define L3_OPEN_OLD 	   1
#define L3_OPEN_RDO 	   2      

#define L3E_NULL  0
#define L3E_C1    1
#define L3E_C1ptr 2
#define L3E_I4    3
#define L3E_I8    4
#define L3E_R4    5
#define L3E_R8    6
#define L3E_I4ptr 7
#define L3E_I8ptr 8
#define L3E_R4ptr 9
#define L3E_R8ptr 10
#define L3E_VOID  11
#define L3E_LK    12
#define L3E_B1    13
#define L3E_U4    14
#define L3E_U8    15

/* --- flags --------------------------------------------------------------- */
#define L3_F_NONE          0x0000
#define L3_F_ALL           0xFFFF
#define L3_F_WITHDATA      0x0001  
#define L3_F_WITHCHILDREN  0x0002
#define L3_F_FAILSONLINK   0x0004
#define L3_F_FOLLOWLINKS   0x0008
#define L3_F_MERGELINKS    0x0010
#define L3_F_COMPRESS      0x0020
#define L3_F_NOTRANSPOSE   0x0040
#define L3_F_OWNDATA       0x0080
#define L3_F_NOCHILDMERGE  0x0100
#define L3_F_WITHLINKINFO  0x0200
#define L3_F_WITHMUTEX     0x0400
#define L3_F_WITHCACHE     0x0800
#define L3_F_BULLETPROOF   0x1000
#define L3_F_HASEXTFLAGS   0x2000
#define L3_F_DEBUG         0x4000
#define L3_F_TRACE         0x8000

#define L3_H_NONE          0x0000

#define L3M_HASFLAG( cursor, flag )   ((cursor->config & flag) == flag)
#define L3M_SETFLAG( cursor, flag )   (cursor->config |= flag)
#define L3M_UNSETFLAG( cursor, flag ) (cursor->config &= ~flag)

#define L3_F_OPEN_DEFAULT L3_F_FOLLOWLINKS|L3_F_OWNDATA

#define CHL_NOERROR -1

#define L3M_ECHECK(ctxt)\
  (ctxt==NULL?0:(ctxt->last_error!=CHL_NOERROR?0:1))

#define L3M_ECLEAR(ctxt)\
  if (ctxt!=NULL){\
  ctxt->ebuffptr=0;\
  ctxt->last_error=CHL_NOERROR;}

/* ------------------------------------------------------------------------- */
/*
   A cursor is a context keeping information on a CGNS tree, file, status
   of a current path, etc...
   The CHLONE allow multiple use of cursors
*/
typedef struct L3_Cursor_t {
  hid_t  file_id;                    	 /* the database file id */
  hid_t  root_id;                    	 /* the database root id */
  hid_t  stack_id;                    	 /* the error stack id */
  herr_t last_error;           	     	 /* last hdf5 error */
  char   ebuff[L3_MAX_SIZE_BUFFER];  	 /* error buffer */
  int    ebuffptr;                   	 /* error buffer index */
  long   config;                     	 /* config flag field */
  hid_t  g_proplist;                 	 /* HDF5 property list for links */
  hid_t *result;                     	 /* queries result ids */
#ifdef CHLONE_HAS_PTHREAD
  pthread_mutex_t g_mutex;               /* global mutex for this context */
  pthread_mutex_t n_mutex[L3_MAX_MUTEX]; /* mutex array for nodes */
#endif
} L3_Cursor_t;

/* ------------------------------------------------------------------------- */
/* 
   Node is a *transient* data structure that should only be used to read/write
   but not for parse or for context storage.
*/
typedef struct L3_Node_t {
  hid_t   id;                    /* node id if relevant */
  hid_t   parentid;              /* parent node id if relevant */
  hid_t  *children;              /* children node ids */
  char    name [L3_MAX_NAME+1];  /* SIDS name */
  char    label[L3_MAX_LABEL+1]; /* SIDS label */
  int     dtype;                 /* SIDS (ADF) data type */
  int     flags;                 /* filemapping flags */
  int     dims[L3_MAX_DIMS];     /* actual data dimensions */
  void   *data;                  /* actual data array */
} L3_Node_t;

#define L3M_CLEARDIMS(dims) \
{int __nn;for (__nn=0;__nn<L3_MAX_DIMS;__nn++){dims[__nn]=(int)-1;};}

#define L3M_NEWNODE(nodeptr) \
nodeptr=(L3_Node_t*)malloc(sizeof(L3_Node_t));\
nodeptr->id=-1;\
nodeptr->parentid=-1;\
nodeptr->children=NULL;\
nodeptr->data=NULL;\
nodeptr->name[0]='\0';\
nodeptr->label[0]='\0';\
nodeptr->dtype=L3E_NULL;\
nodeptr->flags=L3_H_NONE;\
{int __nn;for (__nn=0;__nn<L3_MAX_DIMS;__nn++){nodeptr->dims[__nn]=-1;}}

#define L3M_CLEARNODE(nodeptr) \
nodeptr->id=-1;\
nodeptr->children=NULL;\
nodeptr->data=NULL;\
nodeptr->name[0]='\0';\
nodeptr->label[0]='\0';\
nodeptr->dtype=L3E_VOID;\
nodeptr->flags=L3_H_NONE;\
{int __nn;for (__nn=0;__nn<L3_MAX_DIMS;__nn++){nodeptr->dims[__nn]=-1;}}

/* ------------------------------------------------------------------------- *
   L3 functions API 
 * ------------------------------------------------------------------------- */
hid_t        L3_nodeCreate  (L3_Cursor_t *ctxt, hid_t pid, L3_Node_t *node);
hid_t        L3_nodeUpdate  (L3_Cursor_t *ctxt, L3_Node_t *node);
L3_Cursor_t *L3_nodeLink    (L3_Cursor_t *ctxt, hid_t node, 
			     char *srcname, char *destfile, char *destname);
L3_Cursor_t *L3_nodeMove    (L3_Cursor_t *ctxt, hid_t parent, hid_t newparent,
                                                char *name, char *newname);
L3_Cursor_t *L3_nodeDelete  (L3_Cursor_t *ctxt, hid_t parent, char *name);
hid_t        L3_nodeFind    (L3_Cursor_t *ctxt, hid_t parent, char *path);
L3_Node_t   *L3_nodeRetrieve(L3_Cursor_t *ctxt, hid_t node);
void         L3_nodePrint   (L3_Node_t *node);

L3_Cursor_t *L3_openFile    (char *filename, int mode, long flags);
L3_Cursor_t *L3_openHID     (hid_t root);
int          L3_close       (L3_Cursor_t *ctxt);

L3_Cursor_t *L3_setFlags    (L3_Cursor_t *ctxt, long flags);
L3_Cursor_t *L3_getFlags    (L3_Cursor_t *ctxt, long *flags);

L3_Node_t   *L3_nodeSet     (L3_Cursor_t *ctxt, L3_Node_t *node,
		 	     char *name, char *label, 
			     int  *dims, int dtype, void *data, int flags);
L3_Node_t   *L3_nodeGet     (L3_Cursor_t *ctxt, L3_Node_t *node, 
			     char *name, char *label, 
			     int  *dims, int *dtype, void *data, int *flags);

int          L3_isSameNode  (L3_Cursor_t *ctxt,hid_t id1,hid_t id2);
int          L3_isLinkNode  (L3_Cursor_t *ctxt,hid_t id,char *file,char *name);
char        *L3_node2Path   (L3_Cursor_t *ctxt,hid_t id);
hid_t        L3_path2Node   (L3_Cursor_t *ctxt,char *path);

void   L3_printError (L3_Cursor_t *ctxt);
void   CHL_setError  (L3_Cursor_t *ctxt, int err, ...);
int    CHL_setMessage(L3_Cursor_t* ctxt,char *msg);

int   *L3_initDims  (int  *dims, int d1, ...);
void  *L3_initData  (int  *dims, void *data, int dtype, ...);
void  *L3_fillData  (int  *dims, void *data, int dtype, ...);

char  *L3_typeAsStr(int dtype);
int    L3_typeAsEnum(char *dtype);

int    L3_config(int p);

/* ------------------------------------------------------------------------- */
/* trace/debug macros */
#ifndef CHLONE_PRINTF_TRACE

#define L3M_DBG(crs,msg) 

#define L3M_TRACE(crs,msg) 

#else

#define L3M_DBG(crs,msg) \
{if (L3M_HASFLAG(crs,L3_F_DEBUG))\
{printf("# L3 (dbg) ");printf msg;fflush(stdout);}}

#define L3M_TRACE(crs,msg) \
{if (L3M_HASFLAG(crs,L3_F_TRACE))\
{printf("# L3 ");printf msg ;fflush(stdout);}}

#endif

/* ------------------------------------------------------------------------- */
#endif
/* --- last line ----------------------------------------------------------- */
