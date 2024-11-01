/* ======================================================================
 * CHLone - CGNS HDF5 LIBRARY only node edition
 * See license.txt in the root directory of this source release
 * ====================================================================== */
/* L3 : LOW LEVEL LIBRARY - single node API */
 
#ifndef __CHLONE_L3_H__
#define __CHLONE_L3_H__

#include "CHLone_config.h"

/* first letter of these three strings is a key */
#define L3S_NAME    	   "name"
#define L3S_LABEL   	   "label"
#define L3S_DTYPE   	   "type"

#define L3S_FLAGS   	   "flags"
#define L3S_VERSION 	   " hdf5version"
#define L3S_FORMAT  	   " format"
#define L3S_DATA    	   " data"
#define L3S_FILE    	   " file"
#define L3S_PATH    	   " path"
#define L3S_LINK    	   " link"  

#define L3S_ROOTNODENAME  "HDF5 MotherNode"
#define L3S_ROOTNODEPATH  "/HDF5 MotherNode"
#define L3S_ROOTNODETYPE  "Root Node of HDF5 File"

#define L3C_MAX_DIMS        12
#define L3C_MAX_VALUE_SIZE  32
#define L3C_MAX_LINK_DEPTH  12
#define L3C_MAX_SIZE_BUFFER 4096
#define L3C_MAX_ATTRIB_SIZE L3C_MAX_VALUE_SIZE
#define L3C_MAX_NAME        L3C_MAX_VALUE_SIZE
#define L3C_MAX_LABEL       L3C_MAX_VALUE_SIZE
#define L3C_MAX_DTYPE       2
#define L3C_MAX_VERSION     L3C_MAX_VALUE_SIZE
#define L3C_MAX_FORMAT      20
#define L3C_MAX_FILE        1024
#define L3C_MAX_PATH        4096   

/*
@@ Enumerate: HDF5 storage strategy for small data
*/
#define L3_CONTIGUOUS_STORE 0
#define L3_COMPACT_STORE 1

/*
@@ Enumerate: Open Modes
*/
#define L3E_OPEN_NEW 0 /*=* create a new file that should not already exist */
#define L3E_OPEN_OLD 1 /*=* read or write an existing file */
#define L3E_OPEN_RDO 2 /*=* open an existing file as read only */
#define L3E_OPEN_READ   L3E_OPEN_RDO
#define L3E_OPEN_UPDATE L3E_OPEN_OLD
#define L3E_OPEN_CREATE L3E_OPEN_NEW

/* 
@@ Enumerate: Data Types
*/
#define L3E_NULL  0  /*=* no data */
#define L3E_C1    1  /*=* char */
#define L3E_C1ptr 2  /*=* pointer to char */
#define L3E_I4    3  /*=* simple integer */
#define L3E_I8    4  /*=* double integer */
#define L3E_R4    5  /*=* simple float */
#define L3E_R8    6  /*=* double float */
#define L3E_X4    7  /*=* simple float */
#define L3E_X8    8  /*=* double float */
#define L3E_I4ptr 9  /*=* pointer to simple integer */
#define L3E_I8ptr 10  /*=* pointer to double integer */
#define L3E_R4ptr 11  /*=* pointer to simple float */
#define L3E_R8ptr 12 /*=* pointer to double integer */
#define L3E_X4ptr 13  /*=* pointer to simple float */
#define L3E_X8ptr 14 /*=* pointer to double integer */
#define L3E_VOID  15 /*=* pointer to unknown */

#define L3E_MT L3E_NULL


/*
@@ Enumerate: L3 Flags
*/
#define L3F_NONE          0x0000 /*=* empty flag, all values set to false */
#define L3F_NULL          0xFFFF /*=* unsignificant flags value */
#define L3F_WITHDATA      0x0001 /*=* get/set data into the node */
#define L3F_WITHCHILDREN  0x0002 /*=* parse the children of the node */
#define L3F_FAILSONLINK   0x0004 /*=* raises an error if link unreachable */
#define L3F_FOLLOWLINKS   0x0008 /*=* follow links */
#define L3F_MERGELINKS    0x0010 /*=* unused */
#define L3F_COMPRESS      0x0020 /*=* unused */
#define L3F_SKIPONERROR   0x0040 /*=* errors reported, process continues */
#define L3F_NOALLOCATE    0x0080 /*=* no data allocation in L3 */
#define L3F_NOCHILDMERGE  0x0100 /*=* unused */
#define L3F_LINKOVERWRITE 0x0200 /*=* link create/update overwrite existing */
#define L3F_WITHMUTEX     0x0400 /*=* use mutexes */
#define L3F_FILLNANDATA   0x0800 /*=* fill new data with NaN */
#define L3F_ASCIIORDER    0x1000 /*=* forces ASCII name order for children */
#define L3F_HASEXTFLAGS   0x2000 /*=* another set of flags is present */
#define L3F_DEBUG         0x4000 /*=* set debug traces */
#define L3F_TRACE         0x8000 /*=* set end-user trace */

/*
@@ Enumerate: L3 node release Flags
*/
#define L3F_R_NONE         0x0000 /*=* empty flag, all values set to false */
#define L3F_R_ALL          0xFFFF /*=* unsignificant flags value */
#define L3F_R_HID_NODE     0x0001 /*=* release node hid_t s */
#define L3F_R_MEM_CHILDREN 0x0002 /*=* free children list */
#define L3F_R_HID_CHILDREN 0x0004 /*=* release children hid_t s */
#define L3F_R_MEM_NODE     0x0008 /*=* release node itself */
#define L3F_R_MEM_DATA     0x0010 /*=* actual data memory of node */

#define __L3F_R_ALL L3F_R_HID_NODE|L3F_R_MEM_CHILDREN|L3F_R_HID_CHILDREN|\
L3F_R_MEM_NODE|L3F_R_MEM_DATA 

#define L3F_DEFAULT \
L3F_FOLLOWLINKS|L3F_WITHDATA|L3F_WITHMUTEX|\
L3F_WITHCHILDREN|L3F_SKIPONERROR

/* ----------------------------------------------------------------------------
@@ Interface: L3/Macro
*/
/*
@@ Function:  L3M_HASFLAG
@@ Arg:       cursor:L3_Cursor_t*:Context to use
@@ Arg:       flag:flag enum:Flag to test
@@ Return:    An integer, 1 if the flag is set, 0 if unset
*/
#define L3M_HASFLAG( cursor, flag )   ((cursor->config & flag) == flag)
/*
@@ Function:  L3M_SETFLAG
@@ Arg:       cursor:L3_Cursor_t*:Context to use
@@ Arg:       flag:flag enum:Flag to set
@@ Return:    The flags as integer 
*/
#define L3M_SETFLAG( cursor, flag )   (cursor->config |= flag)
/*
@@ Function:  L3M_UNSETFLAG
@@ Arg:       cursor:L3_Cursor_t*:Context to use
@@ Arg:       flag:flag enum:Flag to unset
@@ Return:    The flags as integer
*/
#define L3M_UNSETFLAG( cursor, flag ) (cursor->config &= ~flag)

#define CHL_NOERROR -1

#define L3M_EGETCODE(ctxt)\
  (ctxt==NULL?-1:(ctxt->last_error))

#define L3M_EGETSTRING(ctxt)\
  (ctxt==NULL?-1:(ctxt->ebuff))

#define L3M_ECHECK(ctxt)\
  (ctxt==NULL?0:(ctxt->last_error!=CHL_NOERROR?0:1))

#define L3M_ECLEAR(ctxt)\
  if (ctxt!=NULL){\
  ctxt->ebuffptr=0;\
  ctxt->last_error=CHL_NOERROR;}

/* ------------------------------------------------------------------------- */
typedef struct L3_PathList_t
{
  struct L3_PathList_t *next;
  char *path;
  int   index;  
} L3_PathList_t;
 
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
  char   ebuff[L3C_MAX_SIZE_BUFFER];  	 /* error buffer */
  int    ebuffptr;                   	 /* error buffer index */
  long   config;                     	 /* config flag field */
  hid_t  l_proplist;                 	 /* HDF5 property list for links */
  hid_t  g_proplist;                 	 /* HDF5 property list for groups */
  hid_t *result;                     	 /* queries result ids */
  L3_PathList_t *pathlist;
  char  *currentpath;
#ifdef CHLONE_HAS_PTHREAD
  pthread_mutex_t g_mutex;               /* global mutex for this context */
#endif
  hid_t  str_cache_label;                /* cached type attribute name */
  hid_t  str_cache_name;                 /* cached type attribute label */
  hid_t  str_cache_dtype;                /* cached type attribute dtype */
#ifdef CHLONE_TRACK_TIME
  struct tms time;                       /* time storage for debug only */
#endif
} L3_Cursor_t;

/* ------------------------------------------------------------------------- */
/* 
   Node is a *transient* data structure that should only be used to read/write
   but not for parse or for context storage.
*/
typedef struct L3_Node_t {
  hid_t    id;                    /* node id if relevant */
  hid_t    parentid;              /* parent node id if relevant */
  hid_t   *children;              /* children node ids */
  char     name [L3C_MAX_NAME+1]; /* SIDS name */
  char     label[L3C_MAX_LABEL+1];/* SIDS label */
  int      dtype;                 /* SIDS (ADF) data type */
  int      flags;                 /* filemapping flags */
  int      dims[L3C_MAX_DIMS];    /* actual data dimensions */
  void    *data;                  /* actual data array */
  void    *base;                  /* base data array (ex: for contiguous)*/
  hsize_t *offset;                /* hdf5/hyperslab spec */
  hsize_t *stride;                /* hdf5/hyperslab spec */
  hsize_t *count;                 /* hdf5/hyperslab spec */
  hsize_t *block;                 /* hdf5/hyperslab spec */
} L3_Node_t;

#define L3M_CLEARDIMS(dims) \
{int __nn;for (__nn=0;__nn<L3C_MAX_DIMS;__nn++){dims[__nn]=(int)-1;};}

#define L3M_NEWNODE(nodeptr) \
nodeptr=(L3_Node_t*)malloc(sizeof(L3_Node_t));\
nodeptr->id=-1;\
nodeptr->parentid=-1;\
nodeptr->children=NULL;\
nodeptr->data=NULL;\
nodeptr->base=NULL;\
nodeptr->offset=NULL;\
nodeptr->stride=NULL;\
nodeptr->count=NULL;\
nodeptr->block=NULL;\
nodeptr->name[0]='\0';\
nodeptr->label[0]='\0';\
nodeptr->dtype=L3E_NULL;\
nodeptr->flags=L3F_NONE;\
{int __nn;for (__nn=0;__nn<L3C_MAX_DIMS;__nn++){nodeptr->dims[__nn]=-1;}}

#define L3M_CLEARNODE(nodeptr) \
nodeptr->id=-1;\
nodeptr->children=NULL;\
nodeptr->data=NULL;\
nodeptr->base=NULL;\
nodeptr->offset=NULL;\
nodeptr->stride=NULL;\
nodeptr->count=NULL;\
nodeptr->block=NULL;\
nodeptr->name[0]='\0';\
nodeptr->label[0]='\0';\
nodeptr->dtype=L3E_VOID;\
nodeptr->flags=L3F_NONE;\
{int __nn;for (__nn=0;__nn<L3C_MAX_DIMS;__nn++){nodeptr->dims[__nn]=-1;}}

/* ------------------------------------------------------------------------- */
/*
@@ Interface: L3/raw 
*/

/*
@@ Function:  L3_nodeCreate
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       pid:hid_t:Parent node
@@ Arg:       node:L3_Node_t*:Node attributes to use
@@ Arg:       HDFstorage:int:Node hdf5 storage strategy
@@ Return:    A successful creation returns the new node hid_t, -1 if failure
@@ Remarks:   
@@ All node arguments are immediatly used and copied, it is then possible
@@ to release L3_Node_t as the function returns
@@ The storage strategy is either L3_COMPACT_STORAGE or L3_CONTIGUOUS_STORAGE
@@ Normal storage is CONTIGUOUS but COMPACT_STORAGE can be useful to store small data in header
*/
/*#*/hid_t L3_nodeCreate(L3_Cursor_t *ctxt, hid_t pid, L3_Node_t *node, const int HDFstorage);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeUpdate
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       node:L3_Node_t*:Node attributes to use
@@ Arg:       HDFstorage:int:Node hdf5 storage strategy 
@@ Return:    The parameter context is returned
@@ Remarks:   
@@ The target node hid_t is in the L3_Node_t* 
@@ If a value is NULL, or L3F_NULL for flags or L3E_NULL for data type,
@@ then previous node value is unchanged for this attribute
@@ The storage strategy is either L3_COMPACT_STORAGE or L3_CONTIGUOUS_STORAGE
*/
/*#*/hid_t L3_nodeUpdate(L3_Cursor_t *ctxt,L3_Node_t *node, const int HDFstorage);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeUpdatePartial
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       node:L3_Node_t*:Node attributes to use
@@ Return:    The parameter context is returned
@@ Remarks:   
@@ The target node hid_t is in the L3_Node_t* 
@@ If a value is NULL, or L3F_NULL for flags or L3E_NULL for data type,
@@ then previous node value is unchanged for this attribute
*/
/*#*/hid_t L3_nodeUpdatePartial(L3_Cursor_t *ctxt,
				hsize_t *src_offset,
				hsize_t *src_stride,
				hsize_t *src_count,
				hsize_t *src_block,
				hsize_t *dst_offset,
				hsize_t *dst_stride,
				hsize_t *dst_count,
				hsize_t *dst_block,
				L3_Node_t *node);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeRelease
@@ Arg:       node:L3_Node_t**:Pointer to L3 Node to release
@@ Return:    Status
@@ Remarks:   
@@ The target L3_Node_t is released depending on flags 
@@ see 'L3 node release Flags' at the top of this file
@@ The node pointer is set to NULL
*/
/*#*/int L3_nodeRelease(L3_Node_t **nodeptr,unsigned int flags);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeChildrenFree
@@ Arg:       node:L3_Node_t**:Pointer to L3 Node to release
@@ Return:    Status
@@ Remarks:   
@@ Release the children list of the L3 node and close the children hid_t
@@ same as with flags: L3F_R_MEM_CHILDREN|L3F_R_HID_CHILDREN
*/
/*#*/int L3_nodeChildrenFree(L3_Node_t **nodeptr);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeFree
@@ Arg:       node:L3_Node_t**:Pointer to node attributes to use
@@ Return:    status -1 if failure
@@ Remarks:   
@@ The target node should be passed as L3_Node_t** and is set to NULL
@@ after the memory is released. Node data is kept untouched.
@@ The embedded hid_t of the node is released, it is no more valid
*/
/*#*/int L3_nodeFree(L3_Node_t **node);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeFreeNoDecRef
@@ Arg:       node:L3_Node_t**:Pointer to node attributes to use
@@ Return:    status -1 if failure
@@ Remarks:   
@@ The target node should be passed as L3_Node_t** and is set to NULL
@@ after the memory is released. Node data is kept untouched.
@@ The embedded hid_t of the node is NOT released, it is still valid
*/
/*#*/int L3_nodeFreeNoDecRef(L3_Node_t **node);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeFree
@@ Arg:       node:L3_Node_t**:Pointer to node attributes to use
@@ Return:    status -1 if failure
@@ Remarks:   
@@ The target node should be passed as L3_Node_t** and is set to NULL
@@ after the memory is released. Node data is released too.
@@ The embedded hid_t of the node is released, it is no more valid
*/
/*#*/int L3_nodeAndDataFree(L3_Node_t **node);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeLink
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       node:hid_t:Parent node
@@ Arg:       srcname:char*:Name of the child node
@@ Arg:       destfile:char*:File name of the actual target node
@@ Arg:       destname:char*:Node name of the actual target node
@@ Return:    The new node id is returned
@@ Remarks:   
@@ There is no check on the target file nor on the target node name
*/
/*#*/hid_t L3_nodeLink(L3_Cursor_t *ctxt,hid_t node,char *srcname,char *destfile,char *destname);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeMove
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       parent:hid_t:Parent node
@@ Arg:       newparent:hid_t:New parent node
@@ Arg:       name:char*:Name of the node to move
@@ Arg:       newname:char*:New node name 
@@ Return:    The new node id is returned
@@ Remarks:   
@@ The move is performed by HDF5, check HDF5 docs for details.
*/
/*#*/hid_t L3_nodeMove(L3_Cursor_t *ctxt,hid_t parent,hid_t newparent,char *name,char *newname);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeDelete
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       parent:hid_t:Parent node
@@ Arg:       name:char*:Name of the node to delete
@@ Return:    The parameter context is returned
@@ Remarks:   
@@ The delete is performed by HDF5, check HDF5 docs for details.
*/
/*#*/L3_Cursor_t *L3_nodeDelete(L3_Cursor_t *ctxt,hid_t parent,char *name);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeFind
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       parent:hid_t:Parent node
@@ Arg:       path:char*:Path of the node to find
@@ Return:    The hid_t of the node, -1 if failure
@@ Remarks:   
@@ The nodeFind retrieves the hid_t of a node from its path.
@@ Forces the link traversal, even if FOLLOWLINKS is set to false
*/
/*#*/hid_t L3_nodeFind(L3_Cursor_t *ctxt,hid_t parent,char *path);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeRetrieve
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       id:hid_t:Node target id
@@ Arg:       node:L3_Node_t*:Pointer to allocated node to fill in
@@ Return:    A L3_Node_t with target node attributes copies
@@ Remarks:   
@@ The return value is allocated by the user and data should have correct
@@ dimensions. If node id is not found or if the context is bad 
@@ a NULL value is returned.
*/
/*#*/L3_Node_t *L3_nodeRetrieve(L3_Cursor_t *ctxt,hid_t id,L3_Node_t *node);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeRetrievePartial
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       id:hid_t:Node target id
@@ Arg:       src_offset:hsize_t*:start index (n dims) of the region to retrieve
@@ Arg:       src_stride:hsize_t*:shift amount index (n dims) between blocks
@@ Arg:       src_count:hsize_t*:number of blocks to retrieve (n dims)
@@ Arg:       src_block:hsize_t*:size of region to retrieve (n dims)
@@ Arg:       dst_offset:hsize_t*:start index (n dims) of target memory
@@ Arg:       dst_stride:hsize_t*:shift amount index (n dims) between blocks
@@ Arg:       dst_count:hsize_t*:number of blocks of target memory (n dims)
@@ Arg:       dst_block:hsize_t*:size of region of target memory (n dims)
@@ Arg:       node:L3_Node_t*:Pointer to allocated node to fill in
@@ Return:    A L3_Node_t with target node attributes copies
@@ Remarks:   
@@ The return value is allocated by the user and data should have correct
@@ dimensions. If node id is not found or if the context is bad 
@@ a NULL value is returned.
@@ NULL pointer for src_* or dst_* dataspaces means H5S_ALL is used.
*/
/*#*/L3_Node_t *L3_nodeRetrievePartial(L3_Cursor_t *ctxt,hid_t id,hsize_t *src_offset,hsize_t *src_stride,hsize_t *src_count,hsize_t *src_block,hsize_t *dst_offset,hsize_t *dst_stride,hsize_t *dst_count,hsize_t *dst_block,L3_Node_t *node);

L3_Node_t *L3_nodeRetrieveContiguous(L3_Cursor_t *ctxt,hid_t oid,int index, int rank, int count, int interlaced, L3_Node_t *node);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodePrint
@@ Arg:       node:L3_Node_t:Node attributes buffer
@@ Remarks:   
@@ Prints node attributes on standard output for debug/trace purpose
*/
/*#*/void L3_nodePrint(L3_Node_t *node);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_openFile
@@ Arg:       filename:char*:File name to handle
@@ Arg:       mode:int:Open mode
@@ Arg:       flags:long:Set of flags
@@ Return:    A new L3_Cursor_t
@@ Remarks:   
@@ The first function to call with L3 interface. All subsequent L3 calls
@@ would us the returned L3_Cursor_t as argument.
@@ Mode enumerate should be in Open Modes list
@@ Flags are a OR-ed list of values from L3 Flags
*/
/*#*/L3_Cursor_t *L3_openFile(char *filename,int mode,long flags);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_openHID
@@ Arg:       flags:long:Set of flags
@@ Return:    A new L3_Cursor_t
@@ Remarks:   
@@ The first function to call with L3 interface. All subsequent L3 calls
@@ would us the returned L3_Cursor_t as argument.
@@ The root id should be a valid HDF5 device.
*/
/*#*/L3_Cursor_t *L3_openHID(hid_t root,long flags);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_close
@@ Arg:       ctxt:L3_Cursor_t*:Context to close
@@ Return:    Status int (1 is ok)
@@ Remarks:   
@@ All cursor contents are released, cursor itself is released, 
@@ pointer is set to NULL
*/
/*#*/int L3_close(L3_Cursor_t **ctxt_ptr);

int L3_closeShutDown(L3_Cursor_t **ctxt_ptr);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_setFlags
@@ Arg:       ctxt:L3_Cursor_t*:Context to update
@@ Arg:       flags:long:Set of flags
@@ Return:    Modified context
@@ Remarks:   
@@ Sets the current context with the flags, all previous flags
@@ are removed
*/
/*#*/L3_Cursor_t *L3_setFlags(L3_Cursor_t *ctxt,long flags);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_getFlags
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       flags:long*:Pointer to the set of flags
@@ Return:    Context itself
@@ Remarks:   
@@ Gets the current context flags
*/
/*#*/L3_Cursor_t *L3_getFlags(L3_Cursor_t *ctxt,long *flags);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeSet
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       node:L3_Node_t*:Old node tu update or NULL
@@ Arg:       name:char*:Name string
@@ Arg:       label:char*:Label string
@@ Arg:       dims:int*:Dimensions array of data
@@ Arg:       dtype:int:Type of data 
@@ Arg:       data:void*:Pointer to data array
@@ Arg:       flags:int:Set of flags
@@ Return:    New node or updated node
@@ Remarks:   
@@ A NULL value for node arg forces the creation of a new (returned) node.
@@ Dims array of int ends with a -1
@@ Data type should be an L3E_<DataType> enumerate
*/
/*#*/L3_Node_t *L3_nodeSet(L3_Cursor_t *ctxt,L3_Node_t *node,char *name,char *label,int  *dims,int dtype,void *data,int flags);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_nodeGet
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       node:L3_Node_t*:Old node tu update or NULL
@@ Arg:       name:char*:Name string
@@ Arg:       label:char*:Label string
@@ Arg:       dims:int*:Dimensions array of data
@@ Arg:       dtype:int*:Type of data 
@@ Arg:       data:void*:Pointer to data array
@@ Arg:       flags:int*:Set of flags
@@ Return:    New node or updated node
@@ Remarks:   
@@ Reads the node structure and fills the arguments with the values.
*/
/*#*/L3_Node_t *L3_nodeGet(L3_Cursor_t *ctxt,L3_Node_t *node,char *name,char *label,int *dims,int *dtype,void *data,int *flags);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_isSameNode
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       id1:hid_t:First HDF5 id
@@ Arg:       id2:hid_t:Second HDF5 id
@@ Return:    Status int (1 is ok)
@@ Remarks:   
@@ Checks if different ids refer to the same HDF5 object.
*/
/*#*/int L3_isSameNode(L3_Cursor_t *ctxt,hid_t id1,hid_t id2);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_isLocalNode
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       id:hid_t:HDF5 id
@@ Return:    Status int (1 is ok, node is local)
@@ Remarks:   
@@ CHecks if id is in the same file than root node, so that it is not a link
*/
/*#*/int L3_isLocalNode(L3_Cursor_t *ctxt,hid_t id);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_incRef
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       id:hid_t:HDF5 id to keep 
@@ Return:    arg id
@@ Remarks:   
@@ Increments the hid_t so that you can call l3_freeNode and still use id
*/
/*#*/hid_t L3_incRef(L3_Cursor_t *ctxt,hid_t id);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_decRef
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       id:hid_t:HDF5 id to release
@@ Return:    void
@@ Remarks:   
@@ Decrements the hid_t once you are finished with it
*/
/*#*/void L3_decRef(L3_Cursor_t *ctxt,hid_t id);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_isLinkNode
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       id1:hid_t:HDF5 id to check
@@ Arg:       file:char*:File name
@@ Arg:       name:char*:Node name
@@ Return:    Status int (1 is ok)
@@ Remarks:   
@@ Checks if a node is a link. If yes, the file name and the node name
@@ of the target of the link are copied into the user-allocated strings.
*/
/*#*/int L3_isLinkNode(L3_Cursor_t *ctxt,hid_t id,char *file,char *name);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_node2Path
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       id:hid_t:HDF5 target node
@@ Arg:       path:char*:Char buff to fill with resulting path
@@ Return:    Path char*
@@ Remarks:   
@@ Finds a path from the file root to the target node.
@@ If path is NULL then the function allocates a char buffer, it is to the
@@ user application to release the memory of this buffer. If the path is not
@@ NULL, the user has to provide enough memory for the function.
*/
/*#*/char *L3_node2Path(L3_Cursor_t *ctxt,hid_t id,char *path);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_path2Node
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       path:char*:Path to search for
@@ Return:    hid_t found node
@@ Remarks:   
@@ Finds a node id in the file from a path.
*/
/*#*/hid_t L3_path2Node(L3_Cursor_t *ctxt,char *path);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_initDims
@@ Arg:       dims:int*:Array of dimensions
@@ Arg:       d1:int:Dimensions one...
@@ Arg:       dn:int:... up to last dimension
@@ Return:    Arrays of dimensions
@@ Remarks:   
@@ Fills the array with the dimensions and fills the rest with -1.
@@ If the arg dims is NULL, the returned array is allocated and has to be
@@ released by the user
@@ If it is not NULL, the user has to insure that the array size is
@@ at least number-of-dimensions + 1 (last value should be -1)
*/
/*#*/int *L3_initDims(int *dims,int d1,...);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_initHyperslab
@@ Arg:       dims:hsize-t*:array of hyperslab info
@@ Arg:       d1:int:info number 1
@@ Arg:       dn:int:... up to last info
@@ Return:    Hyperslab array of hsize_t
@@ Remarks:   
@@ Fills the with the hyperslab info and add a -1 as sentinel
@@ Info are of arbitrary size, but the calls have to be performed for
@@ the attributes: offset, stride, count, block (refer to doc)
*/
/*#*/hsize_t *L3_initHyperslab(hsize_t *hs,int d1,...);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_initData
@@ Arg:       dims:int*:Array of dimensions
@@ Arg:       data:void*:Pointer to data
@@ Arg:       dtype:int:Type of data
@@ Arg:       arglist:data type:variable arguments
@@ Return:    Pointer to data
@@ Remarks:   
@@ Fills the array with variable arguments
*/
/*#*/void *L3_initData(int *dims,void *data,int dtype,...);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_fillData
@@ Arg:       dims:int*:Array of dimensions
@@ Arg:       data:void*:Pointer to data
@@ Arg:       dtype:int:Type of data
@@ Arg:       arglist:data type:variable arguments
@@ Return:    Pointer to data
@@ Remarks:   
@@ Fills the array with variable arguments
*/
/*#*/void *L3_fillData(int  *dims,void *data,int dtype,...);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_typeAsStr
@@ Arg:       dtype:int:Data type
@@ Return:    String type
@@ Remarks:   
@@ Returns the string type corresponding to the int enumerate
@@ Returns NULL if the integer is not found.
*/
/*#*/char *L3_typeAsStr(int dtype);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_typeAsEnum
@@ Arg:       dtype:char*:Data type
@@ Return:    Integer enumerate data type
@@ Remarks:   
@@ Returns the integer enumerate type corresponding to string type.
@@ Returns -1 if the string type is not found.
*/
/*#*/int L3_typeAsEnum(char *dtype);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  L3_config
@@ Arg:       p:int:Parameter
@@ Return:    Parameter
@@ Remarks:   
@@ Unused
*/
/*#*/int    L3_config(int p);

/* ------------------------------------------------------------------------- */
/*
@@ Interface: CHL/raw 
*/

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_printError
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Return:    None
@@ Remarks:   
@@ Prints the message for the last error recorded in the context.
*/
/*#*/void CHL_printError(L3_Cursor_t *ctxt);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_setError
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       err:int:Error code
@@ Arg:       arglist:...:Variable argument list 
@@ Return:    None
@@ Remarks:   
@@ Sets the last context error to error given as argument.
@@ The variable argument list should fit with the corresponding error message
@@ arguments are used to fill % formated string
@@ See the error tables headers: each entry has an integer as key, then an
@@ enumerate to set the level of severity and then the string for the message itself.
*/
/*#*/void CHL_setError(L3_Cursor_t *ctxt,int err,...);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_getError
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Return:    Error code int
@@ Remarks:   
@@ Gets the last context error
*/
/*#*/int CHL_getError(L3_Cursor_t *ctxt);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_setMessage
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       msg:char*:Message to use
@@ Return:    Error code int
@@ Remarks:   
@@ Sets the error message
*/
/*#*/int CHL_setMessage(L3_Cursor_t* ctxt,char *msg);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_getMessage
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Return:    Error message string
@@ Remarks:   
@@ Gets the error message
*/
/*#*/char *CHL_getMessage(L3_Cursor_t* ctxt);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_addLinkSearchPath
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       path:char*:Path string to add
@@ Return:    Index of path in the list int
@@ Remarks:   
@@ Adds the given path in the link search path
*/
/*#*/int CHL_addLinkSearchPath(L3_Cursor_t* ctxt,char *path);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_getFileInSearchPath
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       file:char*:Target file
@@ Return:    Index of path in the list int
@@ Remarks:   
@@ Searches the file in the link path link list, return the index of first
@@ path where the file was found.
@@ Returns -1 if the file was not found.
*/
/*#*/int CHL_getFileInSearchPath(L3_Cursor_t *ctxt,char *file);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_freeLinkSearchPath
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Return:    int:status
@@ Remarks:   
@@ Removes all path entries
*/
/*#*/int CHL_freeLinkSearchPath(L3_Cursor_t* ctxt);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_delLinkSearchPath
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       path:char*:Path to remove
@@ Return:    Index of path in the list int
@@ Remarks:   
@@ Removes the path from path list if found.
@@ The index entry is released but may be re-used by next add.
@@ Returns -1 if the path was not found.
*/
/*#*/int CHL_delLinkSearchPath(L3_Cursor_t* ctxt,char *path);

/* ------------------------------------------------------------------------- */
/*
@@ Function:  CHL_getLinkSearchPath
@@ Arg:       ctxt:L3_Cursor_t*:Context to use
@@ Arg:       index:int:Position in the list
@@ Return:    Path string
@@ Remarks:   
@@ Returns the path in the index entry of link path search list
@@ Returns NULL if the index entry is empty
*/
/*#*/char *CHL_getLinkSearchPath(L3_Cursor_t* ctxt,int index);

/* ------------------------------------------------------------------------- */
/* trace/debug macros */
#ifndef CHLONE_PRINTF_TRACE

#define L3M_DBG(crs,msg) 

#define L3M_TRACE(crs,msg) 

#else

#define L3M_DBG(crs,msg) \
{if (L3M_HASFLAG(crs,L3F_DEBUG))\
{printf("# L3 : +");printf msg;fflush(stdout);}}

#define L3M_TRACE(crs,msg) \
{if (L3M_HASFLAG(crs,L3F_TRACE))\
{printf("# L3 :");printf msg ;fflush(stdout);}}

#endif

/* #define L3_H5F_STRONG_CLOSE */

#endif
/* --- last line ----------------------------------------------------------- */
