/* ======================================================================
 * CHLone - CGNS HDF5 LIBRARY only node edition
 * See license.txt in the root directory of this source release
 * ======================================================================

   ------------------------------------------------------------------------
   l3 has a set of node-level functions to handle an ADF-like
   node with HDF5. You can use it through its own API (see CHLone_l3.h)
   or use the a2 API which defines a more user oriented set of functions.
   ------------------------------------------------------------------------

  LOT OF CODE IN THERE HAS BEEN TAKEN FROM cgnslib/ADF/ADFH.c
  cgnlib/ADF/ADFH.c Credits:
    Bruce Wedan (NASA,ANSYS...),
    Greg Power et al., Arnold Air Force Base,
    Marc Poinot, ONERA
    Other people... sorry, add your name!
  API based on the low level node storage defined by ADF people
  from McDonnell Douglas & Boeing.

   ------------------------------------------------------------------------
   Handle the nodes and get/set their attribute/data without
   taking into account the SIDS specification

   A CGNS/L3/HDF5 node should have:

   L3     : node                 HDF5   : group
   name   : string               name   : string (group attribute)
   label  : string               label  : string (group attribute)
   type   : enum                 type   : string (group attribute)
   flags  : int                  flags  : int    (group attribute)

   special cases (be careful, these are using *datasets*):

   root node                     format :     string (group dataset)
                 hdf5version: string (group dataset)

   link node                     file   : string (group dataset)
                 path   : string (group dataset)
                 link   : actual HDF5 link object
*/
#include "l3P.h"

/* ------------------------------------------------------------------------- */
static char *L3T_MT_s = L3T_MT;
static char *L3T_LK_s = L3T_LK;
static char *L3T_B1_s = L3T_B1;
static char *L3T_C1_s = L3T_C1;
static char *L3T_I4_s = L3T_I4;
static char *L3T_I8_s = L3T_I8;
static char *L3T_U4_s = L3T_U4;
static char *L3T_U8_s = L3T_U8;
static char *L3T_R4_s = L3T_R4;
static char *L3T_R8_s = L3T_R8;
static char *L3T_X4_s = L3T_X4;
static char *L3T_X8_s = L3T_X8;

int __node_count = 0;

/* ------------------------------------------------------------------------- */
/* WATCH OUT ! PASSING A FUNCTION AS arg will DUPLICATE THE CALL             */
/* ------------------------------------------------------------------------- */
#define L3_N_setName(node,aname) \
if (node==NULL){return NULL;} \
if ((aname!=NULL)&&(aname[0]!='\0')) \
 {strncpy(node->name,aname,L3C_MAX_ATTRIB_SIZE); \
  node->name[L3C_MAX_ATTRIB_SIZE]='\0';} 

#define L3_N_setLabel(node,alabel) \
if (node==NULL){return NULL;} \
 if ((alabel!=NULL)&&(alabel[0]!='\0'))\
 {strncpy(node->label,alabel,L3C_MAX_ATTRIB_SIZE);\
  node->label[L3C_MAX_ATTRIB_SIZE]='\0';}

#define L3_N_setFlags(node,aflags) \
if (node==NULL){return NULL;} \
else {node->flags=aflags;}

#define L3_N_setDtype(node,adtype) \
if (node==NULL){return NULL;}\
if (adtype!=L3E_NULL){node->dtype=adtype; }

#define L3_N_setDims(node,adims) \
if (node==NULL){return NULL;} \
if (adims!=NULL) \
{int __nn; \
for (__nn=0;__nn<L3C_MAX_DIMS;__nn++){node->dims[__nn]=adims[__nn];}}

#define L3_N_setData(node,adata) \
if (node==NULL){return NULL;} \
if (adata!=NULL){node->data=adata;}

#define L3_N_setBase(node,adata) \
if (node==NULL){return NULL;} \
if (adata!=NULL){node->base=adata;}

#define L3_N_setParent(node,pid) \
if (node==NULL){return NULL;} \
else {node->parentid=pid;}

#define L3_N_getName(node,aname) \
if (node==NULL){return NULL;} \
if ((aname!=NULL)&&(node->name!=NULL)){strcpy(aname,node->name);} 

#define L3_N_getLabel(node,alabel) \
if (node==NULL){return NULL;} \
if ((alabel!=NULL)&&(node->label!=NULL)){strcpy(alabel,node->label);}

#define L3_N_getDtype(node,adtype) \
if (node==NULL){return NULL;}\
if (adtype!=NULL){*adtype=node->dtype;}

#define L3_N_getDims(node,adims) \
if (node==NULL){return NULL;} \
if ((adims!=NULL)&&(node->adims!=NULL)) \
{int __nn; \
for (__nn=0;__nn<L3C_MAX_DIMS;__nn++){adims[__nn]=node->dims[__nn];}}

#define L3_N_getData(node,data) \
if (node==NULL){return NULL;} \
if ((data!=NULL)&&(node->data!=NULL)){data=node->data;}

#define L3_N_getParent(node,pid) \
if (node==NULL){return NULL;} \
{pid=node->parentid;}

#define L3_N_getFlags(node,aflags) \
if (node==NULL){return NULL;} \
else {aflags=node->flags;}

/* DEBUG TRACE ONLY */
/* #define L3_TRACK_OBJECT_IDS 1 */
#ifdef L3_TRACK_OBJECT_IDS
#define L3_T_ID(tag,id)  \
printf("# L3 :HDF5 OBJ OPEN [%d] {%s} \n",id,tag);fflush(stdout);
#else
#define L3_T_ID(tag,id) 
#endif

#if 0
#define L3_H5_GCLOSE( msg, hid )\
  {\
  H5Gclose(hid);\
  printf( "H5GCLOSE [%d] (L3) @@@",hid);\
  printf( msg );\
  }
#else
#define L3_H5_GCLOSE( msg, hid ) H5Gclose(hid);
#endif
/* To cope with negative value comparison warning */
#define L3_H5_SENTINEL ((hsize_t) -1)

/* ------------------------------------------------------------------------- */
/* To handle HDF5 API changes */
#if H5_VERSION_GE(1,10,3) && !defined(H5_USE_18_API) && !defined(H5_USE_16_API)
#define L3_HDF5_HAVE_110_API 1
#else
#define L3_HDF5_HAVE_110_API 0
#endif

#if H5_VERSION_GE(1,12,0) && !defined(H5_USE_110_API) && !defined(H5_USE_18_API) && !defined(H5_USE_16_API)
#define L3_HDF5_HAVE_112_API 1
#else
#define L3_HDF5_HAVE_112_API 0
#endif

/* ------------------------------------------------------------------------- */
void objlist_status(char *tag)
{
  int n, sname;
  char oname[256];
  hid_t idlist[1024];
  H5O_info_t objinfo;

  n = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL);
  printf("# L3 :HDF5 OBJ COUNT [%d] {%s}\n", n, tag); fflush(stdout);
  n = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_GROUP);
  printf("# L3 :HDF5 GROUP     [%d] {%s}\n", n, tag); fflush(stdout);
  n = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_DATASET);
  printf("# L3 :HDF5 DATASET   [%d] {%s}\n", n, tag); fflush(stdout);
  n = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_DATATYPE);
  printf("# L3 :HDF5 DATATYPE  [%d] {%s}\n", n, tag); fflush(stdout);
  n = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ATTR);
  printf("# L3 :HDF5 ATTR      [%d] {%s}\n", n, tag); fflush(stdout);
  for (n = 0; n < 1024; n++)
  {
    idlist[n] = -1;
  }
  H5Fget_obj_ids(H5F_OBJ_ALL, H5F_OBJ_ALL, 1024, idlist);
  for (n = 0; n < 1024; n++)
  {
    if (H5Iis_valid(idlist[n]))
    {
#if H5_VERSION_GE(1, 12, 0)
      H5Oget_info3(idlist[n], &objinfo, H5O_INFO_BASIC);
#else
      H5Oget_info(idlist[n], &objinfo);
#endif
      memset(oname, '\0', 256);
      sname = H5Iget_name(idlist[n], oname, 0);
      sname = H5Iget_name(idlist[n], oname, sname + 1);
      printf("# L3 :HDF5 ID %ld ALIVE (%s:%d) {%s}\n", \
        idlist[n], oname, objinfo.rc, tag);
    }
  }
}

/* ------------------------------------------------------------------------- */
static herr_t find_name(hid_t id, const char *nm, const H5A_info_t* i, void *snm)
{
  if (!strcmp(nm, (char *)snm)) return 1;
  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t gfind_by_name(hid_t id, const char *nm, const H5L_info_t* linfo, void *snm)
{
  /*  printf("GFIND [%s][%s]\n",nm,snm);fflush(stdout); */
  if (!strcmp(nm, (char *)snm)) return 1;
  return 0;
}
/* ------------------------------------------------------------------------- */
int findExternalLink(hid_t g_id, const char * name,
  const H5L_info_t * info, void * op_data)
{
  if ((info->type == H5L_TYPE_EXTERNAL) && !strcmp(name, (char *)op_data))
  {
    return 1;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */

//#if L3_HDF5_HAVE_112_API
//#define has_child(ID,NAME) H5Literate2(ID, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, gfind_by_name, (void *)NAME)
//#define has_data(ID)       H5Literate2(ID, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, gfind_by_name, (void *)L3S_DATA)
//#else
//#define has_child(ID,NAME) H5Literate(ID, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, gfind_by_name, (void *)NAME)
//#define has_data(ID)       H5Literate(ID, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, gfind_by_name, (void *)L3S_DATA)
//#endif

#define has_child(ID,NAME) H5Lexists(ID, NAME, H5P_DEFAULT) 
#define has_data(ID) H5Lexists(ID, L3S_DATA, H5P_DEFAULT)

/* ------------------------------------------------------------------------- */
static herr_t HDF_Print_Error(unsigned n, H5E_error2_t *desc, void *ctxt)
{
  char localbuff[256]; /* bet ! */
  L3_Cursor_t* c;

  if (ctxt == NULL) { return 0; }

  c = (L3_Cursor_t*)ctxt;
  sprintf(localbuff, "# ### %s: %s\n", desc->func_name, desc->desc);
  CHL_setMessage(c, localbuff);

  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t HDF_Walk_Error(hid_t estack, void *ctxt)
{
  return H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD,
    (H5E_walk2_t)HDF_Print_Error, ctxt);
}
/* ------------------------------------------------------------------------- */
#define HDF_Check_Node(id)  (id==-1)?0:1

/* ------------------------------------------------------------------------- */
static herr_t HDF_link_check(const char *parent_file_name,
  const char *parent_group_name,
  const char *child_file_name,
  const char *child_object_name,
  unsigned *acc_flags, hid_t fapl_id,
  void *op_data)
{
  printf("Take me out to the ball game\n");
  return 0;
}

/* ------------------------------------------------------------------------- */
CHL_INLINE hid_t ADF_to_HDF_datatype(const char *tp)
{
  if (!strcmp(tp, L3T_I4))   return H5Tcopy(H5T_NATIVE_INT32);
  if (!strcmp(tp, L3T_R8)) {
    hid_t tid = H5Tcopy(H5T_NATIVE_DOUBLE);
    H5Tset_precision(tid, 64);
    return tid;

  }
  if (!strcmp(tp, L3T_C1))   return H5Tcopy(H5T_NATIVE_CHAR);
  if (!strcmp(tp, L3T_I8))   return H5Tcopy(H5T_NATIVE_INT64);
  if (!strcmp(tp, L3T_R4)) {
    hid_t tid = H5Tcopy(H5T_NATIVE_FLOAT);
    H5Tset_precision(tid, 32);
    return tid;
  }
  if (!strcmp(tp, L3T_B1))   return H5Tcopy(H5T_NATIVE_UCHAR);
  if (!strcmp(tp, L3T_U4))   return H5Tcopy(H5T_NATIVE_UINT32);
  if (!strcmp(tp, L3T_U8))   return H5Tcopy(H5T_NATIVE_UINT64);
  if (!strcmp(tp, L3T_X4)) {
    hid_t tid = H5Tcreate(H5T_COMPOUND, 8);
    hid_t subid = H5Tcopy(H5T_NATIVE_FLOAT);
    H5Tset_precision(subid, 32);
    H5Tinsert(tid, CMPLX_REAL_NAME, 0, subid);
    H5Tinsert(tid, CMPLX_IMAG_NAME, 4, subid);
    H5Tclose(subid);
    return tid;
  }
  if (!strcmp(tp, L3T_X8)) {
    hid_t tid = H5Tcreate(H5T_COMPOUND, 16);
    hid_t subid = H5Tcopy(H5T_NATIVE_DOUBLE);
    H5Tset_precision(subid, 64);
    H5Tinsert(tid, CMPLX_REAL_NAME, 0, subid);
    H5Tinsert(tid, CMPLX_IMAG_NAME, 8, subid);
    H5Tclose(subid);
    return tid;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
CHL_INLINE int HDF_Get_Attribute_As_Integer(hid_t nodeid,
  const char *name,
  int *value)
{
  hid_t aid;
  herr_t status;

  aid = H5Aopen_by_name(nodeid, ".", name, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0)
  {
    return 0; /* bad return, cannot decide error/value ? */
  }
  status = H5Aread(aid, H5T_NATIVE_INT, value);
  H5Aclose(aid);
  if (status < 0)
  {
    return 0; /* bad return, cannot decide error/value ? */
  }
  return *value;
}

/* ------------------------------------------------------------------------- */
CHL_INLINE char *HDF_Get_Attribute_As_String(L3_Cursor_t *ctxt,
  int stype,
  hid_t nodeid,
  const char *name,
  char *value)
{
  hid_t aid, tid;

  value[0] = '\0';
  aid = H5Aopen_by_name(nodeid, ".", name, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0)
  {
    return value;
  }
  tid = H5Aget_type(aid);
  if (tid < 0)
  {
    H5Aclose(aid);
    return value;
  }
  H5Aread(aid, tid, value);
  H5Tclose(tid);
  H5Aclose(aid);

  return value;
}

/* ------------------------------------------------------------------------- */
CHL_INLINE char *HDF_Get_Dtype(L3_Cursor_t *ctxt,
  hid_t nodeid,
  char *value)
{
  hid_t aid, tid;

  value[0] = '\0';
  //aid = H5Aopen_name(nodeid, L3S_DTYPE);
  aid = H5Aopen_by_name(nodeid, ".", L3S_DTYPE, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0)
  {
    return value;
  }
  tid = H5Aget_type(aid);
  if (tid < 0)
  {
    H5Aclose(aid);
    return value;
  }
  H5Aread(aid, tid, value);
  H5Tclose(tid);
  H5Aclose(aid);

  return value;
}
/* ------------------------------------------------------------------------- */
CHL_INLINE char *HDF_Get_Name(L3_Cursor_t *ctxt,
  hid_t nodeid,
  char *value)
{
  hid_t aid, tid;
  value[0] = '\0';

  aid = H5Aopen_by_name(nodeid, ".", L3S_NAME, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0)
  {
    return value;
  }
  tid = H5Aget_type(aid);
  if (tid < 0)
  {
    H5Aclose(aid);
    return value;
  }
  H5Aread(aid, tid, value);
  H5Tclose(tid);
  H5Aclose(aid);

  value[L3C_MAX_ATTRIB_SIZE] = '\0';/* force size to 32 chars */

  return value;
}
/* ------------------------------------------------------------------------- */
CHL_INLINE char *HDF_Get_Label(L3_Cursor_t *ctxt,
  hid_t nodeid,
  char *value)
{
  hid_t aid, tid;

  value[0] = '\0';
  aid = H5Aopen_by_name(nodeid, ".", L3S_LABEL, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0)
  {
    return value;
  }
  tid = H5Aget_type(aid);
  if (tid < 0)
  {
    H5Aclose(aid);
    return value;
  }
  H5Aread(aid, tid, value);
  H5Tclose(tid);
  H5Aclose(aid);

  return value;
}
/* ------------------------------------------------------------------------- */
CHL_INLINE char *HDF_Get_Attribute_As_Data(hid_t nodeid,
  const char *name,
  char *value)
{
  hid_t did;

  value[0] = '\0';
  did = H5Dopen2(nodeid, name, H5P_DEFAULT);
  if (did > 0)
  {
    H5Dread(did, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, value);
    H5Dclose(did);
  }
  return value;
}
/* ----------------------------------------------------------------- */
CHL_INLINE int is_link(L3_Cursor_t *ctxt, hid_t nodeid)
{
  char ntype[L3C_MAX_DTYPE + 1] = {0};
  
  HDF_Get_Dtype(ctxt, nodeid, ntype);
  return ((ntype[0] != L3T_LK[0]) || (ntype[1] != L3T_LK[1])) ? 0 : 1;
}
/* ----------------------------------------------------------------- */
static hid_t get_link_actual_id(L3_Cursor_t *ctxt, hid_t id)
{
  hid_t lid;
  herr_t herr;
  const char  *file;
  const char  *path;
  char *dfile;
  H5L_info_t lk;
  char querybuff[512];
  int idx;

  lid = -1;

  if (H5Lis_registered(H5L_TYPE_EXTERNAL) != 1)
  {
    return -1;
  }
  herr = H5Lget_info(id, L3S_LINK, &lk, ctxt->l_proplist);
  if (herr < 0)
  {
    return -1;
  }
  /* Soft link                -> link to our current file */
  /* Hard link (User defined) -> link to an external file */
  if (H5L_TYPE_ERROR != lk.type)
  {
    if (H5Lget_val(id, L3S_LINK, querybuff, sizeof(querybuff), H5P_DEFAULT) < 0)
    {
      return -1;
    }
    if (H5L_TYPE_EXTERNAL == lk.type)
    {
      if (H5Lunpack_elink_val(querybuff, lk.u.val_size, NULL, &file, &path) < 0)
      {
        return -1;
      }
      /* open the actual link >> IN THE LINK GROUP << */
      /* only check the file, not the node path inside the file */
      idx = get_file_in_search_path(ctxt, (char*)file);
      if (idx == -1)
      {
        return -1;
      }
      dfile = get_link_search_path(ctxt, idx);
      if (ctxt->currentpath != NULL)
      {
        free(ctxt->currentpath);
      }
      ctxt->currentpath = (char*)malloc(strlen(dfile) + 1);
      H5Pset_elink_prefix(ctxt->l_proplist, ctxt->currentpath);
      if ((lid = H5Gopen2(id, L3S_LINK, H5P_DEFAULT)) < 0)
      {
        return -1;
      }
      L3_T_ID("GL1", lid);
    }
    else
    {
      if ((lid = H5Gopen2(id, L3S_LINK, H5P_DEFAULT)) < 0)
      {
        return -1;
      }
      L3_T_ID("GL3", lid);
    }
  }
  else
  {
    if ((lid = H5Gopen2(id, L3S_LINK, H5P_DEFAULT)) < 0)
    {
      return -1;
    }
    L3_T_ID("GL2", lid);
  }
  return lid;
}
/* ----------------------------------------------------------------- */
static int get_link_data(L3_Cursor_t *ctxt, hid_t id,
  char *rfile, char *rpath)
{
  hsize_t first = 0;
  herr_t herr;
  H5L_info_t lk;
  char  querybuff[512];
  const char *file;
  const char *path;

  if (H5Lis_registered(H5L_TYPE_EXTERNAL) != 1)
  {
    return 0;
  }
  if (!has_child(id, L3S_LINK))
  {
    return 0;
  }
  herr = H5Lget_info(id, L3S_LINK, &lk, ctxt->l_proplist);
  if (herr < 0)
  {
    return 0;
  }
  /* Soft link                -> link to our current file */
  /* Hard link (User defined) -> link to an external file */
  if (H5L_TYPE_ERROR != lk.type)
  {
    if (H5Lget_val(id, L3S_LINK, querybuff, sizeof(querybuff), H5P_DEFAULT) < 0)
    {
      return 0;
    }
    if (H5L_TYPE_EXTERNAL == lk.type)
    {
      if (H5Lunpack_elink_val(querybuff, lk.u.val_size, NULL, &file, &path) < 0)
      {
        return 0;
      }
      strcpy(rfile, file);
      strcpy(rpath, path);
    }
    else
    {
      rfile[0] = '\0';
      strcpy(rpath, querybuff);
    }
    return 1;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t delete_children(hid_t id, const char *name, const H5L_info_t* linfo, void *data)
{
  /* do not change link id with actual here, stop deletion at link node */
  if (name && (name[0] == ' ')) /* leaf node */
  {
    H5Ldelete(id, name, H5P_DEFAULT);
  }
  else
  {
    /* delete children loop */
#if L3_HDF5_HAVE_112_API
    H5Literate_by_name2(id, name, H5_INDEX_CRT_ORDER, H5_ITER_INC, NULL, delete_children, data, H5P_DEFAULT);
#else
    H5Literate_by_name(id, name, H5_INDEX_CRT_ORDER, H5_ITER_INC, NULL, delete_children, data, H5P_DEFAULT);
#endif
    H5Ldelete(id, name, H5P_DEFAULT);
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t print_children(hid_t id, const char *name, const H5L_info_t *info, void *count)
{
  (*((int *)count))++;
  printf("CHILD [%d] [%s]\n", (*((int *)count)), name); fflush(stdout);
  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t count_children(hid_t id, const char *name, const H5L_info_t* linfo, void *count)
{
  if (name && (name[0] != ' '))
  {
    (*((int *)count))++;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t feed_children_ids_list(hid_t id, const char *name,
  const H5L_info_t *linfo, void *idlist)
{
  hid_t cid;
  int n;

  /* skip names starting with a <space> */
  if (name && (name[0] == ' '))
  {
    return 0;
  }
  cid = H5Gopen(id, name, H5P_DEFAULT);
  /* L3_T_ID("FCH",cid); */
  n = 0;
  while (((hid_t*)idlist)[n] + 1) /* != -1 */
  {
    n++;
  }
  ((hid_t*)idlist)[n] = cid;
  return 0;
}
/* ------------------------------------------------------------------------- */
/* WATCHOUT: function does modify input arg */
char *backToParent(char *path)
{
  int n;

  n = strlen(path);
  if (n == 0) return path;
  if (path[n] == '/') n--;
  while (n > 0)
  {
    if (path[n] == '/')
    {
      path[n] = '\0';
      return path;
    }
    n--;
  }
  return path;
}
/* ------------------------------------------------------------------------- */
hid_t *HDF_Get_Children(hid_t nodeid, int asciiorder)
{
  hid_t *idlist, gpl;
  int    nchildren, n;
  unsigned order = 0;

  nchildren = 0;
  /* order not used here */
#if L3_HDF5_HAVE_112_API
  H5Literate2(nodeid, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, count_children, (void *)&nchildren);
#else
  H5Literate(nodeid, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, count_children, (void *)&nchildren);
#endif
  if (!nchildren)
  {
    return NULL;
  }
  idlist = (hid_t*)malloc(sizeof(hid_t)*(nchildren + 1));

  /* use last -1 as sentinel */
  for (n = 0; n <= nchildren; n++) { idlist[n] = (hid_t)-1; }
  /* order used here - if creation fails use name */
  gpl = H5Gget_create_plist(nodeid);
  if (H5Iis_valid(gpl))
  {
    H5Pget_link_creation_order(gpl, &order);
  }
  if (!asciiorder && ((order & H5_INDEX_CRT_ORDER) == H5_INDEX_CRT_ORDER))
  {
#if L3_HDF5_HAVE_112_API
    H5Literate2(nodeid, H5_INDEX_CRT_ORDER, H5_ITER_INC,
        NULL, feed_children_ids_list, (void*)idlist);
#else
    H5Literate(nodeid, H5_INDEX_CRT_ORDER, H5_ITER_INC,
      NULL, feed_children_ids_list, (void*)idlist);
#endif
  }
  if ((nchildren > 0) && (idlist[0] == -1))
  {
#if L3_HDF5_HAVE_112_API
    H5Literate2(nodeid, H5_INDEX_NAME, H5_ITER_INC,
          NULL, feed_children_ids_list, (void*)idlist);
#else
    H5Literate(nodeid, H5_INDEX_NAME, H5_ITER_INC,
      NULL, feed_children_ids_list, (void *)idlist);
#endif
  }
  if (H5Iis_valid(gpl))
  {
    H5Pclose(gpl);
  }
  return idlist;
}
/* ------------------------------------------------------------------------- */
void *HDF_Read_Array(L3_Cursor_t *ctxt, hid_t nid, hid_t did, hid_t yid,
  void *data, hsize_t *int_dim_vals)
{
  herr_t stat;
  int n;
  hsize_t tsize;
  char  name[L3C_MAX_ATTRIB_SIZE + 1];
  char  label[L3C_MAX_ATTRIB_SIZE + 1];
  char  dims[64];
  char  pad = '(';

  L3M_CHECK_CTXT_OR_DIE(ctxt, 0);

  if (L3M_HASFLAG(ctxt, L3F_DEBUG))
  {
    HDF_Get_Name(ctxt, nid, name);
    HDF_Get_Label(ctxt, nid, label);
    L3M_DBG(ctxt, ("HDF_Read_Array [%s][%s]\n", name, label));
    dims[0] = '\0';
  }

  tsize = 1;
  for (n = 0; n < L3C_MAX_DIMS; n++)
  {
    if (int_dim_vals[n] == L3_H5_SENTINEL) { break; }
    tsize *= int_dim_vals[n];
    if (L3M_HASFLAG(ctxt, L3F_DEBUG))
    {
      sprintf(dims, "%s%c%lld", dims, pad, int_dim_vals[n]);
      pad = 'x';
    }
  }
  if (!L3M_HASFLAG(ctxt, L3F_NOALLOCATE))
  {
    data = (void*)malloc(H5Tget_size(yid)*tsize);
    L3M_DBG(ctxt, ("HDF_Read_Array ALLOCATE %p from %ld size %s)x%ld=%lld %s @@@\n",
      data, nid, dims, H5Tget_size(yid), H5Tget_size(yid)*tsize,
      (char*)name));
  }
  else
  {
    L3M_DBG(ctxt, ("HDF_Read_Array NO ALLOCATE %p from %ld @@@\n", data, nid));
  }
  stat = H5Dread(did, yid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  L3M_DBG(ctxt, ("HDF_Read_Array status [%d]\n", stat));
  return data;
}
/* ------------------------------------------------------------------------- */
int HDF_Write_Array(L3_Cursor_t *ctxt, hid_t nid, hid_t did, hid_t yid, hid_t sid,
                    void *data)
{
  herr_t stat;
  char  name[L3C_MAX_ATTRIB_SIZE + 1];
  char  label[L3C_MAX_ATTRIB_SIZE + 1];

  L3M_CHECK_CTXT_OR_DIE(ctxt, 0);

  HDF_Get_Name(ctxt, nid, name);
  HDF_Get_Label(ctxt, nid, label);
  L3M_DBG(ctxt, ("HDF_Write_Array [%s][%s]\n", name, label));

  if (sid == -1)
  {
    sid = H5S_ALL;
  }
  stat = H5Dwrite(did, yid, H5S_ALL, sid, H5P_DEFAULT, (char*)data);

  L3M_DBG(ctxt, ("HDF_Write_Array status [%d]\n", stat));

  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Get_DataDimensions(L3_Cursor_t *ctxt, hid_t nid, int *dims)
{
  int n, ndims;
  hsize_t int_dim_vals[L3C_MAX_DIMS];
  hid_t did, sid;

  L3M_CLEARDIMS(dims);
  L3M_CLEARDIMS(int_dim_vals);

  did = H5Dopen2(nid, L3S_DATA, H5P_DEFAULT);
  sid = H5Dget_space(did);
  ndims = H5Sget_simple_extent_ndims(sid);
  H5Sget_simple_extent_dims(sid, int_dim_vals, NULL);

  for (n = 0; n < ndims; n++) { dims[n] = (int)(int_dim_vals[n]); }

  H5Sclose(sid);
  H5Dclose(did);

  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Get_DataDimensionsPartial(L3_Cursor_t *ctxt, hid_t nid, int *dims,
  hsize_t *dst_offset,
  hsize_t *dst_stride,
  hsize_t *dst_count,
  hsize_t *dst_block)
{
  int n, ndims;

  L3M_CLEARDIMS(dims);

  ndims = 0;
  for (n = 0; dst_count[n] != L3_H5_SENTINEL; n++)
  {
    ndims += 1;
    dims[n] = dst_count[n] * dst_block[n] + dst_offset[n];
    if (dst_block[n] > 1)
    {
      dims[n] += dst_stride[n] * dst_block[n];
    }
  }
  dims[ndims] = -1;

  return 1;
}
/* ------------------------------------------------------------------------- */
void *HDF_Get_DataArrayPartial(L3_Cursor_t *ctxt, hid_t nid, int *dst_dims,
  hsize_t *src_offset,
  hsize_t *src_stride,
  hsize_t *src_count,
  hsize_t *src_block,
  hsize_t *dst_offset,
  hsize_t *dst_stride,
  hsize_t *dst_count,
  hsize_t *dst_block,
  int tsize,
  int rank,
  void **data,
  void **base)
{
  int n, dst_ndims, src_ndims, memshift = 0, eltsize;
  hsize_t src_dim_vals[L3C_MAX_DIMS], dst_dim_vals[L3C_MAX_DIMS];
  hsize_t dst_start[L3C_MAX_DIMS], dst_end[L3C_MAX_DIMS];
  hid_t did, sid, yid, tid, mid;
  hssize_t dst_size, blk_size;
  char  buff[L3C_MAX_ATTRIB_SIZE + 1];
  char  name[L3C_MAX_ATTRIB_SIZE + 1];
  herr_t stat;

  did = H5Dopen2(nid, L3S_DATA, H5P_DEFAULT);
  sid = H5Dget_space(did);
  src_ndims = H5Sget_simple_extent_ndims(sid);
  H5Sget_simple_extent_dims(sid, src_dim_vals, NULL);
  src_dim_vals[src_ndims] = L3_H5_SENTINEL;

  stat = H5Sselect_hyperslab(sid, H5S_SELECT_SET,
    src_offset, src_stride, src_count, src_block);
  dst_size = 1;
  dst_ndims = 0;
  for (n = 0; dst_dims[n] != -1; n++)
  {
    dst_size *= dst_dims[n];
    dst_ndims += 1;
    dst_dim_vals[n] = dst_dims[n];
  }
  dst_dim_vals[dst_ndims] = L3_H5_SENTINEL;
  blk_size = dst_size;
  if (tsize != -1)
  {
    /* obviously to change later on... */
    dst_size = tsize;
    dst_ndims = 1;
    dst_dim_vals[0] = tsize;
    dst_dim_vals[1] = L3_H5_SENTINEL;
    memshift = 1;
  }
  if (L3M_HASFLAG(ctxt, L3F_DEBUG))
  {
    for (n = 0; src_dim_vals[n] != L3_H5_SENTINEL; n++)
    {
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial SRC  [%d]=%lld\n", n, src_dim_vals[n]));
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial OFF S[%d]=%lld\n", n, src_offset[n]));
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial STR S[%d]=%lld\n", n, src_stride[n]));
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial CNT S[%d]=%lld\n", n, src_count[n]));
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial BLK S[%d]=%lld\n", n, src_block[n]));
    }
    for (n = 0; dst_dim_vals[n] != L3_H5_SENTINEL; n++)
    {
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial DST  [%d]=%lld\n", n, dst_dim_vals[n]));
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial OFF D[%d]=%lld\n", n, dst_offset[n]));
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial STR D[%d]=%lld\n", n, dst_stride[n]));
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial CNT D[%d]=%lld\n", n, dst_count[n]));
      L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial BLK D[%d]=%lld\n", n, dst_block[n]));
    }
    L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial S NDIMS %d : D NDIMS %d : D SIZE %lld\n"
      , src_ndims, dst_ndims, dst_size));
  }
  mid = H5Screate_simple(dst_ndims, dst_dim_vals, NULL);
  stat = H5Sselect_hyperslab(mid, H5S_SELECT_SET,
    dst_offset, dst_stride, dst_count, dst_block);
  if (!H5Sselect_valid(mid))
  {
    HDF_Get_Name(ctxt, nid, name);
    H5Sclose(mid);
    H5Sclose(sid);
    H5Dclose(did);
    L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial [%s] bad hyperslab\n", name));
    CHL_setError(ctxt, 3019);
    return NULL;
  }
  H5Sget_select_bounds(mid, dst_start, dst_end);
  dst_dim_vals[dst_ndims] = L3_H5_SENTINEL;
  dst_start[dst_ndims] = L3_H5_SENTINEL;
  dst_end[dst_ndims] = L3_H5_SENTINEL;
  tid = ADF_to_HDF_datatype(HDF_Get_Dtype(ctxt, nid, buff));
  yid = H5Tget_native_type(tid, H5T_DIR_ASCEND);
  eltsize = H5Tget_size(yid);
  if (*base == NULL)
  {
    *base = (void*)malloc(eltsize*dst_size);
  }
  if (!memshift)
  {
    *data = *base;
    L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial DATA==BASE\n"));
  }
  else
  {
    *data = (char*)(*base) + (blk_size*rank*eltsize);
    L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial SIZE %lldx%dx%d = %lld\n", blk_size, rank, eltsize, blk_size*rank*eltsize));
  }
  stat = H5Dread(did, yid, mid, sid, H5P_DEFAULT, *base);/* hyperslab does shift */
  L3M_DBG(ctxt, ("HDF_Get_DataArrayPartial BASE=%p DATA=%p @@@\n", *base, *data));

  H5Tclose(yid);
  H5Tclose(tid);
  H5Sclose(mid);
  H5Sclose(sid);
  H5Dclose(did);

  return *data;
}
/* ------------------------------------------------------------------------- */
void *HDF_Get_DataArray(L3_Cursor_t *ctxt, hid_t nid, int *dims, void *data)
{
  hid_t tid, did, yid;
  char  buff[L3C_MAX_ATTRIB_SIZE + 1];
  char  name[L3C_MAX_ATTRIB_SIZE + 1];
  hsize_t int_dim_vals[L3C_MAX_DIMS];
  int n;

  L3M_ECLEAR(ctxt);
  L3M_CLEARDIMS(int_dim_vals);

  if (!has_data(nid))
  {
    CHL_setError(ctxt, 3020);
  }
  for (n = 0; n < L3C_MAX_DIMS; n++)
  {
    if (dims[n] == -1) { break; }
    int_dim_vals[n] = (hsize_t)(dims[n]);
  }
  did = H5Dopen2(nid, L3S_DATA, H5P_DEFAULT);
  tid = ADF_to_HDF_datatype(HDF_Get_Dtype(ctxt, nid, buff));
  if (!tid)
  {
    HDF_Get_Name(ctxt, nid, name);
    H5Tclose(did);
    L3M_DBG(ctxt, ("HDF_Get_DataArray [%s] bad tid\n", name));
    CHL_setError(ctxt, 3019);
    return NULL;
  }
  yid = H5Tget_native_type(tid, H5T_DIR_ASCEND);

  data = HDF_Read_Array(ctxt, nid, did, yid, data, int_dim_vals);
  L3M_DBG(ctxt, ("HDF_Get_DataArray from %ld/%ld/%ld/%ld @@@\n",
    yid, tid, did, nid));

  H5Tclose(yid);
  H5Tclose(tid);
  H5Dclose(did);

  return data;
}
/* ------------------------------------------------------------------------- */
int HDF_Add_DataArray(L3_Cursor_t *ctxt, hid_t nid, int *dims, void *data, const int HDFstorage)
{
  hid_t tid, sid, did, yid, pid;
  char buff[L3C_MAX_ATTRIB_SIZE + 1];
  char name[L3C_MAX_ATTRIB_SIZE + 1];
  hsize_t int_dim_vals[L3C_MAX_DIMS], chunkdims[L3C_MAX_DIMS];
  int n, rank, totalsize, skipchunk;

  L3M_ECLEAR(ctxt);
  L3M_CLEARDIMS(int_dim_vals);

  HDF_Get_Name(ctxt, nid, name);
  L3M_DBG(ctxt, ("HDF_Add_DataArray [%s]\n", name));

  rank = 0;
  totalsize = 1;
  skipchunk = 0;
  for (n = 0; n < L3C_MAX_DIMS; n++)
  {
    if (!rank) { chunkdims[rank] = (hsize_t)1; }
    else { chunkdims[rank] = (hsize_t)(dims[n]); }
    if (dims[n] == -1) { break; }
    int_dim_vals[n] = (hsize_t)(dims[n]);
    totalsize += dims[n];
    rank++;
  }
  if (rank == 1)
  {
    if (totalsize > 1024) { chunkdims[0] = (hsize_t)1024; }
    else { skipchunk = 1; }
  }
  tid = ADF_to_HDF_datatype(HDF_Get_Dtype(ctxt, nid, buff));
  if (!tid)
  {
    if (!strcmp(buff, L3T_MT))
    {
      return 1;
    }
    else
    {
      HDF_Get_Name(ctxt, nid, name);
      CHL_setError(ctxt, 3017);
      return 0;
    }
  }
  if (tid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_DataArray [%s] bad tid\n", name));
    return 0;
  }
  yid = H5Tget_native_type(tid, H5T_DIR_ASCEND);
  if (yid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_DataArray [%s] bad yid\n", name));
    H5Tclose(tid);
    return 0;
  }
  sid = H5Screate_simple(n, int_dim_vals, NULL);
  if (sid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_DataArray [%s] bad sid %d dims\n", name, n));
    H5Tclose(yid);
    H5Tclose(tid);
    return 0;
  }
  pid = H5Pcreate(H5P_DATASET_CREATE);
  if (pid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_DataArray [%s] bad pid\n", name));
    H5Sclose(sid);
    H5Tclose(yid);
    H5Tclose(tid);
    return 0;
  }
  if (!skipchunk)
  {
    H5Pset_chunk(pid, rank, chunkdims);
    if (L3M_HASFLAG(ctxt, L3F_COMPRESS))
    {
      H5Pset_deflate(pid, 6);
    }
    did = H5Dcreate2(nid, L3S_DATA, tid, sid, H5P_DEFAULT, pid, H5P_DEFAULT);
  }
  else
  {
    hssize_t dset_size = H5Sget_select_npoints(sid);
    size_t dtype_size = H5Tget_size(tid);

    /* Compact storage has a dataset size limit of 64 KiB */
    if (HDFstorage == L3_COMPACT_STORE &&
        dset_size*(hssize_t)dtype_size  < (hssize_t)CGNS_64KB)
    {
      H5Pset_layout(pid, H5D_COMPACT);
    }
    else
    {
      H5Pset_layout(pid, H5D_CONTIGUOUS);
      H5Pset_alloc_time(pid, H5D_ALLOC_TIME_EARLY);
      H5Pset_fill_time(pid, H5D_FILL_TIME_NEVER);
    }
    did = H5Dcreate2(nid, L3S_DATA, tid, sid, H5P_DEFAULT, pid, H5P_DEFAULT);
  }
  if (did < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_DataArray [%s] bad did\n", name));
    H5Pclose(pid);
    H5Sclose(sid);
    H5Tclose(yid);
    H5Tclose(tid);
    return 0;
  }

  HDF_Write_Array(ctxt, nid, did, yid, sid, data);

  H5Pclose(pid);
  H5Dclose(did);
  H5Sclose(sid);
  H5Tclose(yid);
  H5Tclose(tid);

  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Set_DataArray(L3_Cursor_t *ctxt, hid_t nid, int *dims, void *data, const int HDFstorage)
{
  hid_t tid, did, yid, sid;
  char  buff[L3C_MAX_ATTRIB_SIZE + 1];
  char name[L3C_MAX_ATTRIB_SIZE + 1];
  hsize_t int_dim_vals[L3C_MAX_DIMS];
  int dims_old[L3C_MAX_DIMS];
  int n, samedims;

  L3M_ECLEAR(ctxt);
  L3M_CLEARDIMS(int_dim_vals);

  if (!has_data(nid))
  {
    CHL_setError(ctxt, 3021);
  }
  HDF_Get_DataDimensions(ctxt, nid, dims_old);
  samedims = 1;
  for (n = 0; n < L3C_MAX_DIMS; n++)
  {
    if (dims[n] != dims_old[n]) { samedims = 0; }
    if (dims[n] == -1) { break; }
    int_dim_vals[n] = (hsize_t)(dims[n]);
  }
  if (samedims)
  {
    did = H5Dopen2(nid, L3S_DATA, H5P_DEFAULT);
    tid = ADF_to_HDF_datatype(HDF_Get_Dtype(ctxt, nid, buff));
    if (!tid)
    {
      if (!strcmp(buff, L3T_MT))
      {
        H5Ldelete(nid, L3S_DATA, H5P_DEFAULT);
      }
      else
      {
        HDF_Get_Name(ctxt, nid, name);
        H5Dclose(did);
        CHL_setError(ctxt, 3018, name);
        return 0;
      }
    }
    else
    {
      yid = H5Tget_native_type(tid, H5T_DIR_ASCEND);
      sid = -1;

      HDF_Write_Array(ctxt, nid, did, yid, sid, data);

      H5Tclose(yid);
      H5Tclose(tid);
    }
    H5Dclose(did);
  }
  else
  {
    L3M_DBG(ctxt, ("HDF_Set_DataArray change dims\n"));
    H5Ldelete(nid, L3S_DATA, H5P_DEFAULT);
    HDF_Add_DataArray(ctxt, nid, dims, data, HDFstorage);
    H5Fflush(nid, H5F_SCOPE_LOCAL);
  }

  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Add_Attribute_As_Integer(L3_Cursor_t *ctxt,
  hid_t nodeid, const char *name, int value)
{
  hid_t sid, aid;
  hsize_t dim;
  herr_t status;

  L3M_DBG(ctxt, ("HDF_Add_Attribute_As_Integer [%s][%d]\n", name, value));
  dim = 1;
  sid = H5Screate_simple(1, &dim, NULL);
  if (sid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_Integer [%s] bad sid\n", name));
    return 0;
  }

  aid = H5Acreate2(nodeid, name, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_Integer [%s] create attribute failed\n", name));
    H5Sclose(sid);
    return 0;
  }

  status = H5Awrite(aid, H5T_NATIVE_INT, &value);

  H5Aclose(aid);
  H5Sclose(sid);

  if (status < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_Integer [%s] write attribute failed\n", name));
    return 0;
  }
  return 1;

}

/* ------------------------------------------------------------------------- */
int HDF_Add_Attribute_As_String(L3_Cursor_t *ctxt,
  hid_t nodeid, const char *name,
  const char *value)
{
  hid_t sid, tid, aid;
  herr_t status;
  hsize_t dim;
  char buff[L3C_MAX_ATTRIB_SIZE + 1];

  L3M_DBG(ctxt, ("HDF_Add_Attribute_As_String [%s][%s]\n", name, value));
  if (!strcmp(name, L3S_DTYPE))
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_String [%s] datatype string\n", name));
    dim = (hsize_t)(L3C_MAX_DTYPE + 1);
  }
  else
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_String [%s] attribute string\n", name));
    dim = (hsize_t)(L3C_MAX_ATTRIB_SIZE + 1);
  }
  sid = H5Screate(H5S_SCALAR);
  if (sid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_String [%s] bad sid\n", name));
    return 0;
  }
  tid = H5Tcopy(H5T_C_S1);
  if (tid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_String [%s] bad tid\n", name));
    H5Sclose(sid);
    return 0;
  }
  if (H5Tset_size(tid, dim) < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_String [%s] bad data size\n", name));
    H5Tclose(tid);
    H5Sclose(sid);
    return 0;
  }
  aid = H5Acreate2(nodeid, name, tid, sid, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_String [%s] create failed\n", name));
    H5Tclose(tid);
    H5Sclose(sid);
    return 0;
  }
  memset(buff, 0, dim);
  strncpy(buff, value, dim - 1);
  buff[L3C_MAX_ATTRIB_SIZE] = '\0';

  status = H5Awrite(aid, tid, buff);

  H5Aclose(aid);
  H5Tclose(tid);
  H5Sclose(sid);

  if (status < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_String [%s] write failed\n", name));
    return 0;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
static int HDF_Set_Attribute_As_Integer(L3_Cursor_t *ctxt,
                                        hid_t nodeid, const char *name,
                                        int value)
{
  hid_t aid;
  herr_t status;

  L3M_DBG(ctxt, ("HDF_Set_Attribute_As_Integer: [%s]=[%d]\n", name, value));
  aid = H5Aopen_by_name(nodeid, ".", name, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0)
  {
    return 0;
  }
  status = H5Awrite(aid, H5T_NATIVE_INT, &value);
  H5Aclose(aid);
  if (status < 0)
  {
    return 0;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
static int HDF_Set_Attribute_As_String(L3_Cursor_t *ctxt,
  hid_t nodeid, const char *name,
  const char *value)
{
  hid_t aid, tid;
  herr_t status;
  char buff[L3C_MAX_ATTRIB_SIZE + 1];

  L3M_DBG(ctxt, ("HDF_Set_Attribute_As_String: [%s]=[%s]\n", name, value));
  aid = H5Aopen_by_name(nodeid, ".", name, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0)
  {
    return 0;
  }
  tid = H5Aget_type(aid);
  if (tid < 0)
  {
    H5Aclose(aid);
    return 0;
  }
  memset(buff, 0, L3C_MAX_ATTRIB_SIZE + 1);
  strcpy(buff, value);
  status = H5Awrite(aid, tid, buff);
  H5Tclose(tid);
  H5Aclose(aid);
  if (status < 0)
  {
    return 0;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
static int HDF_Add_Attribute_As_Data(L3_Cursor_t *ctxt,
                                     hid_t id, const char *name,
                                     const char *value, int size)
{
  hid_t sid, did;
  hsize_t dim;
  herr_t status;
  hid_t dcpl_id=H5P_DEFAULT;

  L3M_DBG(ctxt, ("HDF_Add_Attribute_As_Data [%s][%s]\n", name, value));
  dim = (hsize_t)(size + 1);
  sid = H5Screate_simple(1, &dim, NULL);
  if (sid < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_Data [%s] bad sid\n", name));
    return 0;
  }

  /* compact storage */  
  dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  if (size+1 < CGNS_64KB)
  {
    H5Pset_layout(dcpl_id, H5D_COMPACT);
  }
  else
  {
    H5Pset_layout(dcpl_id, H5D_CONTIGUOUS);
    H5Pset_alloc_time(dcpl_id, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(dcpl_id, H5D_FILL_TIME_NEVER);
  }

  did = H5Dcreate2(id, name, H5T_NATIVE_CHAR, sid,
    H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
  if (did < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_Data [%s] create data failed\n", name));
    H5Sclose(sid);
    H5Pclose(dcpl_id);
    return 0;
  }
  status = H5Dwrite(did, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, value);
  H5Dclose(did);
  H5Sclose(sid);
  H5Pclose(dcpl_id);

  if (status < 0)
  {
    L3M_DBG(ctxt, ("HDF_Add_Attribute_As_Data [%s] write data failed\n", name));
    return 0;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Create_Root(L3_Cursor_t *ctxt)
{
  unsigned maj, min, rel;
  hid_t gid;
  char  svalue[L3C_MAX_VALUE_SIZE + 1], tvalue[L3C_MAX_VALUE_SIZE + 1];

  L3M_CHECK_CTXT_OR_DIE(ctxt, 0);
  L3M_MXLOCK(ctxt);

  gid = H5Gopen2(ctxt->file_id, "/", H5P_DEFAULT);
  L3_T_ID("ROOT", gid);
  if (gid < 0)
  {
    CHL_setError(ctxt, 3022);
  }

  /* Root node has specific attributes (not used in this package) */
  if (!HDF_Add_Attribute_As_String(ctxt, gid, L3S_NAME, L3S_ROOTNODENAME))
  {
    CHL_setError(ctxt, 3023);
  }
  if (!HDF_Add_Attribute_As_String(ctxt, gid, L3S_LABEL, L3S_ROOTNODETYPE))
  {
    CHL_setError(ctxt, 3024);
  }
  if (!HDF_Add_Attribute_As_String(ctxt, gid, L3S_DTYPE, L3T_MT))
  {
    CHL_setError(ctxt, 3025);
  }
  strcpy(tvalue, "NATIVE");
  sprintf(svalue, "%s", tvalue);
  if (!HDF_Add_Attribute_As_Data(ctxt, gid, L3S_FORMAT, svalue, strlen(svalue)))
  {
    CHL_setError(ctxt, 3026);
  }
  H5get_libversion(&maj, &min, &rel);
  memset(svalue, 0, L3C_MAX_VERSION + 1);
  sprintf(svalue, "HDF5 Version %d.%d.%d", maj, min, rel);
  if (!HDF_Add_Attribute_As_Data(ctxt, gid, L3S_VERSION, svalue, L3C_MAX_VERSION))
  {
    CHL_setError(ctxt, 3027);
  }
  if (H5Iis_valid(gid))
  {
    L3_H5_GCLOSE("CREATE ROOT\n", gid);
  }
  L3M_MXUNLOCK(ctxt);

  return 1;
}
/* ------------------------------------------------------------------------- */
hid_t L3_nodeCreate(L3_Cursor_t *ctxt, hid_t pid, L3_Node_t *node, const int HDFstorage)
{
  hid_t nid = -1;
  int n = 0, s = 0;

  L3M_CHECK_CTXT_OR_DIE(ctxt, -1);
  L3M_MXLOCK(ctxt);
  L3M_ECLEAR(ctxt);
  L3M_ECHECKID(ctxt, pid, -1);
  L3M_TRACE(ctxt, ("L3_nodeCreate\n"));
  L3M_ECHECKL3NODE(ctxt, node, -1);

  s = strlen(node->name);
  if ((s == 0) || (s > L3C_MAX_ATTRIB_SIZE))
  {
    CHL_setError(ctxt, 3036, node->name);
    L3M_MXUNLOCK(ctxt);
    return nid;
  }

  if (has_child(pid, node->name))
  {
    nid = H5Gopen2(pid, node->name, H5P_DEFAULT);
    node->id = nid;
    L3M_MXUNLOCK(ctxt);
    L3_nodeUpdate(ctxt, node, HDFstorage);
    return nid;
  }
  else
  {
    nid = H5Gcreate2(pid, node->name, H5P_DEFAULT, ctxt->g_proplist, H5P_DEFAULT);
  }
  if (!H5Iis_valid(nid))
  {
    CHL_setError(ctxt, 3030, node->name);
    L3M_MXUNLOCK(ctxt);
    return nid;
  }
  node->id = nid;
  if (!HDF_Add_Attribute_As_String(ctxt, nid, L3S_NAME, node->name))
  {
    CHL_setError(ctxt, 3031, node->name);
  }
  if (!HDF_Add_Attribute_As_String(ctxt, nid, L3S_LABEL, node->label))
  {
    CHL_setError(ctxt, 3032);
  }
  if (!HDF_Add_Attribute_As_Integer(ctxt, nid, L3S_FLAGS, node->flags))
  {
    CHL_setError(ctxt, 3036);
  }
  if (!HDF_Add_Attribute_As_String(ctxt, nid, L3S_DTYPE, L3_typeAsStr(node->dtype)))
  {
    CHL_setError(ctxt, 3033);
  }
  /* return nid at this point, allow function embedding */
  if (node->data != NULL)
  {
    if (!HDF_Add_DataArray(ctxt, nid, node->dims, node->data, HDFstorage))
    {
      CHL_setError(ctxt, 3034);
    }
  }
  else
  {
    if (node->dtype != L3E_MT &&
        node->dtype != L3E_VOID)
    {
      CHL_setError(ctxt, 3035);
    }
  }
  L3M_MXUNLOCK(ctxt);
  return nid;
}
/* ------------------------------------------------------------------------- */
hid_t L3_nodeUpdate(L3_Cursor_t *ctxt, L3_Node_t *node, const int HDFstorage)
{
  hid_t nid, pid;
  char oldname[L3C_MAX_ATTRIB_SIZE + 1], oldlabel[L3C_MAX_ATTRIB_SIZE + 1];
  char *ppath;

  L3M_CHECK_CTXT_OR_DIE(ctxt, -1);
  L3M_MXLOCK(ctxt);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt, ("L3_nodeUpdate\n"));
  L3M_ECHECKL3NODE(ctxt, node, -1);

  nid = node->id;

  if (is_link(ctxt, nid))
  {
    /* if the target node is a link, allow to change it to a plain node */
    /*
    CHL_setError(ctxt,3050);
    L3M_MXUNLOCK( ctxt );
    return nid;
    */
  }

  if (!HDF_Check_Node(nid))
  {
    CHL_setError(ctxt, 3051);
    L3M_MXUNLOCK(ctxt);
    return nid;
  }

  HDF_Get_Name(ctxt, nid, oldname);
  L3M_TRACE(ctxt, ("L3_nodeUpdate on [%s]\n", oldname));

  if ((node->name != NULL) && strcmp(node->name, oldname))
  {
    L3M_TRACE(ctxt, ("L3_nodeUpdate update name to [%s]\n", node->name));
    if (!HDF_Set_Attribute_As_String(ctxt, nid, L3S_NAME, node->name))
    {
      CHL_setError(ctxt, 3052, oldname);
    }
    ppath = (char*)malloc(sizeof(char) * 512);
    H5Iget_name(nid, ppath, 256);
    ppath = backToParent(ppath);
    pid = H5Gopen(ctxt->root_id, ppath, H5P_DEFAULT);
    free(ppath);
    //L3M_TRACE(ctxt, ("L3_nodeUpdate H5Lmove [%d:%s] ... @@@\n", pid, oldname));
    //L3M_TRACE(ctxt, ("L3_nodeUpdate H5Lmove to [%d:%s]  @@@\n", pid, node->name));
    H5Lmove(pid, oldname, pid, node->name, H5P_DEFAULT, H5P_DEFAULT);
  }

  HDF_Get_Attribute_As_String(ctxt, L3T_STR_LABEL, nid, L3S_LABEL, oldlabel);
  if ((node->label != NULL) && strcmp(node->label, oldlabel))
  {
    L3M_TRACE(ctxt, ("L3_nodeUpdate update label to [%s]\n", node->label));
    if (!HDF_Set_Attribute_As_String(ctxt, nid, L3S_LABEL, node->label))
    {
      CHL_setError(ctxt, 3053, oldname);
    }
  }

  if (node->flags != L3F_NULL)
  {
    L3M_TRACE(ctxt, ("L3_nodeUpdate update flags\n"));
    if (!HDF_Set_Attribute_As_Integer(ctxt, nid, L3S_FLAGS, node->flags))
    {
      if (!HDF_Add_Attribute_As_Integer(ctxt, nid, L3S_FLAGS, node->flags))
      {
        CHL_setError(ctxt, 3056, oldname);
      }
    }
  }

  if (node->dtype != L3E_NULL)
  {
    L3M_TRACE(ctxt, ("L3_nodeUpdate update datatype\n"));
    if (!HDF_Set_Attribute_As_String(ctxt, nid, L3S_DTYPE,
      L3_typeAsStr(node->dtype)))
    {
      CHL_setError(ctxt, 3054, oldname);
    }
  }

  if (!L3M_HASFLAG(ctxt, L3F_WITHDATA))
  {
    ; /*printf("DATA FLAG FALSE\n"); */
  }
  else
  {
    ; /*printf("DATA FLAG TRUE\n"); */
  }
  if (node->data == NULL)
  {
    ; /*printf("NODE DATA NULL\n"); */
  }
  else
  {
    ; /*printf("NODE DATA NOT NULL\n"); */
  }

  if (L3M_HASFLAG(ctxt, L3F_WITHDATA) && (node->data != NULL))
  {
    L3M_TRACE(ctxt, ("L3_nodeUpdate update data\n"));
    if (!HDF_Set_DataArray(ctxt, nid, node->dims, node->data, HDFstorage))
    {
      CHL_setError(ctxt, 3055, oldname);
    }
  }
  L3M_MXUNLOCK(ctxt);
  return nid;
}
/* ------------------------------------------------------------------------- */
hid_t L3_nodeUpdatePartial(L3_Cursor_t *ctxt,
  hsize_t *src_offset,
  hsize_t *src_stride,
  hsize_t *src_count,
  hsize_t *src_block,
  hsize_t *dst_offset,
  hsize_t *dst_stride,
  hsize_t *dst_count,
  hsize_t *dst_block,
  L3_Node_t *node)
{
  return -1;
}
/* ------------------------------------------------------------------------- */
hid_t L3_nodeLink(L3_Cursor_t *ctxt, hid_t node,
                  char *srcname, char *destfile, char *destname)
{
  hid_t nid;

  L3M_CHECK_CTXT_OR_DIE(ctxt, -1);
  L3M_MXLOCK(ctxt);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt, ("L3_nodeLink [%s]->[%s][%s]\n", srcname, destfile, destname));
  L3M_ECHECKID(ctxt, node, -1);

  if (is_link(ctxt, node))
  {
    CHL_setError(ctxt, 3060);
  }
  if (L3M_HASFLAG(ctxt, L3F_LINKOVERWRITE) && has_child(node, srcname))
  {
#if L3_HDF5_HAVE_112_API
    H5Literate_by_name2(node, srcname, H5_INDEX_CRT_ORDER, H5_ITER_INC, NULL, delete_children, NULL, H5P_DEFAULT);
#else
    H5Literate_by_name(node, srcname, H5_INDEX_CRT_ORDER, H5_ITER_INC, NULL, delete_children, NULL, H5P_DEFAULT);
#endif
    H5Ldelete(node, srcname, H5P_DEFAULT);
  }
  nid = H5Gcreate2(node, srcname, H5P_DEFAULT, ctxt->g_proplist, H5P_DEFAULT);
  if (nid < 0)
  {
    CHL_setError(ctxt, 3061, srcname);
  }
  if (!HDF_Add_Attribute_As_String(ctxt, nid, L3S_NAME, srcname))
  {
    CHL_setError(ctxt, 3062, srcname);
  }
  if (!HDF_Add_Attribute_As_String(ctxt, nid, L3S_DTYPE, L3T_LK))
  {
    CHL_setError(ctxt, 3063, srcname);
  }
  if (!HDF_Add_Attribute_As_String(ctxt, nid, L3S_LABEL, ""))
  {
    CHL_setError(ctxt, 3064, srcname);
  }
  /* Theoretically useless but L3_nodeRetrieve needs it */
  if (!HDF_Add_Attribute_As_Integer(ctxt, nid, L3S_FLAGS, L3F_NONE))
  {
      CHL_setError(ctxt, 3036);
  }
  HDF_Add_Attribute_As_Data(ctxt, nid, L3S_PATH, destname, strlen(destname));

  if (strcmp(destfile, ""))
  {
    H5Lcreate_external(destfile, destname, nid,
                       L3S_LINK, H5P_DEFAULT, ctxt->l_proplist);
    HDF_Add_Attribute_As_Data(ctxt, nid, L3S_FILE, destfile, strlen(destfile));
  }
  else
  {
    H5Lcreate_soft(destname, nid, L3S_LINK, H5P_DEFAULT, ctxt->l_proplist);
  }
  L3M_MXUNLOCK(ctxt);
  return nid;
}
/* ------------------------------------------------------------------------- */
hid_t L3_nodeMove(L3_Cursor_t *ctxt, hid_t pid, hid_t nid,
                  char *oldname, char *newname)
{
  hid_t tid;
  char opn[L3C_MAX_ATTRIB_SIZE + 1];
  char npn[L3C_MAX_ATTRIB_SIZE + 1];

  L3M_CHECK_CTXT_OR_DIE(ctxt, -1);
  L3M_MXLOCK(ctxt);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt, ("L3_nodeMove\n"));
  L3M_ECHECKID(ctxt, pid, -1);
  L3M_ECHECKID(ctxt, nid, -1);

  if (nid == -1) { nid = pid; }
  if (newname == NULL) { newname = oldname; }
  HDF_Get_Name(ctxt, pid, opn);
  HDF_Get_Name(ctxt, nid, npn);

  if (is_link(ctxt, pid) || is_link(ctxt, nid))
  {
    CHL_setError(ctxt, 3040);
  }
  L3M_TRACE(ctxt, ("L3_nodeMove [%s][%s]->[%s][%s]\n", opn, oldname, npn, newname));
  H5Lmove(pid, oldname, nid, newname, H5P_DEFAULT, H5P_DEFAULT);
  tid = H5Gopen2(nid, newname, H5P_DEFAULT);
  L3_T_ID("MV", tid);
  if (!HDF_Set_Attribute_As_String(ctxt, tid, L3S_NAME, newname))
  {
    CHL_setError(ctxt, 3041);
  }
  L3M_MXUNLOCK(ctxt);
  return tid;
}
/* ------------------------------------------------------------------------- */
L3_Cursor_t *L3_nodeDelete(L3_Cursor_t *ctxt, hid_t pid, char *nodename)
{
  L3M_CHECK_CTXT_OR_DIE(ctxt, NULL);
  L3M_MXLOCK(ctxt);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt, ("L3_nodeDelete\n"));
  L3M_ECHECKID(ctxt, pid, ctxt);

  /* do not change link id with actual here, stop deletion at link node */
  if (!HDF_Check_Node(pid))
  {
    CHL_setError(ctxt, 3070);
  }

  if (has_child(pid, nodename))
  {
    /* do not change link id with actual here, stop deletion at link node */
#if L3_HDF5_HAVE_112_API
    H5Literate_by_name2(pid, nodename, H5_INDEX_CRT_ORDER, H5_ITER_INC, NULL, delete_children, NULL, H5P_DEFAULT);
#else
    H5Literate_by_name(pid, nodename, H5_INDEX_CRT_ORDER, H5_ITER_INC, NULL, delete_children, NULL, H5P_DEFAULT);
#endif
    H5Ldelete(pid, nodename, H5P_DEFAULT);
  }
  L3M_MXUNLOCK(ctxt);
  return ctxt;
}
/* ------------------------------------------------------------------------- */
hid_t L3_nodeFind(L3_Cursor_t *ctxt, hid_t parent, char *path)
{
  hid_t rid = -1, cid, pid;
  int   pathlen;
  int   p, s = 0, r = 0, tk = 0;
  char  ppath[L3C_MAX_PATH], pcurrent[L3C_MAX_ATTRIB_SIZE + 1];

  L3M_CHECK_CTXT_OR_DIE(ctxt, -1);
  L3M_MXLOCK(ctxt);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt, ("L3_nodeFind\n"));
  L3M_ECHECKID(ctxt, parent, -1);

  if (path == NULL)
  {
    L3M_MXUNLOCK(ctxt);
    return -1;
  }
  pathlen = strlen(path);
  if (pathlen > (L3C_MAX_PATH - 1))
  {
    CHL_setError(ctxt, 3080, (int)pathlen);
  }
  if (pathlen == 0)
  {
    L3M_MXUNLOCK(ctxt);
    return parent;
  }
  if ((path[0] == '/') && (pathlen == 1))
  {
    L3M_MXUNLOCK(ctxt);
    return parent;
  }
  if (!HDF_Check_Node(parent))
  {
    CHL_setError(ctxt, 3081);
  }

  sprintf(ppath, "/%s/", path);
  cid = parent;
  for (p = 0; (ppath[p] != '\0') && (p < L3C_MAX_PATH) && (r < L3C_MAX_ATTRIB_SIZE + 1); p++)
  {
    if (ppath[p] == '/')
    {
      if (!s && r)
      {
        tk++;
        pcurrent[r] = '\0';
        if (!strcmp(pcurrent, "."))
        {
          ;
        }
        else if (!strcmp(pcurrent, "..") || has_child(cid, pcurrent))
        {
          pid = cid;
          cid = H5Gopen(cid, pcurrent, H5P_DEFAULT);
          if (!H5Iis_valid(cid))
          {
            L3M_MXUNLOCK(ctxt);
            return -1;
          }
          if (H5Iis_valid(pid) && (pid != parent))
          {
            L3_H5_GCLOSE("NODE FIND 1\n", pid);
          }
        }
        else
        {
          L3M_MXUNLOCK(ctxt);
          return -1;
        }
        s = 1;
      }
      r = 0;
    }
    else
    {
      pcurrent[r++] = ppath[p];
      s = 0;
    }
  }
  if (tk == 1)
  {
    while (path[0] == '/') { path++; }
  }
  if (H5Iis_valid(cid) && (cid != parent))
  {
    L3_H5_GCLOSE("NODE FIND 2\n", cid);
  }
  rid = H5Gopen2(parent, path, H5P_DEFAULT);
  L3_T_ID("FIND", rid);

  L3M_MXUNLOCK(ctxt);
  return rid;
}
/* ------------------------------------------------------------------------- */
L3_Node_t *L3_nodeRetrieve(L3_Cursor_t *ctxt, hid_t oid, L3_Node_t *node)
{
  hid_t nid = -1;
  char  buff[L3C_MAX_ATTRIB_SIZE + 1];
  int   dims[L3C_MAX_DIMS];
  int   ibuff, dt, islk, flg;
  void *data;
  char  name[L3C_MAX_ATTRIB_SIZE + 1];
  char  label[L3C_MAX_ATTRIB_SIZE + 1];

  L3M_CHECK_CTXT_OR_DIE(ctxt, NULL);
  L3M_MXLOCK(ctxt);
  L3M_ECLEAR(ctxt);
  //L3M_TRACE(ctxt, ("L3_nodeRetrieve [%d] @@@\n", oid));
  L3M_CLEARDIMS(dims);

  if (node == NULL)
  {
    CHL_setError(ctxt, 3092);
    L3M_MXUNLOCK(ctxt);
    return NULL;
  }
  if (node->children != NULL) { free(node->children); }
  node->children = NULL;

  if (!HDF_Check_Node(oid))
  {
    CHL_setError(ctxt, 3090);
    L3M_MXUNLOCK(ctxt);
    return NULL;
  }
  nid = oid;
  islk = is_link(ctxt, oid);
  if (islk && L3M_HASFLAG(ctxt, L3F_FOLLOWLINKS))
  {
    nid = get_link_actual_id(ctxt, oid);
    if (!HDF_Check_Node(nid) && L3M_HASFLAG(ctxt, L3F_FAILSONLINK))
    {
      CHL_setError(ctxt, 3091);
      L3M_MXUNLOCK(ctxt);
      return NULL;
    }
    /* prevent hdf5 1.1x error */
    if (!HDF_Check_Node(nid)) {
      nid = oid;
    }
    islk = 0;
  }
  else
  {
    islk = 0;
  }
  HDF_Get_Name(ctxt, nid, name);
  HDF_Get_Label(ctxt, nid, label);
  dt = L3_typeAsEnum(HDF_Get_Dtype(ctxt, nid, buff));
  L3_N_setName(node, name);
  L3_N_setLabel(node, label);  
  L3_N_setDtype(node, dt);
  L3_N_setFlags(node, L3F_NONE);
  L3_N_getName(node, buff);
  if (strcmp(buff, L3S_ROOTNODENAME))
  {
    flg = HDF_Get_Attribute_As_Integer(nid, L3S_FLAGS, &ibuff);
    L3_N_setFlags(node, flg);
  }
  node->id = nid;
  if (dt != L3E_VOID)
  {
    HDF_Get_DataDimensions(ctxt, nid, dims);
    L3_N_setDims(node, dims);
    if ((dims[0] != -1) && (L3M_HASFLAG(ctxt, L3F_WITHDATA)))
    {
      data = HDF_Get_DataArray(ctxt, nid, dims, node->data);
      L3_N_setData(node, data);
    }
  }
  if (L3M_HASFLAG(ctxt, L3F_WITHCHILDREN))
  {
    node->children = HDF_Get_Children(nid, L3M_HASFLAG(ctxt, L3F_ASCIIORDER));
  }

  L3M_MXUNLOCK(ctxt);
  return node;
}
/* ------------------------------------------------------------------------- */
L3_Node_t *L3_nodeRetrieveContiguous(L3_Cursor_t *ctxt, hid_t oid,
                                     int index, int rank, int count,
                                     int interlaced,
                                     L3_Node_t *node)
{
  hid_t nid = -1;
  char  buff[L3C_MAX_ATTRIB_SIZE + 1];
  int   dims[L3C_MAX_DIMS], i_dims[L3C_MAX_DIMS];
  int   ibuff, dt, islk, flg, n, tsize = -1;
  void *data;
  char  name[L3C_MAX_ATTRIB_SIZE + 1];
  char  label[L3C_MAX_ATTRIB_SIZE + 1];
  hsize_t src_offset[L3C_MAX_DIMS];
  hsize_t src_stride[L3C_MAX_DIMS];
  hsize_t src_count[L3C_MAX_DIMS];
  hsize_t src_block[L3C_MAX_DIMS];
  hsize_t dst_offset[L3C_MAX_DIMS];
  hsize_t dst_stride[L3C_MAX_DIMS];
  hsize_t dst_count[L3C_MAX_DIMS];
  hsize_t dst_block[L3C_MAX_DIMS];

  L3M_CHECK_CTXT_OR_DIE(ctxt, NULL);
  L3M_MXLOCK(ctxt);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt, ("L3_nodeRetrieveContiguous [%ld] @@@\n", oid));
  L3M_CLEARDIMS(dims);

  if (node == NULL)
  {
    CHL_setError(ctxt, 3092);
    L3M_MXUNLOCK(ctxt);
    return NULL;
  }

  if (node->children != NULL) { free(node->children); }
  node->children = NULL;

  if (HDF_Check_Node(oid))
  {
    nid = oid;
    islk = is_link(ctxt, oid);
    if (islk && L3M_HASFLAG(ctxt, L3F_FOLLOWLINKS))
    {
      nid = get_link_actual_id(ctxt, oid);
      if (!HDF_Check_Node(nid) && L3M_HASFLAG(ctxt, L3F_FAILSONLINK))
      {
        CHL_setError(ctxt, 3091);
        L3M_MXUNLOCK(ctxt);
        return NULL;
      }
      /* prevent hdf5 1.1x error */
      if (!HDF_Check_Node(nid)) {
          nid = oid;
      }
      islk = 0;
    }
    else
    {
      islk = 0;
    }
    HDF_Get_Name(ctxt, nid, name);
    HDF_Get_Label(ctxt, nid, label);
    L3_N_setName(node, name);
    L3_N_setLabel(node, label);
    dt = L3_typeAsEnum(HDF_Get_Dtype(ctxt, nid, buff));
    L3_N_setDtype(node, dt);
    L3_N_getName(node, buff);
    if (strcmp(buff, L3S_ROOTNODENAME))
    {
      flg = HDF_Get_Attribute_As_Integer(nid, L3S_FLAGS, &ibuff);
      L3_N_setFlags(node, flg);
    }
    else
    {
      L3_N_setFlags(node, L3F_NONE);
    }
    node->id = nid;
    if (strcmp(HDF_Get_Dtype(ctxt, nid, buff), L3T_MT))
    {
      HDF_Get_DataDimensions(ctxt, nid, dims);
      tsize = 1;
      if (interlaced)
      {
        for (n = L3C_MAX_DIMS - 1; n > 0; n--)
        {
          i_dims[n] = dims[n - 1];
          printf("DIM %d=%d\n", n, i_dims[n]);
        }
        i_dims[0] = count;
      }
      if (interlaced)
      {
        L3_N_setDims(node, i_dims);
      }
      else
      {
        L3_N_setDims(node, dims);
      }
      for (n = 0; n < L3C_MAX_DIMS; n++)
      {
        if (dims[n] == -1) { break; }
        tsize *= dims[n];
        src_offset[n] = 0;
        src_stride[n] = 1;
        src_count[n] = dims[n];
        src_block[n] = 1;
      }
      src_offset[n] = L3_H5_SENTINEL;
      src_stride[n] = L3_H5_SENTINEL;
      src_count[n] = L3_H5_SENTINEL;
      src_block[n] = L3_H5_SENTINEL;
      if (interlaced)
      {
        dst_offset[0] = (tsize - 1)*rank;
        dst_stride[0] = 1;
      }
      else
      {
        dst_offset[0] = rank;
        dst_stride[0] = count;
      }
      dst_count[0] = tsize;
      dst_block[0] = 1;
      dst_offset[1] = L3_H5_SENTINEL;
      dst_stride[1] = L3_H5_SENTINEL;
      dst_count[1] = L3_H5_SENTINEL;
      dst_block[1] = L3_H5_SENTINEL;
      tsize *= count;
      if ((dims[0] != -1) && (L3M_HASFLAG(ctxt, L3F_WITHDATA)))
      {
        data = HDF_Get_DataArrayPartial(ctxt, nid, dims,
                                        src_offset,
                                        src_stride,
                                        src_count,
                                        src_block,
                                        dst_offset,
                                        dst_stride,
                                        dst_count,
                                        dst_block,
                                        tsize,
                                        rank,
                                        &(node->data),
                                        &(node->base));
        if (interlaced)
        {
          node->data = node->base;
        }
        else
        {
          node->data = data;
        }
      }
    }
    if (L3M_HASFLAG(ctxt, L3F_WITHCHILDREN))
    {
      node->children = HDF_Get_Children(nid, L3M_HASFLAG(ctxt, L3F_ASCIIORDER));
    }
  }
  else
  {
    CHL_setError(ctxt, 3090);
    L3M_MXUNLOCK(ctxt);
    return NULL;
  }

  L3M_MXUNLOCK(ctxt);
  return node;
}
/* ------------------------------------------------------------------------- */
L3_Node_t *L3_nodeRetrievePartial(L3_Cursor_t *ctxt, hid_t oid,
  hsize_t *src_offset,
  hsize_t *src_stride,
  hsize_t *src_count,
  hsize_t *src_block,
  hsize_t *dst_offset,
  hsize_t *dst_stride,
  hsize_t *dst_count,
  hsize_t *dst_block,
  L3_Node_t *node)
{
  hid_t nid = -1;
  char  buff[L3C_MAX_ATTRIB_SIZE + 1];
  int   dims[L3C_MAX_DIMS];
  int   ibuff, dt, islk, flg;
  void *data;
  char  name[L3C_MAX_ATTRIB_SIZE + 1];
  char  label[L3C_MAX_ATTRIB_SIZE + 1];

  L3M_CHECK_CTXT_OR_DIE(ctxt, NULL);
  L3M_MXLOCK(ctxt);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt, ("L3_nodeRetrievePartial [%ld] @@@\n", oid));
  L3M_CLEARDIMS(dims);

  if (node == NULL)
  {
    CHL_setError(ctxt, 3092);
    L3M_MXUNLOCK(ctxt);
    return NULL;
  }

  if (node->children != NULL) { free(node->children); }
  node->children = NULL;

  if (HDF_Check_Node(oid))
  {
    nid = oid;
    islk = is_link(ctxt, oid);
    if (islk && L3M_HASFLAG(ctxt, L3F_FOLLOWLINKS))
    {
      nid = get_link_actual_id(ctxt, oid);
      if (!HDF_Check_Node(nid) && L3M_HASFLAG(ctxt, L3F_FAILSONLINK))
      {
        CHL_setError(ctxt, 3091);
        L3M_MXUNLOCK(ctxt);
        return NULL;
      }
      /* prevent hdf5 1.1x error */
      if (!HDF_Check_Node(nid)) {
          nid = oid;
      }
      islk = 0;
    }
    else
    {
      islk = 0;
    }
    HDF_Get_Name(ctxt, nid, name);
    HDF_Get_Label(ctxt, nid, label);
    L3_N_setName(node, name);
    L3_N_setLabel(node, label);
    dt = L3_typeAsEnum(HDF_Get_Dtype(ctxt, nid, buff));
    L3_N_setDtype(node, dt);
    L3_N_getName(node, buff);
    if (strcmp(buff, L3S_ROOTNODENAME))
    {
      flg = HDF_Get_Attribute_As_Integer(nid, L3S_FLAGS, &ibuff);
      L3_N_setFlags(node, flg);
    }
    else
    {
      L3_N_setFlags(node, L3F_NONE);
    }
    node->id = nid;
    if (strcmp(HDF_Get_Dtype(ctxt, nid, buff), L3T_MT))
    {
      HDF_Get_DataDimensionsPartial(ctxt, nid, dims,
        dst_offset, dst_stride, dst_count, dst_block);
      L3_N_setDims(node, dims);
      if ((dims[0] != -1) && (L3M_HASFLAG(ctxt, L3F_WITHDATA)))
      {
        data = HDF_Get_DataArrayPartial(ctxt, nid, dims,
          src_offset,
          src_stride,
          src_count,
          src_block,
          dst_offset,
          dst_stride,
          dst_count,
          dst_block,
          -1, /* size has to be computed */
          0,
          &(node->data),
          &(node->base));
        node->data = data;
      }
    }
    if (L3M_HASFLAG(ctxt, L3F_WITHCHILDREN))
    {
      node->children = HDF_Get_Children(nid, L3M_HASFLAG(ctxt, L3F_ASCIIORDER));
    }
  }
  else
  {
    CHL_setError(ctxt, 3090);
    L3M_MXUNLOCK(ctxt);
    return NULL;
  }

  L3M_MXUNLOCK(ctxt);
  return node;
}
/* ------------------------------------------------------------------------- */
L3_Cursor_t*
L3_openHID(hid_t root, long flags)
{
  L3_Cursor_t *ctxt;

  ctxt = (L3_Cursor_t*)malloc(sizeof(L3_Cursor_t));
  if (ctxt == NULL)
  {
    return NULL;
  }
  L3M_ECLEAR(ctxt);
  L3M_MXINIT(ctxt);

  ctxt->file_id = -1;
  ctxt->root_id = root;
  ctxt->stack_id = -1;
  ctxt->config = flags;
  ctxt->l_proplist = -1;
  ctxt->g_proplist = -1;
  ctxt->ebuff[0] = '\0';
  ctxt->result = NULL;
  ctxt->pathlist = NULL;
  ctxt->currentpath = NULL;

  return ctxt;
}
/* ------------------------------------------------------------------------- */
L3_Node_t *L3_nodeSet(L3_Cursor_t *ctxt, L3_Node_t *node,
                      char *name, char *label,
                      int  *dims, int dtype, void *data, int flags)
{
  if (node == NULL)
  {
    L3M_NEWNODE(node); // watch out, local modif should be used through return
    __node_count++;
  }
  L3_N_setName(node, name);
  L3_N_setLabel(node, label);
  L3_N_setDtype(node, dtype);
  L3_N_setDims(node, dims);
  L3_N_setData(node, data);
  L3_N_setFlags(node, flags);

  return node;
}
/* ------------------------------------------------------------------------- */
L3_Node_t *L3_nodeGet(L3_Cursor_t *ctxt, L3_Node_t *node,
                      char *name, char *label,
                      int  *dims, int *dtype, void *data, int* flags)
{
  if (node != NULL)
  {
    L3_N_getName(node, name);
    L3_N_getLabel(node, label);
    L3_N_getDtype(node, dtype);
    L3_N_getDims(node, dims);
    L3_N_getData(node, data);
    L3_N_getFlags(node, *flags);
  }
  return node;
}
/* ------------------------------------------------------------------------- */
int L3_nodeAndDataFree(L3_Node_t **nodeptr)
{
  L3_Node_t *node;

  node = *nodeptr;
  if (node && (node->data != NULL)) { free(node->data); }
  return L3_nodeFree(nodeptr);
}
/* ------------------------------------------------------------------------- */
int L3_nodeChildrenFree(L3_Node_t **nodeptr)
{
  L3_Node_t *node;
  int n;
  char name[33];

  node = *nodeptr;
  if (node == NULL) { return 1; }
  if (node && (node->children != NULL))
  {
    n = 0;
    while (node->children[n] != -1)
    {
      if (H5Iis_valid(node->children[n]))
      {
        HDF_Get_Name((L3_Cursor_t *)NULL, node->children[n], name);
        L3_H5_GCLOSE("NODE CHILDREN FREE \n", node->children[n]);
      }
      n++;
    }
    free(node->children);
    node->children = NULL;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
int L3_nodeRelease(L3_Node_t **nodeptr, unsigned int flags)
{
  L3_Node_t *node;

  node = *nodeptr;
  if (node == NULL) { return 1; }

  /*  printf("# L3 :L3_nodeRelease [%d]\n",node->id); */
  if ((flags&L3F_R_HID_CHILDREN) && (node->children != NULL))
  {
    /* printf("# L3 :L3_nodeRelease [%d] R_HID_CHILDREN\n",node->id); */
    L3_nodeChildrenFree(nodeptr);
  }
  else if ((flags&L3F_R_MEM_CHILDREN) && (node->children != NULL))
  {
    /* printf("# L3 :L3_nodeRelease [%d] R_MEM_CHILDREN\n",node->id); */
    free(node->children);
  }
  if ((flags&L3F_R_HID_NODE) && (node->id > 0) && (H5Iis_valid(node->id)))
  {
    /* printf("# L3 :L3_nodeRelease [%d] R_HID_NODE\n",node->id); */
    L3_H5_GCLOSE("NODE RELEASE\n", node->id);
  }
  if ((flags&L3F_R_MEM_DATA) && (node != NULL) && (node->data != NULL))
  {
    /* printf("# L3 :L3_nodeRelease [%d] R_MEM_DATA\n",node->id); */
    free(node->data);
  }
  if ((flags&L3F_R_MEM_NODE) && (node != NULL))
  {
    /* printf("# L3 :L3_nodeRelease [%d] R_MEM_NODE\n",node->id); */
    free(node);
    __node_count--;
    *nodeptr = NULL;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
int L3_nodeFree(L3_Node_t **nodeptr)
{
  L3_Node_t *node;

  if (nodeptr == NULL) { return 1; }
  node = *nodeptr;
  if (node == NULL) { return 1; }
  L3_nodeChildrenFree(nodeptr);
  if (node->id > 0)
  {
    if (H5Iis_valid(node->id))
    {
      L3_H5_GCLOSE("NODE FREE\n", node->id);
    }
  }
  __node_count--;
  free(node);
  *nodeptr = NULL;

  return 1;
}
/* ------------------------------------------------------------------------- */
int L3_nodeFreeNoDecRef(L3_Node_t **nodeptr)
{
  L3_Node_t *node;

  node = *nodeptr;
  if (node == NULL) { return 1; }
  L3_nodeChildrenFree(nodeptr);
  __node_count--;
  free(node);
  *nodeptr = NULL;

  return 1;
}
/* ------------------------------------------------------------------------- */
void L3_nodePrint(L3_Node_t *node)
{
  if (node != NULL)
  {
    printf("Node [%s][%s][%s]\n",
           node->name, node->label, L3_typeAsStr(node->dtype));
    fflush(stdout);
  }
}
/* ------------------------------------------------------------------------- */
L3_Cursor_t*
L3_openFile(char *filename, int mode, long flags)
{
  L3_Cursor_t *ctxt;
  hid_t fapl, fcpl;

  ctxt = (L3_Cursor_t*)malloc(sizeof(L3_Cursor_t));
  if (ctxt == NULL)
  {
    return NULL;
  }
  L3M_ECLEAR(ctxt);
  L3M_MXINIT(ctxt);

  ctxt->file_id = -1;
  ctxt->root_id = -1;
  ctxt->stack_id = -1;
  ctxt->config = flags;
  ctxt->ebuff[0] = '\0';
  ctxt->result = NULL;
  ctxt->pathlist = NULL;
  ctxt->currentpath = NULL;

  H5dont_atexit(); /* MANDATORY FIRST HDF5 function to call for Threading */

  if (!L3M_HASFLAG(ctxt, L3F_DEBUG))
  {
    /* fails, the error is set for all HDF5 context but ctxt is local !
       see test case 0012 CGNS.MAP without file access check in pyCHLone.pyx */
    H5Eset_auto2(H5E_DEFAULT, HDF_Walk_Error, ctxt);
  }
  else
  {
    ctxt->config |= L3F_TRACE;
  }

  fapl = H5Pcreate(H5P_FILE_ACCESS);
  if (fapl < 0)
  {
    CHL_setError(ctxt, 3011);
    return ctxt;
  }

#if defined(L3_H5F_STRONG_CLOSE)
  H5Pset_fclose_degree(fapl, H5F_CLOSE_STRONG);
#else
  H5Pset_fclose_degree(fapl, H5F_CLOSE_WEAK);
#endif

  if (mode == L3E_OPEN_RDO)
  {
    H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
  }
  else
  {
#if L3_HDF5_HAVE_110_API
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18);
#else
    H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
#endif
  }

  fcpl = H5Pcreate(H5P_FILE_CREATE);
  if (fcpl < 0)
  {
    CHL_setError(ctxt, 3011);
    return ctxt;
  }
  H5Pset_link_creation_order(fcpl,
                             H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);

  ctxt->g_proplist = H5Pcreate(H5P_GROUP_CREATE);
  if (ctxt->g_proplist < 0)
  {
    CHL_setError(ctxt, 3013);
    return ctxt;
  }
  H5Pset_link_creation_order(ctxt->g_proplist,
    H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);

  ctxt->l_proplist = H5Pcreate(H5P_LINK_ACCESS);
  if (ctxt->l_proplist < 0)
  {
    CHL_setError(ctxt, 3012);
    return ctxt;
  }
  H5Pset_nlinks(ctxt->l_proplist, L3C_MAX_LINK_DEPTH);
  /*  H5Pset_elink_cb(ctxt->l_proplist, HDF_link_check, NULL); */

  switch (mode)
  {
  case L3E_OPEN_NEW:
    L3M_TRACE(ctxt, ("newL3_Cursor_t open new\n"));
    ctxt->file_id = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl);
    if (ctxt->file_id < 0)
    {
      CHL_setError(ctxt, 3002, filename);
    }
    else
    {
      HDF_Create_Root(ctxt);
    }
    break;
  case L3E_OPEN_OLD:
    L3M_TRACE(ctxt, ("newL3_Cursor_t open old\n"));
    //printf("ACC RDWR\n");
    //objlist_status("OPEN");
    ctxt->file_id = H5Fopen(filename, H5F_ACC_RDWR, fapl);
    //H5Eprint1(stdout);
    if (ctxt->file_id < 0)
    {
      CHL_setError(ctxt, 3003, filename);
    }
    break;
  case L3E_OPEN_RDO:
  default:
    L3M_TRACE(ctxt, ("newL3_Cursor_t read only\n"));
    ctxt->file_id = H5Fopen(filename, H5F_ACC_RDONLY, fapl);
    if (ctxt->file_id < 0)
    {
      CHL_setError(ctxt, 3004, filename);
    }
    break;
  }
  ctxt->root_id = H5Gopen2(ctxt->file_id, "/", H5P_DEFAULT);
  H5Pclose(fcpl);
  H5Pclose(fapl);

  return ctxt;
}
/* ------------------------------------------------------------------------- */
int L3_close(L3_Cursor_t **ctxt_ptr)
{
  herr_t err;
  L3_Cursor_t *ctxt;

  if (ctxt_ptr == NULL) return 0;

  L3M_CHECK_CTXT_OR_DIE((*ctxt_ptr), 0);
  L3M_MXLOCK((*ctxt_ptr));
  L3M_TRACE((*ctxt_ptr), ("Close\n"));

  ctxt = *ctxt_ptr;

  if (ctxt->file_id != -1)
  {
    H5Pclose(ctxt->l_proplist);
    H5Pclose(ctxt->g_proplist);
    if (H5Iis_valid(ctxt->root_id))
    {
      L3_H5_GCLOSE("L3_Close @@@\n", ctxt->root_id);
    }
    if (H5Iis_valid(ctxt->file_id))
    {
#if !defined(L3_H5F_STRONG_CLOSE)
      H5Fflush(ctxt->file_id, H5F_SCOPE_LOCAL);
#endif
      err = H5Fclose(ctxt->file_id);
      H5garbage_collect();
    }
    ctxt->file_id = -1;
    ctxt->root_id = -1;
    if (ctxt->currentpath != NULL)
    {
      free(ctxt->currentpath);
    }
    if (ctxt->pathlist != NULL)
    {
      CHL_freeLinkSearchPath(ctxt);
    }
  }
  L3M_MXUNLOCK(ctxt);
  L3M_MXDESTROY(ctxt);

  if (ctxt->result != NULL) { printf("RESULT NOT EMPTY\n"); }

  free(ctxt);
  ctxt_ptr = NULL;

  return 0;
}
/* ------------------------------------------------------------------------- */
int L3_closeShutDown(L3_Cursor_t **ctxt_ptr)
{
  herr_t err;
  L3_Cursor_t *ctxt;
  ssize_t n, p;
  hid_t obj_id_list[1024];

  if (ctxt_ptr == NULL) return 0;

  L3M_CHECK_CTXT_OR_DIE((*ctxt_ptr), 0);
  L3M_MXLOCK((*ctxt_ptr));
  L3M_TRACE((*ctxt_ptr), ("Close\n"));

  ctxt = *ctxt_ptr;

  printf("force release 1\n");
  if (ctxt->file_id != -1)
  {
    H5Pclose(ctxt->l_proplist);
    H5Pclose(ctxt->g_proplist);
    if (H5Iis_valid(ctxt->root_id))
    {
      L3_H5_GCLOSE("L3_Close @@@\n", ctxt->root_id);
    }
    if (0 && H5Iis_valid(ctxt->file_id))
    {
      printf("force release 2\n");
      n = H5Fget_obj_count(ctxt->file_id, H5F_OBJ_ALL);
      p = H5Fget_obj_ids(ctxt->file_id, H5F_OBJ_ALL, 1024, obj_id_list);
      for (n = 0; n < p; n++)
      {
        printf("N %ld > P %ld\n", n, p);
        H5Oclose(obj_id_list[n]);
      }
#if !defined(L3_H5F_STRONG_CLOSE)
      H5Fflush(ctxt->file_id, H5F_SCOPE_LOCAL);
#endif
      err = H5Fclose(ctxt->file_id);
      H5garbage_collect();
    }
    ctxt->file_id = -1;
    ctxt->root_id = -1;
    if (ctxt->currentpath != NULL)
    {
      free(ctxt->currentpath);
    }
    if (ctxt->pathlist != NULL)
    {
      CHL_freeLinkSearchPath(ctxt);
    }
  }
  L3M_MXUNLOCK(ctxt);
  L3M_MXDESTROY(ctxt);

  if (ctxt->result != NULL) { printf("RESULT NOT EMPTY\n"); }

  free(ctxt);
  ctxt_ptr = NULL;

  return 0;
}
/* ------------------------------------------------------------------------- */
int L3_isLinkNode(L3_Cursor_t *ctxt, hid_t id, char *file, char *name)
{
  int r = 0;
  char bfile[512]; /* to fix */
  char bpath[512];

  if (!is_link(ctxt, id) || (file == NULL) || (name == NULL))
  {
    return 0;
  }
  bfile[0] = '\0';
  bpath[0] = '\0';
  r = get_link_data(ctxt, id, bfile, bpath);
  if (r && (bfile != NULL) && (bpath != NULL))
  {
    strcpy(file, bfile);
    strcpy(name, bpath);
  }
  return r;
}
/* ------------------------------------------------------------------------- */
hid_t L3_incRef(L3_Cursor_t *ctxt, hid_t id)
{
  H5Iinc_ref(id);
  return id;
}
/* ------------------------------------------------------------------------- */
void L3_decRef(L3_Cursor_t *ctxt, hid_t id)
{
  H5Idec_ref(id);
}
/* ------------------------------------------------------------------------- */
int L3_isLocalNode(L3_Cursor_t *ctxt, hid_t id)
{
  H5O_info_t info1, info2;
#if L3_HDF5_HAVE_112_API
  H5Oget_info3(ctxt->root_id, &info1, H5O_INFO_BASIC);
  H5Oget_info3(id, &info2, H5O_INFO_BASIC);
#else
  H5Oget_info(ctxt->root_id, &info1);
  H5Oget_info(id, &info2);
#endif
  printf("CHECK %d %d\n", (int)(info1.fileno), (int)(info2.fileno));
  if (info1.fileno != info2.fileno) return 0;

  return 1;
}
/* ------------------------------------------------------------------------- */
int L3_isSameNode(L3_Cursor_t *ctxt, hid_t id1, hid_t id2)
{
  H5O_info_t info1, info2;
#if L3_HDF5_HAVE_112_API
  int token_cmp;
  H5Oget_info3(id1, &info1, H5O_INFO_BASIC);
  H5Oget_info3(id2, &info2, H5O_INFO_BASIC);

  if (info1.fileno != info2.fileno) return 0;
  token_cmp = 1;
  H5Otoken_cmp(id1, &info1.token, &info2.token, &token_cmp);
  if (token_cmp != 0) return 0;
#else
  H5Oget_info(id1, &info1);
  H5Oget_info(id2, &info2);

  if (info1.fileno != info2.fileno) return 0;
  if (info1.addr != info2.addr)   return 0;
#endif
  return 1;
}
/* ------------------------------------------------------------------------- */
int *L3_initDims(int *dims, int d1, ...)
{
  va_list intptr;
  int n, *ndims;

  ndims = dims;
  va_start(intptr, d1);
  if (dims == NULL)
  {
    ndims = (int*)malloc(sizeof(int)*L3C_MAX_DIMS);
    L3M_CLEARDIMS(ndims);
  }
  ndims[0] = d1;
  n = 1;
  while (ndims[n - 1] != -1)
  {
    ndims[n++] = va_arg(intptr, int);
  }
  va_end(intptr);
  return ndims;
}
/* ------------------------------------------------------------------------- */
hsize_t *L3_initHyperslab(hsize_t *hs, int d1, ...)
{
  va_list intptr;
  int n;
  hsize_t *nhs;

  nhs = hs;
  va_start(intptr, d1);
  if (hs == NULL)
  {
    nhs = (hsize_t*)malloc(sizeof(hsize_t)*L3C_MAX_DIMS);
    L3M_CLEARDIMS(nhs);
  }
  nhs[0] = d1;
  n = 1;
  while (nhs[n - 1] != L3_H5_SENTINEL)
  {
    nhs[n++] = va_arg(intptr, int);
  }
  va_end(intptr);
  return nhs;
}
/* ------------------------------------------------------------------------- */
void  *L3_initData(int  *dims, void *data, int dtype, ...)
{
  int dsize, n;
  int d_i;
  long d_l;
  float d_f;
  double d_d;
  va_list stack;

  if (dims == NULL) { return NULL; }
  dsize = 1;
  n = 0;
  va_start(stack, dtype);

  while ((n < L3C_MAX_DIMS) && (dims[n] != -1))
  {
    dsize *= dims[n++];
  }
  switch (dtype)
  {
  case L3E_I4:
  {
    data = (void*)malloc(dsize * sizeof(int));
    n = 0;
    while (n < dsize)
    {
      d_i = va_arg(stack, int);
      ((int*)data)[n++] = d_i;
    }
    break;
  }
  case L3E_I8:
  {
    data = (void*)malloc(dsize * sizeof(long));
    n = 0;
    while (n < dsize)
    {
      d_l = va_arg(stack, long);
      ((long*)data)[n++] = d_l;
    }
    break;
  }
  case L3E_R4:
  {
    data = (void*)malloc(dsize * sizeof(float));
    n = 0;
    while (n < dsize)
    {
      d_f = (float) va_arg(stack, double);
      ((float*)data)[n++] = d_f;
    }
    break;
  }
  case L3E_R8:
  {
    data = (void*)malloc(dsize * sizeof(double));
    n = 0;
    while (n < dsize)
    {
      d_d = va_arg(stack, double);
      ((double*)data)[n++] = d_d;
    }
    break;
  }
  case L3E_X4:
  {
    data = (void*)malloc(2 * dsize * sizeof(float));
    n = 0;
    while (n < 2*dsize)
    {
      d_f = (float) va_arg(stack, double);
      ((float*)data)[n++] = d_f;
    }
    break;
  }
  case L3E_X8:
  {
    data = (void*)malloc(2 * dsize * sizeof(double));
    n = 0;
    while (n < 2*dsize)
    {
      d_d = va_arg(stack, double);
      ((double*)data)[n++] = d_d;
    }
    break;
  }
  default: break;
  }
  va_end(stack);
  return data;
}
/* ------------------------------------------------------------------------- */
void  *L3_fillData(int  *dims, void *data, int dtype, ...)
{
  int dsize, n;
  int d_i;
  double d_d;
  float d_f;
  char d_c;
  va_list stack;

  if (dims == NULL) { return NULL; }
  dsize = 1;
  n = 0;
  va_start(stack, dtype);

  while ((n < L3C_MAX_DIMS) && (dims[n] != -1))
  {
    dsize *= dims[n++];
  }
  switch (dtype)
  {
  case L3E_C1:
  {
    data = (void*)malloc(dsize);
    d_c = va_arg(stack, unsigned int);
    n = 0; while (n < dsize) { ((char*)data)[n++] = d_c; }
    break;
  }
  case L3E_I4:
  {
    data = (void*)malloc(dsize * sizeof(int));
    d_i = va_arg(stack, int);
    n = 0; while (n < dsize) { ((int*)data)[n++] = d_i; }
    break;
  }
  case L3E_R8:
  {
    data = (void*)malloc(dsize * sizeof(double));
    d_d = va_arg(stack, double);
    n = 0; while (n < dsize) { ((double *)data)[n++] = d_d; }
    break;
  }
  case L3E_R4:
  {
    data = (void*)malloc(dsize * sizeof(float));
    d_f = (float) va_arg(stack, double);
    n = 0; while (n < dsize) { ((float *)data)[n++] = d_f; }
    break;
  }
  case L3E_X8:
  {
    data = (void*)malloc(dsize * sizeof(double));
    d_d = va_arg(stack, double);
    n = 0; while (n < 2*dsize) { ((double *)data)[n++] = d_d; }
    break;
  }
  case L3E_X4:
  {
    data = (void*)malloc(dsize * sizeof(float));
    d_f = (float) va_arg(stack, double);
    n = 0; while (n < 2*dsize) { ((float *)data)[n++] = d_f; }
    break;
  }
  default: break;
  }
  va_end(stack);
  return data;
}
/* ------------------------------------------------------------------------- */
char *L3_node2Path(L3_Cursor_t *ctxt, hid_t id, char *retpath)
{
  int nsize;
  char trybuff[1], *path;

  if (!H5Iis_valid(id))
  {
    CHL_setError(ctxt, 3093, id);
    return NULL;
  }

  path = retpath;
  nsize = H5Iget_name(id, trybuff, 1);
  if (path == NULL)
  {
    path = (char*)malloc(nsize + 1);
  }
  nsize = H5Iget_name(id, path, nsize + 1);

  return path;
}
/* ------------------------------------------------------------------------- */
hid_t L3_path2Node(L3_Cursor_t *ctxt, char *path)
{
  hid_t id = -1;

  L3M_MXLOCK(ctxt);
  id = H5Gopen2(ctxt->root_id, path, H5P_DEFAULT);
  L3_T_ID("PTH", id);
  L3M_MXUNLOCK(ctxt);

  return id;
}
/* ------------------------------------------------------------------------- */
char *L3_typeAsStr(int dtype)
{
  switch (dtype)
  {
  case L3E_C1:
  case L3E_C1ptr: return L3T_C1_s;
  case L3E_I4:
  case L3E_I4ptr: return L3T_I4_s;
  case L3E_I8:
  case L3E_I8ptr: return L3T_I8_s;
  case L3E_R4:
  case L3E_R4ptr: return L3T_R4_s;
  case L3E_R8:
  case L3E_R8ptr: return L3T_R8_s;
  case L3E_X4:
  case L3E_X4ptr: return L3T_X4_s;
  case L3E_X8:
  case L3E_X8ptr: return L3T_X8_s;
  case L3E_VOID:
  default:        return L3T_MT_s;
  }
}
/* ------------------------------------------------------------------------- */
int L3_typeAsEnum(char *dtype)
{
//  if (!strcmp(dtype, L3T_MT_s)) { return L3E_MT; }
  if (!strcmp(dtype, L3T_C1_s)) { return L3E_C1ptr; }
  if (!strcmp(dtype, L3T_I4_s)) { return L3E_I4ptr; }
  if (!strcmp(dtype, L3T_I8_s)) { return L3E_I8ptr; }
  if (!strcmp(dtype, L3T_R4_s)) { return L3E_R4ptr; }
  if (!strcmp(dtype, L3T_R8_s)) { return L3E_R8ptr; }
  if (!strcmp(dtype, L3T_X4_s)) { return L3E_X4ptr; }
  if (!strcmp(dtype, L3T_X8_s)) { return L3E_X8ptr; }
  return L3E_VOID;
}
/* ------------------------------------------------------------------------- */
int L3_config(int p)
{
  return 0;
}
/* --- last line */
