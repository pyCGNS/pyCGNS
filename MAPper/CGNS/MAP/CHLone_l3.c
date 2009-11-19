/* 
#  -------------------------------------------------------------------------
#  pyCGNS.MAP - Python package for CFD General Notation System - MAPper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  ------------------------------------------------------------------------- 
*/

/* 
   ------------------------------------------------------------------------
   l3 has a set of node-level functions to handle an ADF-like
   node with HDF5. You can use it through its own API (see CHLone_l3.h)
   or use the a2 API which defines a more user oriented set of functions.
   ------------------------------------------------------------------------

    LOT OF CODE IN THERE HAS BEEN TAKEN FROM cgnlib/ADF/ADFH.c
    cgnlib/ADF/ADFH.c Credits: 
      Bruce Wedan (formerly ANSYS), 
      Greg Power et al., Arnold Air Force Base,
      Marc Poinot, ONERA
      Other people... sorry, add your name!
    Based on the node format defined by ADF people 
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

   root node                     format     : string (group dataset)
                                 hdf5version: string (group dataset)

   link node                     file   : string (group dataset)
                                 path   : string (group dataset)
                                 link   : actual HDF5 link object
*/
#include "CHLone_l3.h"

/* ------------------------------------------------------------------------- */
#define L3M_ECHECKID(id,ret) \
if (id == -1){return ret;}

#define L3M_ECHECKL3NODE(id,ret) \
if (id == NULL){return ret;}

#define  L3M_CHECK_CTXT_OR_DIE( ctxt, rval )\
if ((ctxt == NULL)||(((L3_Cursor_t*)ctxt)->file_id<0)){ return rval ;}

/* ------------------------------------------------------------------------- */
#define L3_T_MT "MT"
#define L3_T_LK "LK"
#define L3_T_B1 "B1"
#define L3_T_C1 "C1"
#define L3_T_I4 "I4"
#define L3_T_I8 "I8"
#define L3_T_U4 "U4"
#define L3_T_U8 "U8"
#define L3_T_R4 "R4"
#define L3_T_R8 "R8"       

static char *L3_T_MT_s=L3_T_MT;
static char *L3_T_LK_s=L3_T_LK;
static char *L3_T_B1_s=L3_T_B1;
static char *L3_T_C1_s=L3_T_C1;
static char *L3_T_I4_s=L3_T_I4;
static char *L3_T_I8_s=L3_T_I8;
static char *L3_T_U4_s=L3_T_U4;
static char *L3_T_U8_s=L3_T_U8;
static char *L3_T_R4_s=L3_T_R4;
static char *L3_T_R8_s=L3_T_R8;

/* ------------------------------------------------------------------------- */
#define L3_N_setName(node,aname) \
if (node==NULL){return NULL;} \
if ((aname!=NULL)&&(aname[0]!='\0')){strcpy(node->name,aname);} 

#define L3_N_setLabel(node,alabel) \
if (node==NULL){return NULL;} \
if ((alabel!=NULL)&&(alabel[0]!='\0')){strcpy(node->label,alabel);}

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
for (__nn=0;__nn<L3_MAX_DIMS;__nn++){node->dims[__nn]=adims[__nn];}}

#define L3_N_setData(node,adata) \
if (node==NULL){return NULL;} \
if (adata!=NULL){node->data=adata;}

#define L3_N_setParent(node,pid) \
if (node==NULL){return NULL;} \
else {node->parentid=pid;}

/* ------------------------------------------------------------------------- */
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
for (__nn=0;__nn<L3_MAX_DIMS;__nn++){adims[__nn]=node->dims[__nn];}}

#define L3_N_getData(node,data) \
if (node==NULL){return NULL;} \
if ((data!=NULL)&&(node->data!=NULL)){data=node->data;}

#define L3_N_getParent(node,pid) \
if (node==NULL){return NULL;} \
{pid=node->parentid;}

#define L3_N_getFlags(node,aflags) \
if (node==NULL){return NULL;} \
else {aflags=node->flags;}

/* ------------------------------------------------------------------------- */
static herr_t find_name(hid_t id,const char *nm,const H5A_info_t* i,void *snm)
{
  if (!strcmp(nm,(char *)snm)) return 1;
  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t gfind_name(hid_t id, const char *nm, void *snm)
{
  if (!strcmp(nm,(char *)snm)) return 1;
  return 0;
}
/* ------------------------------------------------------------------------- */
#define has_att(ID,NAME) \
H5Aiterate2(ID,H5_INDEX_NAME,H5_ITER_NATIVE,NULL,find_name,(void *)NAME)
#define has_data(ID) \
H5Giterate(ID,".",NULL,gfind_name,(void *)L3_S_DATA)
#define has_child(ID,NAME) \
H5Giterate(ID,".",NULL,gfind_name,(void *)NAME)
/* ------------------------------------------------------------------------- */
static herr_t HDF_Print_Error(unsigned n, H5E_error2_t *desc, void *ctxt)
{
  char localbuff[256]; /* bet ! */
  L3_Cursor_t* c;

  if (ctxt == NULL) { return 0;}

  c=(L3_Cursor_t*)ctxt;
  sprintf(localbuff, "%s (%u): %s\n",desc->func_name,desc->line,desc->desc);
  CHL_setMessage(c,localbuff);

  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t HDF_Walk_Error(hid_t estack, void *ctxt)
{
  return H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, 
		  (H5E_walk2_t)HDF_Print_Error, ctxt);
}
/* ------------------------------------------------------------------------- */
int HDF_Check_Node(hid_t parentid)
{
  return 1;
}
/* ------------------------------------------------------------------------- */
hid_t ADF_to_HDF_datatype(const char *tp)
{
  if (!strcmp(tp, L3_T_B1))   return H5Tcopy(H5T_NATIVE_UCHAR);
  if (!strcmp(tp, L3_T_C1))   return H5Tcopy(H5T_NATIVE_CHAR);
  if (!strcmp(tp, L3_T_I4))   return H5Tcopy(H5T_NATIVE_INT32);
  if (!strcmp(tp, L3_T_I8))   return H5Tcopy(H5T_NATIVE_INT64);
  if (!strcmp(tp, L3_T_U4))   return H5Tcopy(H5T_NATIVE_UINT32);
  if (!strcmp(tp, L3_T_U8))   return H5Tcopy(H5T_NATIVE_UINT64);
  if (!strcmp(tp, L3_T_R4)) {
                              hid_t tid = H5Tcopy(H5T_NATIVE_FLOAT);
			      H5Tset_precision(tid, 32); 
			      return tid;
                            }
  if (!strcmp(tp, L3_T_R8)) {
                              hid_t tid = H5Tcopy(H5T_NATIVE_DOUBLE);
			      H5Tset_precision(tid, 64);
			      return tid;

                              }
  if (!strcmp(tp, L3_T_LK))   return 0;
  if (!strcmp(tp, L3_T_B1))   return 0;
  if (!strcmp(tp, L3_T_U4))   return 0;
  if (!strcmp(tp, L3_T_U8))   return 0;

  if (tp!=NULL) printf("TYPE NOT FOUND [%s]\n",tp);
  else          printf("TYPE NULL\n");
  return 0;
}
/* ------------------------------------------------------------------------- */
int HDF_Is_Fortranable(L3_Cursor_t *ctxt,int ndim,char *dtype,char *label)
{ 
  /* zero shape should not appear at this place */

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);

  if (L3M_HASFLAG(ctxt, L3_F_NOTRANSPOSE))   return 0;
  if (ndim==1)            	      	    return 0;
  if (!strcmp(dtype,L3_T_C1))        	    return 0;
  if (strcmp(label,"DataArray_t"))    	    return 0;
  if (strcmp(label,"DimensionalUnits_t"))   return 0;

  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Get_Attribute_As_Integer(hid_t nodeid, const char *name, int *value)
{
  hid_t aid;
  herr_t status;

  if (!has_att(nodeid,name))
  {
    return 0;
  }
  aid=H5Aopen_name(nodeid, name);
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
char *HDF_Get_Attribute_As_String(hid_t nodeid, const char *name, char *value)
{
  hid_t aid,tid;

  value[0]='\0';
  if (!has_att(nodeid,name))
  {
    return value;
  }
  aid=H5Aopen_name(nodeid, name);
  if (aid > 0)
  {
    tid = H5Aget_type(aid);
    if (tid < 0) 
    {
      H5Aclose(aid);
      return 0;
    }
    H5Aread(aid,tid,value);
    H5Tclose(tid);
    H5Aclose(aid);
  }
  return value;
}
/* ------------------------------------------------------------------------- */
char *HDF_Get_Attribute_As_Data(hid_t nodeid, const char *name, char *value)
{
  hid_t did ;

  value[0]='\0';
  did = H5Dopen2(nodeid, name, H5P_DEFAULT);
  if (did > 0)
  {
    H5Dread(did, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, value);
    H5Dclose(did);
  }
  return value;
}
/* ----------------------------------------------------------------- */
static int is_link(hid_t nodeid)
{
  char ntype[L3_MAX_DTYPE+1];

  HDF_Get_Attribute_As_String(nodeid,L3_S_DTYPE,ntype);
  if (!strcmp(L3_T_LK, ntype))
  {
    return 1;
  }
  return 0;
}
/* ----------------------------------------------------------------- */
static hid_t get_link_actual_id(hid_t id)
{
  hid_t lid;
  herr_t herr;
  const char  *file;
  const char  *path;
  H5G_stat_t	sb; /* Object information */
  char querybuff[512];

  if (H5Lis_registered(H5L_TYPE_EXTERNAL) != 1)
  {
    return -1;
  }
  herr=H5Gget_objinfo(id, L3_S_LINK, (hbool_t)0, &sb);

  if (herr<0)
  {
    return -1;
  } 

  /* Soft link                -> link to our current file */
  /* Hard link (User defined) -> link to an external file */
  if (H5G_LINK != sb.type) 
  {
    if (H5G_UDLINK != sb.type) 
    {
      return -1;
    }

    if (H5Lget_val(id,L3_S_LINK,querybuff,sizeof(querybuff),H5P_DEFAULT)<0)
    {
      return -1;
    } 

    if (H5Lunpack_elink_val(querybuff,sb.linklen,NULL,&file,&path)<0)
    {
      return -1;
    } 
    /* open the actual link >> IN THE LINK GROUP << */
    if ((lid = H5Gopen2(id, L3_S_LINK, H5P_DEFAULT)) < 0)
      {
	return -1;
      }
  }
  else
  {
    if ((lid = H5Gopen2(id, L3_S_LINK, H5P_DEFAULT)) < 0)
      {
	return -1;
      }
  }
  return lid;

}
/* ----------------------------------------------------------------- */
static int get_link_data(hid_t id, const char *file, const char *path)
{
  herr_t herr;
  H5G_stat_t	sb; /* Object information */
  char querybuff[512];

  if (H5Lis_registered(H5L_TYPE_EXTERNAL) != 1)
  {
    return 0;
  }
  herr=H5Gget_objinfo(id, L3_S_LINK, (hbool_t)0, &sb);

  if (herr<0)
  {
    return 0;
  } 

  /* Soft link                -> link to our current file */
  /* Hard link (User defined) -> link to an external file */
  if (H5G_LINK != sb.type) 
  {
    if (H5G_UDLINK != sb.type) 
    {
      return 0;
    }

    if (H5Lget_val(id,L3_S_LINK,querybuff,sizeof(querybuff),H5P_DEFAULT)<0)
    {
      return 0;
    } 

    if (H5Lunpack_elink_val(querybuff,sb.linklen,NULL,&file,&path)<0)
    {
      return 0;
    } 
    return 1;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t delete_children(hid_t id, const char *name, void *data)
{
  /* do not change link id with actual here, stop deletion at link node */
  if (name && (name[0] == ' ')) /* leaf node */
  {
    H5Gunlink(id, name);
  }
  else 
  {
    H5Giterate(id, name, NULL, delete_children, data);
    H5Gunlink(id, name);
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t count_children(hid_t id, const char *name, void *count)
{
  if (name && (name[0] != ' '))
  {
    (*((int *)count))++;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static herr_t feed_children_ids_list(hid_t id, const char *name, void *idlist)
{
  hid_t cid;
  int n;

  /* skip names starting with a <space> */
  if (name && (name[0] == ' '))
  {
    return 0;
  }
  cid=H5Gopen(id, name, H5P_DEFAULT);
  /* set id  */
  n=0;
  while (((hid_t*)idlist)[n]!=-1) 
  {
    n++;
  }
  ((hid_t*)idlist)[n]=cid;
  return 0;
}
/* ------------------------------------------------------------------------- */
hid_t *HDF_Get_Children(hid_t nodeid)
{
  hid_t *idlist;
  int    nchildren,n;

  nchildren=0; 
  H5Giterate(nodeid,".",NULL,count_children,(void *)&nchildren);

  idlist=(hid_t*)malloc(sizeof(hid_t)*(nchildren+1));
  /* use last -1 as sentinel */
  for (n=0; n<=nchildren; n++) {idlist[n]=(hid_t)-1;}
  H5Giterate(nodeid,".",NULL,feed_children_ids_list,(void *)idlist);

  return idlist;
}
/* ------------------------------------------------------------------------- */
void *HDF_Read_Array(L3_Cursor_t *ctxt,hid_t nid,hid_t did,hid_t yid,
		     void *data,hsize_t *int_dim_vals)
{
  herr_t stat; 
  int n;
  hsize_t tsize;
  char  name[L3_MAX_ATTRIB_SIZE+1];
  char  label[L3_MAX_ATTRIB_SIZE+1];

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);

  HDF_Get_Attribute_As_String(nid,L3_S_NAME,name);
  HDF_Get_Attribute_As_String(nid,L3_S_LABEL,label);
  L3M_DBG(ctxt,("HDF_Read_Array [%s][%s]\n",name,label));

  tsize=1;
  for (n=0; n<L3_MAX_DIMS; n++)
  {
    if (int_dim_vals[n]==-1){break;}
    tsize*=int_dim_vals[n];
  }
  data=(void*)malloc(H5Tget_size(yid)*tsize);
  stat=H5Dread(did,yid,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);

  L3M_DBG(ctxt,("HDF_Read_Array status [%d]\n",stat));

  return data;
}
/* ------------------------------------------------------------------------- */
int HDF_Write_Array(L3_Cursor_t *ctxt,hid_t nid,hid_t did,hid_t yid,
		    void *data,hsize_t *int_dim_vals)
{
  herr_t stat;
  char  name[L3_MAX_ATTRIB_SIZE+1];
  char  label[L3_MAX_ATTRIB_SIZE+1];

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);

  HDF_Get_Attribute_As_String(nid,L3_S_NAME,name);
  HDF_Get_Attribute_As_String(nid,L3_S_LABEL,label);
  L3M_DBG(ctxt,("HDF_Write_Array [%s][%s]\n",name,label));

  stat=H5Dwrite(did,yid,H5S_ALL,H5S_ALL,H5P_DEFAULT,(char*)data);

  L3M_DBG(ctxt,("HDF_Write_Array status [%d]\n",stat));

  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Get_DataDimensions(L3_Cursor_t *ctxt, hid_t nid, int *dims)
{
  int n,ndims;
  hsize_t int_dim_vals[L3_MAX_DIMS];
  hid_t did,sid;

  L3M_CLEARDIMS(dims);
  L3M_CLEARDIMS(int_dim_vals);

  did = H5Dopen(nid,L3_S_DATA,H5P_DEFAULT);
  sid = H5Dget_space(did);
  ndims = H5Sget_simple_extent_ndims(sid);
  H5Sget_simple_extent_dims(sid,int_dim_vals,NULL);

  for (n=0;n<ndims;n++){dims[n]=(int)(int_dim_vals[n]);}

  return 1;
}
/* ------------------------------------------------------------------------- */
void *HDF_Get_DataArray(L3_Cursor_t *ctxt, hid_t nid, int *dims)
{
  hid_t tid,did,yid;
  char  buff[L3_MAX_ATTRIB_SIZE+1];
  hsize_t int_dim_vals[L3_MAX_DIMS];
  int n;
  void *data=NULL;

  L3M_CHECK_CTXT_OR_DIE(ctxt,NULL);
  L3M_ECLEAR(ctxt);
  L3M_ECHECKID(nid,NULL);
  L3M_CLEARDIMS(int_dim_vals);

  if (!has_data(nid))
  {
    CHL_setError(ctxt,3020);
  }
  for (n=0; n<L3_MAX_DIMS; n++)
  {
    if (dims[n]==-1){break;}
    int_dim_vals[n]=(hsize_t)(dims[n]);
  }
  did = H5Dopen2(nid,L3_S_DATA,H5P_DEFAULT);
  tid = ADF_to_HDF_datatype(HDF_Get_Attribute_As_String(nid,L3_S_DTYPE,buff));
  yid = H5Tget_native_type(tid,H5T_DIR_ASCEND);

  data=HDF_Read_Array(ctxt,nid,did,yid,data,int_dim_vals);

  H5Tclose(yid);
  H5Tclose(tid);
  H5Dclose(did);

  return data;
}
/* ------------------------------------------------------------------------- */
int HDF_Set_DataArray(L3_Cursor_t *ctxt, hid_t nid, int *dims, void *data)
{
  hid_t tid,did,yid;
  char  buff[L3_MAX_ATTRIB_SIZE+1];
  hsize_t int_dim_vals[L3_MAX_DIMS];
  int n;

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);
  L3M_ECLEAR(ctxt);
  L3M_ECHECKID(nid,0);
  L3M_CLEARDIMS(int_dim_vals);

  if (!has_data(nid))
  {
    CHL_setError(ctxt,3021);
  }
  for (n=0; n<L3_MAX_DIMS; n++)
  {
    if (dims[n]==-1){break;}
    int_dim_vals[n]=(hsize_t)(dims[n]);
  }
  did = H5Dopen2(nid,L3_S_DATA,H5P_DEFAULT);
  tid = ADF_to_HDF_datatype(HDF_Get_Attribute_As_String(nid,L3_S_DTYPE,buff));
  yid = H5Tget_native_type(tid,H5T_DIR_ASCEND);

  HDF_Write_Array(ctxt,nid,did,yid,data,int_dim_vals);
  
  H5Tclose(yid);
  H5Tclose(tid);
  H5Dclose(did);

  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Add_DataArray(L3_Cursor_t *ctxt,hid_t nid,int *dims,void *data)
{
  hid_t tid,sid,did,yid;
  char buff[L3_MAX_ATTRIB_SIZE+1];
  char name[L3_MAX_ATTRIB_SIZE+1];
  hsize_t int_dim_vals[L3_MAX_DIMS];
  int n;

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);
  L3M_ECLEAR(ctxt);
  L3M_ECHECKID(nid,0);
  L3M_CLEARDIMS(int_dim_vals);

  HDF_Get_Attribute_As_String(nid,L3_S_NAME,name);

  for (n=0; n<L3_MAX_DIMS; n++)
  {
    if (dims[n]==-1){break;}
    int_dim_vals[n]=(hsize_t)(dims[n]);
  }
  tid = ADF_to_HDF_datatype(HDF_Get_Attribute_As_String(nid,L3_S_DTYPE,buff));
  if (tid < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_DataArray [%s] bad tid\n",name));
    return 0;
  }
  yid = H5Tget_native_type(tid,H5T_DIR_ASCEND);
  if (yid < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_DataArray [%s] bad yid\n",name));
    H5Tclose(tid);
    return 0;
  }
  sid = H5Screate_simple(n,int_dim_vals,NULL);
  if (sid < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_DataArray [%s] bad sid %d dims\n",name,n));
    H5Tclose(yid);
    H5Tclose(tid);
    return 0;
  }
  did = H5Dcreate2(nid,L3_S_DATA,tid,sid,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
  if (did < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_DataArray [%s] bad did\n",name));
    H5Tclose(sid);
    H5Tclose(yid);
    H5Tclose(tid);
    return 0;
  }

  HDF_Write_Array(ctxt,nid,did,yid,data,int_dim_vals);
  
  H5Dclose(did);
  H5Sclose(sid);
  H5Tclose(yid);
  H5Tclose(tid);

  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Add_Attribute_As_Integer(L3_Cursor_t *ctxt,
				 hid_t nodeid, const char *name, int value)
{
  hid_t sid, aid;
  hsize_t dim;
  herr_t status;

  L3M_DBG(ctxt,("HDF_Add_Attribute_As_Integer [%s][%d]\n",name,value));
  dim = 1;
  sid = H5Screate_simple(1, &dim, NULL);
  if (sid < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_Integer [%s] bad sid\n",name));
    return 0;
  }

  aid = H5Acreate(nodeid, name, H5T_NATIVE_INT, sid, H5P_DEFAULT, H5P_DEFAULT);
  if (aid < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_Integer [%s] create attribute failed\n",name));
    H5Sclose(sid);
    return 0;
  }

  status = H5Awrite(aid, H5T_NATIVE_INT, &value);

  H5Aclose(aid);
  H5Sclose(sid);

  if (status < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_Integer [%s] write attribute failed\n",name));
    return 0;
  }
  return 1;

}

/* ------------------------------------------------------------------------- */
int HDF_Add_Attribute_As_String(L3_Cursor_t *ctxt,
				hid_t nodeid, const char *name, 
				const char *value)
{
  hid_t sid,tid,aid;
  herr_t status;
  hsize_t dim;
  char buff[L3_MAX_ATTRIB_SIZE+1];

  L3M_DBG(ctxt,("HDF_Add_Attribute_As_String [%s][%s]\n",name,value));
  if (!strcmp(name,L3_S_DTYPE))
  {
    dim=(hsize_t)(L3_MAX_DTYPE+1);
  }
  else
  {
    dim=(hsize_t)(L3_MAX_ATTRIB_SIZE+1);
  }
  sid = H5Screate(H5S_SCALAR);
  if (sid < 0)
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_String [%s] bad sid\n",name));
    return 0;
  }
  tid = H5Tcopy(H5T_C_S1);
  if (tid < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_String [%s] bad tid\n",name));
    H5Sclose(sid);
    return 0;
  }
  if (H5Tset_size(tid,dim)<0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_String [%s] bad data size\n",name));
    H5Tclose(tid);
    H5Sclose(sid);
    return 0;
  }
  aid = H5Acreate(nodeid,name,tid,sid,H5P_DEFAULT,H5P_DEFAULT);
  if (aid < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_String [%s] create attribute failed\n",name));
    H5Tclose(tid);
    H5Sclose(sid);
    return 0;
  }
  memset(buff, 0, dim);
  strcpy(buff, value);
  status = H5Awrite(aid, tid, buff);

  H5Aclose(aid);
  H5Tclose(tid);
  H5Sclose(sid);

  if (status < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_String [%s] write attribute failed\n",name));
    return 0;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
static int HDF_Set_Attribute_As_Integer(L3_Cursor_t *ctxt,
					hid_t nodeid,const char *name,
					int value)
{
  hid_t aid;
  herr_t status;

  L3M_DBG(ctxt,("HDF_Set_Attribute_As_Integer: [%s]=[%d]\n",name,value));
  aid=H5Aopen_name(nodeid,name);
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
				       hid_t nodeid,const char *name,
				       const char *value)
{
  hid_t aid,tid;
  herr_t status;

  L3M_DBG(ctxt,("HDF_Set_Attribute_As_String: [%s]=[%s]\n",name,value));
  aid=H5Aopen_name(nodeid,name);
  if (aid < 0) 
  {
    return 0;
  }
  tid=H5Aget_type(aid);
  if (tid < 0) 
  {
    H5Aclose(aid);
    return 0;
  }
  status=H5Awrite(aid,tid,value);
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
				     const char *value,int size)
{
  hid_t sid,did;
  hsize_t dim;
  herr_t status;

  L3M_DBG(ctxt,("HDF_Add_Attribute_As_Data [%s][%s]\n",name,value));
  dim = (hsize_t)(size+1);
  sid = H5Screate_simple(1,&dim,NULL);
  if (sid < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_Data [%s] bad sid\n",name));
    return 0;
  }
  did=H5Dcreate2(id,name,H5T_NATIVE_CHAR,sid,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
  if (did < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_Data [%s] create data failed\n",name));
    H5Sclose(sid);
    return 0;
  }
  status=H5Dwrite(did,H5T_NATIVE_CHAR,H5S_ALL,H5S_ALL,H5P_DEFAULT,value);
  H5Dclose(did);
  H5Sclose(sid);

  if (status < 0) 
  {
    L3M_DBG(ctxt,("HDF_Add_Attribute_As_Data [%s] write data failed\n",name));
    return 0;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
int HDF_Create_Root(L3_Cursor_t *ctxt)
{
  hid_t gid;
  char  svalue[64];

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);

  gid = H5Gopen2(ctxt->file_id, "/", H5P_DEFAULT);
  if (gid < 0)
  {
    CHL_setError(ctxt,3022);
  }

  /* Root node has specific attributes (not used in this package) */
  if (!HDF_Add_Attribute_As_String(ctxt,gid,L3_S_NAME,L3_S_ROOTNODENAME))
  {
    CHL_setError(ctxt,3023);
  }
  if (!HDF_Add_Attribute_As_String(ctxt,gid,L3_S_LABEL,L3_S_ROOTNODETYPE))
  {
    CHL_setError(ctxt,3024);
  }
  if (!HDF_Add_Attribute_As_String(ctxt,gid,L3_S_DTYPE,"MT"))
  {
    CHL_setError(ctxt,3025);
  }
  strcpy(svalue,"NATIVE"); 
  if (!HDF_Add_Attribute_As_Data(ctxt,gid,L3_S_FORMAT,svalue,strlen(svalue)))
  {
    CHL_setError(ctxt,3026);
  }
  strcpy(svalue,CHLONE_CGNS_LIBRARY_VERSION); 
  if (!HDF_Add_Attribute_As_Data(ctxt,gid,L3_S_VERSION,svalue,strlen(svalue)))
  {
    CHL_setError(ctxt,3027);
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
hid_t L3_nodeCreate(L3_Cursor_t *ctxt, hid_t pid, L3_Node_t *node)
{
  hid_t nid=-1;

  L3M_CHECK_CTXT_OR_DIE(ctxt,-1);
  L3M_ECLEAR(ctxt);
  L3M_ECHECKID(pid,-1);
  L3M_ECHECKL3NODE(node,-1);

  L3M_TRACE(ctxt,("nodeCreate: [%s][%s]\n",node->name,node->label));
  nid=H5Gcreate2(pid,node->name,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
  node->id=nid;
  if (nid==-1)
  {
    CHL_setError(ctxt,3030,node->name);
  }
  if (!HDF_Add_Attribute_As_String(ctxt,nid,L3_S_NAME,node->name))  
  {
    CHL_setError(ctxt,3031,node->name);
  }
  if (!HDF_Add_Attribute_As_String(ctxt,nid,L3_S_LABEL,node->label))
  {
    CHL_setError(ctxt,3032);
  }
  if (!HDF_Add_Attribute_As_Integer(ctxt,nid,L3_S_FLAGS,node->flags))
  {
    CHL_setError(ctxt,3036);
  }
  if (!HDF_Add_Attribute_As_String(ctxt,nid,L3_S_DTYPE,L3_typeAsStr(node->dtype)))
  {
    CHL_setError(ctxt,3033);
  }
  /* return nid at this point, allow function embedding */
  if (node->data != NULL)
  {
    if (!HDF_Add_DataArray(ctxt,nid,node->dims,node->data))
    {
      CHL_setError(ctxt,3034);
    }
  }
  else
  {
    if (node->dtype!=L3E_VOID)
    {
      CHL_setError(ctxt,3035);
    }
  }
  return nid;
}
/* ------------------------------------------------------------------------- */
hid_t L3_nodeUpdate(L3_Cursor_t *ctxt, L3_Node_t *node)
{
  hid_t nid;
  char oldname[L3_MAX_ATTRIB_SIZE+1];

  L3M_CHECK_CTXT_OR_DIE(ctxt,-1);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt,("L3_nodeUpdate\n"));
  L3M_ECHECKL3NODE(node,-1);

  nid=node->id;
  HDF_Get_Attribute_As_String(nid,L3_S_NAME,oldname);

  /* you should not have such a link hid_t here... 
     retrieve follows the links 
     and gives the actual hid_t of the target node 
  */
  if (is_link(nid)) 
  {
    CHL_setError(ctxt,3050);
  }

  if (!HDF_Check_Node(nid))
  {
    CHL_setError(ctxt,3051);
  }

  if (node->name != NULL)
  {
    if (!HDF_Set_Attribute_As_String(ctxt,nid,L3_S_NAME,node->name))
    {
      CHL_setError(ctxt,3052,oldname);
    }
    H5Gmove2(nid,oldname,nid,node->name);
  }

  if (node->label != NULL)
  {
    if (!HDF_Set_Attribute_As_String(ctxt,nid,L3_S_LABEL,node->label))
    {
      CHL_setError(ctxt,3053,oldname);
    }
  }

  if (!HDF_Set_Attribute_As_Integer(ctxt,nid,L3_S_FLAGS,node->flags))
  {
    CHL_setError(ctxt,3056,oldname);
  }

  if (!HDF_Set_Attribute_As_String(ctxt,nid,L3_S_DTYPE,L3_typeAsStr(node->dtype)))
  {
    CHL_setError(ctxt,3054,oldname);
  }

  if (L3M_HASFLAG(ctxt,L3_F_WITHDATA)&&(node->data != NULL))
  {
    if (!HDF_Set_DataArray(ctxt,nid,node->dims,node->data))
    {
      CHL_setError(ctxt,3055,oldname);
    }
  }
  return nid;
}
/* ------------------------------------------------------------------------- */
L3_Cursor_t *L3_nodeLink(L3_Cursor_t *ctxt, hid_t node, 
			 char *srcname, char *destfile, char *destname)
{
  hid_t nid;

  L3M_CHECK_CTXT_OR_DIE(ctxt,NULL);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt,("L3_nodeLink\n"));
  L3M_ECHECKID(node,NULL);

  if (is_link(node))
  {
    CHL_setError(ctxt,3060);
  }
  nid=H5Gcreate2(node,srcname,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
  if (nid < 0)
  {
    CHL_setError(ctxt,3061,srcname);
  }
  if (!HDF_Add_Attribute_As_String(ctxt,nid,L3_S_NAME,srcname))  
  {
    CHL_setError(ctxt,3062,srcname);
  }
  if (!HDF_Add_Attribute_As_String(ctxt,nid,L3_S_DTYPE, L3_T_LK))  
  {
    CHL_setError(ctxt,3063,srcname);
  }
  if (!HDF_Add_Attribute_As_String(ctxt,nid,L3_S_LABEL,""))  
  {
    CHL_setError(ctxt,3064,srcname);
  }
  HDF_Add_Attribute_As_Data(ctxt,nid,L3_S_PATH,destname,strlen(destname));

  if (strcmp(destfile,""))
  {
    H5Lcreate_external(destfile,destname,nid,
		       L3_S_LINK,H5P_DEFAULT,ctxt->g_proplist);
    HDF_Add_Attribute_As_Data(ctxt,nid,L3_S_FILE,destfile,strlen(destfile));
  }
  else
  {
    H5Glink(nid, H5G_LINK_SOFT,destname,L3_S_LINK);
  }
  return ctxt;
}
/* ------------------------------------------------------------------------- */
L3_Cursor_t *L3_nodeMove(L3_Cursor_t *ctxt, 
			hid_t pid, hid_t nid, 
			char *oldname, char *newname)
{
  hid_t tid;
  char opn[L3_MAX_ATTRIB_SIZE+1];
  char npn[L3_MAX_ATTRIB_SIZE+1];

  L3M_CHECK_CTXT_OR_DIE(ctxt,NULL);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt,("L3_nodeMove\n"));
  L3M_ECHECKID(pid,ctxt);
  L3M_ECHECKID(nid,ctxt);

  if (nid==-1)      {nid=pid;}
  if (newname==NULL){newname=oldname;}
  HDF_Get_Attribute_As_String(pid,L3_S_NAME,opn);
  HDF_Get_Attribute_As_String(nid,L3_S_NAME,npn);

  if (is_link(pid) || is_link(nid))
  {
    CHL_setError(ctxt,3040);
  }
  L3M_TRACE(ctxt,("L3_nodeMove [%s][%s]->[%s][%s]\n",opn,oldname,npn,newname));
  H5Gmove2(pid, oldname, nid, newname);
  tid=H5Gopen2(nid, newname, H5P_DEFAULT);
  if (!HDF_Set_Attribute_As_String(ctxt,tid,L3_S_NAME,newname))
  {
    CHL_setError(ctxt,3041);
  }  
  return ctxt;
}
/* ------------------------------------------------------------------------- */
L3_Cursor_t *L3_nodeDelete(L3_Cursor_t *ctxt,hid_t pid,char *nodename)
{
  L3M_CHECK_CTXT_OR_DIE(ctxt,NULL);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt,("L3_nodeDelete\n"));
  L3M_ECHECKID(pid,ctxt);

  /* do not change link id with actual here, stop deletion at link node */
  if (!HDF_Check_Node(pid))
  {
    CHL_setError(ctxt,3070);
  }

  if (has_child(pid,nodename))
  {
    /* do not change link id with actual here, stop deletion at link node */
    H5Giterate(pid, nodename, NULL, delete_children, NULL);
    H5Gunlink(pid, nodename);
  }
  return ctxt;
}
/* ------------------------------------------------------------------------- */
hid_t L3_nodeFind(L3_Cursor_t *ctxt, hid_t parent, char *path)
{
  hid_t rid=-1;

  L3M_CHECK_CTXT_OR_DIE(ctxt,-1);
  L3M_ECLEAR(ctxt);
  L3M_TRACE(ctxt,("L3_nodeFind\n"));
  L3M_ECHECKID(parent,-1);

  if (path == NULL) {return -1;}
  if (strlen(path)>(L3_MAX_PATH-1))
  {
    CHL_setError(ctxt,3080,(int)strlen(path));
  }
  if (!HDF_Check_Node(parent))
  {
    CHL_setError(ctxt,3081);
  }
  if (strlen(path)==0) {return parent;}

  rid=H5Gopen2(parent,path,H5P_DEFAULT);

  return rid;
}
/* ------------------------------------------------------------------------- */
L3_Node_t *L3_nodeRetrieve(L3_Cursor_t *ctxt, hid_t oid)
{
  hid_t nid;
  char  buff[L3_MAX_ATTRIB_SIZE+1];
  int   dims[L3_MAX_DIMS];
  L3_Node_t *node;
  int ibuff;

  L3M_CHECK_CTXT_OR_DIE(ctxt,NULL);
  L3M_TRACE(ctxt,("L3_nodeRetrieve\n"));
  L3M_ECLEAR(ctxt);
  L3M_ECHECKID(oid,NULL);
  L3M_NEWNODE(node); // node allocated here: macro call, not a function !
  L3M_CLEARDIMS(dims);

  if (is_link(oid))
  {
    nid=get_link_actual_id(oid);
  }
  else
  {
    nid=oid;
  }
  if (!HDF_Check_Node(nid))
  {
    CHL_setError(ctxt,3090);
  }

  L3_N_setName(node,HDF_Get_Attribute_As_String(nid,L3_S_NAME,buff));
  L3_N_setLabel(node,HDF_Get_Attribute_As_String(nid,L3_S_LABEL,buff));
  L3_N_setDtype(node,L3_typeAsEnum(HDF_Get_Attribute_As_String(nid,L3_S_DTYPE,
							       buff)));
  L3_N_setFlags(node,HDF_Get_Attribute_As_Integer(nid,L3_S_FLAGS,&ibuff));
  node->id=nid;
  if (    (L3M_HASFLAG(ctxt,L3_F_WITHDATA)) 
       && (strcmp(HDF_Get_Attribute_As_String(nid,L3_S_DTYPE,buff),L3_T_MT)))
  {
    HDF_Get_DataDimensions(ctxt,nid,dims);
    L3_N_setDims(node,dims);
    if (dims[0]!=-1)
    {
      L3_N_setData(node,HDF_Get_DataArray(ctxt,nid,dims));
    }
  }

  if (   L3M_HASFLAG(ctxt,L3_F_WITHCHILDREN) 
      || L3M_HASFLAG(ctxt,L3_F_WITHLINKINFO))
  {
    node->children=HDF_Get_Children(nid);
  }
  return node;
}
/* ------------------------------------------------------------------------- */
L3_Cursor_t*
L3_openHID(hid_t root)
{
  return NULL;
}
/* ------------------------------------------------------------------------- */
L3_Node_t *L3_nodeSet(L3_Cursor_t *ctxt, L3_Node_t *node,
		     char *name, char *label, 
		     int  *dims, int dtype, void *data,int flags)
{
  L3M_CHECK_CTXT_OR_DIE(ctxt,NULL);
  L3M_ECLEAR(ctxt);
  if (node==NULL)
  {
    L3M_NEWNODE(node); // watch out, local modif should be used through return
  }
  L3_N_setName(node,name);
  L3_N_setLabel(node,label);
  L3_N_setDtype(node,dtype);
  L3_N_setDims(node,dims);
  L3_N_setData(node,data);
  L3_N_setFlags(node,flags);

  return node;
}
/* ------------------------------------------------------------------------- */
L3_Node_t *L3_nodeGet(L3_Cursor_t *ctxt, L3_Node_t *node,
		     char *name, char *label, 
		     int  *dims, int *dtype, void *data, int* flags)
{
  L3M_CHECK_CTXT_OR_DIE(ctxt,NULL);
  L3M_ECLEAR(ctxt);
  if (node!=NULL)
  {
    L3_N_getName(node,name);
    L3_N_getLabel(node,label);
    L3_N_getDtype(node,dtype);
    L3_N_getDims(node,dims);
    L3_N_getData(node,data);
    L3_N_getFlags(node,*flags);
  }

  return node;
}
/* ------------------------------------------------------------------------- */
void L3_nodePrint(L3_Node_t *node)
{
  if (node!=NULL)
  {
    printf("Node [%s][%s][%s]\n",
	   node->name,node->label,L3_typeAsStr(node->dtype));
    fflush(stdout);
  }
}
/* ------------------------------------------------------------------------- */
L3_Cursor_t*
L3_openFile(char *filename, int mode, long flags)
{
  L3_Cursor_t *ctxt;
  hid_t fapl;

  ctxt = (L3_Cursor_t*) malloc(sizeof(L3_Cursor_t));
  if (ctxt == NULL) 
  {
    return NULL;
  }
  L3M_ECLEAR(ctxt);
  H5Eset_auto2(H5E_DEFAULT,HDF_Walk_Error,ctxt); 
/*   ctxt->stack_id=H5Eget_current_stack(); */
  fapl = H5Pcreate(H5P_FILE_ACCESS);
  if (fapl < 0)
  {
    CHL_setError(ctxt,3011);
  }
  H5Pset_fclose_degree(fapl, H5F_CLOSE_STRONG);

  ctxt->file_id=-1;
  ctxt->root_id=-1;
  ctxt->stack_id=-1;
  ctxt->config=flags;
  ctxt->g_proplist=H5Pcreate(H5P_LINK_ACCESS);
  ctxt->ebuff[0]='\0';
  ctxt->result=NULL;
  if (ctxt->g_proplist < 0)
  {
    CHL_setError(ctxt,3012);
  }
  H5Pset_nlinks(ctxt->g_proplist, L3_MAX_LINK_DEPTH);

  switch (mode)
  {
  case L3_OPEN_NEW:
    L3M_TRACE(ctxt,("newL3_Cursor_t open new\n"));
    ctxt->file_id=H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (ctxt->file_id < 0) 
    {
      H5Pclose(fapl);
      CHL_setError(ctxt,3002,filename);
    }
    if (!HDF_Create_Root(ctxt))
    {
      
    }
    break;
  case L3_OPEN_OLD:
    L3M_TRACE(ctxt,("newL3_Cursor_t open old\n"));
    ctxt->file_id = H5Fopen(filename, H5F_ACC_RDWR, fapl);
    if (ctxt->file_id < 0) 
    {
      H5Pclose(fapl);
      CHL_setError(ctxt,3003,filename);
    }
   break;
  case L3_OPEN_RDO:
  default:
    L3M_TRACE(ctxt,("newL3_Cursor_t read only\n"));
    ctxt->file_id = H5Fopen(filename, H5F_ACC_RDONLY, fapl);
    if (ctxt->file_id < 0) 
    {
      H5Pclose(fapl);
      CHL_setError(ctxt,3004,filename);
    }
    break;
  }
  ctxt->root_id=H5Gopen2(ctxt->file_id, "/", H5P_DEFAULT);
  H5Pclose(fapl);

#ifdef CHLONE_HAS_PTHREAD
  int nmtx;
  L3M_TRACE(ctxt,("Mutex init\n"));
  pthread_mutex_init(&ctxt->g_mutex,NULL);
  for (nmtx=0;nmtx<L3_MAX_MUTEX;nmtx++)
  {
    pthread_mutex_init(&ctxt->n_mutex[nmtx],NULL);    
  }
#endif  
  return ctxt;
}
/* ------------------------------------------------------------------------- */
int L3_close(L3_Cursor_t *ctxt)
{
  L3M_CHECK_CTXT_OR_DIE(ctxt,0);
  L3M_TRACE(ctxt,("Close\n"));

  if (ctxt->file_id != -1)
  {
    H5Pclose(ctxt->g_proplist);
    H5Gclose(ctxt->root_id);
    H5Fclose(ctxt->file_id);
/*     H5Eclose_stack(ctxt->stack_id); */
    ctxt->file_id=-1;
    ctxt->root_id=-1;
#ifdef CHLONE_HAS_PTHREAD
  int nmtx;
  L3M_TRACE(ctxt,("Mutex destroy\n"));
  pthread_mutex_destroy(&ctxt->g_mutex);
  for (nmtx=0;nmtx<L3_MAX_MUTEX;nmtx++)
  {
    pthread_mutex_destroy(&ctxt->n_mutex[nmtx]);    
  }
#endif  
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
int L3_isLinkNode  (L3_Cursor_t *ctxt,hid_t id,char *file,char *name)
{
  int r=0;
  const char *bfile=NULL, *bpath=NULL;

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);
  L3M_ECLEAR(ctxt);
  if ((file == NULL) || (name == NULL))
  {
    return 0; /* oups!... */
  }
  r=get_link_data(id,bfile,bpath);
  if (r && (bfile!=NULL) && (bpath!=NULL))
  {
    strcpy(file,bfile);
    strcpy(name,bpath);
  }
  return r;
}

/* ------------------------------------------------------------------------- */
int L3_isSameNode(L3_Cursor_t *ctxt,hid_t id1,hid_t id2)
{
  H5O_info_t info1,info2;

  H5Oget_info(id1,&info1);
  H5Oget_info(id2,&info2);

  if (info1.fileno!=info2.fileno) return 0;
  if (info1.type!=info2.type)     return 0;
  if (info1.addr!=info2.addr)     return 0;

  return 1;
}
/* ------------------------------------------------------------------------- */
int *L3_initDims(int *dims, int d1, ...)
{
  va_list intptr;
  int n;

  va_start(intptr,d1);
  L3M_CLEARDIMS(dims);
  dims[0]=d1;
  n=1;  
  while (dims[n-1]!=-1)
  {
    dims[n++]=va_arg(intptr,int);
  }
  va_end(intptr);
  return dims;
}
/* ------------------------------------------------------------------------- */
void  *L3_initData(int  *dims, void *data, int dtype, ...)
{
  int dsize,n,d;
  va_list stack;

  if (dims==NULL) {return NULL;}
  dsize=1;
  n=0;
  va_start(stack, dtype);

  while ((n<L3_MAX_DIMS)&&(dims[n]!=-1))
  {
    dsize*=dims[n++];
  }
  switch(dtype)
  {
    case L3E_I4:
    {
      data=(void*)malloc(dsize*sizeof(int));
      n=0;
      while (n<dsize)
      {
	d=va_arg(stack,int );
	((int*)data)[n++]=d;
      }      
      break;
    }
    default: break;
  }
  va_end(stack);
  return data;
}
/* ------------------------------------------------------------------------- */
void  *L3_fillData  (int  *dims, void *data, int dtype, ...)
{
  int dsize,n;
  int d_i;
  double d_d;
  float d_f;
  char d_c;
  va_list stack;

  if (dims==NULL) {return NULL;}
  dsize=1;
  n=0;
  va_start(stack, dtype);

  while ((n<L3_MAX_DIMS)&&(dims[n]!=-1))
  {
    dsize*=dims[n++];
  }
  switch(dtype)
  {
    case L3E_C1:
    {
      data=(void*)malloc(dsize);
      d_c=va_arg(stack,unsigned int );
      n=0;while (n<dsize){((char*)data)[n++]=d_c;}      
      break;
    }
    case L3E_I4:
    {
      data=(void*)malloc(dsize*sizeof(int));
      d_i=va_arg(stack,int );
      n=0;while (n<dsize){((int*)data)[n++]=d_i;}      
      break;
    } 
    case L3E_R8:
    {
      data=(void*)malloc(dsize*sizeof(double));
      d_d=va_arg(stack,double );
      n=0;while (n<dsize){((double *)data)[n++]=d_d;}      
      break;
    }
    case L3E_R4:
    {
      data=(void*)malloc(dsize*sizeof(float));
      d_f=va_arg(stack,double );
      n=0;while (n<dsize){((float *)data)[n++]=d_f;}      
      break;
    }
    default: break;
  }
  va_end(stack);
  return data;
}
/* ------------------------------------------------------------------------- */
char *L3_node2Path(L3_Cursor_t *ctxt, hid_t id)
{
  int nsize;
  char trybuff[1],*path;

  nsize=H5Iget_name(id,trybuff,1);
  path=(char*)malloc(nsize+1);
  nsize=H5Iget_name(id,path,nsize+1);

  return path;
}
/* ------------------------------------------------------------------------- */
hid_t L3_path2Node(L3_Cursor_t *ctxt, char *path)
{
  return H5Gopen2(ctxt->root_id,path,H5P_DEFAULT);
}
/* ------------------------------------------------------------------------- */
void L3_printError(L3_Cursor_t *ctxt)
{
  if (ctxt==NULL)
  {
    printf("# CHLone error: [NULL context]\n");
  }
  else
  {
    printf("# CHLone error: [%s]\n",ctxt->ebuff);
  }
}
/* ------------------------------------------------------------------------- */
char *L3_typeAsStr(int dtype)
{
  switch(dtype)
  {
    case L3E_C1:
    case L3E_C1ptr: return L3_T_C1_s;
    case L3E_I4:
    case L3E_I4ptr: return L3_T_I4_s;
    case L3E_I8:
    case L3E_I8ptr: return L3_T_I8_s;
    case L3E_R4:
    case L3E_R4ptr: return L3_T_R4_s;
    case L3E_R8:
    case L3E_R8ptr: return L3_T_R8_s;
    case L3E_LK:    return L3_T_LK_s;
    case L3E_B1:    return L3_T_B1_s;
    case L3E_U4:    return L3_T_U4_s;
    case L3E_U8:    return L3_T_U8_s;
    case L3E_VOID:
    default:        return L3_T_MT_s;
  }
}
/* ------------------------------------------------------------------------- */
int L3_typeAsEnum(char *dtype)
{
  if(!strcmp(dtype,L3_T_C1_s)){return L3E_C1ptr;}
  if(!strcmp(dtype,L3_T_I4_s)){return L3E_I4ptr;}
  if(!strcmp(dtype,L3_T_I8_s)){return L3E_I8ptr;}
  if(!strcmp(dtype,L3_T_R4_s)){return L3E_R4ptr;}
  if(!strcmp(dtype,L3_T_R8_s)){return L3E_R8ptr;}
  if(!strcmp(dtype,L3_T_LK_s)){return L3E_LK;}
  if(!strcmp(dtype,L3_T_B1_s)){return L3E_B1;}
  if(!strcmp(dtype,L3_T_U4_s)){return L3E_U4;}
  if(!strcmp(dtype,L3_T_U8_s)){return L3E_U8;}
  return L3E_VOID;
}
/* ------------------------------------------------------------------------- */
int L3_config(int p)
{
  return 0;
}
/* --- last line */
