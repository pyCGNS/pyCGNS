/* ------------------------------------------------------------------------- */
/* pyCGNS - CFD General Notation System - SIDS-to-Python MAPping             */
/* $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $          */
/* See license file in the root directory of this Python module source       */
/* ------------------------------------------------------------------------- */
#include "SIDStoPython.h"
#include "ADF.h"
#include "numpy/arrayobject.h"
/* ------------------------------------------------------------------------- */
#define MAXCHILDRENCOUNT   1024
#define MAXLINKNAMESIZE    1024
#define MAXPATHSIZE        1024
#define MAXFILENAMESIZE    1024
#define MAXDATATYPESIZE    32
#define MAXDIMENSIONVALUES 12
#define MAXFORMATSIZE      20
#define MAXERRORSIZE       80
#define MAXVERSIONSIZE     32

#define S2P_CLEARCONTEXTPTR( ctxt ) \

#define S2P_FREECONTEXTPTR( ctxt ) \
if (ctxt->_c_float !=NULL){free(ctxt->_c_float);};\
if (ctxt->_c_double!=NULL){free(ctxt->_c_double);};\
if (ctxt->_c_int   !=NULL){free(ctxt->_c_int);};\
ctxt->_c_float =NULL;\
ctxt->_c_int   =NULL;\
ctxt->_c_double=NULL;\
ctxt->_c_char  =NULL;

#define S2P_HASFLAG( flag ) ((context->flg & flag) == flag)
#define S2P_SETFLAG( flag ) ( context->flg |=  flag)
#define S2P_CLRFLAG( flag ) ( context->flg &= ~flag)

#define S2P_TRACE( txt ) \
if (S2P_HASFLAG(S2P_FTRACE)){printf txt ;fflush(stdout);}

#define S2P_TRACE__( txt ) \
if (S2P_HASFLAG(S2P_FTRACE)){ txt ;}

static char *DT_MT="MT";
static char *DT_I4="I4";
static char *DT_I8="I8";
static char *DT_R4="R4";
static char *DT_R8="R8";
static char *DT_C1="C1";

/* ------------------------------------------------------------------------- */
static double s2p_addoneADF(char *filename,s2p_ctx_t *context)
{
  double root_id;
  int openfile,error;
  s2p_ent_t *nextdbs,*prevdbs;

  root_id=0;
  openfile=1;
  nextdbs=context->dbs;
  if (nextdbs==NULL)
  {
    nextdbs=(s2p_ent_t*)malloc(sizeof(s2p_ent_t));
    context->dbs=nextdbs;
  }
  else
  {
    prevdbs=context->dbs;
    while (nextdbs!=NULL)
    {
      if (!strcmp(nextdbs->filename,filename))
      {
	root_id=nextdbs->root_id;
	openfile=0;
	break;
      }
      prevdbs=nextdbs;
      nextdbs=nextdbs->next;
    }
    if (openfile)
    {
      prevdbs->next=(s2p_ent_t*)malloc(sizeof(s2p_ent_t));
      nextdbs=prevdbs->next;
    }
  }
  if (openfile)
  {
    ADF_Database_Open(filename,"OLD","NATIVE",&root_id,&error);
    nextdbs->filename=(char*)malloc(sizeof(char)*strlen(filename));
    strcpy(nextdbs->filename,filename);
    nextdbs->dirname=NULL;
    nextdbs->root_id=root_id;
    nextdbs->next=NULL;
  }
  return root_id;
}
/* ------------------------------------------------------------------------- */
static void s2p_closeallADF(s2p_ctx_t *context)
{
  s2p_ent_t *nextdbs,*dbs;
  int error;

  dbs=context->dbs;
  if (dbs!=NULL)
  {
    while (dbs->next!=NULL)
    {
      ADF_Database_Close(dbs->root_id,&error);
      if (dbs->filename!=NULL){ free(dbs->filename); }
      if (dbs->dirname!=NULL){ free(dbs->dirname); }
      nextdbs=dbs->next;
      free(dbs);
      dbs=nextdbs;
    }
  }
  context->dbs=NULL;
}
/* ------------------------------------------------------------------------- */
static s2p_ctx_t *s2p_filllinktable(PyObject *linktable, s2p_ctx_t *context)
{
  int linktablesize,n,sz;
  char *st;
  s2p_lnk_t *nextlink;

  if ((linktable == NULL) || (!PyList_Check(linktable))){ return NULL; }
  linktablesize=PyList_Size(linktable);
  if (!linktablesize) { return NULL; }
  nextlink=context->lnk;
  if (nextlink == NULL)
  {
    nextlink=(s2p_lnk_t*)malloc(sizeof(s2p_lnk_t));
    context->lnk=nextlink;
  }
  else
  {
    while (nextlink->next != NULL )
    {
      nextlink=nextlink->next;
    }
  }
  for (n=0;n<linktablesize;n++)
  {
    /* Remark 05:
       No check on list contents here, it's far easier at the Python level
       and it would lead to something too hairy here. */
    sz=PyString_Size(PySequence_GetItem(PyList_GetItem(linktable,n),0));
    st=PyString_AsString(PySequence_GetItem(PyList_GetItem(linktable,n),0));
    nextlink->targetfilename=(char*)malloc(sizeof(char)*sz+1);
    strcpy(nextlink->targetfilename,st);
    sz=PyString_Size(PySequence_GetItem(PyList_GetItem(linktable,n),1));
    st=PyString_AsString(PySequence_GetItem(PyList_GetItem(linktable,n),1));
    nextlink->targetnodename=(char*)malloc(sizeof(char)*sz+1);
    strcpy(nextlink->targetnodename,st);
    sz=PyString_Size(PySequence_GetItem(PyList_GetItem(linktable,n),2));
    st=PyString_AsString(PySequence_GetItem(PyList_GetItem(linktable,n),2));
    nextlink->localnodename=(char*)malloc(sizeof(char)*sz+1);
    strcpy(nextlink->localnodename,st);
    nextlink->next=NULL;
  }         
  return context;
}
/* ------------------------------------------------------------------------- */
static void s2p_freelinktable(s2p_ctx_t *context)
{
  s2p_lnk_t *nextlink,*links;

  links=context->lnk;
  if (links!=NULL)
  {
    while (links->next!=NULL)
    {
      free(links->targetfilename);
      free(links->targetnodename);
      free(links->localnodename);
      nextlink=links->next;
      free(links);
      links=nextlink;
    }
  }
  context->lnk=NULL;
}
/* ------------------------------------------------------------------------- */
static int s2p_checklinktable(s2p_ctx_t *context,char *nodename)
{
  s2p_lnk_t *links;

  links=context->lnk;
  if (links!=NULL)
  {
    while (links->next!=NULL)
    {
      if (!strcmp(links->localnodename,nodename))
      {
	return 1;
      }
      links=links->next;
    }
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static PyObject *s2p_getlinktable(s2p_ctx_t *context)
{
  PyObject *rt,*lk;
  s2p_lnk_t *links;

  rt=PyList_New(0);
  links=context->lnk;
  while (links!=NULL)
  {
    lk=Py_BuildValue("[sss]",
    		       links->targetfilename,
    		       links->targetnodename,
    		       links->localnodename);
    PyList_Append(rt,lk);
    links=links->next;
  }
  return rt;
}
/* ------------------------------------------------------------------------- */
static void s2p_linkstack(char 	     *curpath,
                          char 	     *destfile,
			  char 	     *destnode,
                          s2p_ctx_t  *context)
{
  s2p_lnk_t *nextlink,*curlink;
  int sz;

  nextlink=context->lnk;
  if (nextlink!=NULL)
  {
    while (nextlink->next != NULL )
    {
      nextlink=nextlink->next;
    }
    nextlink->next=(s2p_lnk_t*)malloc(sizeof(s2p_lnk_t));
    curlink=nextlink->next;
  }
  else
  {
    context->lnk=(s2p_lnk_t*)malloc(sizeof(s2p_lnk_t));
    curlink=context->lnk;
  }
  sz=strlen(destfile);
  curlink->targetfilename=(char*)malloc(sizeof(char)*sz+1);
  sz=strlen(destnode);
  curlink->targetnodename=(char*)malloc(sizeof(char)*sz+1);
  sz=strlen(curpath);
  curlink->localnodename=(char*)malloc(sizeof(char)*sz+1);
  curlink->next=NULL;
  strcpy(curlink->targetfilename,destfile);
  strcpy(curlink->targetnodename,destnode);
  strcpy(curlink->localnodename,curpath);
}
/* ------------------------------------------------------------------------- */
static double s2p_linktrack(double      rootid,
			    char       *nodename,
			    s2p_ctx_t  *context)
{
  char curpath[MAXPATHSIZE];
  char name[ADF_NAME_LENGTH+1];
  char destnode[ADF_NAME_LENGTH+1];
  char destfile[MAXFILENAMESIZE];
  int error,pathlength,parsepath,c;
  double id,nid,lkrootid;
  char *p;

  parsepath=1;
  p=nodename;
  nid=rootid;
  curpath[0]='\0';

  while (parsepath)
  {
    id=nid;
    while ((*p!='\0')&&(*p!='/')) { p++; }
    if (*p=='\0'){ break; }
    p++;
    c=0;
    while ((*p!='\0')&&(*p!='/')) 
    { 
      name[c]=*p; 
      p++;
      c++;
    }
    name[c]='\0';
    strcat(curpath,"/");
    strcat(curpath,name);

    /*    printf("linktrack [%s][%s]\n",curpath,name);fflush(stdout); */
    ADF_Get_Node_ID(id,name,&nid,&error);
    ADF_Is_Link(nid,&pathlength,&error);
    if (pathlength)
    {
      ADF_Get_Link_Path(nid,destfile,destnode,&error);
      S2P_TRACE(("\n### Link Follow link: [%s][%s]",destfile,destnode));

      s2p_linkstack(curpath,destfile,destnode,context);
      lkrootid=s2p_addoneADF(destfile,context);
      nid=s2p_linktrack(lkrootid,destnode,context);
    } 
  }
  return nid;
}
/* ------------------------------------------------------------------------- */
static int s2p_trustlink(char *file, char *name)
{
  return 1;
}
/* ------------------------------------------------------------------------- */
#define TRANSP_LOOP( nnn ) level-=1; for (l[nnn]=0;l[nnn]<dims[nnn];l[nnn]++)
#define TRANSP_LEV2( ttt ) \
(( ttt *)tdata)[l[0]+l[1]*dims[0]]=(( ttt *)ddata)[l[1]+l[0]*dims[1]];
#define TRANSP_LEV3( ttt ) \
(( ttt *)tdata)[l[0]+l[1]*dims[0]+l[2]*dims[0]*dims[1]]=(( ttt *)ddata)[l[2]+l[1]*dims[2]+l[0]*dims[2]*dims[1]];
#define TRANSP_LEV4( ttt ) \
(( ttt *)tdata)[l[0]+l[1]*dims[0]+l[2]*dims[0]*dims[1]+l[3]*dims[0]*dims[1]*dims[2]]=(( ttt *)ddata)[l[3]+l[2]*dims[3]+l[1]*dims[2]*dims[3]+l[0]*dims[2]*dims[1]*dims[3]];
#define TRANSP_LEV5( ttt ) \
(( ttt *)tdata)[l[0]+l[1]*dims[0]+l[2]*dims[0]*dims[1]+l[3]*dims[0]*dims[1]*dims[2]+l[4]*dims[0]*dims[1]*dims[2]*dims[3]]=(( ttt *)ddata)[l[4]+l[3]*dims[4]+l[2]*dims[3]*dims[4]+l[1]*dims[2]*dims[3]*dims[4]+l[0]*dims[2]*dims[1]*dims[3]*dims[4]];
#define TRANSP_BLOC( tttt ) \
TRANSP_LOOP(0){TRANSP_LOOP(1){if (!level){TRANSP_LEV2( tttt );}\
else{TRANSP_LOOP(2){if (!level){TRANSP_LEV3( tttt );}\
else{TRANSP_LOOP(3){if (!level){TRANSP_LEV4( tttt );}\
else{TRANSP_LOOP(4){TRANSP_LEV5( tttt );}\
level+=1;}}level+=1;}}level+=1;}}level+=1;}
/* ------------------------------------------------------------------------- */
static char *s2p_transpose(PyObject *dobject, 
			   int ptype, int ndims, int *dims, int total, 
			   s2p_ctx_t  *context)
{
  char *ddata,*tdata=NULL;
  int  l[MAXDIMENSIONVALUES],level,didx,tidx;

  ddata=(char*)PyArray_DATA(dobject);
  level=ndims;

  if (ndims==1)
  {
    return ddata;
  }
  if (ptype==NPY_STRING)
  {
    context->_c_char=(char*)malloc(total*sizeof(char));
    tdata=(char*)context->_c_char;
    TRANSP_BLOC( char );
  }
  if (ptype==NPY_DOUBLE)
  {
    context->_c_double=(double*)malloc(total*sizeof(double));
    tdata=(char*)context->_c_double;
    TRANSP_BLOC( double );
  }
  if (ptype==NPY_FLOAT)
  {
    context->_c_float=(float*)malloc(total*sizeof(float));
    tdata=context->_c_float;
    TRANSP_BLOC( float );
  }
  if (ptype==NPY_LONG)
  {
    context->_c_long=(long*)malloc(total*sizeof(long));
    tdata=context->_c_long;
    TRANSP_BLOC( long );
  }
  if (ptype==NPY_INT)
  {
    context->_c_int=(int*)malloc(total*sizeof(int));
    tdata=context->_c_int;
    TRANSP_BLOC( int );
  }

  if (tdata!=NULL){ return tdata;}
  else            { return ddata;}
}
/* ------------------------------------------------------------------------- */
static int s2p_getData(PyObject *dobject, 
		       char **dtype, int *ddims, int *dshape, char **dvalue,
		       s2p_ctx_t  *context)
{
  int n,total;

  *dtype=DT_MT;
  ddims[0]=0;
  *dvalue=NULL;

  /* --- Integer */
  if (   (PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_INT))
      || (PyLong_Check(dobject)))
  {
    *dtype=DT_I4;
    if (PyLong_Check(dobject))
    {
      ddims[0]=1;
      dshape[0]=1;
      context->_c_int=(int*)malloc(sizeof(int));
      *context->_c_int=(int)PyLong_AsLong(dobject);
      *dvalue=(char*)&context->_c_int;
    }
    else
    {
      ddims[0]=PyArray_NDIM(dobject);
      for (n=0; n<ddims[0]; n++)
      {
        dshape[n]=(int)PyArray_DIM(dobject,n);
      } 
      *dvalue=(char*)PyArray_DATA(dobject);
    }
    return 1;
  }
  /* --- Long */
  if (PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_LONG))
  {
    *dtype=DT_I8;
    ddims[0]=PyArray_NDIM(dobject);
    for (n=0; n<ddims[0]; n++)
    {
      dshape[n]=(int)PyArray_DIM(dobject,n);
    } 
    *dvalue=(char*)PyArray_DATA(dobject);
    return 1;
  }
  /* --- Double */
  if (   (PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_DOUBLE))
      || (PyFloat_Check(dobject)))
  {
    if (PyFloat_Check(dobject))
    {
      ddims[0]=1;
      dshape[0]=1;
      context->_c_double=(double*)malloc(sizeof(double));
      *context->_c_double=(double)PyFloat_AsDouble(dobject);
      *dvalue=(char*)&context->_c_double;
    }
    else
    {
      *dtype=DT_R8;
      ddims[0]=PyArray_NDIM(dobject);
      for (n=0; n<ddims[0]; n++)
      {
        dshape[n]=(int)PyArray_DIM(dobject,n);
      } 
      total=1;
      for (n=0; n<ddims[0]; n++)
      {
        dshape[n]=(int)PyArray_DIM(dobject,n);
        total*=dshape[n];
      } 
      if (!PyArray_ISFORTRAN(dobject))
      {
        *dvalue=s2p_transpose(dobject,
			      NPY_DOUBLE,ddims[0],dshape,total,context);
      }
      else
      {
        *dvalue=(char*)PyArray_DATA(dobject);
      } 
    }
    return 1;
  }
  /* --- Float */
  if ((PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_FLOAT)))
  {
    *dtype=DT_R4;
    ddims[0]=PyArray_NDIM(dobject);
    for (n=0; n<ddims[0]; n++)
    {
      dshape[n]=(int)PyArray_DIM(dobject,n);
    } 
    *dvalue=(char*)PyArray_DATA(dobject);
    return 1;
  }
  /* --- String */
  if ((PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_STRING)))
  {
    *dtype=DT_C1;
    if (PyString_Check(dobject))
    {
      ddims[0]=1;
      dshape[0]=strlen((char*)PyString_AsString(dobject));
      context->_c_char=(char*)PyString_AsString(dobject);
      *dvalue=(char*)&context->_c_char;
    }
    else
    {
      ddims[0]=PyArray_NDIM(dobject);
      total=1;
      for (n=0; n<ddims[0]; n++)
      {
        dshape[n]=(int)PyArray_DIM(dobject,n);
	total*=dshape[n];
      } 
      if (PyArray_ISFORTRAN(dobject))
      {
	*dvalue=s2p_transpose(dobject,
			      NPY_STRING,ddims[0],dshape,total,context);
      }
      else
      {
        *dvalue=(char*)PyArray_DATA(dobject);
      }
    }
    return 1;
  }
}
/* ------------------------------------------------------------------------- */
static PyObject* s2p_parseAndReadADF(double    	 id,
                                     char      	*name,
                                     char*     	 curpath,    
                                     char*     	 path,    
                                     s2p_ctx_t  *context)
{
  char      label[ADF_NAME_LENGTH+1];
  char      childname[ADF_NAME_LENGTH+1];
  char      destnode[ADF_NAME_LENGTH+1];
  char      destfile[MAXFILENAMESIZE];
  char      datatype[MAXDATATYPESIZE];
  int       ndim,tsize,n,pathlength,count,child,rcount;
  int       dim_vals[MAXDIMENSIONVALUES];
  int       error,isdataarray,arraytype,toknpath;
  double    childid,actualid,lkrootid;
  PyObject *o_clist,*o_value,*o_child,*o_node;
  npy_intp  npy_dim_vals[MAXDIMENSIONVALUES];
  void     *data;
  char     *nextpath;

  context->dpt-=1;

  o_value=NULL;
  data=NULL;
  count=0;
  error=-1;
  
  /* In case of path, we are in a link search or sub-tree retrieval. We
     skip the Python node creation but we keep track of links. */
  ADF_Get_Label(id,label,&error);
  S2P_TRACE(("### %s [%s]",curpath,label));
  if (    !strcmp(label,"DataArray_t") 
       || !strcmp(label,"ExponentUnits_t") 
       || !strcmp(label,"DimensionalUnits_t"))
  {
    isdataarray=1;
  }
  else
  {
    isdataarray=0;
  }
  ADF_Is_Link(id,&pathlength,&error);
  actualid=id;
  if (pathlength)
  {
    ADF_Get_Link_Path(id,destfile,destnode,&error);
    if (S2P_HASFLAG(S2P_FFOLLOWLINKS) && s2p_trustlink(destfile,destnode))
    {
      S2P_TRACE(("\n### Parse Follow link: [%s][%s]",destfile,destnode));
      /* Remark 01:
         We recurse on destination file, S2P functions should be used and
         not ADF function, because ADF lib would hide links to S2P.
         Then we start our parse from the root node and keep track of links,
         the actual node is finally used at the end. See also remark 02. */
      s2p_linkstack(curpath,destfile,destnode,context);
      lkrootid=s2p_addoneADF(destfile,context);
      actualid=s2p_linktrack(lkrootid,destnode,context);
    }
  }
  ADF_Get_Data_Type(actualid,datatype,&error);
  if (strcmp(datatype,"MT"))
  {
    ADF_Get_Number_of_Dimensions(actualid,&ndim,&error);
    ADF_Get_Dimension_Values(actualid,dim_vals,&error);
    tsize=1;
    S2P_TRACE(("[%s]",datatype));
    for (n=0; n<ndim; n++)
    {
      tsize*=dim_vals[n];
      npy_dim_vals[n]=dim_vals[n];
        S2P_TRACE(("[%d]",dim_vals[n]));
    } 
    S2P_TRACE(("=%d",tsize));
    if (!strcmp(datatype,"I4"))
    {
      arraytype=NPY_INT;
      data=(void *)malloc((tsize)*sizeof(int));
    }
    else if (!strcmp(datatype,"C1"))
    {
      arraytype=NPY_CHAR;
      data=(void *)malloc((tsize)*sizeof(char));
      memset(data,0,tsize+1);
    }
    else if (!strcmp(datatype,"R8"))
    {
      arraytype=NPY_DOUBLE;
      data=(void *)malloc((tsize)*sizeof(double));
    }
    else if (!strcmp(datatype,"I8"))
    {
      arraytype=NPY_LONG;
      data=(void *)malloc((tsize)*sizeof(long));
    }
    else if (!strcmp(datatype,"R4"))
    {
      arraytype=NPY_FLOAT;
      data=(void *)malloc((tsize)*sizeof(float));
    }
    else
    {
      data=NULL;
    }
    if (data!=NULL)
    {
      ADF_Read_All_Data(actualid,(char*)data,&error);
      if (isdataarray)
      {
        o_value=(PyObject*)PyArray_New(&PyArray_Type,
                                       ndim,npy_dim_vals, 
                                       arraytype,(npy_intp*)NULL,
                                       (void*)data,0, 
                                       NPY_OWNDATA|NPY_FORTRAN,
                                       (PyObject*)NULL);
      }
      else
      {
        o_value=(PyObject*)PyArray_New(&PyArray_Type,
                                       ndim,npy_dim_vals, 
                                       arraytype,(npy_intp *)NULL,
                                       (void*)data,0, 
                                       NPY_OWNDATA,
                                       (PyObject*)NULL);
      }
    }
  }
  S2P_TRACE(("\n"));
  /* Loop on children. This is a depth first recurse. In case of a path search,
     skip until we have the right name. */
  ADF_Number_of_Children(actualid,&count,&error);
  o_clist=PyList_New(0);
  nextpath=path;
  for (child=1;child<=count;child++)
  {
    /* Remark 03:
       ADF API uses node ids and the list of chidren returns the names, not the
       ids. So we avoid a call to Get_Node_Name: we pass the child name as a
       recurse function parameter. */
    ADF_Children_Names(actualid,child,1,ADF_NAME_LENGTH+1,
		       &rcount,childname,&error);
    strcat(curpath,"/");
    strcat(curpath,childname);
    if ((path!=NULL) && (strlen(path)!=0))
    {
      nextpath=strchr(path,'/');
      if (nextpath)
      {
        toknpath=(int)(nextpath-path); /* should work for chars */
        S2P_TRACE(("### Path compare [%s] to [%s] up to [%d]\n",
                   path+1,childname,toknpath));
        if (!strncmp(path+1,childname,toknpath))
        {
          S2P_TRACE(("### Path compare OK\n"));
        } 
      }
    }
    /* Remark 02:
       ADF_Get_Node_ID can parse paths, i.e. a node name can be a path and the
       resulting ID is the actual last node. However, we SHOULD not use that
       because we want to have control on link parse. */
    ADF_Get_Node_ID(actualid,childname,&childid,&error);
    o_child=s2p_parseAndReadADF(childid,childname,curpath,nextpath,context);
    curpath[strlen(curpath)-strlen(childname)-1]='\0';
    PyList_Append(o_clist,o_child);
  }
  if (o_value==NULL)
  {
    Py_INCREF(Py_None);
    o_value=Py_None;
  };
  o_node=Py_BuildValue("[sOOs]",name,o_value,o_clist,label);

  return o_node;
}
/* ------------------------------------------------------------------------- */
static int s2p_parseAndWriteADF(double     id,
                                PyObject  *tree,
                                char      *curpath,
                                char      *path,
                                s2p_ctx_t *context)
{
  char *name,*label,*tdat=NULL;
  int sz,n,error,ret=0;
  double nid;
  int ndat=0,ddat[NPY_MAXDIMS];
  char *vdat=NULL;

  if (    (PyList_Check(tree))
       && (PyList_Size(tree) == 4)
       && (PyString_Check(PyList_GetItem(tree,0)))
       && (PyString_Check(PyList_GetItem(tree,3))))
  {
    name=PyString_AsString(PyList_GetItem(tree,0));
    label=PyString_AsString(PyList_GetItem(tree,3));
    strcat(curpath,"/");
    strcat(curpath,name);
    S2P_TRACE(("### create [%s][%s]\n",curpath,label));
    if (s2p_checklinktable(context,curpath))
    {
      S2P_TRACE(("### linked to [%s][%s]\n",curpath,label));
    }
    S2P_FREECONTEXTPTR(context);
    s2p_getData(PyList_GetItem(tree,1),&tdat,&ndat,ddat,&vdat,context);
    ADF_Create(id,name,&nid,&error);
    printf("ERR %d\n",error);
    ADF_Set_Label(nid,label,&error);
    printf("ERR %d\n",error);
    ADF_Put_Dimension_Information(nid,tdat,ndat,ddat,&error);
    printf("ERR %d\n",error);
    ADF_Write_All_Data(nid,(char*)vdat,&error);
    printf("ERR %d\n",error);
    if (PyList_Check(PyList_GetItem(tree,2)))
    {
      sz=PyList_Size(PyList_GetItem(tree,2));
      for (n=0;n<sz;n++)
      {
        ret=s2p_parseAndWriteADF(nid,PyList_GetItem(PyList_GetItem(tree,2),n),
                                 curpath,path,context); 
      } 
    }
    curpath[strlen(curpath)-strlen(name)-1]='\0';
  }
  return ret;
}
/* ------------------------------------------------------------------------- */
/* Interface Functions                                                       */
/* ------------------------------------------------------------------------- */
PyObject* s2p_loadAsADF(char *filename,
                        int   flags,
                        int   threshold,
                        int   depth,
                        char *path)
{
  double root_id;
  int error;
  char name[ADF_NAME_LENGTH+1];
  char cpath[MAXPATHSIZE];
  PyObject *tree,*ret,*links;
  s2p_ctx_t *context;

  context=(s2p_ctx_t*)malloc(sizeof(s2p_ctx_t));
  context->flg=flags;
  context->lnk=NULL;
  context->dbs=NULL;
  context->thh=threshold;
  context->dpt=depth;
  context->_c_float =NULL;
  context->_c_double=NULL;
  context->_c_int=NULL;
  context->_c_char=NULL;
  name[0]='\0';
  cpath[0]='\0';

  /* Remark 04:
     We do NOT check file name or file access, it's up to the caller to make
     such checks. */
  root_id=s2p_addoneADF(filename,context);
  ADF_Get_Name(root_id,name,&error);
  ret=s2p_parseAndReadADF(root_id,name,cpath,path,context);
  links=s2p_getlinktable(context);
  s2p_freelinktable(context);

  tree=Py_BuildValue("([sOOs]O)",
		     "CGNSTree",Py_None,PyList_GetItem(ret,2),"CGNSTree_t",
		     links);
  s2p_closeallADF(context);
  Py_INCREF(Py_None);
  S2P_FREECONTEXTPTR(context);
  free(context);

  return tree;
}
/* ------------------------------------------------------------------------- */
int s2p_saveAsADF(char      *filename,
                  PyObject  *tree,
                  PyObject  *links,
                  int        flags,
                  int        threshold,
                  int        depth,
                  char*      path)    
{
  double root_id,nid,n;
  int error,ret=0,sz;
  char cpath[MAXPATHSIZE],*tdat=NULL;
  s2p_ctx_t *context;
  PyObject *rtree,*otree;
  int ndat=0,ddat[NPY_MAXDIMS];
  char *vdat=NULL;

  context=(s2p_ctx_t*)malloc(sizeof(s2p_ctx_t));
  context->flg=flags;
  context->lnk=NULL;
  context->dbs=NULL;
  context->thh=threshold;
  context->dpt=depth;
  context->_c_float =NULL;
  context->_c_int=NULL;
  context->_c_double=NULL;
  context->_c_char=NULL;
  cpath[0]='\0';

  if (    (PyList_Check(tree))
       && (PyList_Size(tree) == 4)
       && (PyString_Check(PyList_GetItem(tree,0)))
       && (PyList_Check(PyList_GetItem(tree,2)))
       && (PyString_Check(PyList_GetItem(tree,3))))
  {
    s2p_filllinktable(links,context);
    ADF_Database_Open(filename,"NEW","NATIVE",&root_id,&error);
    if (error == REQUESTED_NEW_FILE_EXISTS)
    {
      printf("Error %d\n",error);fflush(stdout);
      return -1;
    }
    rtree=PyList_GetItem(tree,2);
    if (PyList_Check(rtree))
    {
      sz=PyList_Size(rtree);
      for (n=0;n<sz;n++)
      {
	otree=PyList_GetItem(rtree,n);
	if (   PyList_Check(otree)
            && PyList_Size(otree) == 4
	    && PyString_Check(PyList_GetItem(otree,0))
	    && PyString_Check(PyList_GetItem(otree,3))
	    && !strcmp(PyString_AsString(PyList_GetItem(otree,3)),
		       "CGNSLibraryVersion_t"))
	{
	  S2P_TRACE(("### create [CGNSLibraryVersion][CGNSLibraryVersion_t]"));
	  S2P_FREECONTEXTPTR(context);
	  s2p_getData(PyList_GetItem(otree,1),&tdat,&ndat,ddat,&vdat,context);
	  ADF_Create(root_id,"CGNSLibraryVersion",&nid,&error);
	  ADF_Set_Label(nid,"CGNSLibraryVersion_t",&error);
	  ADF_Put_Dimension_Information(nid,tdat,ndat,ddat,&error);
	  ADF_Write_All_Data(nid,(char*)vdat,&error);
	  S2P_TRACE(("### value [CGNSLibraryVersion][%.4f]",*((float*)vdat)));
	}
	else
	{
	  ret=s2p_parseAndWriteADF(root_id,otree,cpath,path,context);
	}
      } 

    }
    ADF_Database_Close(root_id,&error);
    s2p_freelinktable(context);
    S2P_FREECONTEXTPTR(context);
    free(context);
  }

  return ret;
}
/* ------------------------------------------------------------------------- */

