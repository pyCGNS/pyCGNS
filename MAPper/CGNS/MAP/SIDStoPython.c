/* 
#  -------------------------------------------------------------------------
#  pyCGNS.MAP - Python package for CFD General Notation System - MAPper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  ------------------------------------------------------------------------- 
*/
#include "SIDStoPython.h"
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

static char *DT_MT="MT";
static char *DT_I4="I4";
static char *DT_I8="I8";
static char *DT_R4="R4";
static char *DT_R8="R8";
static char *DT_C1="C1";

/* ------------------------------------------------------------------------- */
static L3_Cursor_t *s2p_addoneHDF(char *filename,s2p_ctx_t *context)
{
  L3_Cursor_t *l3db;
  int openfile;
  s2p_ent_t *nextdbs,*prevdbs;

  l3db=NULL;
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
	l3db=nextdbs->l3db;
	openfile=0;
	S2P_TRACE(("### open [%s] (already opened)\n",filename));
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
    /* L3_F_OWNDATA|L3_F_WITHCHILDREN */
    nextdbs->l3db=L3_openFile(filename,L3E_OPEN_OLD,L3F_DEFAULT);
    nextdbs->filename=(char*)malloc(sizeof(char)*strlen(filename)+1);
    strcpy(nextdbs->filename,filename);
    nextdbs->dirname=NULL;
    nextdbs->next=NULL;
    l3db=nextdbs->l3db;
    S2P_TRACE(("### open [%s]\n",filename));
  }
  return l3db;
}
/* ------------------------------------------------------------------------- */
static void s2p_closeallHDF(s2p_ctx_t *context)
{
  s2p_ent_t *nextdbs,*dbs;

  dbs=context->dbs;
  if (dbs!=NULL)
  {
    while (dbs->next!=NULL)
    {
      L3_close(dbs->l3db);
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
    /* No check on list contents here, it's far easier at the Python level
       and it would lead to something too hairy here. */
    sz=PyString_Size(PySequence_GetItem(PyList_GetItem(linktable,n),0));
    st=PyString_AsString(PySequence_GetItem(PyList_GetItem(linktable,n),0));
    nextlink->targetdirname=(char*)malloc(sizeof(char)*sz+1);
    strcpy(nextlink->targetdirname,st);

    sz=PyString_Size(PySequence_GetItem(PyList_GetItem(linktable,n),1));
    st=PyString_AsString(PySequence_GetItem(PyList_GetItem(linktable,n),1));
    nextlink->targetfilename=(char*)malloc(sizeof(char)*sz+1);
    strcpy(nextlink->targetfilename,st);

    sz=PyString_Size(PySequence_GetItem(PyList_GetItem(linktable,n),2));
    st=PyString_AsString(PySequence_GetItem(PyList_GetItem(linktable,n),2));
    nextlink->targetnodename=(char*)malloc(sizeof(char)*sz+1);
    strcpy(nextlink->targetnodename,st);

    sz=PyString_Size(PySequence_GetItem(PyList_GetItem(linktable,n),3));
    st=PyString_AsString(PySequence_GetItem(PyList_GetItem(linktable,n),3));
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
      free(links->targetdirname);
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
    lk=Py_BuildValue("[ssss]",
    		       links->targetdirname,
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
                          char 	     *destdir,
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
  sz=strlen(destdir);
  curlink->targetdirname=(char*)malloc(sizeof(char)*sz+1);
  sz=strlen(destfile);
  curlink->targetfilename=(char*)malloc(sizeof(char)*sz+1);
  sz=strlen(destnode);
  curlink->targetnodename=(char*)malloc(sizeof(char)*sz+1);
  sz=strlen(curpath);
  curlink->localnodename=(char*)malloc(sizeof(char)*sz+1);
  curlink->next=NULL;
  strcpy(curlink->targetdirname,destdir);
  strcpy(curlink->targetfilename,destfile);
  strcpy(curlink->targetnodename,destnode);
  strcpy(curlink->localnodename,curpath);
}
/* ------------------------------------------------------------------------- */
static hid_t s2p_linktrack(L3_Cursor_t *l3db,
			    char        *nodename,
			    s2p_ctx_t   *context)
{
  char curpath[MAXPATHSIZE];
  char name[L3C_MAX_NAME+1];
  char destnode[L3C_MAX_NAME+1];
  char destdir[MAXFILENAMESIZE];
  char destfile[MAXFILENAMESIZE];
  int parsepath,c,ix;
  hid_t nid;
  char *p;
  L3_Cursor_t *lkhdfdb;

  parsepath=1;
  p=nodename;
  nid=-1;
  curpath[0]='\0';

  while (parsepath)
  {
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
    nid=L3_path2Node(l3db,curpath);
    if ((nid != -1) && L3_isLinkNode(l3db,nid,destfile,destnode))
    {
      /* L3 layer has its own link search path, we have to ask it
         the directory used to open the actual file. */
      ix=CHL_getFileInSearchPath(l3db,destfile);
      strcpy(destdir,CHL_getLinkSearchPath(l3db,ix));
      S2P_TRACE(("\n### Link Follow link: [%s][%s][%s]\n",
		 destdir,destfile,destnode));
      s2p_linkstack(curpath,destdir,destfile,destnode,context);
      lkhdfdb=s2p_addoneHDF(destfile,context);
      nid=s2p_linktrack(lkhdfdb,destnode,context);
    } 
    if (nid==-1) {break;}
  }
  return nid;
}

/* ------------------------------------------------------------------------- */
/* Fill the link path table for the context. The searchpath should be a 
   single string where paths are separated by : (classical way for paths) 
*/
static int s2p_setlinksearchpath(L3_Cursor_t *l3db,
				 char *searchpath,
				 s2p_ctx_t   *context)
{
  char *ptr,*path,*rpath;
  int parse;

  if (! searchpath)
  {
    return 1;
  }
  rpath=(char*)malloc(sizeof(char)*strlen(searchpath)+1);
  strcpy(rpath,searchpath);
  path=rpath;
  parse=1;
  while (parse)
  {
    ptr=path;
    while ((ptr[0]!=':') && (ptr[0]!='\0'))
    {
      ptr++;
    }
    if (ptr[0]=='\0')
    {
      parse=0;
    }
    ptr[0]='\0';
    S2P_TRACE(("### Add search path :[%s]\n",path));
    CHL_addLinkSearchPath(l3db,path);
    ptr++;
    path=ptr;
  }
  free(rpath);
  return 1;
}

/* ------------------------------------------------------------------------- */
static int s2p_trustlink(char *file, char *name)
{
  return 1;
}
/* ------------------------------------------------------------------------- */
static PyObject* s2p_getObjectByPath(PyObject* updict, char *path)
{
  PyObject *ret=NULL;

  if ((updict !=NULL) && PyDict_Check(updict))
  {
    ret=PyDict_GetItemString(updict,path);
  }
  return ret;
}

/* ------------------------------------------------------------------------- */
static int s2p_getData(PyObject *dobject, 
		       char **dtype, int *ddims, int *dshape, char **dvalue,
		       int isdataarray, s2p_ctx_t  *context)
{
  int n,total;

/*   if (   (!PyArray_ISFORTRAN(dobject)) */
/*       && (PyArray_NDIM(dobject)>1) */
/*       && (PyArray_NDIM(dobject)<MAXDIMENSIONVALUES)) */
/*   { */
/*     S2P_TRACE(("\n ERROR: ARRAY SHOULD BE FORTRAN\n")); */
/*     return 0; */
/*   } */

  ddims[0]=0;
  *dtype=DT_MT;
  *dvalue=NULL;

  L3M_CLEARDIMS(dshape);

  if (PyArray_Check(dobject))
  {
     ddims[0]=PyArray_NDIM(dobject);
     total=1;
     for (n=0; n<ddims[0]; n++)
     {
     	if (S2P_HASFLAG(S2P_FREVERSEDIMS) || !isdataarray)
     	{ 
     	  dshape[n]=(int)PyArray_DIM(dobject,ddims[0]-n-1);
     	  total*=dshape[ddims[0]-n-1];
     	}
     	else
     	{
     	  dshape[n]=(int)PyArray_DIM(dobject,n);
     	  total*=dshape[n];
     	}
     } 
     if (isdataarray)
     {
       dobject=(PyObject*)(PyArray_Transpose((PyArrayObject*)dobject,NULL));
     }
     *dvalue=(char*)PyArray_DATA(dobject);
  }

  /* --- Integer */
  if (PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_INT))
  {
    *dtype=DT_I4;
    return 1;
  }
  /* --- Long */
  if (PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_LONG))
  {
    *dtype=DT_I8;
    return 1;
  }
  /* --- Double */
  if ((PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_DOUBLE)))
  {
    *dtype=DT_R8;
    return 1;
  }
  /* --- Float */
  if ((PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_FLOAT)))
  {
    *dtype=DT_R4;
    return 1;
  }
  /* --- String */
  if ((PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_STRING)))
  {
    *dtype=DT_C1;
     return 1;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static PyObject* s2p_parseAndReadHDF(hid_t    	  id,
                                     char      	 *name,
                                     char        *curpath,    
                                     char        *path,    
                                     s2p_ctx_t   *context,
				     L3_Cursor_t *l3db)
{
  char      localnode[MAXPATHSIZE];
  char      destnode[L3C_MAX_NAME+1];
  char      destdir[MAXFILENAMESIZE];
  char      destfile[MAXFILENAMESIZE];
  int       ndim,tsize,n,child,arraytype,psize,trackpath,ix,skipnewarray;
  hid_t     actualid;
  PyObject *o_clist,*o_value,*o_child,*o_node,*u_value;
  npy_intp  npy_dim_vals[MAXDIMENSIONVALUES];
  L3_Cursor_t *lkl3db;
  L3_Node_t *rnode,*cnode;

  destnode[0]='\0';
  destdir[0]='\0';
  destfile[0]='\0';
  context->dpt-=1;
  trackpath=1;
  if (    (path==NULL) 
      || ((path!=NULL) && (path[0]=='\0'))
      || ((path!=NULL) && (!strcmp(path,"/"))))
  {
    trackpath=0;
  }
  else
  {
    S2P_TRACE(("### Target path (%s)\n",path));
  }
  o_value=NULL;
  
  /* In case of path, we are in a link search or sub-tree retrieval. We
     skip the Python node creation but we keep track of links. */
  actualid=id;
  if (L3_isLinkNode(l3db,id,destfile,destnode))
  {
    if (S2P_HASFLAG(S2P_FFOLLOWLINKS) && s2p_trustlink(destfile,destnode))
    {
      /* L3 layer has its own link search path, we have to ask it
         the directory used to open the actual file. */
      ix=CHL_getFileInSearchPath(l3db,destfile);
      if (ix!=-1)
      {
	strcpy(destdir,CHL_getLinkSearchPath(l3db,ix));
      }
      else
      {
	S2P_TRACE(("### Linked-to file not readable (%s)\n",destfile));
      }
      S2P_TRACE(("\n### Parse Follow link: [%s][%s][%s]\n",
		 destdir,destfile,destnode));
      sprintf(localnode,"%s/%s",curpath,name);
      s2p_linkstack(localnode,destdir,destfile,destnode,context);
      /* We recurse on destination file, S2P functions should be used and
         not HDF functions, because HDF lib would hide links to S2P.
         Then we start our parse from the root node and keep track of links,
         the actual node is finally used at the end. */
      lkl3db=s2p_addoneHDF(destfile,context);
      actualid=s2p_linktrack(lkl3db,destnode,context);
      if (actualid==-1)
      {
	S2P_TRACE(("### Linked-to node (%s) does not exist\n",destnode));
	return NULL;
      }
    }
  }
  L3M_SETFLAG(l3db,L3F_WITHCHILDREN);
  L3M_UNSETFLAG(l3db,L3F_WITHDATA);
  L3M_NEWNODE(rnode);
  rnode=L3_nodeRetrieve(l3db,actualid,rnode);
  if (rnode == NULL)
  {
    S2P_TRACE(("### (%s) Retrieve returns NULL POINTER\n",curpath));
    return NULL;
  }
  strcat(curpath,"/");
  strcat(curpath,rnode->name);
  skipnewarray=0;
  if (!S2P_HASFLAG(S2P_FNODATA))
  {
    u_value=s2p_getObjectByPath(context->obj,curpath);
    if (u_value!=NULL)
    {
      rnode->data=PyArray_DATA(u_value);
      skipnewarray=1;
      printf("##### DATA FOUND\n");
    }
    L3M_SETFLAG(l3db,L3F_WITHDATA);
    rnode=L3_nodeRetrieve(l3db,actualid,rnode);
  }
  if (S2P_HASFLAG(S2P_FNODATA))
  {
    rnode->dtype=L3E_VOID;
  }
  if (path != NULL)
  {
    psize=(strlen(path)>strlen(curpath)?strlen(curpath):strlen(path));
  }
  else
  {
    psize=strlen(curpath);
  }
  if (trackpath && strcmp(curpath,"/HDF5 MotherNode"))
  {
      S2P_TRACE(("### Path filter \'%s\' \'%s\'[:%d]\n",path,curpath,psize));
      if (strncmp(path,curpath,psize))
      {
	curpath[strlen(curpath)-strlen(rnode->name)-1]='\0';
	if (rnode!=NULL){ free(rnode);}
        return NULL;
      } 
  }
  if (trackpath && !strcmp(curpath,"/HDF5 MotherNode"))
  {
    curpath[strlen(curpath)-strlen(rnode->name)-1]='\0';
  }
  if (S2P_HASFLAG(S2P_FALTERNATESIDS))
  {
    if (!strcmp(rnode->label,"\"int[1+...+IndexDimension]\""))
    {
      strcpy(rnode->label,"DiffusionModel_t");
    }
    if (!strcmp(rnode->label,"\"int[IndexDimension]\""))
    {
      strcpy(rnode->label,"Transform_t");
    }
  }
  S2P_TRACE(("### (%s) [%s]",curpath,rnode->label));
  if (!skipnewarray || (rnode->dtype!=L3E_VOID))
  {
    S2P_TRACE(("[%s]",L3_typeAsStr(rnode->dtype)));
    tsize=1;
    n=0;
    ndim=0;
    npy_dim_vals[0]=0;
    while ((n<L3C_MAX_DIMS)&&(rnode->dims[n]!=-1))
    {
      tsize*=rnode->dims[n];
      ndim++;
      n++;
    }
    n=0;
    while ((n<L3C_MAX_DIMS)&&(rnode->dims[n]!=-1))
    {
      if (    (!strcmp(rnode->label,"DataArray_t"))
           || (!strcmp(rnode->label,"DimensionalUnits_t"))
           || (!strcmp(rnode->label,"AdditionalUnits_t"))
           || (!strcmp(rnode->label,"DimensionalExponents_t"))
           || (!strcmp(rnode->label,"AdditionalExponents_t")) 
           || (S2P_HASFLAG(S2P_FREVERSEDIMS)) )
      {
	npy_dim_vals[ndim-n-1]=rnode->dims[n];
      }
      else
      {
	npy_dim_vals[n]=rnode->dims[n];
      }
      n++;
    } 
    S2P_TRACE(("{")); 
    for (n=0;n<ndim;n++)
    {
      S2P_TRACE(("%d",(int)(npy_dim_vals[n])));
      if (n<ndim-1)
      {
	S2P_TRACE(("x"));
      }
    } 
    S2P_TRACE(("}=%d",tsize));
    if ((rnode->dtype==L3E_I4) || (rnode->dtype==L3E_I4ptr))
    {
      arraytype=NPY_INT;
    }
    else if ((rnode->dtype==L3E_C1) || (rnode->dtype==L3E_C1ptr))
    {
      arraytype=NPY_CHAR;
    }
    else if ((rnode->dtype==L3E_R8) || (rnode->dtype==L3E_R8ptr))
    {
      arraytype=NPY_DOUBLE;
    }
    else if ((rnode->dtype==L3E_I8) || (rnode->dtype==L3E_I8ptr))
    {
      arraytype=NPY_LONG;
    }
    else if ((rnode->dtype==L3E_R4) || (rnode->dtype==L3E_R4ptr))
    {
      arraytype=NPY_FLOAT;
    }
    else
    {
      arraytype=-1;
    }
    if (arraytype!=-1)
    {
      o_value=(PyObject*)PyArray_New(&PyArray_Type,
				     ndim,npy_dim_vals, 
				     arraytype,(npy_intp *)NULL,
				     (void*)rnode->data,0, 
				     NPY_OWNDATA|NPY_FORTRAN,
				     (PyObject*)NULL);
      if (    (!strcmp(rnode->label,"DataArray_t"))
           || (!strcmp(rnode->label,"DimensionalUnits_t"))
           || (!strcmp(rnode->label,"AdditionalUnits_t"))
           || (!strcmp(rnode->label,"DimensionalExponents_t"))
           || (!strcmp(rnode->label,"AdditionalExponents_t")) )
      {
	o_value=(PyObject*)(PyArray_Transpose((PyArrayObject*)o_value,NULL));
      }
    }
  }
  S2P_TRACE(("\n"));
  /* Loop on children. This is a depth first recurse. In case of a path search,
     skip until we have the right name. */
  o_clist=PyList_New(0);
  child=0;
  L3M_NEWNODE(cnode);
  while ((rnode->children != NULL) && 
         (rnode->children[child] != -1) &&
	 (context->dpt > 0)
        )
  {
    if (S2P_HASFLAG(S2P_FFOLLOWLINKS))
    {
      L3M_UNSETFLAG(l3db,L3F_FOLLOWLINKS);
    }
    cnode=L3_nodeRetrieve(l3db,rnode->children[child],cnode);
    /* HDF can parse paths, i.e. a node name can be a path and the
       resulting ID is the actual last node. However, we SHOULD not use that
       because we want to have control on link parse. */
    if (!strcmp(curpath,"/HDF5 MotherNode"))
    {
      curpath[0]='\0';
    }
    o_child=s2p_parseAndReadHDF(cnode->id,cnode->name,curpath,path,
				context,l3db);
    if (S2P_HASFLAG(S2P_FFOLLOWLINKS))
    {
      L3M_SETFLAG(l3db,L3F_FOLLOWLINKS);
    }
    if (o_child != NULL)
    {
      PyList_Append(o_clist,o_child);
    }
    child++;
  }
  if (cnode!=NULL){ free(cnode);}
  if (strcmp(rnode->name,"HDF5 MotherNode"))
  {
    curpath[strlen(curpath)-strlen(rnode->name)-1]='\0';
  }
  if (o_value==NULL)
  {
    Py_INCREF(Py_None);
    o_value=Py_None;
  };
  o_node=Py_BuildValue("[sOOs]",name,o_value,o_clist,rnode->label);
  context->dpt+=1;

  if (rnode!=NULL){ free(rnode);}

  return o_node;
}
/* ------------------------------------------------------------------------- */
static int s2p_parseAndWriteHDF(hid_t     id,
                                PyObject  *tree,
                                char      *curpath,
                                char      *path,
                                s2p_ctx_t *context,
				L3_Cursor_t *l3db)
{
  char *name=NULL,*label=NULL,*tdat=NULL,altlabel[L3C_MAX_NAME+1];
  int sz=0,n=0,ret=0,tsize=1,isdataarray=0;
  int ndat=0,ddat[NPY_MAXDIMS];
  char *vdat=NULL;
  L3_Node_t *node=NULL;

  if (    (PyList_Check(tree))
       && (PyList_Size(tree) == 4)
       && (PyString_Check(PyList_GetItem(tree,0)))
       && (PyString_Check(PyList_GetItem(tree,3))))
  {
    name=PyString_AsString(PyList_GetItem(tree,0));
    label=PyString_AsString(PyList_GetItem(tree,3));
    strcpy(altlabel,label);
    if (!strcmp(altlabel,"DiffusionModel_t"))
    {
      strcpy(altlabel,"\"int[1+...+IndexDimension]\"");
    }
    if (!strcmp(altlabel,"Transform_t"))
    {
      strcpy(altlabel,"\"int[IndexDimension]\"");
    }
    strcat(curpath,"/");
    strcat(curpath,name);
    if (S2P_HASFLAG(S2P_FREVERSEDIMS))
    { 
      S2P_TRACE(("### create (swap dims) [%s][%s]",curpath,altlabel));
    }
    else
    { 
      S2P_TRACE(("### create [%s][%s]",curpath,altlabel));
    }
    if (s2p_checklinktable(context,curpath))
    {
      S2P_TRACE(("### linked to [%s][%s]\n",curpath,altlabel));
    }
    S2P_FREECONTEXTPTR(context);
    if (    (!strcmp(altlabel,"DataArray_t"))
         || (!strcmp(altlabel,"DimensionalUnits_t"))
         || (!strcmp(altlabel,"AdditionalUnits_t"))
         || (!strcmp(altlabel,"DimensionalExponents_t"))
         || (!strcmp(altlabel,"AdditionalExponents_t")) )
    {
      isdataarray=1;
    }
    if (s2p_getData(PyList_GetItem(tree,1),&tdat,&ndat,ddat,&vdat,
		    isdataarray,context))
    {
    }
    n=0;
    tsize=1;
    S2P_TRACE(("{"));
    while ((n<L3C_MAX_DIMS)&&(ddat[n]!=-1))
    {
      tsize*=ddat[n];
      S2P_TRACE(("%d",ddat[n]));
      n++;
      if ((n<L3C_MAX_DIMS)&&(ddat[n]!=-1))
      {
	S2P_TRACE(("x"));
      }
    } 
    S2P_TRACE(("}=%d\n",tsize));
    node=L3_nodeSet(l3db,node,name,altlabel,ddat,
		    L3_typeAsEnum(tdat),vdat,L3F_NONE);
    L3_nodeCreate(l3db,id,node);
    if (PyList_Check(PyList_GetItem(tree,2)))
    {
      sz=PyList_Size(PyList_GetItem(tree,2));
      for (n=0;n<sz;n++)
      {
        ret=s2p_parseAndWriteHDF(node->id,
				 PyList_GetItem(PyList_GetItem(tree,2),n),
                                 curpath,path,context,l3db); 
      } 
    }
    curpath[strlen(curpath)-strlen(name)-1]='\0';
  }
  return ret;
}
/* ------------------------------------------------------------------------- */
/* Interface Functions                                                       */
/* ------------------------------------------------------------------------- */
PyObject* s2p_loadAsHDF(char *filename,
                        int   flags,
                        int   depth,
                        char *path,
			char *searchpath,
                        PyObject *update)
{
  char cpath[MAXPATHSIZE];
  PyObject *tree,*ret,*links;
  s2p_ctx_t *context;
  L3_Cursor_t *l3db;
  L3_Node_t   *rnode;

  context=(s2p_ctx_t*)malloc(sizeof(s2p_ctx_t));
  context->flg=flags;
  context->lnk=NULL;
  context->dbs=NULL;
  context->dpt=depth;
  context->obj=update;
  context->_c_float =NULL;
  context->_c_double=NULL;
  context->_c_int=NULL;
  context->_c_char=NULL;
  cpath[0]='\0';

  /* We do NOT check file name or file access, it's up to the caller to make
     such checks. Anyway, HDF will check. */
  S2P_TRACE(("### SIDS-to-python v%d.%d\n",
	     SIDSTOPYTHON_MAJOR,SIDSTOPYTHON_MINOR));
  S2P_TRACE(("### load file [%s]\n",filename));

  l3db=s2p_addoneHDF(filename,context);
  s2p_setlinksearchpath(l3db,searchpath,context);
  L3M_NEWNODE(rnode);
  rnode=L3_nodeRetrieve(l3db,l3db->root_id,rnode);
  ret=s2p_parseAndReadHDF(l3db->root_id,rnode->name,cpath,path,context,l3db);
  links=s2p_getlinktable(context);
  s2p_freelinktable(context);

  if (ret==NULL)
  {
    ret=Py_None;
    tree=Py_BuildValue("([sOOs]O)",
		       CGNSTree_n,Py_None,ret,CGNSTree_ts,links);
  }
  else
  {
    tree=Py_BuildValue("([sOOs]O)",
		       CGNSTree_n,Py_None,PyList_GetItem(ret,2),CGNSTree_ts,
		       links);
  }
  s2p_closeallHDF(context);
  Py_INCREF(Py_None);
  if (rnode!=NULL) { free(rnode); }
  S2P_FREECONTEXTPTR(context);
  free(context);

  return tree;
}
/* ------------------------------------------------------------------------- */
int s2p_saveAsHDF(char      *filename,
                  PyObject  *tree,
                  PyObject  *links,
                  int        flags,
                  int        depth,
                  char*      path)    
{
  int ret=0,sz=-1;
  char cpath[MAXPATHSIZE],*tdat=NULL;
  s2p_ctx_t *context=NULL;
  PyObject *rtree,*otree=NULL;
  int dims[L3C_MAX_DIMS];
  int ndat=0,ddat[NPY_MAXDIMS],n=0;
  char *vdat=NULL;
  L3_Cursor_t *l3db=NULL;
  L3_Node_t *node=NULL;

  context=(s2p_ctx_t*)malloc(sizeof(s2p_ctx_t));
  context->flg=flags;
  context->lnk=NULL;
  context->dbs=NULL;
  context->dpt=depth;
  context->_c_float =NULL;
  context->_c_int=NULL;
  context->_c_double=NULL;
  context->_c_char=NULL;
  cpath[0]='\0';

  S2P_TRACE(("### SIDS-to-python v%d.%d\n",
	     SIDSTOPYTHON_MAJOR,SIDSTOPYTHON_MINOR));
  S2P_TRACE(("### save in file [%s]\n",filename));
  if (    (PyList_Check(tree))
       && (PyList_Size(tree) == 4)
       && (PyString_Check(PyList_GetItem(tree,0)))
       && (PyList_Check(PyList_GetItem(tree,2)))
       && (PyString_Check(PyList_GetItem(tree,3))))
  {
    s2p_filllinktable(links,context);
    l3db=L3_openFile(filename,L3E_OPEN_NEW,L3F_DEFAULT);
    if (!L3M_ECHECK(l3db))
    {
      CHL_printError(l3db);
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
		       CGNSLibraryVersion_ts))
	{
	  S2P_TRACE(("### create [CGNSLibraryVersion]\n"));
	  S2P_FREECONTEXTPTR(context);
	  s2p_getData(PyList_GetItem(otree,1),&tdat,&ndat,ddat,&vdat,
		      0,context);
	  L3_initDims(dims,1,-1);
	  node=L3_nodeSet(l3db,node,CGNSLibraryVersion_n,CGNSLibraryVersion_ts,
			  dims,L3E_R4,vdat,L3F_NONE);
	  L3_nodeCreate(l3db,l3db->root_id,node);
	}
	else
	{
	  ret=s2p_parseAndWriteHDF(l3db->root_id,otree,
				   cpath,path,context,l3db);
	}
      } 
    }
    L3_close(l3db);
    s2p_freelinktable(context);
    S2P_FREECONTEXTPTR(context);
    free(context);
  }

  return ret;
}
/* ------------------------------------------------------------------------- */

