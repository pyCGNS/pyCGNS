/* ------------------------------------------------------------------------- */
/* pyCGNS.MAP - CFD General Notation System - SIDS-to-Python MAPping         */
/* See license.txt file in the root directory of this Python module source   */
/* ------------------------------------------------------------------------- */
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

#define S2P_TRACE__( txt ) \
if (S2P_HASFLAG(S2P_FTRACE)){ txt ;}

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
    nextdbs->l3db=L3_openFile(filename,L3_OPEN_OLD,L3_F_DEFAULT);
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
static hid_t s2p_linktrack(L3_Cursor_t *l3db,
			    char        *nodename,
			    s2p_ctx_t   *context)
{
  char curpath[MAXPATHSIZE];
  char name[L3_MAX_NAME+1];
  char destnode[L3_MAX_NAME+1];
  char destfile[MAXFILENAMESIZE];
  int parsepath,c;
  hid_t nid;
  char *p;
  L3_Cursor_t *lkhdfdb;

  parsepath=1;
  p=nodename;
  nid=l3db->root_id;
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
    if (L3_isLinkNode(l3db,nid,destfile,destnode))
    {
      S2P_TRACE(("\n### Link Follow link: [%s][%s]\n",destfile,destnode));

      s2p_linkstack(curpath,destfile,destnode,context);
      lkhdfdb=s2p_addoneHDF(destfile,context);
      nid=s2p_linktrack(lkhdfdb,destnode,context);
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
static int s2p_getData(PyObject *dobject, 
		       char **dtype, int *ddims, int *dshape, char **dvalue,
		       s2p_ctx_t  *context)
{
  int n,total;
  PyObject *ostr;

  // code modification: Return ERROR code if ARRAY is not FORTRAN

  if (!PyArray_ISFORTRAN(dobject) && PyArray_NDIM(dobject)>1 && PyArray_NDIM(dobject)<MAXDATATYPESIZE)
  {
    S2P_TRACE(("\n ERROR: ARRAY SHOULD BE FORTRAN\n"));
    // Launch error
    //    return 1;
  }

  *dtype=DT_MT;
  ddims[0]=0;
  *dvalue=NULL;

  L3M_CLEARDIMS(dshape);

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
	// code modification: check array fortran order 
	if (S2P_HASFLAG(S2P_FREVERSEDIMS))
	{ 
	  S2P_TRACE(("\n COMMENT PIJE: apply reverse L335\n"));
        dshape[ddims[0]-n-1]=(int)PyArray_DIM(dobject,n);
        total*=dshape[ddims[0]-n-1];
      } 
	else
	{
	  S2P_TRACE(("\n COMMENT PIJE: apply NO reverse L348\n"));
          return 1;
	  dshape[n]=(int)PyArray_DIM(dobject,n);
	  total*=dshape[n];
	}
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
      // code modification: check array fortran order 
      if (S2P_HASFLAG(S2P_FREVERSEDIMS))
      { 
	S2P_TRACE(("\n COMMENT PIJE: apply reverse L361\n"));
      dshape[ddims[0]-n-1]=(int)PyArray_DIM(dobject,n);
      total*=dshape[ddims[0]-n-1];
    } 
      else
      {
	S2P_TRACE(("\n COMMENT PIJE: apply NO reverse L367\n"));
	dshape[n]=(int)PyArray_DIM(dobject,n);
        total*=dshape[n];
      }
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
      total=1;
      for (n=0; n<ddims[0]; n++)
      {
	// code modification: check array fortran order 
	if (S2P_HASFLAG(S2P_FREVERSEDIMS))
        { 
	  S2P_TRACE(("\n COMMENT PIJE: apply reverse L398\n"));
        dshape[ddims[0]-n-1]=(int)PyArray_DIM(dobject,n);
        total*=dshape[ddims[0]-n-1];
      } 
        else
        {
	  S2P_TRACE(("\n COMMENT PIJE: apply NO reverse L404\n"));
	  dshape[n]=(int)PyArray_DIM(dobject,n);
          total*=dshape[n];
        }
      }
      *dvalue=(char*)PyArray_DATA(dobject);
    }
    return 1;
  }
  /* --- Float */
  if ((PyArray_Check(dobject) && (PyArray_TYPE(dobject)==NPY_FLOAT)))
  {
    *dtype=DT_R4;
    ddims[0]=PyArray_NDIM(dobject);
    /* index is the C order for numpy. Even if array is FORTRAN, you should
       pass an index with the C order, i.e. imax,ijmax,kmax...
       Then you have to reverse the size if you detect a fortran */
    for (n=0; n<ddims[0]; n++)
    {
      // code modification: check array fortran order
      if (S2P_HASFLAG(S2P_FREVERSEDIMS))
      { 
	S2P_TRACE(("\n COMMENT PIJE: apply reverse L426\n"));
      dshape[ddims[0]-n-1]=(int)PyArray_DIM(dobject,n);
        total*=dshape[ddims[0]-n-1];
      }
      else
      {
	S2P_TRACE(("\n COMMENT PIJE: apply NO reverse L432\n"));
        dshape[n]=(int)PyArray_DIM(dobject,n);
        total*=dshape[n];
      }
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
      ostr=dobject;
    }
    else
    {
      ddims[0]=PyArray_NDIM(dobject);
      total=1;
      for (n=0; n<ddims[0]; n++)
      {
	// code modification: check array fortran order 
	if (S2P_HASFLAG(S2P_FREVERSEDIMS))
	{ 
	  S2P_TRACE(("\n COMMENT PIJE: apply reverse L461\n"));
        dshape[ddims[0]-n-1]=(int)PyArray_DIM(dobject,n);
        total*=dshape[ddims[0]-n-1];
      } 
        else
        {
	  S2P_TRACE(("\n COMMENT PIJE: apply NO reverse L467\n"));
          dshape[n]=(int)PyArray_DIM(dobject,n);
          total*=dshape[n];
        }
      } 
      *dvalue=(char*)PyArray_DATA(dobject);
      ostr=PyArray_ToString((PyArrayObject*)dobject,NPY_ANYORDER);
    }
    ddims[0]=1;
    dshape[0]=strlen((char*)PyString_AsString(ostr));
    context->_c_char=(char*)malloc(dshape[0]*sizeof(char)+1);
    strcpy(context->_c_char,(char*)PyString_AsString(ostr));
    *dvalue=(char*)context->_c_char;
    return 1;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
static PyObject* s2p_parseAndReadHDF(hid_t    	  id,
                                     char      	 *name,
                                     char*     	  curpath,    
                                     char*     	  path,    
                                     s2p_ctx_t   *context,
				     L3_Cursor_t *l3db)
{
  char      destnode[L3_MAX_NAME+1];
  char      destfile[MAXFILENAMESIZE];
  int       ndim,tsize,n,child;
  int       arraytype,toknpath;
  hid_t    actualid;
  PyObject *o_clist,*o_value,*o_child,*o_node;
  npy_intp  npy_dim_vals[MAXDIMENSIONVALUES];
  char     *nextpath;
  L3_Cursor_t *lkl3db;
  L3_Node_t *rnode,*cnode;

  context->dpt-=1;

  o_value=NULL;
  
  /* In case of path, we are in a link search or sub-tree retrieval. We
     skip the Python node creation but we keep track of links. */
  actualid=id;
  if (L3_isLinkNode(l3db,id,destfile,destnode))
  {
    if (S2P_HASFLAG(S2P_FFOLLOWLINKS) && s2p_trustlink(destfile,destnode))
    {
      S2P_TRACE(("\n### Parse Follow link: [%s][%s]\n",destfile,destnode));
      /* We recurse on destination file, S2P functions should be used and
         not HDF functions, because HDF lib would hide links to S2P.
         Then we start our parse from the root node and keep track of links,
         the actual node is finally used at the end. */
      s2p_linkstack(curpath,destfile,destnode,context);
      lkl3db=s2p_addoneHDF(destfile,context);
      actualid=s2p_linktrack(lkl3db,destnode,context);
    }
  }
  L3M_SETFLAG(l3db,L3_F_WITHCHILDREN|L3_F_WITHDATA);
  rnode=L3_nodeRetrieve(l3db,actualid);
  if (rnode == NULL)
  {
    S2P_TRACE(("### (%s) Retrieve returns NULL POINTER",curpath));
    return NULL;
  }
  S2P_TRACE(("### (%s) [%s]",curpath,rnode->label));
  if (rnode->dtype!=L3E_VOID)
  {
    S2P_TRACE(("[%s]",L3_typeAsStr(rnode->dtype)));
    tsize=1;
    n=0;
    ndim=0;
    npy_dim_vals[0]=0;
    while ((n<L3_MAX_DIMS)&&(rnode->dims[n]!=-1))
    {
      tsize*=rnode->dims[n];
      ndim++;
      n++;
    }
    n=0;
    while ((n<L3_MAX_DIMS)&&(rnode->dims[n]!=-1))
    {
      // code modification:
      if (S2P_HASFLAG(S2P_FREVERSEDIMS))
      {
	S2P_TRACE(("\n COMMENT PIJE: READ apply reverse L546\n"));
	npy_dim_vals[ndim-n-1]=rnode->dims[n];
      }
      else
      {
	S2P_TRACE(("\n COMMENT PIJE: F order, apply NO reverse L551\n"));
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
    }
  }
  S2P_TRACE(("\n"));
  /* Loop on children. This is a depth first recurse. In case of a path search,
     skip until we have the right name. */
  o_clist=PyList_New(0);
  nextpath=path;
  child=0;
  if (rnode->children == NULL)
  {
    S2P_TRACE(("### No child \n"));
  }
  while ((rnode->children != NULL) && (rnode->children[child] != -1))
  {
    cnode=L3_nodeRetrieve(l3db,rnode->children[child]);
    strcat(curpath,"/");
    strcat(curpath,cnode->name);
    if ((path!=NULL) && (strlen(path)!=0))
    {
      nextpath=strchr(path,'/');
      if (nextpath)
      {
        toknpath=(int)(nextpath-path); /* should work for chars */
        S2P_TRACE(("### Path compare [%s] to [%s] up to [%d]\n",
                   path+1,cnode->name,toknpath));
        if (!strncmp(path+1,cnode->name,toknpath))
        {
          S2P_TRACE(("### Path compare OK\n"));
        } 
      }
    }
    /* HDF can parse paths, i.e. a node name can be a path and the
       resulting ID is the actual last node. However, we SHOULD not use that
       because we want to have control on link parse. */
    o_child=s2p_parseAndReadHDF(cnode->id,cnode->name,curpath,nextpath,
				context,l3db);
    curpath[strlen(curpath)-strlen(cnode->name)-1]='\0';
    PyList_Append(o_clist,o_child);
    child++;
  }
  if (o_value==NULL)
  {
    Py_INCREF(Py_None);
    o_value=Py_None;
  };
  o_node=Py_BuildValue("[sOOs]",name,o_value,o_clist,rnode->label);

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
  char *name=NULL,*label=NULL,*tdat=NULL;
  int sz=0,n=0,ret=0,tsize=1;
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
    strcat(curpath,"/");
    strcat(curpath,name);
    S2P_TRACE(("### create [%s][%s]",curpath,label));
    if (s2p_checklinktable(context,curpath))
    {
      S2P_TRACE(("### linked to [%s][%s]\n",curpath,label));
    }
    S2P_FREECONTEXTPTR(context);
    s2p_getData(PyList_GetItem(tree,1),&tdat,&ndat,ddat,&vdat,context);
    n=0;
    tsize=1;
    S2P_TRACE(("{"));
    while ((n<L3_MAX_DIMS)&&(ddat[n]!=-1))
    {
      tsize*=ddat[n];
      S2P_TRACE(("%d",ddat[n]));
      n++;
      if ((n<L3_MAX_DIMS)&&(ddat[n]!=-1))
      {
	S2P_TRACE(("x"));
      }
    } 
    S2P_TRACE(("}=%d\n",tsize));
    node=L3_nodeSet(l3db,node,name,label,ddat,L3_typeAsEnum(tdat),vdat,L3_F_NONE);
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
                        int   threshold,
                        int   depth,
                        char *path)
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
  context->thh=threshold;
  context->dpt=depth;
  context->_c_float =NULL;
  context->_c_double=NULL;
  context->_c_int=NULL;
  context->_c_char=NULL;
  cpath[0]='\0';

  /* We do NOT check file name or file access, it's up to the caller to make
     such checks. Anyway, HDF will check. */
  l3db=s2p_addoneHDF(filename,context);
  rnode=L3_nodeRetrieve(l3db,l3db->root_id);
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
  S2P_FREECONTEXTPTR(context);
  free(context);

  return tree;
}
/* ------------------------------------------------------------------------- */
int s2p_saveAsHDF(char      *filename,
                  PyObject  *tree,
                  PyObject  *links,
                  int        flags,
                  int        threshold,
                  int        depth,
                  char*      path)    
{
  int ret=0,sz=-1;
  char cpath[MAXPATHSIZE],*tdat=NULL;
  s2p_ctx_t *context=NULL;
  PyObject *rtree,*otree=NULL;
  int dims[L3_MAX_DIMS];
  int ndat=0,ddat[NPY_MAXDIMS],n=0;
  char *vdat=NULL;
  L3_Cursor_t *l3db=NULL;
  L3_Node_t *node=NULL;

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

  S2P_TRACE(("### save in file [%s]\n",filename));
  if (    (PyList_Check(tree))
       && (PyList_Size(tree) == 4)
       && (PyString_Check(PyList_GetItem(tree,0)))
       && (PyList_Check(PyList_GetItem(tree,2)))
       && (PyString_Check(PyList_GetItem(tree,3))))
  {
    s2p_filllinktable(links,context);
    l3db=L3_openFile(filename,L3_OPEN_NEW,L3_F_DEFAULT);
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
	  s2p_getData(PyList_GetItem(otree,1),&tdat,&ndat,ddat,&vdat,context);
	  L3_initDims(dims,1,-1);
	  node=L3_nodeSet(l3db,node,CGNSLibraryVersion_n,CGNSLibraryVersion_ts,
			  dims,L3E_R4,vdat,L3_F_NONE);
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

