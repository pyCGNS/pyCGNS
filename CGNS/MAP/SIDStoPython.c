/*
#  -------------------------------------------------------------------------
#  pyCGNS.MAP - Python package for CFD General Notation System - MAPper
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "SIDStoPython.h"
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "Python.h"

#ifndef CHLONE_ON_WINDOWS
#define S2P_PLATFORM_CURRENT S2P_PLATFORM_UNIX
#else
#define S2P_PLATFORM_CURRENT S2P_PLATFORM_WINDOWS
#endif

#if !defined(NPY_ARRAY_OWNDATA)
#define NPY_ARRAY_OWNDATA      NPY_OWNDATA
#define NPY_ARRAY_ALIGNED      NPY_ALIGNED
#define NPY_ARRAY_WRITEABLE    NPY_WRITEABLE
#define NPY_ARRAY_F_CONTIGUOUS NPY_F_CONTIGUOUS
#define AS_NPY_ARRAY_PTR( ar ) ((PyArrayObject*)ar)
#else
#define AS_NPY_ARRAY_PTR( ar ) ((PyArrayObject_fields*)ar)
#endif

#if !defined(NPY_ARRAY_BEHAVED)
#define NPY_ARRAY_BEHAVED      (NPY_ARRAY_ALIGNED | \
                                NPY_ARRAY_WRITEABLE)
#define NPY_ARRAY_C_CONTIGUOUS    0x0001

static NPY_INLINE void
PyArray_ENABLEFLAGS(PyArrayObject *arr, int flags)
{
  ((PyArrayObject *)arr)->flags |= flags;
}
static NPY_INLINE void
PyArray_CLEARFLAGS(PyArrayObject *arr, int flags)
{
  ((PyArrayObject *)arr)->flags &= ~flags;
}

#endif

#ifdef CHLONE_TIMING
#include <sys/times.h>
#define TIMESTAMP( ctxt, msg ) \
{times(&(((L3_Cursor_t*)ctxt)->time));  \
printf("# L3 :: ");\
printf("U:[%.4d] S:[%.4d]\n",\
((L3_Cursor_t*)ctxt)->time.tms_utime,\
((L3_Cursor_t*)ctxt)->time.tms_stime);\
printf msg;fflush(stdout);}
#endif

extern int __node_count;

/* ------------------------------------------------------------------------- */
#define MAXCHILDRENCOUNT   1024
#define MAXLINKNAMESIZE    1024
#define MAXPATHSIZE        1024
#define MAXFILENAMESIZE    1024
#define MAXDATATYPESIZE    32

#if defined(WIN32) || (defined(__APPLE__) && defined(__clang__))
#define CHL_import_array() {if (_import_array() < 0)\
 {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "CHLone: numpy.core.multiarray failed to import"); return 0;} }
#else
#define CHL_import_array() import_array()
#endif

/* --- warning: NPY_MAXDIMS (numpy) is larger than MAXDIMENSIONVALUES (CGNS)

   L3C_MAX_DIMS       12
   MAXDIMENSIONVALUES 12

   NPY_MAX_DIMS       32

 */
#define MAXDIMENSIONVALUES 12

#define MAXFORMATSIZE      20
#define MAXERRORSIZE       80
#define MAXVERSIONSIZE     32

#define MAXERRORMESSAGE    256

#define __MAXOBJTABLE      256

#define  __THREADING__     1

#define S2P_NEWCONTEXTPTR( ctxt ) \
ctxt=(s2p_ctx_t*)malloc(sizeof(s2p_ctx_t));\
ctxt->flg=S2P_FNONE;\
ctxt->pth=NULL;\
ctxt->lnk=NULL;\
ctxt->hdf_dbs=NULL;\
ctxt->lnk_obj=NULL;\
ctxt->upd_pth=NULL;\
ctxt->upd_pth_lk=NULL;\
ctxt->flt_dct=NULL;\
ctxt->skp_pth=NULL;\
ctxt->skp_pth_lk=NULL;\
ctxt->lsp=NULL;\
ctxt->hdf_idx=-1;\
ctxt->flg=0;\
ctxt->dpt=-1;\
ctxt->mxs=-1;\
ctxt->pol_max=0;\
ctxt->pol_cur=0;\
ctxt->pol_oid=NULL;\
ctxt->sha256=(sha256_t *)malloc(sizeof(sha256_t));\
ctxt->platform=S2P_PLATFORM_CURRENT;\
ctxt->err=NULL;

#define S2P_CLEARCONTEXTPTR( ctxt ) \
ctxt->flg=S2P_FNONE;\
ctxt->pth=NULL;\
ctxt->lnk=NULL;\
ctxt->hdf_dbs=NULL;\
ctxt->lnk_obj=NULL;\
ctxt->upd_pth=NULL;\
ctxt->upd_pth_lk=NULL;\
ctxt->flt_dct_pth=NULL;\
ctxt->skp_pth=NULL;\
ctxt->skp_pth_lk=NULL;\
ctxt->lsp=NULL;\
ctxt->hdf_idx=-1;\
ctxt->flg=0;\
ctxt->dpt=-1;\
ctxt->mxs=-1;\
ctxt->pol_max=0;\
ctxt->pol_cur=0;\
ctxt->pol_oid=NULL;\
ctxt->platform=S2P_PLATFORM_CURRENT;\
ctxt->err=NULL;

#define S2P_HASFLAG( flag ) ((context->flg & flag) == flag)
#define S2P_SETFLAG( flag ) ( context->flg |=  flag)
#define S2P_CLRFLAG( flag ) ( context->flg &= ~flag)

#define S2P_TRACE( txt ) \
if (S2P_HASFLAG(S2P_FTRACE)){printf txt ;fflush(stdout);}

static char *DT_MT = "MT";
static char *DT_I4 = "I4";
static char *DT_I8 = "I8";
static char *DT_R4 = "R4";
static char *DT_R8 = "R8";
static char *DT_X4 = "X4";
static char *DT_X8 = "X8";
static char *DT_C1 = "C1";

#define S2P_CHECKNODE( node, ctxt )                       \
  ((PyList_Check(node)) && (PyList_Size(node) == 4) \
  && (PyUnicode_Check(PyList_GetItem(node, 0))) \
  && (PyList_Check(PyList_GetItem(node,2))) \
  && (PyUnicode_Check(PyList_GetItem(node, 3))) \
  && (s2p_nodeNotAlreadyParsed(node,ctxt)))

#define S2P_CHECKNODENOCACHE( node, ctxt )                        \
  ((PyList_Check(node)) && (PyList_Size(node) == 4) \
  && (PyUnicode_Check(PyList_GetItem(node,0))) \
  && (PyList_Check(PyList_GetItem(node,2))) \
  && (PyUnicode_Check(PyList_GetItem(node,3))))

#define S2P_AS_HSIZE(val,validx) \
  arg=PyList_GetItem(obj,validx); \
  for (n=0;n<PyList_Size(arg);n++){val[n]=PyLong_AsLong(PyList_GetItem(arg,n));} \
  val[n]=-1;

#ifdef __THREADING__
#define ENTER_NOGIL_BLOCK( tst ) \
{\
  int _py_th_ok = tst;\
  PyThreadState *_py_th_save;\
  if (_py_th_ok) { _py_th_save = PyEval_SaveThread();}

#define LEAVE_NOGIL_BLOCK( ) \
  if (_py_th_ok) { PyEval_RestoreThread(_py_th_save);} }

#else
#define ENTER_NOGIL_BLOCK( tst ) ;
#define LEAVE_NOGIL_BLOCK( ) ;
#endif

#define STR_ALLOC( st, sz ) \
st = (char*)malloc(sizeof(char)*MAXPATHSIZE); st [0]='\0';

#define STR_FREE( st ) {free( st );st = NULL;}

#define DIM_ALLOC( dm, tp, sz ) dm =( tp * )malloc(sizeof( tp )* sz );

#define DIM_FREE( dm ) free( dm );


/* DEBUG TRACE ONLY */
extern void objlist_status(char *tag);
#define TRACE_HDF5_LEAK( msg ) \
  if (S2P_HASFLAG(S2P_FDEBUG)){ objlist_status( msg ); }

#define S2P_H5_GCLOSE( msg, hid )\
  {\
  H5Gclose(hid);\
  if (0 || S2P_HASFLAG(S2P_FDEBUG)){\
    printf( "H5GCLOSE [%ld]",hid);\
    printf( msg );\
  }}

/* Handle HDF5 API variability */
#if H5_VERSION_GE(1,12,0) && !defined(H5_USE_110_API) && !defined(H5_USE_18_API) && !defined(H5_USE_16_API)
#define SIDS_HDF5_HAVE_112_API 1
#else
#define SIDS_HDF5_HAVE_112_API 0
#endif

/* ------------------------------------------------------------------------- */
#ifdef CHLONE_TIMING
rusage_snapshot(char *tag)
{
  static long last = 0L;
  long rss = 0L, current;
  FILE* fp = NULL;

  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t)0L;          /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1)
  {
    fclose(fp);
    return (size_t)0L;                /* Can't read? */
  }
  fclose(fp);
  current = ((size_t)rss * (size_t)sysconf(_SC_PAGESIZE));

  printf(">%s %12.d %12.d", tag, current, current - last);
  last = current;
}
#endif
/* ------------------------------------------------------------------------- */
static int is_fortran_contiguous(PyArrayObject *ap)
{
  /* array_has_column_major_storage(a) is equivalent to
     transpose(a).iscontiguous() but more efficient.

     This function can be used in order to decide whether to use a
     Fortran or C version of a wrapped function. This is relevant, for
     example, in choosing a clapack or flapack function depending on
     the storage order of array arguments.

     This function was taken (with a small modification
     for broadcast numarray) from f2py.
  */
  npy_intp sd;
  npy_intp dim;
  int i;
  sd = PyArray_ITEMSIZE(ap);
  for (i = 0; i < PyArray_NDIM(ap); ++i)
  {
    dim = PyArray_DIMS(ap)[i];
    if (dim == 0)
    {
      return 0;  /* broadcast array */
    }
    if (dim != 1) {
      if (PyArray_STRIDES(ap)[i] != sd)
      {
        return 0;
      }
      sd *= dim;
    }
  }
  return 1;
}
/* -------------------------------------------------------------------------- */
/* finds a subpath inside the current path, starting from first char.
   if exact is 1 then the current path should have at least the size of
   the sub path, so that it is completly included inside the current path     */
static int s2p_issubpath(char *subpathtofind, char* currentpath, int exact)
{
  size_t l_subpathtofind, l_currentpath;

  l_subpathtofind = strlen(subpathtofind);
  l_currentpath = strlen(currentpath);
  if ((l_subpathtofind != l_currentpath) && exact)
  {
    return 0;
  }
  if (l_subpathtofind > l_currentpath)
  {
    if (!strncmp(subpathtofind, currentpath, l_currentpath) &&
        subpathtofind[l_currentpath] == '/')
    {
      return 1;
    }
  }
  else
  {
    /* (l_subpathtofind <= l_currentpath) */
    if (!strncmp(subpathtofind, currentpath, l_subpathtofind))
    {
      if (l_currentpath == l_subpathtofind) { return 1; }
      else if (currentpath[l_subpathtofind] == '/') { return 1; }
    }
  }
  return 0;
}
/* -------------------------------------------------------------------------- */
static void setError(int code, char* message, char* value, s2p_ctx_t *context)
{
  PyObject *otype = NULL;
  char buff[256];

  if (value != NULL)
  {
    snprintf(buff, MAXERRORMESSAGE, message, value);
  }
  else
  {
    strncat(buff, message, MAXERRORMESSAGE);
  }

  PyErr_SetObject(context->err, Py_BuildValue("(is)", code, buff));

}
/* ------------------------------------------------------------------------- */
static s2p_ent_t *s2p_getcurrentHDF(s2p_ctx_t *context)
{
  return context->hdf_stk[context->hdf_idx];
}
/* ------------------------------------------------------------------------- */
static int s2p_popHDF(s2p_ctx_t *context)
{
  S2P_TRACE(("# CHL:pop context [%d]\n", context->hdf_idx));
  if (context->hdf_idx > 0)
  {
    //L3_decRef(context->hdf_stk[context->hdf_idx]->l3db,
    //  context->hdf_stk[context->hdf_idx]->l3db->root_id);
    context->hdf_idx--;
  }
  return context->hdf_idx;
}
/* ------------------------------------------------------------------------- */
static L3_Cursor_t *s2p_addoneHDF(char* dirname, char *filename,
                                  s2p_ctx_t *context, int excpt)
{
  L3_Cursor_t *l3dbptr = NULL;
  int newentry = 0, ishdf = 0;
  size_t szd, szf;
  s2p_ent_t *nextdbs = NULL, *prevdbs = NULL;
  struct stat *sbuff = NULL;
  long l3flag = L3F_DEFAULT;
  char *fullpath;

  if ((context->hdf_idx != -1)
      && !strcmp(context->hdf_stk[context->hdf_idx]->dirname, dirname)
      && !strcmp(context->hdf_stk[context->hdf_idx]->filename, filename))
  {
    S2P_TRACE(("# CHL:same context [%d]:[%s][%s]\n",
      context->hdf_idx,
      context->hdf_stk[context->hdf_idx]->dirname,
      context->hdf_stk[context->hdf_idx]->filename));
    return context->hdf_stk[context->hdf_idx]->l3db;
  }

  szd = strlen(dirname);
  szf = strlen(filename);
  if (szd && (dirname[szd - 1] != '/'))
  {
    fullpath = (char*)malloc(sizeof(char*)*(szd + szf + 2));
    sprintf(fullpath, "%s/%s", dirname, filename);
  }
  else
  {
    fullpath = (char*)malloc(sizeof(char*)*(szd + szf + 1));
    sprintf(fullpath, "%s%s", dirname, filename);
  }
  newentry = 1;
  nextdbs = context->hdf_dbs;
  if (nextdbs == NULL)
  {
    nextdbs = (s2p_ent_t*)malloc(sizeof(s2p_ent_t));
    nextdbs->filename = NULL;
    nextdbs->dirname = NULL;
    nextdbs->l3db = NULL;
    nextdbs->next = NULL;
    context->hdf_dbs = nextdbs;
  }
  else
  {
    prevdbs = context->hdf_dbs;
    while ((nextdbs != NULL) && (nextdbs->filename != NULL))
    {
      if (!strcmp(nextdbs->filename, filename))
      {
        l3dbptr = nextdbs->l3db;
        if (!H5Iis_valid(l3dbptr->root_id))
        {
          S2P_TRACE(("# CHL:BAD ROOT [%s] (broken ID %ld)\n", \
            filename, l3dbptr->root_id));
        }
        newentry = 0;
        S2P_TRACE(("# CHL:open [%s] (already open)\n", filename));
        break;
      }
      prevdbs = nextdbs;
      nextdbs = nextdbs->next;
    }
    if (newentry)
    {
      prevdbs->next = (s2p_ent_t*)malloc(sizeof(s2p_ent_t));
      nextdbs = prevdbs->next;
      nextdbs->filename = NULL;
      nextdbs->dirname = NULL;
      nextdbs->l3db = NULL;
      nextdbs->next = NULL;
    }
  }
  if (newentry)
  {
    if (S2P_HASFLAG(S2P_FUPDATE))
    {
      sbuff = (struct stat*)malloc(sizeof(struct stat));
      if (stat(fullpath, sbuff) == -1)
      {
        if (excpt)
        {
          setError(S2P_EFILEUNKWOWN, "Cannot find file [%s]",
            fullpath, context);
        }
        free(sbuff);
        free(fullpath);
        return NULL;
      }
      free(sbuff);
      if (CHL_ACCESS_READ(fullpath))
      {
        if (excpt)
        {
          setError(S2P_EFILEUNKWOWN, "Cannot read file [%s]",
            fullpath, context);
        }
        free(fullpath);
        return NULL;
      }
#if SIDS_HDF5_HAVE_112_API
      ishdf = H5Fis_accessible(fullpath, H5P_DEFAULT);
#else
      ishdf = H5Fis_hdf5(fullpath);
#endif
      if (!ishdf)
      {
        if (excpt)
        {
          setError(S2P_ENOTHDF5FILE, "Target is not an HDF5 file [%s]",
            fullpath, context);
        }
        free(fullpath);
        return NULL;
      }
    }
    if (!S2P_HASFLAG(S2P_FUPDATE) && !S2P_HASFLAG(S2P_FNEW))
    {
      ENTER_NOGIL_BLOCK(1);
      if (S2P_HASFLAG(S2P_FDEBUG)) { l3flag |= L3F_DEBUG; }
      l3dbptr = L3_openFile(fullpath, L3E_OPEN_RDO, l3flag);
      S2P_TRACE(("# CHL:open '%s' READ ONLY\n", fullpath));
      LEAVE_NOGIL_BLOCK();
      if (!L3M_ECHECK(l3dbptr))
      {
        if (excpt)
        {
          setError(S2P_EFAILOLDOPEN, "Cannot read file [%s]",
            fullpath, context);
        }
        free(fullpath);
        return NULL;
      }
    }
    else if (!S2P_HASFLAG(S2P_FNEW))
    {
      ENTER_NOGIL_BLOCK(1);
      if (S2P_HASFLAG(S2P_FDEBUG)) { l3flag |= L3F_DEBUG; }
      l3dbptr = L3_openFile(fullpath, L3E_OPEN_OLD, l3flag);
      S2P_TRACE(("# CHL:open '%s' UPDATE\n", fullpath));
      LEAVE_NOGIL_BLOCK();
      if (!L3M_ECHECK(l3dbptr))
      {
        if (excpt)
        {
          setError(S2P_EFAILOLDOPEN, "Cannot modify existing file [%s]",
            fullpath, context);
        }
        free(fullpath);
        return NULL;
      }
    }
    else
    {
      ENTER_NOGIL_BLOCK(1);
      if (S2P_HASFLAG(S2P_FDEBUG)) { l3flag |= L3F_DEBUG; }
      l3dbptr = L3_openFile(fullpath, L3E_OPEN_NEW, l3flag);
      S2P_TRACE(("# CHL:open '%s' NEW\n", fullpath));
      LEAVE_NOGIL_BLOCK();
      if (!L3M_ECHECK(l3dbptr))
      {
        if (excpt)
        {
          setError(S2P_EFAILNEWOPEN, "Cannot create new file [%s]",
            fullpath, context);
        }
        free(fullpath);
        return NULL;
      }
    }
    nextdbs->filename = (char*)malloc(sizeof(char)*(szf + 1));
    strcpy(nextdbs->filename, filename);
    nextdbs->dirname = (char*)malloc(sizeof(char)*(szd + 1));
    strcpy(nextdbs->dirname, dirname);
    nextdbs->next = NULL;
    l3dbptr = nextdbs->l3db = l3dbptr;
    S2P_TRACE(("# CHL:open '%s' [%ld]\n", fullpath, l3dbptr->root_id));
  }
  if (context->hdf_idx == S2P_MAX_LINK_STACK)
  {
    if (excpt)
    {
      setError(S2P_EMAXLINKSTCK, "Link depth exceed CGNS/HDF5 standard max",
        "", context);
    }
  }
  context->hdf_idx++;
  context->hdf_stk[context->hdf_idx] = nextdbs;
  S2P_TRACE(("# CHL:push context [%d]:[%s][%s]\n",
    context->hdf_idx, nextdbs->dirname, nextdbs->filename));
  free(fullpath);
  return l3dbptr;
}
/* ------------------------------------------------------------------------- */
static void s2p_closeallHDF(s2p_ctx_t *context)
{
  s2p_ent_t *nextdbs = NULL, *dbs = NULL;

  S2P_TRACE(("# CHL:close all\n"));

  dbs = context->hdf_dbs;
  while (dbs != NULL)
  {
    if (dbs->l3db != NULL)
    {
      S2P_TRACE(("# CHL:close one\n"));
      L3_close(&(dbs->l3db));
      TRACE_HDF5_LEAK("CLOSE");
    }
    if (dbs->filename != NULL)
    {
      S2P_TRACE(("# CHL:close '%s'\n", dbs->filename));
      free(dbs->filename);
    }
    if (dbs->dirname != NULL) { free(dbs->dirname); }
    nextdbs = dbs->next;
    free(dbs);
    dbs = nextdbs;
  }
  context->hdf_dbs = NULL;
}
/* ------------------------------------------------------------------------- */
static s2p_ctx_t *s2p_filllinktable(PyObject *linktable, s2p_ctx_t *context)
{
  //int linktablesize = 0, n = 0;
  Py_ssize_t linktablesize = 0, n = 0, sz = 0;
  char *st = NULL;
  s2p_lnk_t *nextlink = NULL, *previouslink = NULL;
  PyObject  *lke = NULL;
  PyObject * ascii_string = NULL;

  S2P_TRACE(("# CHL:fill link table\n"));

  if ((linktable == NULL) || (!PyList_Check(linktable))) { return NULL; }
  linktablesize = PyList_Size(linktable);
  if (!linktablesize) { return NULL; }
  nextlink = context->lnk;
  previouslink = NULL;
  if (nextlink != NULL)
  {
    while (nextlink->next != NULL)
    {
      previouslink = nextlink;
      nextlink = nextlink->next;
    }
  }
  for (n = 0; n < linktablesize; n++)
  {
    lke = PyList_GetItem(linktable, n);

    if (PyList_Check(lke)
      && (PyList_Size(lke) >= 4)
      && ((PyBytes_Check(PySequence_GetItem(lke, 0)))
        || (PyUnicode_Check(PySequence_GetItem(lke, 0)))
        || (PySequence_GetItem(lke, 0) == Py_None))
      && (PyBytes_Check(PySequence_GetItem(lke, 1))
        || PyUnicode_Check(PySequence_GetItem(lke, 1)))
      && (PyBytes_Check(PySequence_GetItem(lke, 2))
        || PyUnicode_Check(PySequence_GetItem(lke, 2)))
      && (PyBytes_Check(PySequence_GetItem(lke, 3))
        || PyUnicode_Check(PySequence_GetItem(lke, 3))))
    {
      if (previouslink != NULL)
      {
        previouslink->next = (s2p_lnk_t*)malloc(sizeof(s2p_lnk_t));
        nextlink = previouslink->next;
      }
      else
      {
        context->lnk = (s2p_lnk_t*)malloc(sizeof(s2p_lnk_t));
        nextlink = context->lnk;
      }
      nextlink->next = NULL;
      previouslink = nextlink;

      if (PySequence_GetItem(lke, 0) != Py_None)
      {
        ascii_string = PyUnicode_AsASCIIString(PySequence_GetItem(lke, 0));
        sz = PyBytes_Size(ascii_string);
        st = PyBytes_AsString(ascii_string);
        nextlink->dst_dirname = (char*)malloc(sizeof(char)*sz + 1);
        strcpy(nextlink->dst_dirname, st);
        Py_DECREF(ascii_string);
      }
      else
      {
        nextlink->dst_dirname = (char*)malloc(sizeof(char));
        strcpy(nextlink->dst_dirname, "");
      }

      ascii_string = PyUnicode_AsASCIIString(PySequence_GetItem(lke, 1));
      sz = PyBytes_Size(ascii_string);
      st = PyBytes_AsString(ascii_string);

      nextlink->dst_filename = (char*)malloc(sizeof(char)*sz + 1);
      strcpy(nextlink->dst_filename, st);
      if (ascii_string != NULL) { Py_DECREF(ascii_string); }

      ascii_string = PyUnicode_AsASCIIString(PySequence_GetItem(lke, 2));
      sz = PyBytes_Size(ascii_string);
      st = PyBytes_AsString(ascii_string);
      nextlink->dst_nodename = (char*)malloc(sizeof(char)*sz + 1);
      strcpy(nextlink->dst_nodename, st);
      if (ascii_string != NULL) { Py_DECREF(ascii_string); }
      ascii_string = PyUnicode_AsASCIIString(PySequence_GetItem(lke, 3));
      sz = PyBytes_Size(ascii_string);
      st = PyBytes_AsString(ascii_string);
      nextlink->src_nodename = (char*)malloc(sizeof(char)*sz + 1);
      strcpy(nextlink->src_nodename, st);
      if (ascii_string != NULL) { Py_DECREF(ascii_string); }
      nextlink->status = S2P_LKFAIL;

      S2P_TRACE(("# CHL:link data [%s]->[%s][%s][%s]\n", \
        nextlink->src_nodename, \
        nextlink->dst_dirname, \
        nextlink->dst_filename, \
        nextlink->dst_nodename));
    }
  }
  return context;
}
/* ------------------------------------------------------------------------- */
static void s2p_freelinktable(s2p_ctx_t *context)
{
  s2p_lnk_t *nextlink = NULL, *links = NULL;
  return;
  links = context->lnk;
  if (links != NULL)
  {
    while (links->next != NULL)
    {
      free(links->dst_dirname);
      free(links->dst_filename);
      free(links->dst_nodename);
      free(links->src_dirname);
      free(links->src_filename);
      free(links->src_nodename);
      nextlink = links->next;
      free(links);
      links = nextlink;
    }
  }
  context->lnk = NULL;
}
/* ------------------------------------------------------------------------- */
static int s2p_atleastonelink(s2p_ctx_t *context)
{
  s2p_lnk_t *link = NULL;

  link = context->lnk;
  while (link != NULL)
  {
    if (link->status != S2P_LKUPDATED) { return 1; }
    link = link->next;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static s2p_lnk_t *s2p_checklinktable(s2p_ctx_t *context, char *nodename)
{
  s2p_lnk_t *link = NULL;
  int n = 0;

  link = context->lnk;
  while (link != NULL)
  {
    if (!strcmp(link->src_nodename, nodename)) { return link; }
    link = link->next;
  }
  return (s2p_lnk_t *)NULL;
}
/* ------------------------------------------------------------------------- */
static PyObject *s2p_getlinktable(s2p_ctx_t *context)
{
  PyObject *rt = NULL, *lk = NULL;
  s2p_lnk_t *links = NULL;

  rt = PyList_New(0);
  links = context->lnk;
  while (links != NULL)
  {
    lk = Py_BuildValue("[ssssssi]",
      links->src_dirname,
      links->src_filename,
      links->src_nodename,
      links->dst_dirname,
      links->dst_filename,
      links->dst_nodename,
      links->status);
    PyList_Append(rt, lk);
    links = links->next;
  }
  return rt;
}
#define __ALLOCANDCOPYORZERO( from, to )       \
if ( from == NULL ) {sz=0; }                   \
else { sz=strlen( from ); }                    \
curlink-> to =(char*)malloc(sizeof(char)*sz+1);\
if (sz) { strcpy(curlink-> to , from ); }      \
else { curlink-> to [0]='\0'; }

/* ------------------------------------------------------------------------- */
static void s2p_linkstack(char       *curdir,
  char       *curfile,
  char       *curnode,
  char       *destdir,
  char       *destfile,
  char       *destnode,
  int         status,
  s2p_ctx_t  *context)
{
  s2p_lnk_t *nextlink = NULL, *curlink = NULL;
  int sz = 0;

  nextlink = context->lnk;
  if (nextlink != NULL)
  {
    while (nextlink->next != NULL)
    {
      nextlink = nextlink->next;
    }
    nextlink->next = (s2p_lnk_t*)malloc(sizeof(s2p_lnk_t));
    curlink = nextlink->next;
    curlink->next = NULL;
  }
  else
  {
    context->lnk = (s2p_lnk_t*)malloc(sizeof(s2p_lnk_t));
    curlink = context->lnk;
    curlink->next = NULL;
  }
  __ALLOCANDCOPYORZERO(destdir, dst_dirname);
  __ALLOCANDCOPYORZERO(destfile, dst_filename);
  __ALLOCANDCOPYORZERO(destnode, dst_nodename);
  __ALLOCANDCOPYORZERO(curdir, src_dirname);
  __ALLOCANDCOPYORZERO(curfile, src_filename);
  __ALLOCANDCOPYORZERO(curnode, src_nodename);

  curlink->dst_object = (PyObject*)NULL;
  curlink->status = status;
  S2P_TRACE(("# CHL:link stack [%s][%s][%s]->[%s][%s][%s]\n",
    curdir, curfile, curnode, destdir, destfile, destnode));

}
static int s2p_setlinksearchpath(L3_Cursor_t *l3db, s2p_ctx_t *context);

/* ------------------------------------------------------------------------- */
static void s2p_createlinkoffparse(L3_Cursor_t *l3db,
                                   s2p_lnk_t *link,
                                   s2p_ctx_t *context)
{
  hid_t lkid = -1, parentid = -1;
  char leafnodename[33], *parentnodename, *p;
  int islocal, skiponerror;
  size_t i;
  s2p_lnk_t *plink = NULL;

  parentnodename = (char*)malloc(strlen(link->src_nodename) + 1);
  strcpy(parentnodename, link->src_nodename);
  p = parentnodename;
  for (i = strlen(link->src_nodename); i > 0; i--)
  {
    if (*(p + i) == '/')
    {
      *(p + i) = '\0';
      break;
    }
  }
  if (i == 0) { return; }
  strcpy(leafnodename, p + i + 1);
  parentid = L3_path2Node(l3db, parentnodename);
  /*
     Link granted only for current file as source node
     If the requested local node is a link itsef, it is rejected
     Such a check is useless if the node is in the CGNS/Python tree,
     as this node would be actually created.
  */
  islocal = 1;
  plink = context->lnk;
  i = strlen(parentnodename);
  while (plink != NULL)
  {
    if (strcmp(plink->src_nodename, link->src_nodename)
      && !strncmp(plink->src_nodename, parentnodename, i)
      && (i == strlen(plink->src_nodename)))
    {
      islocal = 0;
      break;
    }
    plink = plink->next;
  }
  if (islocal && H5Iis_valid(parentid))
  {
    if (L3M_HASFLAG(l3db, L3F_SKIPONERROR)) { skiponerror = 1; }
    else { skiponerror = 0; }
    L3M_UNSETFLAG(l3db, L3F_SKIPONERROR);
    lkid = L3_nodeLink(l3db, parentid,
      leafnodename,
      link->dst_filename,
      link->dst_nodename);
    if (skiponerror) { L3M_SETFLAG(l3db, L3F_SKIPONERROR); }
    S2P_TRACE(("# CHL:link [%s/%s]->[%s][%s][%s]\n", \
      parentnodename, \
      leafnodename, \
      link->dst_dirname, \
      link->dst_filename, \
      link->dst_nodename));
    if (H5Iis_valid(lkid))
    {
      S2P_H5_GCLOSE("LINK OPEN\n", lkid);
    }
    else
    {
      S2P_TRACE(("# CHL:link failed\n"));
      /* CHL_printError(l3db); */
    }
    S2P_H5_GCLOSE("PARENT LINK\n", parentid);
  }
  else
  {
    S2P_TRACE(("# CHL:link [%s/%s]->[%s][%s][%s] rejected\n", \
      parentnodename, \
      leafnodename, \
      link->dst_dirname, \
      link->dst_filename, \
      link->dst_nodename));
  }
  free(parentnodename);
}

/* ------------------------------------------------------------------------- */
static hid_t s2p_linktrack(L3_Cursor_t *l3db,
                           char        *nodename,
                           s2p_ctx_t   *context)
{
  int parsepath = 0, c = 0, ix = 0, linkstatus = 0;
  hid_t nid = -1;
  char *p = NULL, *curdir = NULL, *curfile = NULL;
  L3_Cursor_t *lkhdfdb = NULL;
  s2p_ent_t *curhdf;
  char *curpath, *name, *destnode, *destdir, *destfile, *targetfile;

  STR_ALLOC(curpath, MAXPATHSIZE);
  STR_ALLOC(name, (L3C_MAX_NAME + 1));
  STR_ALLOC(destnode, (L3C_MAX_NAME + 1));
  STR_ALLOC(destdir, MAXFILENAMESIZE);
  STR_ALLOC(destfile, MAXFILENAMESIZE);
  STR_ALLOC(targetfile, MAXFILENAMESIZE);

  parsepath = 1;
  p = nodename;
  nid = -1;

  curhdf = s2p_getcurrentHDF(context);
  curdir = curhdf->dirname;
  curfile = curhdf->filename;

  while (parsepath)
  {
    while ((*p != '\0') && (*p != '/')) { p++; }
    if (*p == '\0') { break; }
    p++;
    c = 0;
    while ((*p != '\0') && (*p != '/'))
    {
      name[c] = *p;
      p++;
      c++;
    }
    name[c] = '\0';
    strcat(curpath, "/");
    strcat(curpath, name);
    if (H5Iis_valid(nid))
    {
      S2P_H5_GCLOSE("LINKTRACK 1\n", nid);
    }
    nid = L3_path2Node(l3db, curpath);
    if ((nid != -1) && L3_isLinkNode(l3db, nid, destfile, destnode))
    {
      linkstatus = S2P_LKOK;
      /* L3 layer has its own link search path,
         we have to ask L3 the directory used to open the actual file. */
      ix = CHL_getFileInSearchPath(l3db, destfile);
      strcpy(destdir, CHL_getLinkSearchPath(l3db, ix));
      S2P_TRACE(("# CHL:link follow link: [%s][%s][%s]\n",
        destdir, destfile, destnode));
      s2p_linkstack(curdir, curfile, curpath,
        destdir, destfile, destnode, linkstatus, context);
      sprintf(targetfile, "%s/%s", destdir, destfile);
      lkhdfdb = s2p_addoneHDF(destdir, destfile, context, 0);
      if (!L3M_ECHECK(l3db))
      {
        setError(S2P_EFAILLNKOPEN,
          "Cannot open file for LINK '%s'", targetfile, context);
        STR_FREE(curpath);
        STR_FREE(name);
        STR_FREE(destnode);
        STR_FREE(destdir);
        STR_FREE(destfile);
        STR_FREE(targetfile);
        return -1;
      }
      s2p_setlinksearchpath(lkhdfdb, context);
      if (H5Iis_valid(nid))
      {
        S2P_H5_GCLOSE("LINKTRACK 2\n", nid);
      }
      nid = s2p_linktrack(lkhdfdb, destnode, context);
    }
    if (nid == -1) { parsepath = 0; }
  }
  STR_FREE(curpath);
  STR_FREE(name);
  STR_FREE(destnode);
  STR_FREE(destdir);
  STR_FREE(destfile);
  STR_FREE(targetfile);
  return nid;
}

/* ------------------------------------------------------------------------- */
/* Fill the link path table for the context. The searchpath should be a
   single string where paths are separated by : or ; (classical way for paths)
*/
static int s2p_setlinksearchpath(L3_Cursor_t *l3db,
                                 s2p_ctx_t *context)
{
  char *ptr = NULL, *path = NULL, *rpath = NULL, path_sep;
  int parse = 0;

  if (context->lsp == NULL)
  {
    return 1;
  }
  rpath = (char*)malloc(sizeof(char)*strlen(context->lsp) + 1);
  strcpy(rpath, context->lsp);
  path = rpath;
  parse = 1;
  if (context->platform == S2P_PLATFORM_UNIX) path_sep = ':';
  else path_sep = ';';
  while (parse)
  {
    ptr = path;
    while ((ptr[0] != path_sep) && (ptr[0] != '\0'))
    {
      ptr++;
    }
    if (ptr[0] == '\0')
    {
      parse = 0;
    }
    ptr[0] = '\0';
    S2P_TRACE(("# CHL:add search path :[%s]\n", path));
    CHL_addLinkSearchPath(l3db, path);
    ptr++;
    path = ptr;
  }
  free(rpath);
  return 1;
}
/* ------------------------------------------------------------------------- */
static void s2p_freepathtable(s2p_ctx_t *context)
{
  s2p_pth_t *nextpath = NULL, *paths = NULL;

  paths = context->pth;
  if (paths != NULL)
  {
    while (paths->next != NULL)
    {
      free(paths->path);
      nextpath = paths->next;
      free(paths);
      paths = nextpath;
    }
  }
  context->pth = NULL;
}
/* ------------------------------------------------------------------------- */
static void s2p_pathstack(char *path, int state, int dtype, int *dims,
                          s2p_ctx_t *context)
{
  s2p_pth_t *nextpath = NULL, *curpath = NULL;
  int n;
  size_t sz = 0;

  nextpath = context->pth;
  if (nextpath != NULL)
  {
    while (nextpath->next != NULL)
    {
      nextpath = nextpath->next;
    }
    nextpath->next = (s2p_pth_t*)malloc(sizeof(s2p_pth_t));
    curpath = nextpath->next;
  }
  else
  {
    context->pth = (s2p_pth_t*)malloc(sizeof(s2p_pth_t));
    curpath = context->pth;
  }
  sz = strlen(path);
  curpath->path = (char*)malloc(sizeof(char)*sz + 1);
  curpath->state = state;
  curpath->dtype = dtype;
  for (n = 0; n < L3C_MAX_DIMS; n++) { curpath->dims[n] = dims[n]; }
  curpath->next = NULL;
  strcpy(curpath->path, path);
}
/* ------------------------------------------------------------------------- */
static PyObject *s2p_getpathtable(s2p_ctx_t *context)
{
  PyObject *rt = NULL, *pth = NULL, *tp = NULL;
  s2p_pth_t *paths = NULL;
  int n;

  rt = PyList_New(0);
  paths = context->pth;
  while (paths != NULL)
  {
    for (n = 0; n < L3C_MAX_DIMS; n++)
    {
      if (paths->dims[n] == -1) break;
    }
    tp = PyTuple_New(n);
    for (n = 0; n < L3C_MAX_DIMS; n++)
    {
      if (paths->dims[n] != -1)
      {
        PyTuple_SetItem(tp, n, PyLong_FromLong((long)(paths->dims[n])));
      }
    }
    pth = Py_BuildValue("(sisO)", paths->path, paths->state,
      L3_typeAsStr(paths->dtype), tp);
    PyList_Append(rt, pth);
    paths = paths->next;
  }
  return rt;
}
/* ------------------------------------------------------------------------- */
static PyObject* s2p_getUpdateObjectByPath(PyObject* updict, char *path)
{
  PyObject *obj = NULL, *tab = NULL;

  if (updict != NULL)
  {
    obj = PyDict_GetItemString(updict, path);
    if (obj != NULL)
    {
      if (PyList_Check(obj) && (PyList_Size(obj) == 4)
        && (PyUnicode_Check(PyList_GetItem(obj, 0)))
        && (PyUnicode_Check(PyList_GetItem(obj, 3))))
      {
        tab = PyList_GetItem(obj, 1);
        if ((PyArray_Check(tab)) || (tab == Py_None))
        {
          return tab;
        }
      }
    }
  }
  return NULL;
}
/* ------------------------------------------------------------------------- */
static int s2p_pathToSkip(s2p_ctx_t *context, char *path)
{
  Py_ssize_t sz, n;
  PyObject *opth;

  if (context->skp_pth == NULL) { return S2P_HASFLAG(S2P_FKEEPLIST) ? 1 : 0; }
  if (!PyList_Check(context->skp_pth)) { return S2P_HASFLAG(S2P_FKEEPLIST) ? 1 : 0; }

  sz = PyList_Size(context->skp_pth);

  for (n = 0; n < sz; n++)
  {
    opth = PyList_GetItem(context->skp_pth, n);
    if (PyUnicode_Check(opth))
    {
      PyObject *ascii = PyUnicode_AsASCIIString(opth);
      if (s2p_issubpath(PyBytes_AsString(ascii), path, 1))
      {
        Py_DECREF(ascii);
        return S2P_HASFLAG(S2P_FKEEPLIST) ? 0 : 1;
      }
      Py_DECREF(ascii);
    }
  }
  return S2P_HASFLAG(S2P_FKEEPLIST) ? 1 : 0;
}

/* ------------------------------------------------------------------------- */
static int s2p_filterDataContiguous(s2p_ctx_t *context, char *path,
                                    int *index, int *rank, int *count,
                                    int *interlaced)
{
  int s;
  PyObject *obj;

  if (context->flt_dct == NULL) { return 0; }
  obj = PyDict_GetItemString(context->flt_dct, path);
  /* checks performed at python level */
  if (obj != NULL)
  {
    s = PyLong_AsLong(PyTuple_GetItem(obj, 0));
    if (s&S2P_SCONTIGUOUS)
    {
      *index = PyLong_AsLong(PyTuple_GetItem(obj, 1));
      *rank = PyLong_AsLong(PyTuple_GetItem(obj, 2));
      *count = PyLong_AsLong(PyTuple_GetItem(obj, 3));

      if (s&S2P_SINTERLACED)
      {
        *interlaced = 1;
      }

      /* WRONG should del only CONTIGUOUS entry for this path */
      PyDict_DelItemString(context->flt_dct, path);
      return 1;
    }
  }
  return 0;
}

/* ------------------------------------------------------------------------- */
static int s2p_filterDataPartial(s2p_ctx_t *context, char *path,
  hsize_t *src_offset,
  hsize_t *src_stride,
  hsize_t *src_count,
  hsize_t *src_block,
  hsize_t *dst_offset,
  hsize_t *dst_stride,
  hsize_t *dst_count,
  hsize_t *dst_block)
{
  int n, s;
  PyObject *obj, *arg;

  if (context->flt_dct == NULL) { return 0; }
  obj = PyDict_GetItemString(context->flt_dct, path);
  /* checks performed at python level */
  if (obj != NULL)
  {
    s = PyLong_AsLong(PyTuple_GetItem(obj, 0));
    if (s == S2P_SPARTIAL)
    {
      obj = PyTuple_GetItem(obj, 1);
      S2P_AS_HSIZE(src_offset, 0);
      S2P_AS_HSIZE(src_stride, 1);
      S2P_AS_HSIZE(src_count, 2);
      S2P_AS_HSIZE(src_block, 3);
      S2P_AS_HSIZE(dst_offset, 4);
      S2P_AS_HSIZE(dst_stride, 5);
      S2P_AS_HSIZE(dst_count, 6);
      S2P_AS_HSIZE(dst_block, 7);

      /* WRONG should del only PARTIAL entry for this path */
      PyDict_DelItemString(context->flt_dct, path);
      return 1;
    }
  }
  return 0;
}

/* ------------------------------------------------------------------------- */
static int s2p_getData(PyArrayObject *dobject,
                       char **dtype, int *ddims, int *dshape, char **dvalue,
                       int reversedims, int transposedata, s2p_ctx_t  *context)
{
  int n = 0;
  int pshape = 1;

  ddims[0] = 0;
  *dtype = DT_MT;
  *dvalue = NULL;

  L3M_CLEARDIMS(dshape);

  if (PyArray_Check(dobject))
  {
    if (PyArray_SIZE(dobject) == 0)
    {
      S2P_TRACE(("\n# CHL:warning - numpy array has a zero size\n"));
      return 0;
    }
    ddims[0] = PyArray_NDIM(dobject);
    transposedata = !is_fortran_contiguous(dobject);
    if (PyArray_BASE(dobject) != NULL)
    {
      // This is a view of an array
      if (transposedata) {
        S2P_TRACE(("\n# CHL:warning - array view should be fortran contiguous\n"));
        return 0;
      }
    }
    /* S2P_TRACE(("\n# CHL:is_fortran_contiguous : %d\n",transposedata)); */
    if (transposedata)
    {
      dobject = (PyArrayObject*)(PyArray_Transpose(dobject, NULL));
    }
    if (reversedims)
    {
        for (n = 0; n < ddims[0]; n++)
        {
          dshape[n] = (int)PyArray_DIM(dobject, ddims[0] - n - 1);
        }
    }
    else
    {
      for (n = 0; n < ddims[0]; n++)
      {
        dshape[n] = (int)PyArray_DIM(dobject, n);
      }
    }

    *dvalue = (char*)PyArray_DATA(dobject);
  }
  else if ((PyObject*)dobject == Py_None)
  {
    return 0;
  }
  else
  {
    S2P_TRACE(("\n# CHL:warning - not a numpy array\n"));
    return 0;
  }

  for (n = 0; n < ddims[0]; n++)
  {
    pshape *= dshape[n];
  }
  if (S2P_HASFLAG(S2P_FFORTRANFLAG)
    && (pshape > 1 && !PyArray_IS_F_CONTIGUOUS(dobject))
    && (PyArray_NDIM(dobject) > 1)
    && (PyArray_NDIM(dobject) < MAXDIMENSIONVALUES))
  {
    S2P_TRACE(("\n# CHL:warning - array should be fortran\n"));
    return 0;
  }

  switch(PyArray_TYPE(dobject))
  {
  /* --- Integer */
  case NPY_INT32:
    *dtype = DT_I4;
    break;
  /* --- Long */
  case NPY_INT64:
    *dtype = DT_I8;
    break;
  /* --- Float */
  case NPY_FLOAT32:
    *dtype = DT_R4;
    break;
  /* --- Double */
  case NPY_FLOAT64:
    *dtype = DT_R8;
    break;
  /* --- String */
  case NPY_STRING:
    *dtype = DT_C1;
    break;
  case NPY_COMPLEX64:
    *dtype = DT_X4;
    break;
  case NPY_COMPLEX128:
    *dtype = DT_X8;
  default:
    S2P_TRACE(("\n# CHL: ERROR - numpy array dtype not in [C1,I4,I8,R4,R8,X4,X8]\n"));
    return 0;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
static int s2p_getFortranContiguousData(PyArrayObject *dobject,
                       char **dtype, int *ddims, int *dshape, char **dvalue,
                       int reversedims, int transposedata, s2p_ctx_t  *context)
{
  int n = 0;
  int pshape = 1;
  int viewdata = 0;

  ddims[0] = 0;
  *dtype = DT_MT;
  *dvalue = NULL;

  L3M_CLEARDIMS(dshape);

  if (PyArray_Check(dobject))
  {
    if (PyArray_SIZE(dobject) == 0)
    {
      S2P_TRACE(("\n# CHL:warning - numpy array has a zero size\n"));
      return 0;
    }
    ddims[0] = PyArray_NDIM(dobject);
    if (!is_fortran_contiguous(dobject)) {
      dobject = (PyArrayObject*)(PyArray_GETCONTIGUOUS(dobject));
    }
    transposedata = !is_fortran_contiguous(dobject);
    /* S2P_TRACE(("\n# CHL:is_fortran_contiguous : %d\n",transposedata)); */
    if (transposedata)
    {
      dobject = (PyArrayObject*)(PyArray_Transpose(dobject, NULL));
    }
    if (reversedims)
    {
        for (n = 0; n < ddims[0]; n++)
        {
          dshape[n] = (int)PyArray_DIM(dobject, ddims[0] - n - 1);
        }
    }
    else
    {
      for (n = 0; n < ddims[0]; n++)
      {
        dshape[n] = (int)PyArray_DIM(dobject, n);
      }
    }

    *dvalue = (char*)PyArray_DATA(dobject);
  }
  else if ((PyObject*)dobject == Py_None)
  {
    return 0;
  }
  else
  {
    S2P_TRACE(("\n# CHL:warning - not a numpy array\n"));
    return 0;
  }

  for (n = 0; n < ddims[0]; n++)
  {
    pshape *= dshape[n];
  }
  if (S2P_HASFLAG(S2P_FFORTRANFLAG)
    && (pshape > 1 && !PyArray_IS_F_CONTIGUOUS(dobject))
    && (PyArray_NDIM(dobject) > 1)
    && (PyArray_NDIM(dobject) < MAXDIMENSIONVALUES))
  {
    S2P_TRACE(("\n# CHL:warning - array should be fortran\n"));
    return 0;
  }

  switch(PyArray_TYPE(dobject))
  {
  /* --- Integer */
  case NPY_INT32:
    *dtype = DT_I4;
    break;
  /* --- Long */
  case NPY_INT64:
    *dtype = DT_I8;
    break;
  /* --- Float */
  case NPY_FLOAT32:
    *dtype = DT_R4;
    break;
  /* --- Double */
  case NPY_FLOAT64:
    *dtype = DT_R8;
    break;
  /* --- String */
  case NPY_STRING:
    *dtype = DT_C1;
    break;
  case NPY_COMPLEX64:
    *dtype = DT_X4;
    break;
  case NPY_COMPLEX128:
    *dtype = DT_X8;
  default:
    S2P_TRACE(("\n# CHL: ERROR - numpy array dtype not in [C1,I4,I8,R4,R8,X4,X8]\n"));
    return 0;
  }
  return 1;
}
/* ------------------------------------------------------------------------- */
static int s2p_hasToReverseDims(char *name, char *label, s2p_ctx_t *context)
{
  if (S2P_HASFLAG(S2P_FREVERSEDIMS)) { return 1; }
  return 0;
  if (!strcmp(label, "DataArray_t")) { return 1; }
  if (!strcmp(label, "DimensionalUnits_t")) { return 1; }
  if (!strcmp(label, "AdditionalUnits_t")) { return 1; }
  if (!strcmp(label, "DimensionalExponents_t")) { return 1; }
  if (!strcmp(label, "AdditionalExponents_t")) { return 1; }
  return 0;
}
/* ------------------------------------------------------------------------- */
static int s2p_hasToTransposeData(char *name, char *label, s2p_ctx_t *context)
{
  return 1;
  if (!strcmp(label, "DataArray_t")) { return 1; }
  if (!strcmp(label, "DimensionalUnits_t")) { return 1; }
  if (!strcmp(label, "AdditionalUnits_t")) { return 1; }
  if (!strcmp(label, "DimensionalExponents_t")) { return 1; }
  if (!strcmp(label, "AdditionalExponents_t"))
    if ((!strcmp(label, "IndexRange_t")
      && (!strcmp(name, "PointRange")))) {
      return 1;
    }
  if ((!strcmp(label, "IndexRange_t")
    && (!strcmp(name, "PointRangeDonor")))) {
    return 1;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
static void s2p_trackRefs(s2p_ctx_t *context, PyObject *tree)
{
  Py_ssize_t n, ct, st;

  if (S2P_HASFLAG(S2P_FDEBUG))
  {
    st = PyList_GetItem(tree, 0)->ob_refcnt;
    if (PyList_GetItem(tree, 1) == Py_None)
    {
      ct = 0;
    }
    else
    {
      ct = PyList_GetItem(tree, 1)->ob_refcnt;
    }
    fprintf(stdout, "[%ld,%ld,%ld,%ld]\n",
      st, ct,
      PyList_GetItem(tree, 2)->ob_refcnt, PyList_GetItem(tree, 3)->ob_refcnt);
    fflush(stdout);
    if (PyList_Size(tree) > 3)
    {
      for (n = 0; n < PyList_Size(PyList_GetItem(tree, 2)); n++)
      {
        s2p_trackRefs(context, PyList_GetItem(PyList_GetItem(tree, 2), n));
      }
    }
  }
}
/* ------------------------------------------------------------------------- */
static void s2p_freenodetable(s2p_ctx_t *context)
{
  size_t s = sizeof(PyObject*);

  if (context->pol_oid != NULL)
  {
    free(context->pol_oid);
    context->pol_oid = NULL;
  }
}
/* ------------------------------------------------------------------------- */
static int s2p_nodeAlreadyParsedCheck(PyObject *node, s2p_ctx_t *context)
{
  int n;

  if (context->pol_cur == 0)
  {
    return 0;
  }
  for (n = 0; n < context->pol_cur; n++)
  {
    if (context->pol_oid[n] == node)
    {
      return 1;
    }
  }
  return 0;

}/* ------------------------------------------------------------------------- */
static int s2p_nodeNotAlreadyParsed(PyObject *node, s2p_ctx_t *context)
{
  int n;
  size_t s = sizeof(PyObject*);

  if (context->pol_cur == 0)
  {
    context->pol_max = __MAXOBJTABLE;
    context->pol_oid = (PyObject**)malloc(s*context->pol_max);
  }
  else
  {
    for (n = 0; n < context->pol_cur; n++)
    {
      if (context->pol_oid[n] == node)
      {
        return 0;
      }
    }
    if (context->pol_cur == context->pol_max)
    {
      context->pol_max += __MAXOBJTABLE;
      context->pol_oid = (PyObject**)realloc(context->pol_oid, s*context->pol_max);
    }
  }
  context->pol_oid[context->pol_cur++] = node;
  return 1;
}
/* ------------------------------------------------------------------------- */
static int s2p_freeContext(s2p_ctx_t **context_ptr)
{
  s2p_ctx_t *context;

  context = *context_ptr;
  s2p_closeallHDF(context);
  s2p_freelinktable(context);
  s2p_freepathtable(context);
  s2p_freenodetable(context);
  free(context->sha256);
  free(context);
  context_ptr = NULL;

  return 1;
}
/* ------------------------------------------------------------------------- */
static void s2p_removeMissingChildren(hid_t        id,
                                      PyObject    *tree,
                                      s2p_ctx_t   *context,
                                      L3_Cursor_t *l3db)
{
  Py_ssize_t sz, n, child, toremove = 0;
  L3_Node_t *node = NULL, *cnode = NULL;
  PyObject *ctree;
  char *tnodename;

  L3M_NEWNODE(node);
  /* SHOULD delete after parsing subnodes...
     Do **not** release argument id
  */
  if (S2P_HASFLAG(S2P_FDELETEMISSING))
  {
    child = 0;
    ctree = PyList_GetItem(tree, 2);
    sz = PyList_Size(ctree);
    PyObject** ascii_strings = NULL;
    if (sz > 0)
    {
       ascii_strings = (PyObject **)malloc(sz*sizeof(PyObject*));
       for (n = 0; n < sz; n++)
       {
           ascii_strings[n] = NULL;
           ascii_strings[n] = PyUnicode_AsASCIIString(PyList_GetItem(PyList_GetItem(ctree, n), 0));
       }
    }
    L3M_NEWNODE(cnode);
    __node_count++;
    L3M_UNSETFLAG(l3db, L3F_WITHDATA);
    L3M_SETFLAG(l3db, L3F_WITHCHILDREN);
    node = L3_nodeRetrieve(l3db, id, node);
    while ((node != NULL) &&
      (node->children != NULL) &&
      (node->children[child] != -1))
    {
      L3M_UNSETFLAG(l3db, L3F_WITHDATA);
      L3M_UNSETFLAG(l3db, L3F_WITHCHILDREN);
      cnode = L3_nodeRetrieve(l3db, node->children[child], cnode);
      toremove = 1;
      /* Not efficient for large sz */
      for (n = 0; n < sz; n++)
      {
        if (ascii_strings[n] == NULL) continue;
        tnodename = PyBytes_AsString(ascii_strings[n]);

        if (!strcmp(tnodename, cnode->name))
        {
          S2P_TRACE(("# CHL:found in tree and file [%s]\n", tnodename));
          Py_DECREF(ascii_strings[n]);
          ascii_strings[n] = NULL;
          toremove = 0;
          break;
        }
      }
      if (toremove)
      {
        S2P_TRACE(("# CHL:node child remove [%s]\n", cnode->name));
        L3_nodeDelete(l3db, node->id, cnode->name);
      }
      child++;
    }
    L3_nodeFree(&cnode);
    if (sz >0)
    {
      for (n = 0; n < sz; n++)
      {
        if (ascii_strings[n] != NULL)
            Py_DECREF(ascii_strings[n]);
        ascii_strings[n] = NULL;
      }
      free(ascii_strings);
    }
  }
  L3_nodeRelease(&node, L3F_R_MEM_CHILDREN | L3F_R_HID_CHILDREN | L3F_R_MEM_DATA);
}
/* ------------------------------------------------------------------------- */
/* Main read function, parses a node and recurse on its children, creates
   the Python node structure.
*/
static PyObject* s2p_parseAndReadHDF(L3_Node_t   *anode,
                                     char        *curpath,
                                     char        *subpath,
                                     s2p_ctx_t   *context,
                                     L3_Cursor_t *l3db)
{
  int ndim = 0, tsize = 0, n = 0, child = 0, arraytype = -1, skip = 0, c = 0, p1 = 0, p2 = 0, soft = 0;
  int index = -1, rank = -1, count = -1, isinterlaced = 0;
  int trackpath = 0, ix = 0, skipnewarray = 0, linkstatus = 0, islinknode = 0;
  int pathinlist = 0, memsize = 0, ispartial = 0, iscontiguous = 0, itemsize_ = 0;
  hid_t actualid = -1, id;
  PyObject *o_clist = NULL, *o_value = NULL, *o_child = NULL;
  PyObject *o_node = NULL, *u_value = NULL, *c_value = NULL;
  npy_uint32 npyflags = -1;
  L3_Cursor_t *lkl3db = NULL;
  L3_Node_t *rnode = NULL, *cnode = NULL;
  char *curdir = NULL, *curfile = NULL, *name;
  s2p_ent_t *curhdf;
  char destnode[MAXPATHSIZE];
  char destdir[MAXFILENAMESIZE];
  char destfile[MAXFILENAMESIZE];
  void *data_ptr = NULL;
  char localnode[MAXPATHSIZE];
  char targetfile[MAXFILENAMESIZE];
  npy_intp npy_dim_vals[NPY_MAXDIMS];
  hsize_t s_offset[L3C_MAX_DIMS], s_stride[L3C_MAX_DIMS];
  hsize_t s_count[L3C_MAX_DIMS], s_block[L3C_MAX_DIMS];
  hsize_t d_offset[L3C_MAX_DIMS], d_stride[L3C_MAX_DIMS];
  hsize_t d_count[L3C_MAX_DIMS], d_block[L3C_MAX_DIMS];
  char altlabel[L3C_MAX_NAME + 1];

  PyObject *o_name = NULL;
  PyObject *b_name = NULL;
  PyObject *o_altlabel = NULL;
  PyObject *b_altlabel = NULL;

  id = anode->id;
  name = anode->name;

  context->dpt -= 1;
  trackpath = 1;
  if ((subpath == NULL)
    || ((subpath != NULL) && (subpath[0] == '\0'))
    || ((subpath != NULL) && (!strcmp(subpath, "/"))))
  {
    trackpath = 0;
  }
  else
  {
    S2P_TRACE(("# CHL:target path (%s)\n", subpath));
  }
  o_value = NULL;

  /* In case of subpath, we are in a link search or sub-tree retrieval. We
     skip the Python node creation but we keep track of links.
     We retrieve the actual node to create (destination) and resume the
     parse from this node.
  */
  actualid = id;
  curhdf = s2p_getcurrentHDF(context);
  curdir = curhdf->dirname;
  curfile = curhdf->filename;
  /* step 1: link management ---------------------------------------------- */
  islinknode = L3_isLinkNode(l3db, id, destfile, destnode);
  if (islinknode)
  {
    linkstatus = S2P_LKOK;
    strcpy(localnode, curpath);
    strcat(localnode, "/");
    strcat(localnode, name);
    if (destfile[0] == '\0')
    {
      S2P_TRACE(("# CHL:link node found (internal): [%s/%s]\n", curpath, name));
      soft = 1;
    }
    else
    {
      S2P_TRACE(("# CHL:link node found (external): [%s/%s]\n", curpath, name));
      soft = 0;
    }
    if (s2p_pathToSkip(context,localnode))
    {
      S2P_TRACE(("# CHL:path skip \'%s\'\n", localnode));
      context->dpt+=1;
      return NULL;
    }
    if (soft)
    {
      /* destination node is in the same file */
      actualid = L3_nodeFind(l3db, l3db->root_id, destnode);
      s2p_linkstack(curdir, curfile, localnode,
        curdir, curfile, destnode, linkstatus, context);
      if (H5Iis_valid(id))
      {
        S2P_H5_GCLOSE("LINK LOCAL\n", id);
      }
    }
    else
    {
      /* L3 layer has its own link search path,
         we have to ask L3 the directory used to open the actual file. */
      destdir[0] = '\0';
      ix = CHL_getFileInSearchPath(l3db, destfile);
      if (ix != -1)
      {
        strcpy(destdir, CHL_getLinkSearchPath(l3db, ix));
      }
      else
      {
        linkstatus = S2P_LKFAIL | S2P_LKNOFILE;
        S2P_TRACE(("# CHL:linked-to file not found (no [%s] in search path)\n",
          destfile));
        if (S2P_HASFLAG(S2P_FFOLLOWLINKS))
        {
          s2p_linkstack(curdir, curfile, localnode,
            destdir, destfile, destnode, linkstatus, context);
          if (H5Iis_valid(id))
          {
            S2P_H5_GCLOSE("LINK NO FILE\n", id);
          }
          context->dpt += 1;
          return NULL;
        }
      }
      if (S2P_HASFLAG(S2P_FFOLLOWLINKS))
      {
        S2P_TRACE(("# CHL:parse follow link: [%s][%s][%s]\n",
          destdir, destfile, destnode));
        if (ix != -1)
        {
          /* We recurse on destination file, S2P functions should be used and
             not HDF functions, because HDF lib would hide links to S2P.
             Then we start our parse from the root node and keep track
             of links, the actual node is finally used at the end. */
          sprintf(targetfile, "%s/%s", destdir, destfile);
          lkl3db = s2p_addoneHDF(destdir, destfile, context, 0);
          if (!lkl3db)
          {
            S2P_TRACE(("# CHL:linked-to file [%s] found unreadable\n",
              targetfile));
            linkstatus = S2P_LKFAIL | S2P_LKFILENOREAD;
          }
          else
          {
            s2p_setlinksearchpath(lkl3db, context);
            actualid = s2p_linktrack(lkl3db, destnode, context);
            if (actualid == -1)
            {
              S2P_TRACE(("# CHL:linked-to node (%s) does not exist\n",
                destnode));
              linkstatus = S2P_LKFAIL | S2P_LKNONODE;
            }
            if (H5Iis_valid(id))
            {
              S2P_H5_GCLOSE("LINK REPLACE BY ACTUAL NODE\n", id);
            }
          }
        }
        s2p_linkstack(curdir, curfile, localnode,
          destdir, destfile, destnode, linkstatus, context);
        if ((ix == -1) || !lkl3db || (actualid == -1))
        {
          context->dpt += 1;
          return NULL;
        }
      }
      else
      {
        if (ix == -1)
        {
          strcpy(destdir, "");
        }
        s2p_linkstack(curdir, curfile, localnode,
          destdir, destfile, destnode,
          S2P_LKIGNORED, context);
        //context->dpt+=1;
        //return NULL;
      }
    }
  }
  /* step 2: retrieve actual object id ------------------------------------ */
  ENTER_NOGIL_BLOCK(1);
  L3_nodeRelease(&rnode, L3F_R_ALL);
  L3M_NEWNODE(rnode);
  __node_count++;
  L3M_SETFLAG(l3db, L3F_WITHCHILDREN);
  L3M_UNSETFLAG(l3db, L3F_WITHDATA);
  rnode = L3_nodeRetrieve(l3db, actualid, rnode);
  anode->id = actualid;
  LEAVE_NOGIL_BLOCK();
  if (rnode == NULL)
  {
    S2P_TRACE(("# CHL:retrieve returns NULL POINTER at (%s)\n", curpath));
    context->dpt += 1;
    if (islinknode)
    {
      s2p_popHDF(context);
    }
    return NULL;
  }
  strcat(curpath, "/");
  strcat(curpath, rnode->name);
  skipnewarray = 0;
  ispartial = s2p_filterDataPartial(context, curpath,
    s_offset, s_stride, s_count, s_block,
    d_offset, d_stride, d_count, d_block);
  iscontiguous = s2p_filterDataContiguous(context, curpath,
    &index, &rank, &count, &isinterlaced);
  /* step 3: -------------------------------------------------------------- */
  if (!S2P_HASFLAG(S2P_FNODATA))
  {
    /* 'update' always overwrites 'contiguous' */
    u_value = s2p_getUpdateObjectByPath(context->upd_pth, curpath);
    if (S2P_HASFLAG(S2P_FUPDATEONLY))
    {
      skipnewarray = 1;
    }
    if (u_value != NULL)
    {
      /* path is in the update list */
      /* if not None: update numpy.ndarray */
      /* if     None: create new numpy.ndarray */
      if (u_value != Py_None)
      {
        /* TODO: check size, data type here */
        /* TODO: check ref count newobj/updated obj */
        rnode->data = PyArray_DATA((PyArrayObject*)u_value);
        skipnewarray = 1;
        L3M_SETFLAG(l3db, L3F_NOALLOCATE);
        S2P_TRACE(("# CHL:data update found [%s]\n", curpath));
      }
      else
      {
        skipnewarray = 0;
      }
    }
    ENTER_NOGIL_BLOCK(1);
    L3M_SETFLAG(l3db, L3F_WITHDATA);
    L3M_SETFLAG(l3db, L3F_WITHCHILDREN);
    L3_nodeRelease(&rnode, L3F_R_MEM_CHILDREN | L3F_R_HID_CHILDREN | L3F_R_MEM_DATA);
    /* node retrieve with not NODATA flag on */
    if (ispartial)
    {
      rnode = L3_nodeRetrievePartial(l3db, actualid,
        s_offset, s_stride, s_count, s_block,
        d_offset, d_stride, d_count, d_block,
        rnode);
      s2p_pathstack(curpath, S2P_SPARTIAL, rnode->dtype, rnode->dims, context);
    }
    else if ((iscontiguous) && (index != -1) && (index < S2P_EMAXCTGINDEX))
    {
      /* input ptr is either NULL or the BASE ptr for all contiguous memory */
      rnode->base = context->ctg_obj[index];
      rnode = L3_nodeRetrieveContiguous(l3db, actualid,
        index, rank, count, isinterlaced,
        rnode);
      /* output data ptr is the numpy array base ptr,
       then NOT the allocated base ptr which is stored for other fills */
      context->ctg_obj[index] = rnode->base;
      s2p_pathstack(curpath, S2P_SCONTIGUOUS, rnode->dtype, rnode->dims, context);
      if (rnode->data == NULL) { printf("NULL RETURN\n"); }
    }
    else
    {
      rnode = L3_nodeRetrieve(l3db, actualid, rnode);
    }
    L3M_UNSETFLAG(l3db, L3F_NOALLOCATE);
    LEAVE_NOGIL_BLOCK();
  }
  /* step 4: check if the exact path is to be skipped --------------------- */
  if (s2p_pathToSkip(context, curpath) && strcmp(curpath, L3S_ROOTNODEPATH))
  {
    S2P_TRACE(("# CHL:path skip \'%s\'\n", curpath));
    curpath[strlen(curpath) - strlen(rnode->name) - 1] = '\0';
    L3_nodeAndDataFree(&rnode);
    context->dpt += 1;
    if (islinknode)
    {
      s2p_popHDF(context);
    }
    return NULL;
  }
  /* step 5: check if there is a start path (to be merged with step 4) ---- */
  if (trackpath && strcmp(curpath, L3S_ROOTNODEPATH))
  {
    ENTER_NOGIL_BLOCK(1);
    S2P_TRACE(("# CHL:path filter \'%s\' \'%s\'", subpath, curpath));
    skip = 0;
    if (subpath != NULL)
    {
      skip = !s2p_issubpath(subpath, curpath, 0);
    }
    LEAVE_NOGIL_BLOCK();
    if (skip)
    {
      curpath[strlen(curpath) - strlen(rnode->name) - 1] = '\0';
      L3_nodeAndDataFree(&rnode);
      S2P_TRACE((" skip\n"));
      context->dpt += 1;
      return NULL;
    }
    S2P_TRACE((" ok\n"));
  }
  if (trackpath && !strcmp(curpath, L3S_ROOTNODEPATH))
  {
    curpath[strlen(curpath) - strlen(rnode->name) - 1] = '\0';
  }
  /* step 6: change label if alternate requested -------------------------- */
  strcpy(altlabel, rnode->label);
  if (S2P_HASFLAG(S2P_FALTERNATESIDS))
  {
    if (!strcmp(rnode->label, "\"int[1+...+IndexDimension]\""))
    {
      strcpy(altlabel, "DiffusionModel_t");
    }
    if (!strcmp(rnode->label, "\"int[IndexDimension]\""))
    {
      if (!strcmp(rnode->name, "InwardNormalIndex"))
      {
        strcpy(altlabel, "InwardNormalIndex_t");
      }
      else if (!strcmp(rnode->name, "Transform"))
      {
        strcpy(altlabel, "Transform_t");
      }
      else
      {
        strcpy(altlabel, "Transform_t");
      }
    }
  }
  /* step 7: create python node with retrieved infos ---------------------- */
  S2P_TRACE(("# CHL:node %ld (%s) [%s] ", rnode->id, curpath, rnode->label));
  if (!skipnewarray || (rnode->dtype != L3E_VOID))
  {
    ENTER_NOGIL_BLOCK(1);
    S2P_TRACE(("[%s]", L3_typeAsStr(rnode->dtype)));
    tsize = 1;
    n = 0;
    ndim = 0;
    npy_dim_vals[0] = 0;
    while ((n < L3C_MAX_DIMS) && (rnode->dims[n] != -1))
    {
      tsize *= rnode->dims[n];
      ndim++;
      n++;
    }
    n = 0;
    while ((n < L3C_MAX_DIMS) && (rnode->dims[n] != -1))
    {
      if (s2p_hasToReverseDims(rnode->name, rnode->label, context))
      {
        npy_dim_vals[ndim - n - 1] = rnode->dims[n];
      }
      else
      {
        npy_dim_vals[n] = rnode->dims[n];
      }
      n++;
    }
    S2P_TRACE(("{"));
    for (n = 0; n < ndim; n++)
    {
      S2P_TRACE(("%d", (int)(npy_dim_vals[n])));
      if (n < ndim - 1)
      {
        S2P_TRACE(("x"));
      }
    }
    S2P_TRACE(("}=%d (max=%d)", tsize, context->mxs));
    if (S2P_HASFLAG(S2P_FNODATA))
    {
      if ((context->mxs == -1) || (tsize > context->mxs))
      {
        s2p_pathstack(curpath, S2P_SNODATA, rnode->dtype, rnode->dims, context);
        rnode->dtype = L3E_VOID;
      }
      else
      {
        /* Load actual data ptr that should be
           shared with numpy.ndarray object.
           Do not de-allocate node data ptr */
        L3_nodeRelease(&rnode,
          L3F_R_MEM_DATA | L3F_R_HID_CHILDREN | L3F_R_MEM_CHILDREN);
        L3M_NEWNODE(rnode);
        __node_count++;
        L3M_SETFLAG(l3db, L3F_WITHDATA);
        L3M_SETFLAG(l3db, L3F_WITHCHILDREN);
        /* node retrieve with NODATA flag on but below maxdata */
        if (ispartial)
        {
          rnode = L3_nodeRetrievePartial(l3db, actualid,
            s_offset, s_stride, s_count, s_block,
            d_offset, d_stride, d_count, d_block,
            rnode);
          s2p_pathstack(curpath, S2P_SPARTIAL, rnode->dtype, rnode->dims, context);
        }
        else if (iscontiguous)
        {
          printf("CONTIGUOUS 1\n");
        }
        else
        {
          rnode = L3_nodeRetrieve(l3db, actualid, rnode);
        }
      }
    }
    switch(rnode->dtype)
    {
    case L3E_I4:
    case L3E_I4ptr:
      arraytype = NPY_INT32;
      memsize = sizeof(int)*tsize;
      break;
    case L3E_C1:
    case L3E_C1ptr:
      arraytype = NPY_STRING;
      memsize = sizeof(char)*tsize;
      break;
    case L3E_R8:
    case L3E_R8ptr:
      arraytype = NPY_FLOAT64;
      memsize = sizeof(double)*tsize;
      break;
    case L3E_I8:
    case L3E_I8ptr:
      arraytype = NPY_INT64;
      memsize = sizeof(long)*tsize;
      break;
    case L3E_R4:
    case L3E_R4ptr:
      arraytype = NPY_FLOAT32;
      memsize = sizeof(float)*tsize;
      break;
    case L3E_X4:
    case L3E_X4ptr:
      arraytype = NPY_COMPLEX64;
      memsize = 2*sizeof(float)*tsize;
      break;
    case L3E_X8:
    case L3E_X8ptr:
      arraytype = NPY_COMPLEX128;
      memsize = 2*sizeof(double)*tsize;
      break;
    default:
      arraytype = -1;
      memsize = 0;
      break;
    }
    LEAVE_NOGIL_BLOCK();
    if (arraytype != -1)
    {
      npyflags = NPY_ARRAY_BEHAVED;
      if (S2P_HASFLAG(S2P_FOWNDATA)) { npyflags |= NPY_ARRAY_OWNDATA; }
      else { npyflags &= ~NPY_ARRAY_OWNDATA; }
      if (S2P_HASFLAG(S2P_FFORTRANFLAG)) { npyflags |= NPY_ARRAY_F_CONTIGUOUS; }
      else { npyflags |= NPY_ARRAY_C_CONTIGUOUS; }
      if (iscontiguous && (index != -1) && (index < S2P_EMAXCTGINDEX) && (rank != 0))
      {
        npyflags &= ~NPY_ARRAY_OWNDATA;
      }
      data_ptr = rnode->data;
      rnode->data = NULL;
      itemsize_ = arraytype == NPY_STRING ? 1 : 0;
      o_value = PyArray_New(&PyArray_Type, ndim, npy_dim_vals, arraytype,
                            NULL, data_ptr, itemsize_, npyflags, NULL);
      /* force flags that may be reset by API calls */
      PyArray_CLEARFLAGS((PyArrayObject*)o_value, 0xFFFF);
      PyArray_ENABLEFLAGS((PyArrayObject*)o_value, npyflags);
    }
  }
  S2P_TRACE(("\n"));
  /* Loop on children. This is a depth first recurse. In case of a path search,
     skip until we have the right name. */
  o_clist = PyList_New(0);

  child = 0;
  while ((rnode->children != NULL) &&
         (rnode->children[child] != -1) &&
         (context->dpt > 0))
  {
    ENTER_NOGIL_BLOCK(1);
    L3_nodeRelease(&cnode, L3F_R_ALL);
    L3M_NEWNODE(cnode);
    __node_count++;
    L3M_UNSETFLAG(l3db, L3F_FOLLOWLINKS);
    L3M_UNSETFLAG(l3db, L3F_WITHDATA);
    L3M_UNSETFLAG(l3db, L3F_WITHCHILDREN);
    cnode = L3_nodeRetrieve(l3db, rnode->children[child], cnode);
    LEAVE_NOGIL_BLOCK();
    /* HDF can parse paths, i.e. a node name can be a path and the
       resulting ID is the actual last node. However, we SHOULD NOT use that
       because we want to have control on LINK PARSE. */
    if (!strcmp(curpath, L3S_ROOTNODEPATH))
    {
      curpath[0] = '\0';
    }
    o_child = s2p_parseAndReadHDF(cnode,
      curpath, subpath, context, l3db);
    L3_nodeRelease(&cnode, L3F_R_MEM_CHILDREN);
    if (S2P_HASFLAG(S2P_FFOLLOWLINKS))
    {
      L3M_SETFLAG(l3db, L3F_FOLLOWLINKS);
    }
    if (o_child != NULL)
    {
      PyList_Append(o_clist, o_child);
      Py_DECREF(o_child);
    }
    child++;
  }
  /* end chidren loop */

  L3_nodeRelease(&cnode, L3F_R_ALL);
  if (skipnewarray)
  {
    o_value = u_value;
  }
  if (o_value == NULL)
  {
    Py_INCREF(Py_None);
    o_value = Py_None;
  }
  if (strcmp(rnode->name, L3S_ROOTNODENAME))
  {
    curpath[strlen(curpath) - strlen(rnode->name) - 1] = '\0';
  }
  if (S2P_HASFLAG(S2P_FCHECKSUM))
  {
    int i, j;
    unsigned char  buf1[512];
    unsigned char  buf2[512];
    unsigned char *val;

    strcpy(buf1, "CGNS/PYTHON CHECKSUM CHLONE "); /* +28 */
    if (o_value != Py_None)
    {
      sprintf(buf2, "%.12d%.12d%.12d",
        PyArray_FLAGS((PyArrayObject*)o_value),
        PyArray_NDIM((PyArrayObject*)o_value),
        PyArray_ITEMSIZE((PyArrayObject*)o_value));
      strcat(buf1, buf2);/* 28+36=64 */
      j = PyArray_NDIM((PyArrayObject*)o_value);
      for (i = 0; i < 12; i++)
      {
        if (i < j)
        {
          sprintf(buf2, " %.12d %.12d",
            PyArray_DIM((PyArrayObject*)o_value, i),
            PyArray_STRIDE((PyArrayObject*)o_value, i));
        }
        else
        {
          sprintf(buf2, " %.12d %.12", 0, 0);
        }
        strcat(buf1, buf2);/* 64+(2+12+12)*12=376 */
      }
      val = (unsigned char *)PyArray_DATA((PyArrayObject*)o_value);
      i = PyArray_NBYTES((PyArrayObject*)o_value);
      sha256_update(context->sha256, val, i);
    }
    sprintf(buf2, " %-32s %64s %.16d %.16d ",
      name, rnode->label, child, context->dpt);
    strcat(buf1, buf2); /* 376+5+128=509 */
    i = strlen(buf1);
    sha256_update(context->sha256, buf1, i);
    /* printf("NODE [%s]\n",buf1); */
  }
  o_node = PyList_New(4);
  b_name = PyBytes_FromString(name);
  b_altlabel = PyBytes_FromString(altlabel);
  o_name = PyUnicode_FromEncodedObject(b_name, "ascii", "strict");
  o_altlabel = PyUnicode_FromEncodedObject(b_altlabel, "ascii", "strict");
  Py_DECREF(b_name);
  Py_DECREF(b_altlabel);
  PyList_SetItem(o_node, 0, o_name);
  PyList_SetItem(o_node, 1, o_value);
  PyList_SetItem(o_node, 2, o_clist);
  PyList_SetItem(o_node, 3, o_altlabel);
  ENTER_NOGIL_BLOCK(1);
  context->dpt += 1;
  if (islinknode)
  {
    s2p_popHDF(context);
  }
  else
  {
    L3_nodeRelease(&rnode, L3F_R_HID_CHILDREN);
  }
  /*
  if (H5Iis_valid(actualid))
  {
    H5Gclose(actualid);
  }
  */
  LEAVE_NOGIL_BLOCK();

  return o_node;
}
/* ------------------------------------------------------------------------- */
static int s2p_parseAndWriteHDF(hid_t        id,
                                PyObject    *tree,
                                char        *curpath,
                                char        *path,
                                s2p_ctx_t   *context,
                                L3_Cursor_t *l3db)
{
  char *name = NULL, *label = NULL, *tdat = NULL, *altlabel;
  int tsize = 1, transpose = 0, reverse = 0, toupdate = 0, toskip = 0;
  int storage = L3_CONTIGUOUS_STORE;
  Py_ssize_t sz = 0, n = 0;
  int ndat = 0, ret = 1, child = 0, ispartial = 0;
  char *vdat = NULL;
  L3_Node_t *node = NULL, *cnode = NULL;
  s2p_lnk_t *lke = NULL;
  hid_t lid = -1, oldid = -1;
  int hadWithData = 0, *ddat;
  hsize_t s_offset[L3C_MAX_DIMS], s_stride[L3C_MAX_DIMS];
  hsize_t s_count[L3C_MAX_DIMS], s_block[L3C_MAX_DIMS];
  hsize_t d_offset[L3C_MAX_DIMS], d_stride[L3C_MAX_DIMS];
  hsize_t d_count[L3C_MAX_DIMS], d_block[L3C_MAX_DIMS];
  PyObject *ascii_name = NULL;
  PyObject *ascii_label = NULL;

  STR_ALLOC(altlabel, L3C_MAX_NAME + 1);
  DIM_ALLOC(ddat, int, NPY_MAXDIMS);

  if ((curpath[0] != '\0') && s2p_nodeAlreadyParsedCheck(tree, context))
  {
    setError(S2P_EDUPLICATEUP,
      "Double update for node [%s]",
      curpath, context);
    ret = 0;
  }
  else if (S2P_CHECKNODENOCACHE(tree, context))
  {
    if (PyUnicode_Check(PyList_GetItem(tree, 0)))
    {
      ascii_name = PyUnicode_AsASCIIString(PyList_GetItem(tree, 0));
      if (ascii_name != NULL)
      {
        name = PyBytes_AsString(ascii_name);
      }
      else
      {
        setError(S2P_EBADSTRUCTOB,
          "Failed to retrieve an ASCII name for node [%s]",
          curpath, context);
        ret = 0;
        name = "\0";
      }
    }
    else
    {
      name = PyBytes_AsString(PyList_GetItem(tree, 0));
    }
    if (PyUnicode_Check(PyList_GetItem(tree, 3)))
    {
      ascii_label = PyUnicode_AsASCIIString(PyList_GetItem(tree, 3));
      if (ascii_label != NULL)
      {
        label = PyBytes_AsString(ascii_label);
      }
      else
      {
        setError(S2P_EBADSTRUCTOB,
          "Failed to retrieve an ASCII label for node [%s]",
          curpath, context);
        ret = 0;
        label = "\0";
      }
    }
    else
    {
      label = PyBytes_AsString(PyList_GetItem(tree, 3));
    }
    strcpy(altlabel, label);
    if (!strcmp(altlabel, "DiffusionModel_t"))
    {
      strcpy(altlabel, "\"int[1+...+IndexDimension]\"");
    }
    else if (!strcmp(altlabel, "Transform_t"))
    {
      strcpy(altlabel, "\"int[IndexDimension]\"");
    }
    else if (!strcmp(altlabel, "InwardNormalIndex_t"))
    {
      strcpy(altlabel, "\"int[IndexDimension]\"");
    }
    strcat(curpath, "/");
    strcat(curpath, name);
    toskip = s2p_pathToSkip(context, curpath);
    if (toskip)
    {
      S2P_TRACE(("# CHL:path skip \'%s\'\n", curpath));
    }
    lke = s2p_checklinktable(context, curpath);
    if (lke != NULL)
    {
      S2P_TRACE(("# CHL: [%s] linked to [%s][%s]\n", \
        curpath, lke->dst_filename, lke->dst_nodename));
      if (S2P_HASFLAG(S2P_FUPDATE))
      {
        L3M_SETFLAG(l3db, L3F_LINKOVERWRITE);
      }
      lid = L3_nodeLink(l3db, id, name, lke->dst_filename, lke->dst_nodename);
      lke->status = S2P_LKUPDATED;
      if (H5Iis_valid(lid))
      {
        S2P_H5_GCLOSE("LINK PARSE\n", lid);
      }
      if (S2P_HASFLAG(S2P_FPROPAGATE))
      {
        /* we managed the link, now if PROPAGATE is set continue to
           recurse on CGNS/Python tree in the target dir/file
           in the sub-tree starting from the current linked-to path
           call again save with (tree, links, paths) flags and set
           starting subtree to current linked-to node.
           if there is incomplete nodes data listed in paths var,
           these have to be skipped using their remote path, not local,
           then we have first to identify these nodes in order to
           update the paths var - check also this name change does not
           overwrite existing node paths (possible?) */
           /*
           ret&=s2p_parseAndWriteHDF(parentid,
                       PyDict_GetItemString(context->upd_pth,tdat),
                         parentnodename,"",context,l3db);
           s2p_saveAsHDF(lke->dst_dirname,lke->dst_filename,subtree,
                 context->lnk_obj,context->flg,context->dpt,context->lsp,
                 PyObject *update,
                 PyObject *skip,
                 context->upd_pth_lk,context->skp_pth_lk,
                 context->err)    */

      }
    }
    else
    {
      /* if (S2P_HASFLAG(S2P_FUPDATE)) */
      /* { */
      /*        oldid=L3_nodeFind(l3db,l3db->root_id,curpath); */
      /*        if (H5Iis_valid(oldid)) */
      /*        { */
      /*          L3M_NEWNODE(node); */
      /*          L3M_UNSETFLAG(l3db,L3F_WITHDATA); */
      /*          L3M_UNSETFLAG(l3db,L3F_WITHCHILDREN); */
      /*          node=L3_nodeRetrieve(l3db,oldid,node); */
      /*          toupdate=1; */
      /*        } */
      /* } */
      transpose = s2p_hasToTransposeData(name, altlabel, context);
      reverse = s2p_hasToReverseDims(name, altlabel, context);
      if (reverse)
      {
        if (transpose)
        {
          S2P_TRACE(("# CHL:node [%s][%s]", curpath, altlabel));
        }
        else
        {
          S2P_TRACE(("# CHL:node (no transpose) [%s][%s]", curpath, altlabel));
        }
      }
      else
      {
        if (transpose)
        {
          S2P_TRACE(("# CHL:node (no swap dims) [%s][%s]", curpath, altlabel));
        }
        else
        {
          S2P_TRACE(("# CHL:node (no swap dims no transpose) [%s][%s]",
            curpath, altlabel));
        }
      }
      if (!s2p_getData((PyArrayObject*)PyList_GetItem(tree, 1),
        &tdat, &ndat, ddat, &vdat,
        reverse, transpose, context))
      {
        //Nasty hack
        if (!s2p_getFortranContiguousData((PyArrayObject*)PyList_GetItem(tree, 1),
            &tdat, &ndat, ddat, &vdat,
            reverse, transpose, context)){
          // Hack did not work
          //TODO
            ;
          }
      }
      n = 0;
      tsize = 1;
      S2P_TRACE(("{"));
      while ((n < L3C_MAX_DIMS) && (ddat[n] != -1))
      {
        tsize *= ddat[n];
        S2P_TRACE(("%d", ddat[n]));
        n++;
        if ((n < L3C_MAX_DIMS) && (ddat[n] != -1))
        {
          S2P_TRACE(("x"));
        }
      }
      if (n) { S2P_TRACE(("}=%d\n", tsize)); }
      else { S2P_TRACE(("} (no data)\n")); }
      node = L3_nodeSet(l3db, node, name, altlabel, ddat,
        L3_typeAsEnum(tdat), vdat, L3F_NONE);
      ispartial = s2p_filterDataPartial(context, curpath,
        s_offset, s_stride, s_count, s_block,
        d_offset, d_stride, d_count, d_block);
#if (CHLONE_USE_COMPACT_STORAGE == 1)
      if ((strcmp(altlabel, "DataArray_t") == 0) ||
            (strcmp(altlabel, "IndexArray_t") == 0)) {
        storage = L3_CONTIGUOUS_STORE;
      } else {
        storage = L3_COMPACT_STORE;
      }
#endif
      if (toupdate)
      {
        if (lke == NULL)
        {
          S2P_TRACE(("# CHL:node update existing\n"));
          L3M_SETFLAG(l3db, L3F_WITHDATA);
          L3_nodeUpdate(l3db, node, storage);
        }
        else if (lke != NULL)
        {
          S2P_TRACE(("# CHL:node update link\n"));
        }
      }
      else if (!toskip)
      {
        hadWithData = L3M_HASFLAG(l3db, L3F_WITHDATA);
        if (S2P_HASFLAG(S2P_FUPDATE))
        {
          L3M_SETFLAG(l3db, L3F_WITHDATA);
        }
        L3_nodeCreate(l3db, id, node, storage);
        if (!hadWithData) { L3M_UNSETFLAG(l3db, L3F_WITHDATA); }
      }
      else
      {
        S2P_TRACE(("# CHL:node [%s] skip update\n", curpath));
        oldid = L3_nodeFind(l3db, l3db->root_id, curpath);
        if (H5Iis_valid(oldid))
        {
          L3M_NEWNODE(node);
          L3M_UNSETFLAG(l3db, L3F_WITHDATA);
          L3M_UNSETFLAG(l3db, L3F_WITHCHILDREN);
          node = L3_nodeRetrieve(l3db, oldid, node);
        }
      }
      if ((H5Iis_valid(node->id)) && (PyList_Check(PyList_GetItem(tree, 2))))
      {
        sz = PyList_Size(PyList_GetItem(tree, 2));
        for (n = 0; n < sz; n++)
        {
          ret = s2p_parseAndWriteHDF(node->id,
            PyList_GetItem(PyList_GetItem(tree, 2), n),
            curpath, path, context, l3db);
          if (!ret) { break; }
        }
        s2p_removeMissingChildren(node->id, tree, context, l3db);
      }
      if ((!toskip) && (!H5Iis_valid(node->id)))
      {
        setError(S2P_ECANNOTCREAT,
          "CGNS/HDF5 cannot create node [%s]",
          curpath, context);
        ret = 0;
      }
      L3_nodeFree(&node);
    }
    curpath[strlen(curpath) - strlen(name) - 1] = '\0';
    if (ascii_name != NULL)
    {
      Py_DECREF(ascii_name);
    }
    if (ascii_label != NULL)
    {
      Py_DECREF(ascii_label);
    }
  }
  else
  {
    setError(S2P_EBADSTRUCTOB,
      "CGNS/Python incorrect node structure, node or parent '%s'",
      curpath, context);
    ret = 0;
  }
  STR_FREE(altlabel);
  DIM_FREE(ddat);

  return ret;
}
/* ------------------------------------------------------------------------- */
/* Interface Functions                                                       */
/* ------------------------------------------------------------------------- */
PyObject* s2p_loadAsHDF(char     *dirname,
                        char     *filename,
                        int       flags,
                        int       depth,
                        int       maxdata,
                        char     *path,
                        char     *searchpath,
                        PyObject *update,
                        PyObject *filter,
                        PyObject *skip,
                        PyObject *except)
{
  PyObject *tree = NULL, *links = NULL, *paths = NULL, *load_return = NULL;
  PyObject *ret = NULL, *ret_children = NULL;
  s2p_ctx_t *context = NULL;
  L3_Cursor_t *l3db = NULL;
  L3_Node_t *rnode = NULL;
  char cpath[MAXPATHSIZE];
  int n;

  CHL_import_array();

  S2P_NEWCONTEXTPTR(context);
  context->flg = flags;
  context->dpt = depth;
  context->skp_pth = skip;
  context->mxs = maxdata;
  if (PyDict_Check(update))
  {
    context->upd_pth = update;
  }
  if (PyDict_Check(filter))
  {
    context->flt_dct = filter;
  }
  context->err = except;
  context->lsp = searchpath;
  for (n = 0; n < S2P_EMAXCTGINDEX; n++)
  {
    context->ctg_obj[n] = NULL;
  }

  /* We do NOT check file name or file access, it's up to the caller to make
     such checks. Anyway, HDF will check. */
  S2P_TRACE(("# CHL:CHLone v%d.%d\n",
    SIDSTOPYTHON_MAJOR, SIDSTOPYTHON_MINOR));
  S2P_TRACE(("# CHL:load file [%s/%s]\n", dirname, filename));

#ifdef __THREADING__
  H5dont_atexit(); /* MANDATORY FIRST HDF5 function to call */
#endif

  l3db = s2p_addoneHDF(dirname, filename, context, 1);
  if (!L3M_ECHECK(l3db))
  {
    return NULL;
  }
  s2p_setlinksearchpath(l3db, context);
  L3M_NEWNODE(rnode);
  __node_count++;
  L3M_UNSETFLAG(l3db, L3F_WITHDATA);
  L3M_UNSETFLAG(l3db, L3F_WITHCHILDREN);
  rnode = L3_nodeRetrieve(l3db, l3db->root_id, rnode);

  cpath[0] = '\0';
  S2P_TRACE(("# CHL:start tree read parse\n"));
  if (S2P_HASFLAG(S2P_FCHECKSUM))
  {
    sha256_starts(context->sha256);
  }
  ret = s2p_parseAndReadHDF(rnode, cpath, path, context, l3db);
  if (S2P_HASFLAG(S2P_FCHECKSUM))
  {
  }
  links = s2p_getlinktable(context);
  paths = s2p_getpathtable(context);
  if (ret == NULL)
  {
    tree = Py_BuildValue("([sO[]s]OO)",
      CG_CGNSTree_n, Py_None, CG_CGNSTree_ts, links, paths);
  }
  else
  {
    load_return = PyList_New(0);
    tree = PyList_New(0);
    PyList_Append(tree, PyUnicode_DecodeASCII(CG_CGNSTree_n, 8, "strict"));
    if (S2P_HASFLAG(S2P_FCHECKSUM))
    {
      npy_intp npy_dim_vals[NPY_MAXDIMS];
      char buf4[65], buf3[3];
      int j;

      buf4[0] = 0;
      sha256_finish(context->sha256, context->checksum);
      for (j = 0; j < 32; j++)
      {
        sprintf(buf3, "%02X", context->checksum[j]);
        strcat(buf4, buf3);
      }
      npy_dim_vals[0] = 64;
      PyList_Append(tree, PyArray_New(&PyArray_Type, 1, npy_dim_vals, NPY_STRING,
        NULL, buf4, 1,
        NPY_ARRAY_BEHAVED | NPY_ARRAY_OWNDATA,
        NULL));
    }
    else
    {
      PyList_Append(tree, Py_None);
    }
    PyList_Append(tree, ret);
    PyList_Append(tree, PyUnicode_DecodeASCII(CG_CGNSTree_ts, 10, "strict"));
    PyList_Append(load_return, tree);
    PyList_Append(load_return, links);
    PyList_Append(load_return, paths);
    Py_DECREF(tree);
    Py_DECREF(links);
    Py_DECREF(paths);
    Py_DECREF(ret);
  }
  Py_INCREF(Py_None);
  L3_nodeRelease(&rnode, L3F_R_ALL);
  TRACE_HDF5_LEAK("LOAD LEAVE");
  S2P_TRACE(("# CHL:remaining nodes [%d]\n", __node_count));

  s2p_freeContext(&context);

  return load_return;
}
/* ------------------------------------------------------------------------- */
PyObject* s2p_saveAsHDF(char     *dirname,
                        char     *filename,
                        PyObject *tree,
                        PyObject *links,
                        int       flags,
                        int       depth,
                        char     *searchpath,
                        PyObject *update,
                        PyObject *filter,
                        PyObject *skip,
                        PyObject *lkupdate,
                        PyObject *lkskip,
                        PyObject *except)
{
  int toupdate = 0;
  Py_ssize_t sz = -1, i = 0;
  Py_ssize_t n = 0;
  char *tdat = NULL, parentnodename[256], *pt, *cpath, *path;
  s2p_ctx_t *context = NULL;
  PyObject *rtree = NULL, *otree = NULL, *paths = NULL;
  int ndat = 0, ret = 1, *dims, *ddat;
  char *vdat = NULL;
  char *clabel = NULL;
  L3_Cursor_t *l3db = NULL;
  L3_Node_t *node = NULL, *rnode = NULL;
  hid_t nodeid = -1, oldid = -1, parentid = -1;
  s2p_lnk_t *link = NULL;

  STR_ALLOC(cpath, MAXPATHSIZE);
  STR_ALLOC(path, MAXPATHSIZE);
  DIM_ALLOC(dims, int, L3C_MAX_DIMS);
  DIM_ALLOC(ddat, int, NPY_MAXDIMS);

  CHL_import_array();

  S2P_NEWCONTEXTPTR(context);
  context->flg = flags;
  context->dpt = depth;
  context->lnk_obj = links;
  context->upd_pth = update;
  context->skp_pth = skip;
  context->upd_pth_lk = lkupdate;
  context->skp_pth_lk = lkskip;
  context->err = except;
  context->lsp = searchpath;

  if (PyDict_Check(filter))
  {
    context->flt_dct = filter;
  }

#ifdef __THREADING__
  H5dont_atexit(); /* MANDATORY FIRST HDF5 function to call */
#endif

  /* --------------------------------------------------------------------- */
  /* update paths at save time:
   - start to save from each path sub-tree, do not avoid links that
     would lead to another subtree.
   - update dict value is actual CGNS/Python node for this path key */

  if (PyDict_Size(context->upd_pth) && S2P_HASFLAG(S2P_FUPDATE))
  {
    S2P_TRACE(("# CHL:update file [%s]\n", filename));
    if (S2P_CHECKNODENOCACHE(tree, context))
    {
      s2p_filllinktable(links, context);
      l3db = s2p_addoneHDF(dirname, filename, context, 1);
      if (!L3M_ECHECK(l3db))
      {
        s2p_freelinktable(context);
        STR_FREE(cpath);
        STR_FREE(path);
        DIM_FREE(dims);
        DIM_FREE(ddat);
        return NULL;
      }
      s2p_setlinksearchpath(l3db, context);
      paths = PyDict_Keys(context->upd_pth);
      for (n = 0; n < PyList_Size(paths); n++)
      {
        PyObject *ascii_tdat = NULL;
        ascii_tdat = PyUnicode_AsASCIIString(PyList_GetItem(paths, n));
        tdat = PyBytes_AsString(ascii_tdat);

        L3M_ECLEAR(l3db);
        nodeid = L3_nodeFind(l3db, l3db->root_id, tdat);
        L3M_NEWNODE(node);
        __node_count++;
        rnode = L3_nodeRetrieve(l3db, nodeid, node);
        if (rnode == NULL)
        {
          S2P_TRACE(("# CHL:update path not found '%s'\n", tdat));
          ret &= 1;
        }
        else
        {
          S2P_TRACE(("# CHL:update path found '%s'\n", tdat));
          strcpy(parentnodename, tdat);
          pt = parentnodename;
          for (i = strlen(tdat); i > 0; i--)
          {
            if (*(pt + i) == '/')
            {
              *(pt + i) = '\0';
              break;
            }
          }
          parentid = L3_path2Node(l3db, parentnodename);
          L3M_SETFLAG(l3db, L3F_DEBUG);
          if (S2P_HASFLAG(S2P_FCOMPRESS))
          {
            L3M_SETFLAG(l3db, L3F_COMPRESS);
          }
          ret &= s2p_parseAndWriteHDF(parentid,
            PyDict_GetItemString(context->upd_pth, tdat),
            parentnodename, "", context, l3db);
        }
        L3_nodeFree(&node);
        if (ascii_tdat != NULL) Py_DECREF(ascii_tdat);
      }
      s2p_closeallHDF(context);
      s2p_freelinktable(context);
      s2p_freenodetable(context);
      free(context);
    }
  }
  /* --------------------------------------------------------------------- */
  else
  {
    S2P_TRACE(("# CHL:create/full update file [%s]\n", filename));
    if (S2P_CHECKNODENOCACHE(tree, context))
    {
      s2p_filllinktable(links, context);
      if (!S2P_HASFLAG(S2P_FUPDATE))
      {
        S2P_SETFLAG(S2P_FNEW);
      }
      else
      {
        S2P_CLRFLAG(S2P_FNEW);
      }
      rtree = PyList_GetItem(tree, 2);
      sz = 0;
      if (PyList_Check(rtree))
      {
        sz = PyList_Size(rtree);
        if (!sz)
        {
          Py_INCREF(Py_None);
          return Py_None;
        }
      }
      l3db = s2p_addoneHDF(dirname, filename, context, 1);
      if (!L3M_ECHECK(l3db))
      {
        STR_FREE(cpath);
        STR_FREE(path);
        DIM_FREE(dims);
        DIM_FREE(ddat);
        return NULL;
      }
      s2p_setlinksearchpath(l3db, context);
      rtree = PyList_GetItem(tree, 2);
      if (PyList_Check(rtree))
      {
        sz = PyList_Size(rtree);
        for (n = 0; n < sz; n++)
        {
          PyObject *ascii_label = NULL;
          otree = PyList_GetItem(rtree, n);

          ascii_label = PyUnicode_AsASCIIString(PyList_GetItem(otree, 3));
          clabel = PyBytes_AsString(ascii_label);

          if (S2P_CHECKNODENOCACHE(otree, context)
            && !strcmp(clabel, CG_CGNSLibraryVersion_ts))
          {
            S2P_TRACE(("# CHL:node [CGNSLibraryVersion]\n"));
            s2p_getData((PyArrayObject*)PyList_GetItem(otree, 1),
              &tdat, &ndat, ddat, &vdat,
              0, 0, context);
            L3_initDims(dims, 1, -1);
            if (S2P_HASFLAG(S2P_FUPDATE))
            {
              strcpy(cpath, "/");
              strcat(cpath, CG_CGNSLibraryVersion_n);
              oldid = L3_nodeFind(l3db, l3db->root_id, cpath);
              if (H5Iis_valid(oldid))
              {
                L3M_NEWNODE(node);
                __node_count++;
                node = L3_nodeRetrieve(l3db, oldid, node);
                toupdate = 1;
              }
            }
            node = L3_nodeSet(l3db, node,
              CG_CGNSLibraryVersion_n, CG_CGNSLibraryVersion_ts,
              dims, L3E_R4, vdat, L3F_NONE);
            if (toupdate)
            {
#if (CHLONE_USE_COMPACT_STORAGE == 1)
              L3_nodeUpdate(l3db, node, L3_COMPACT_STORE);
#else
              L3_nodeUpdate(l3db, node, L3_CONTIGUOUS_STORE);
#endif
            }
            else
            {
#if (CHLONE_USE_COMPACT_STORAGE == 1)
            L3_nodeCreate(l3db, l3db->root_id, node, L3_COMPACT_STORE);
#else
            L3_nodeCreate(l3db, l3db->root_id, node, L3_CONTIGUOUS_STORE);
#endif
            }
          }
          else
          {
            if (S2P_HASFLAG(S2P_FCOMPRESS))
            {
              L3M_SETFLAG(l3db, L3F_COMPRESS);
            }
            strcpy(cpath, "");
            S2P_TRACE(("# CHL:start tree update parse\n"));
            ret = s2p_parseAndWriteHDF(l3db->root_id, otree,
              cpath, path, context, l3db);
            if (!ret) { break; }
          }
          if (ascii_label != NULL) { Py_DECREF(ascii_label); }
        }
        s2p_removeMissingChildren(l3db->root_id, tree, context, l3db);
        if (s2p_atleastonelink(context))
        {
          S2P_TRACE(("# CHL:create remaining links\n"));
          link = context->lnk;
          while (link != NULL)
          {
            if (link->status != S2P_LKUPDATED)
            {
              s2p_createlinkoffparse(l3db, link, context);
            }
            link = link->next;
          }
          S2P_TRACE(("# CHL:unsolved links still remain\n"));
        }
        else
        {
          S2P_TRACE(("# CHL:no remaining links\n"));
        }
      }
      L3_nodeFree(&node); /* free releases hid_t, before actual close */
      s2p_closeallHDF(context);
      s2p_freelinktable(context);
      s2p_freenodetable(context);
      free(context);
    }
    else
    {
      setError(S2P_EBADTREEROOT, "Bad tree structure at %s level",
        "root", context);
      return NULL;
    }
  }
  /* --------------------------------------------------------------------- */
  STR_FREE(cpath);
  STR_FREE(path);
  DIM_FREE(dims);
  DIM_FREE(ddat);
  TRACE_HDF5_LEAK("SAVE LEAVE");

  if (ret)
  {
    Py_INCREF(Py_None);
    return Py_None;
  }
  return NULL;
}
/* ------------------------------------------------------------------------- */
int s2p_garbage(PyObject *tree)
{
  int n;
  PyObject *ar;
  PyArrayObject *aro;

  PyObject *ascii_string = PyUnicode_AsASCIIString(PyList_GetItem(tree, 0));
  printf("\n> %s", PyBytes_AsString(ascii_string)); fflush(stdout);
  Py_DECREF(ascii_string);

  Py_DECREF(PyList_GetItem(tree, 0));
  ar = PyList_GetItem(tree, 1);
  if (PyArray_Check(ar))
  {
    aro = (PyArrayObject *)ar;
    printf("[%p]", PyArray_DATA(aro)); fflush(stdout);
  }
  else
  {
    Py_DECREF(PyList_GetItem(tree, 1));
  }
  Py_DECREF(PyList_GetItem(tree, 3));
  if (PyList_Size(PyList_GetItem(tree, 2)) == 0)
  {
    printf("<E\n"); fflush(stdout);
    Py_DECREF(tree);
    return 0;
  }
  for (n = 0; n < PyList_Size(PyList_GetItem(tree, 2)); n++)
  {
    s2p_garbage(PyList_GetItem(PyList_GetItem(tree, 2), n));
  }
  printf("\n<C"); fflush(stdout);
  return 1;
}
/* ------------------------------------------------------------------------- */
int s2p_probe(char *filename, char *path)
{
  L3_Cursor_t *l3db = NULL;
  long l3flag = L3F_DEFAULT;
  int r;

  l3db = L3_openFile(filename, L3E_OPEN_RDO, l3flag);
  if (!L3M_ECHECK(l3db))
  {
    return 0;
  }
  r = 1;
  if (path[0] != '\0')
  {
    r = L3_path2Node(l3db, path);
  }
  L3_close(&l3db);

  return r;
}
/* ------------------------------------------------------------------------- */

