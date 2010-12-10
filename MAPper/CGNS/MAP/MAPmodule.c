/* 
#  -------------------------------------------------------------------------
#  pyCGNS.MAP - Python package for CFD General Notation System - MAPper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  ------------------------------------------------------------------------- 
*/
#include "SIDStoPython.c"
/* ------------------------------------------------------------------------- */
static PyObject *
MAP_load(PyObject *self, PyObject *args)
{
  char *filename, *path;
  int flags,threshold,depth;

  threshold=0;
  depth=999;
  path=NULL;

  if (!PyArg_ParseTuple(args,"si|iiz",
			&filename,&flags,
			&threshold,&depth,&path))
  {
    return NULL;
  }
  return s2p_loadAsHDF(filename,flags,threshold,depth,path);
}
/* ------------------------------------------------------------------------- */
static PyObject *
MAP_save(PyObject *self, PyObject *args)
{
  char *filename, *path;
  int flags,threshold,depth,ret;
  PyObject *tree,*links;

  threshold=0;
  depth=999;
  path=NULL;

  if (!PyArg_ParseTuple(args,"sOOi|iis",
			&filename,&tree,&links,&flags,
			&threshold,&depth,&path))
  {
    return NULL;
  }
  ret=s2p_saveAsHDF(filename,tree,links,flags,threshold,depth,path);
  return PyInt_FromLong(ret);
}
/* ------------------------------------------------------------------------- */
static PyMethodDef MAP_methods[] = {
  {"load",MAP_load,METH_VARARGS},
  {"save",MAP_save,METH_VARARGS},  
  {NULL,NULL}
};

#define S2P_SETCONSTINDICT(sss,vvv) \
 s2p_v=PyInt_FromLong((long)vvv);\
 s2p_s=PyString_FromString(sss);\
 PyList_Append(s2p_l,s2p_s);\
 PyDict_SetItem(s2p_d,s2p_s,s2p_v);\
 Py_DECREF(s2p_s);\
 Py_DECREF(s2p_v);

/* ------------------------------------------------------------------------- */
DL_EXPORT(void)
initMAP(void)
{
  PyObject *s2p_m,*s2p_d,*s2p_s,*s2p_v,*s2p_l; 

  s2p_m=Py_InitModule("MAP", MAP_methods);
  import_array();
  s2p_d=PyModule_GetDict(s2p_m);
  s2p_l=PyList_New(0);
  PyDict_SetItemString(s2p_d,"flags",s2p_l);
  S2P_SETCONSTINDICT("S2P_NONE",S2P_FNONE);
  S2P_SETCONSTINDICT("S2P_ALL",S2P_FALL);
  S2P_SETCONSTINDICT("S2P_ALTERNATESIDS",S2P_FALTERNATESIDS);
  S2P_SETCONSTINDICT("S2P_TRACE",S2P_FTRACE);
  S2P_SETCONSTINDICT("S2P_FOLLOWLINKS",S2P_FFOLLOWLINKS);
  S2P_SETCONSTINDICT("S2P_MERGELINKS",S2P_FMERGELINKS);
  S2P_SETCONSTINDICT("S2P_COMPRESS",S2P_FCOMPRESS);
  S2P_SETCONSTINDICT("S2P_REVERSEDIMS",S2P_FREVERSEDIMS);
  S2P_SETCONSTINDICT("S2P_OWNDATA",S2P_FOWNDATA);
  S2P_SETCONSTINDICT("S2P_NODATA",S2P_FNODATA);
  S2P_SETCONSTINDICT("S2P_UPDATE",S2P_FUPDATE);
  S2P_SETCONSTINDICT("S2P_DELETEMISSING",S2P_FDELETEMISSING);
  S2P_SETCONSTINDICT("S2P_DEFAULT",S2P_FDEFAULT);
}
/* ------------------------------------------------------------------------- */

