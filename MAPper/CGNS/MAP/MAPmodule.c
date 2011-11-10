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
  char *filename, *path, *pth, *ptr;
  int flags,depth,totalsize,n;
  PyObject *lksearch,*ret,*update;

  depth=999;
  path=NULL;
  lksearch=NULL;
  pth=NULL;

  if (!PyArg_ParseTuple(args,"si|izOO",
			&filename,&flags,
			&depth,&path,&lksearch,&update))
  {
    return NULL;
  }
  if (depth < 1){ depth=999; }
  if ( lksearch && 
       PyList_Check(lksearch) && 
       PyList_Size(lksearch) &&
       PyString_Check(PyList_GetItem(lksearch,0))
     )
  {
    totalsize=0;
    for (n=0;n<PyList_Size(lksearch);n++)
    {
      if (PyString_Check(PyList_GetItem(lksearch,n)))
      {
	totalsize+=PyString_Size(PyList_GetItem(lksearch,n))+1;
      }
    }
    pth=(char*)malloc(sizeof(char)*totalsize);
    ptr=pth;
    for (n=0;n<PyList_Size(lksearch);n++)
    {
      if (PyString_Check(PyList_GetItem(lksearch,n)))
      {
	strcpy(ptr,PyString_AsString(PyList_GetItem(lksearch,n)));
	ptr+=PyString_Size(PyList_GetItem(lksearch,n));
	if (n!=PyList_Size(lksearch)-1)
        {
	  ptr[0]=':';
	}
	else
        {
	  ptr[0]='\0';
	}
	ptr++;
      }
    }
  }
  ret=s2p_loadAsHDF(filename,flags,depth,path,pth,update);
  if (pth){free(pth);};
  return ret;
}
/* ------------------------------------------------------------------------- */
static PyObject *
MAP_save(PyObject *self, PyObject *args)
{
  char *filename, *path;
  int flags,depth;
  PyObject *tree,*links,*ret;

  depth=999;
  path=NULL;

  if (!PyArg_ParseTuple(args,"sOOi|is",
			&filename,&tree,&links,&flags,
			&depth,&path))
  {
    return NULL;
  }
  ret=s2p_saveAsHDF(filename,tree,links,flags,depth,path);
  return ret;
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

  S2P_SETCONSTINDICT("NONE",S2P_FNONE);
  S2P_SETCONSTINDICT("ALL",S2P_FALL);
  S2P_SETCONSTINDICT("ALTERNATESIDS",S2P_FALTERNATESIDS);
  S2P_SETCONSTINDICT("TRACE",S2P_FTRACE);
  S2P_SETCONSTINDICT("FOLLOWLINKS",S2P_FFOLLOWLINKS);
  S2P_SETCONSTINDICT("MERGELINKS",S2P_FMERGELINKS);
  S2P_SETCONSTINDICT("COMPRESS",S2P_FCOMPRESS);
  S2P_SETCONSTINDICT("REVERSEDIMS",S2P_FREVERSEDIMS);
  S2P_SETCONSTINDICT("OWNDATA",S2P_FOWNDATA);
  S2P_SETCONSTINDICT("NODATA",S2P_FNODATA);
  S2P_SETCONSTINDICT("UPDATE",S2P_FUPDATE);
  S2P_SETCONSTINDICT("DELETEMISSING",S2P_FDELETEMISSING);
  S2P_SETCONSTINDICT("DEFAULT",S2P_FDEFAULT);
  S2P_SETCONSTINDICT("DEFAULTS",S2P_FDEFAULT);

  if (MAPErrorObject == NULL) 
  {
    MAPErrorObject = PyErr_NewException("CGNS.MAP.error", NULL, NULL);
    if (MAPErrorObject == NULL) { return; }
  }
  Py_INCREF(MAPErrorObject);
  PyModule_AddObject(s2p_m, "error", MAPErrorObject);
}
/* ------------------------------------------------------------------------- */

