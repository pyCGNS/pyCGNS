/* 
#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  ------------------------------------------------------------------------- 
*/
/* DbMIDLEVEL objects */

/* KEYWORDS are now in cgnskeywords.py file */

#include "Python.h"
#include "cgnslib.h"
#include "cgnsstrings.h"

#ifndef CG_MODE_READ
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      modes for cgns file                                              *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define CG_MODE_READ	0
#define CG_MODE_WRITE	1
#define CG_MODE_CLOSED	2
#define CG_MODE_MODIFY	3

/* file types */

#define CG_FILE_NONE 0
#define CG_FILE_ADF  1
#define CG_FILE_HDF5 2
#define CG_FILE_XML  3

/* function return codes */

#define CG_OK		  0
#define CG_ERROR	  1
#define CG_NODE_NOT_FOUND 2
#define CG_INCORRECT_PATH 3
#define CG_NO_INDEX_DIM   4

/* Null and UserDefined enums */

#define CG_Null        0
#define CG_UserDefined 1

/* max goto depth */

#define CG_MAX_GOTO_DEPTH 20

/* configuration options */

#define CG_CONFIG_ERROR     1
#define CG_CONFIG_COMPRESS  2
#define CG_CONFIG_SET_PATH  3
#define CG_CONFIG_ADD_PATH  4
#define CG_CONFIG_FILE_TYPE 5

#define CG_CONFIG_XML_DELETED     301
#define CG_CONFIG_XML_NAMESPACE   302
#define CG_CONFIG_XML_THRESHOLD   303
#define CG_CONFIG_XML_COMPRESSION 304

/* legacy code support */

#define MODE_READ	CG_MODE_READ
#define MODE_WRITE	CG_MODE_WRITE
#define MODE_MODIFY	2
#define Null            CG_Null
#define UserDefined	CG_UserDefined

#endif

/* macros for heavy use of Python dicts ;) */
/* PyDict_SetItemString(xd, xdn##"_", xdd); \ */

#define createDict(xd,xdn,xdd,xddr) \
xdd = PyDict_New(); \
xddr = PyDict_New(); \
PyDict_SetItemString(dtop, xdn, xdd); \
PyDict_SetItemString(xd, xdn "_", xddr); \
PyDict_SetItemString(xd, xdn, xdd);

#define addConstInDict2(xd,xxd,xxdr,xs,xv) \
v= PyInt_FromLong((long)xv);\
s= PyString_FromString(xs);\
PyDict_SetItem(xd,s,s);\
PyDict_SetItem(xxd,s,v);\
PyDict_SetItem(xxdr,v,s);\
Py_DECREF(v);\
Py_DECREF(s); 

#define addStringInDict(xd,xs) \
s= PyString_FromString(xs);\
PyDict_SetItemString(xd,xs,s);\
Py_DECREF(s); 

#define addStringInDict2(xd,xxd,xs) \
s= PyString_FromString(xs);\
PyDict_SetItemString(xxd,xs,s);\
PyDict_SetItemString(xd,xs,s);\
Py_DECREF(s); 

void midleveldictionnary_init(PyObject *d)
{
  PyObject *v, *s, *dr, *ddr, *dtop;

  /* This top dictionnary stores the names of the enum dictionnaries */
  dtop = PyDict_New();
  s= PyString_FromString("Enumerates");

  /* d --> the CGNS module dictionnary */
  PyDict_SetItem(d, s, dtop);
  
  Py_DECREF(dtop);
}
