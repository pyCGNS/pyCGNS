/* ====================================================================== */
/* pyCGNS - CFD General Notation System - CGNS lib wrapper */
/* ONERA/DSNA/ELSA - poinot@onera.fr */
/* $Rev: 72 $ $Date: 2009-02-04 10:02:12 +0100 (Wed, 04 Feb 2009) $ */
/* See file COPYING in the root directory of this Python module source */
/* tree for license information. */
/* ====================================================================== */

#include "Python.h"
#define  PY_ARRAY_UNIQUE_SYMBOL CGNS_PY_ARRAY_UNIQUE_SYMBOL
#include "numpy/oldnumeric.h"
#include "numpy/ndarrayobject.h"

#include "ADF__.h"
 
/* --- Fixed magic values
       - these should be taken from some lib header file
       - note: there is NO max number of children :( 
*/
#define MAXCHILDRENCOUNT   1024
#define MAXLINKNAMESIZE    1024
#define MAXPATHSIZE        1024
#define MAXFILENAMESIZE    1024
#define MAXDATATYPESIZE    32
#define MAXDIMENSIONVALUES 12
#define MAXFORMATSIZE      20
#define MAXERRORSIZE       80
#define MAXVERSIONSIZE     32

/* --- Local enumerates   */
typedef enum {UNKNOWN, OPEN, REOPEN, CLOSED} ADFStatus;

#define DbMIDLEVELObject_Check(v)	((v)->ob_type == &DbMIDLEVEL_Type)

/* --- ADF object structure */
typedef struct {
  PyObject_HEAD

  ADFStatus status;      /* last status of python adf object */
  int storagetype;  	 /* type of storage adf/hdf       	*/
  double    root_id;     /* copy of the database root id */
  int       last_error;  /* copy of the last adf error code */
  int       flags;       /* see __ADF.h set of flags */
  char      msg_error[MAXERRORSIZE];
} DBADFObject;

/* --- Python stuff */
static PyObject *ADFErrorObject;
static PyTypeObject DBADF_Type;
#define DBADFObject_Check(v)    ((v)->ob_type == &DBADF_Type)

extern int cg_is_cgns(const char *filename, int *file_type);

/* ======================================================================
   DBADFObject methods
   - Bindings to adf calls (with *ADF* tag). These calls do *NOT* check
     adf errors, these should be done in the upper layer, using adf
     error routines.
   - Extra methods, some hard coded adf calls patterns
*/


/* ---------------------------------------------------------------------
 *ADF* ADF_Database_Open
*/
static DBADFObject *
newDBADFObject(char *fn, char *fs, char *ff)
{
  DBADFObject *self;

  if (! strlen(fn))
  {
    PyErr_SetString(ADFErrorObject,
                    "File name is empty");
    return NULL;
  }
  self = PyObject_New(DBADFObject, &DBADF_Type);
  if (self == NULL) return NULL;
  self->storagetype=CG_FILE_ADF;
  self->status=UNKNOWN;
  cg_is_cgns(fn,&(self->storagetype));
  if (! self->storagetype)
  {
    self->storagetype=CG_FILE_ADF;
  }
  self->last_error=0;
  if (self->storagetype == CG_FILE_ADF)
  {
    ADF_Database_Open(fn,fs,ff,&(self->root_id),&(self->last_error));
    if (self->last_error > 0)
    {
      ADF__Error_Message(self->last_error,self->msg_error);
      PyErr_SetString(ADFErrorObject,self->msg_error);
      return NULL;
    }
    self->storagetype=STORAGE_ADF;
  }
  else
  {
   /* try HDF now */
    if (self->storagetype == CG_FILE_HDF5)
    {
      ADFH_Database_Open(fn,fs,ff,&(self->root_id),&(self->last_error));
      if (self->last_error != -1)
      {
	ADF__Error_Message(self->last_error,self->msg_error);
	PyErr_SetString(ADFErrorObject,self->msg_error);
	return NULL;
      }
      self->storagetype=STORAGE_HDF;
    }
  }
  self->status=OPEN; /* should check error */
  self->flags=__CGNS__DEFAULT__FLAGS__;

  return self;
}

/* ---------------------------------------------------------------------
 *ADF* ADF_Database_Close
*/
static PyObject*
DBADF_close(DBADFObject *self)
{
  if (self->status==OPEN)
  {
    ADF__Database_Close(self->root_id,&(self->last_error));
  }
  Py_INCREF(Py_None);
  self->status=CLOSED;
  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_database_delete
       - can we delete self here ?
*/
static PyObject*
DBADF_delete(DBADFObject *self, PyObject *args)
{
  char *name;
  
  if (!PyArg_ParseTuple(args, "s",&name))
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.delete(file-name:string)");
    return NULL;
  }
  ADF__Database_Delete(name,&(self->last_error));
  Py_INCREF(Py_None);

  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_database_get_format
*/
static PyObject*
DBADF_getFormat(DBADFObject *self, PyObject *args)
{
  double rootid;
  char   format[MAXFORMATSIZE];
  
  if (!PyArg_ParseTuple(args, "d",&rootid))
  {
    PyErr_SetString(ADFErrorObject,
                    "database-format:string=adf.database_get_format(root-id:double)");
    return NULL;
  }
  ADF__Database_Get_Format(rootid,format,&(self->last_error));

  return PyString_FromString(format);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_database_set_format
*/
static PyObject*
DBADF_setFormat(DBADFObject *self, PyObject *args)
{
  double rootid;
  char  *format;
  
  if (!PyArg_ParseTuple(args, "ds",&rootid,&format))
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.database_set_format(root-id:double,database-format:string)");
    return NULL;
  }
  ADF__Database_Set_Format(rootid,format,&(self->last_error));
  Py_INCREF(Py_None);

  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_database_garbage_collection
*/
static PyObject*
DBADF_dbgc(DBADFObject *self, PyObject *args)
{
  double id;
  if (!PyArg_ParseTuple(args, "d",&id))
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.database_garbage_collection(node-id:double)");
    return NULL;
  }
  ADF__Database_Garbage_Collection(id,&(self->last_error));
  Py_INCREF(Py_None);

  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_database_version
*/
static PyObject*
DBADF_dbVersion(DBADFObject *self, PyObject *args)
{
  double rootid;
  char   v[MAXVERSIONSIZE],c[MAXVERSIONSIZE],m[MAXVERSIONSIZE];

  if (!PyArg_ParseTuple(args, "d",&rootid))
  {
    PyErr_SetString(ADFErrorObject,
                    "(version:string,creation-date:string,modification-date:string)=adf.database_version(root-id:double)");
    return NULL;
  }
  ADF__Database_Version(rootid,v,c,m,&(self->last_error));
  
  return Py_BuildValue("(sss)",v,c,m);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_library_version
*/
static PyObject*
DBADF_libVersion(DBADFObject *self)
{
  char   v[MAXVERSIONSIZE];
  
  ADF__Library_Version(v,&(self->last_error));

  return PyString_FromString(v);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_get_root_id
*/
static PyObject*
DBADF_getRootId(DBADFObject *self, PyObject *args)
{
  double id,root;

  if (!PyArg_ParseTuple(args, "d",&id))
  {
    PyErr_SetString(ADFErrorObject,
                    "root-id:double=adf.get_root_id(node-id:double)");
    return NULL;
  }
  ADF__Get_Root_ID(id,&root,&(self->last_error));

  return PyFloat_FromDouble(root);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_create
*/
static PyObject*
DBADF_nodeCreate(DBADFObject *self, PyObject *args)
{
  double parent,id;
  char  *name;
  
  if (!PyArg_ParseTuple(args, "ds",&parent,&name))
  {
    PyErr_SetString(ADFErrorObject,
                    "node-id:double=adf.create(parent-id:double, node-name:string)");
    return NULL;
  }
  ADF__Create(parent,name,&id,&(self->last_error));
  
  return PyFloat_FromDouble(id);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_delete
*/
static PyObject*
DBADF_nodeDelete(DBADFObject *self, PyObject *args)
{
  double parent,node;

  if (!PyArg_ParseTuple(args, "dd",&parent,&node))
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.delete(parent-id:double, node-id:double)");
    return NULL;
  }
  ADF__Delete(parent,node,&(self->last_error));
  Py_INCREF(Py_None);

  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_put_name
*/
static PyObject*
DBADF_setNodeName(DBADFObject *self, PyObject *args)
{
  double parent,node;
  char  *name;
  
  if (!PyArg_ParseTuple(args, "dds",&parent,&node,&name))
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.put_name(parent-id:double, node-id:double, node-name:string)");
    return NULL;
  }
  ADF__Put_Name(parent,node,name,&(self->last_error));
  Py_INCREF(Py_None);
  
  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_get_name
*/
static PyObject*
DBADF_getNodeName(DBADFObject *self, PyObject *args)
{
  double node;
  char   name[ADF_NAME_LENGTH+1];
  
  if (!PyArg_ParseTuple(args, "d",&node))
  {
    PyErr_SetString(ADFErrorObject,
                    "node-name:string=adf.get_name(node-id:double)");
    return NULL;
  }
  ADF__Get_Name(node,name,&(self->last_error));

  return PyString_FromString(name);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_children_names
       - we have to get the number of children first, to allocate
         the returned tuple of strings
*/
static PyObject*
DBADF_getChildrenNames(DBADFObject *self, PyObject *args)
{
  double    node;
  int       num,i,c;
  char      cn[ADF_NAME_LENGTH+1];
  PyObject *listofnames;
  
  if (!PyArg_ParseTuple(args, "d",&node))
  {
    PyErr_SetString(ADFErrorObject,
                    "(children-name,*)=adf.children_names(parent-id:double)");
    return NULL;
  }
  ADF__Number_of_Children(node,&num,&(self->last_error));
  if (self->last_error != -1)
  {
    ADF__Error_Message(self->last_error,self->msg_error);
    PyErr_SetString(ADFErrorObject,self->msg_error);
    return NULL;
  }
  
  listofnames=PyTuple_New(num);

  /* beware, index is n+1 for children, n for tuple */
  for (i=1;i<=num;i++)
  {
    ADF__Children_Names(node,i,1,ADF_NAME_LENGTH+1,&c,cn,&(self->last_error));
    PyTuple_SET_ITEM(listofnames, i-1, PyString_FromString(cn));
  }

  return listofnames;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_move_child
*/
static PyObject*
DBADF_nodeMove(DBADFObject *self, PyObject *args)
{
  double parent,node,newparent;
  
  if (!PyArg_ParseTuple(args, "ddd",&parent,&node,&newparent))
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.mode_child(parent-id:double, node-id:double, new-parent-id:double)");
    return NULL;
  }
  ADF__Move_Child(parent,node,newparent,&(self->last_error));
  Py_INCREF(Py_None);

  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_set_label
*/
static PyObject*
DBADF_setLabel(DBADFObject *self, PyObject *args)
{
  double id;
  char  *label;
  
  if (!PyArg_ParseTuple(args, "ds",&id,&label))
  {
    PyErr_SetString(ADFErrorObject,
                    "=adf.set_label(node-id:double,node-label:string)");
    return NULL;
  }
  ADF__Set_Label(id,label,&(self->last_error));
  Py_INCREF(Py_None);

  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_get_data_type
*/
static PyObject*
DBADF_getDataType(DBADFObject *self, PyObject *args)
{
  double node;
  char   datatype[MAXDATATYPESIZE];
  
  if (!PyArg_ParseTuple(args, "d",&node))
  {
    PyErr_SetString(ADFErrorObject,
                    "data-type:string=adf.get_data_type(node-id:double)");
    return NULL;
  }
  ADF__Get_Data_Type(node,datatype,&(self->last_error));

  return PyString_FromString(datatype);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_get_number_of_dimensions
*/
static PyObject*
DBADF_getNumDim(DBADFObject *self, PyObject *args)
{
  double node;
  int    ndim=0;
  
  if (!PyArg_ParseTuple(args, "d",&node))
  {
    PyErr_SetString(ADFErrorObject,
                    "number-of-dimensions:int=adf.get_number_of_dimensions(node-id:double)");
    return NULL;
  }
  ADF__Get_Number_of_Dimensions(node,&ndim,&(self->last_error));
  return PyInt_FromLong(ndim);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_get_dimension_values
       - we have to call get_number_of_dimensions here, because we
         have to parse the returned array to be able to return a tuple.
         then, the first call has to be error-proof, otherwise we leave.
*/
static PyObject*
DBADF_getDimValues(DBADFObject *self, PyObject *args)
{
  double    node;
  int       dimval[MAXDIMENSIONVALUES],ndim,n;
  PyObject *tp;

  if (!PyArg_ParseTuple(args, "d",&node))
  {
    PyErr_SetString(ADFErrorObject,
                    "(dimension:integer,*)=adf.get_dimension_values(node-id:double)");
    return NULL;
  }
  ADF__Get_Number_of_Dimensions(node,&ndim,&(self->last_error));
  if (self->last_error != -1) /* ADF NO ERROR IS -1 */
  {
    PyErr_SetString(ADFErrorObject,
                    "cannot read the dimension number (check with get_number_of_dimensions)");
    return NULL;
  }
  ADF__Get_Dimension_Values(node, dimval,&(self->last_error));
  tp = PyTuple_New(ndim);
  for (n=0; n<ndim; n++)
  {
    PyTuple_SET_ITEM(tp, n, PyInt_FromLong(dimval[n]));
  }
  
  return tp;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_put_dimension_information
       - the arguments are slightly different from the adf args. we do not
         pass the dimension, the length of the tuple of dimensions can
         be found, the dimensions themselves are into the tuple.
*/
static PyObject*
DBADF_putDimInfo(DBADFObject *self, PyObject *args)
{
  double node;
  char  *datatype;
  int    ndim,*dimval,n;
  PyObject *dimensions;
  
  if (!PyArg_ParseTuple(args, "dsO",&node,&datatype,&dimensions))
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.put_dimension_information(node-id:double,data-type:string,dimensions:(integer,*))");
    return NULL;
  }
  if (!PyTuple_Check(dimensions))
  {
    PyErr_SetString(ADFErrorObject,
                    "second argument should be a tuple of integers");
    return NULL;
  }
  ndim=PyTuple_Size(dimensions);
  dimval=(int*) malloc(sizeof(int)*ndim); /* allocated dimval */
  for (n=0;n<ndim;n++)
  {
    dimval[n]=PyInt_AsLong(PyTuple_GetItem(dimensions,n));
  }
  
  ADF__Put_Dimension_Information(node,datatype,ndim,dimval,&(self->last_error));
  free(dimval); /* de-allocated dimval */
  Py_INCREF(Py_None);

  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_is_link
*/
static PyObject*
DBADF_isLink(DBADFObject *self, PyObject *args)
{
  double id;
  int    pathlength;
  
  if (!PyArg_ParseTuple(args, "d",&id))
  {
    PyErr_SetString(ADFErrorObject,
                    "path-length:integer=adf.is_link(node-id:double)");
    return NULL;
  }
  ADF__Is_Link(id,&pathlength,&(self->last_error));

  return PyInt_FromLong(pathlength);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_link
*/
static PyObject*
DBADF_addLink(DBADFObject *self, PyObject *args)
{
  double parent,id;
  char  *name,*destfile,*destnode;
  
  if (!PyArg_ParseTuple(args, "dsss",&parent,&name,&destfile,&destnode))
  {
    PyErr_SetString(ADFErrorObject,
                    "node-id:double=adf.link(parent-id:double, node-name:string, destination-file:string, destination-node:string)");
    return NULL;
  }  
  ADF__Link(parent,name,destfile,destnode,&id,&(self->last_error));
  
  return PyFloat_FromDouble(id);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_get_link_path
*/
static PyObject*
DBADF_getLink(DBADFObject *self, PyObject *args)
{
  double id;
  char   filename[MAXFILENAMESIZE];
  char   nodename[MAXPATHSIZE];
  
  if (!PyArg_ParseTuple(args, "d",&id))
  {
    PyErr_SetString(ADFErrorObject,
                    "(file:string,node:string)=adf.get_link_path(node-id:double)");
    return NULL; 
  } 
  ADF__Get_Link_Path(id,filename,nodename,&(self->last_error));
  
  return Py_BuildValue("(ss)",filename,nodename);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_read_data
*/
static PyObject*
DBADF_readData(DBADFObject *self)
{
  PyErr_SetString(ADFErrorObject,"Sorry, not implemented yet");
  return NULL;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_read_all_data
       - we have to get a lot of information in order to actually get
         the data. The returned array has to have to correct memory size.
         we do *NOT* want to make this allocation external, because it would
         lead to potential segfaults inside Python.
    - the routine is copied from the cgi_read_node cgns internal routine
*/
static PyObject*
DBADF_readAllData(DBADFObject *self, PyObject *args)
{
  double         node_id;
  void          *data;
  char           name[ADF_NAME_LENGTH+1];
  char           label[ADF_LABEL_LENGTH+1];
  char           data_type[MAXDATATYPESIZE];
  int            ndim,size,n;
  npy_intp       npy_dim_vals[MAXDIMENSIONVALUES];
  int            dim_vals[MAXDIMENSIONVALUES];
  int            arraytype,isdataarray;
  char           sterror[256];
  PyArrayObject *array;
  
  if (!PyArg_ParseTuple(args, "d",&node_id))
  {
    PyErr_SetString(ADFErrorObject,
                    "data:numpy-array=adf.read_all_data(node-id:double)");
    return NULL;
  }
  ADF__Get_Name(node_id, name, &(self->last_error));
  if (self->last_error != -1)
  {
    strcpy(sterror,"node not found while reading data");
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }
  /* data is an allocated memory space for US, we have to allocate it
     and ADF directly fills it with the data.
     - the following part is taken from cgi_read_node, in cgns_internals.c
  */

  /* read node data type */
  ADF__Get_Data_Type(node_id, data_type, &(self->last_error));
  if (self->last_error != -1)
  {
    strcpy(sterror,"datatype error while reading node: ");
    strcat(sterror,name);
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }

  ADF__Get_Label(node_id,label,&(self->last_error));
  isdataarray=0;
  if (!strcmp(label,"DataArray_t") || !strcmp(label,"DimensionalUnits_t"))
  {
    isdataarray=1;
  }

  if (strcmp(data_type,"MT")==0)
  {
    strcpy(sterror,"no data in this node: ");
    strcat(sterror,name);
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }
  
  /* number of dimension */
  ADF__Get_Number_of_Dimensions(node_id, &ndim, &(self->last_error));
  if (self->last_error != -1)
  {
    PyErr_SetString(ADFErrorObject,
                    "number of dimensions error while reading node");
    return NULL;
  }

  /* dimension vector */
  ADF__Get_Dimension_Values(node_id, dim_vals, &(self->last_error));
  if (self->last_error != -1)
  {
    PyErr_SetString(ADFErrorObject,
                    "dimension values error while reading node");
    return NULL;
  }

  /* Skipping data - ?? */
  /* if (!data_flag) return 0; */

  /* allocate data */
  size=1;
  for (n=0; n<ndim; n++)
  {
    size*=dim_vals[n];
    npy_dim_vals[n]=dim_vals[n];
  }
  if (size<=0)
  {
    PyErr_SetString(ADFErrorObject,
                    "size error while reading node");
    return NULL;
  }

#define _cgnew(t,s) (void *) malloc((s)*sizeof(t))

  if (strcmp(data_type,"I4")==0)
  {
    arraytype=PyArray_INT;
    data=_cgnew(int, size);
  }
  else if (strcmp(data_type,"I8")==0)
  {
    arraytype=PyArray_LONG;
    data=_cgnew(long, size);
  }
  else if (strcmp(data_type,"R4")==0)
  {
    arraytype=PyArray_FLOAT;
    data=_cgnew(float, size);
  }
  else if (strcmp(data_type,"R8")==0)
  {
    arraytype=PyArray_DOUBLE;
    data=_cgnew(double, size);
  }
  else if (strcmp(data_type,"C1")==0)
  {
    arraytype=PyArray_CHAR;
    data=_cgnew(char, size+1);
    memset(data,0,size+1);
  }
  else
  {
    PyErr_SetString(ADFErrorObject,
                    "cannot handle this datatype in a Numpy array"); 
    return NULL;
  }

  /* read data - last error has to be handle by user call */
  ADF__Read_All_Data(node_id, (char*)data, &(self->last_error));

  if (isdataarray)
  {
  array=(PyArrayObject*)PyArray_New(&PyArray_Type, ndim, npy_dim_vals, 
				    arraytype, NULL, (void*)data, 0, 
				    NPY_OWNDATA | NPY_FORTRAN, NULL);
  }
  else
  {
  array=(PyArrayObject*)PyArray_New(&PyArray_Type, ndim, npy_dim_vals, 
				    arraytype, NULL, (void*)data, 0, 
				    NPY_OWNDATA, NULL);
  }
  if (!array)
  {
    return NULL;
  }

  return (PyObject*)array;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_write_data
*/
static PyObject*
DBADF_writeData(DBADFObject *self)
{    
  PyErr_SetString(ADFErrorObject,"Sorry, not implemented yet");
  return NULL;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_write_all_data
       - should make a BUNCH of checks in there... one day
       - transpose has to be done ONLY FOR DataArray_t types
       - NO TRANSPOSE for IndexArray_t, IndexRange_t...
*/
static PyObject*
DBADF_writeAllData(DBADFObject *self, PyObject *args)
{
  char           data_type[MAXDATATYPESIZE];
  double         node_id;
  char           name[ADF_NAME_LENGTH+1];
  char           label[ADF_LABEL_LENGTH+1];
  char           sterror[256];
  int            ndim,localptr,isdataarray;
  PyObject      *oarray,*trsp;
  int            i,j,k,imax,jmax,kmax,fx,cx;
  void          *ptrs,*ptrd=NULL;

  if (!PyArg_ParseTuple(args, "dO",&node_id,&oarray))
  { 
    PyErr_SetString(ADFErrorObject,
                    "adf.read_all_data(node-id:double,data:numpy-array)");
    return NULL;
  }
  ADF__Get_Name(node_id, name, &(self->last_error));
  if (self->last_error != -1)
  {
    strcpy(sterror,"node not found while trying to write data");
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }
  ADF__Get_Label(node_id,label,&(self->last_error));
  isdataarray=0;
  if (!strcmp(label,"DataArray_t") || !strcmp(label,"DimensionalUnits_t"))
  {
    isdataarray=1;
  }
  ADF__Get_Data_Type(node_id, data_type, &(self->last_error));
  if (self->last_error != -1)
  {
    strcpy(sterror,"datatype error while reading node: ");
    strcat(sterror,name);
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }
  if ((strcmp(data_type,"I4")==0)&&(PyArray_TYPE(oarray)!=PyArray_INT))
  {
    strcpy(sterror,"array type is not expected I4 data type for this node");
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }
  else if ((strcmp(data_type,"I8")==0)&&(PyArray_TYPE(oarray)!=PyArray_LONG))
  {
    strcpy(sterror,"array type is not expected I8 data type for this node");
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }
  else if ((strcmp(data_type,"R4")==0)&&(PyArray_TYPE(oarray)!=PyArray_FLOAT))
  {
    strcpy(sterror,"array type is not expected R4 data type for this node");
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }
  else if ((strcmp(data_type,"R8")==0)&&(PyArray_TYPE(oarray)!=PyArray_DOUBLE))
  {
    strcpy(sterror,"array type is not expected R8 data type for this node");
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }
  else if ((strcmp(data_type,"C1")==0)&&(PyArray_TYPE(oarray)!=PyArray_STRING))
  {
    strcpy(sterror,"array type is not expected C1 data type for this node");
    PyErr_SetString(ADFErrorObject,sterror);
    return NULL;
  }
  if (!PyArray_Check(oarray))
  {
    PyErr_SetString(ADFErrorObject,
                    "second argument should be a Numpy array");
    return NULL;
  }
  /* If array is not fortran contiguous we copy (check if flag allows this), 
     this allocates a new data zone 
  */
  localptr=0;
  if ( (!PyArray_ISFORTRAN(oarray)) && isdataarray)
  {
    if (self->flags && __CGNS__FORCE_FORTRAN_INDEX__) 
    {
      ndim=PyArray_NDIM(oarray);
      if (ndim == 1)
      {
	ptrd=PyArray_DATA(oarray);
      }
      else if ((ndim == 2) || (ndim == 3))
      {
	if (ndim == 2)
	{
	  trsp=PyArray_Transpose((PyArrayObject*)oarray,NULL);
	  ptrd=PyArray_DATA(trsp);
	}
	else if (ndim == 3)
	{
	  imax=PyArray_DIM(oarray,0);
	  jmax=PyArray_DIM(oarray,1);
	  kmax=PyArray_DIM(oarray,2);
	  switch (PyArray_TYPE(oarray))
	  {
	    case PyArray_DOUBLE:
	      ptrd=(void*)malloc(sizeof(double)*imax*jmax*kmax);     break;
	    case PyArray_FLOAT:
	      ptrd=(void*)malloc(sizeof(float)*imax*jmax*kmax);      break;
	    case PyArray_LONG:
	      ptrd=(void*)malloc(sizeof(long)*imax*jmax*kmax);       break;
	    case PyArray_INT:
	      ptrd=(void*)malloc(sizeof(int)*imax*jmax*kmax);        break;
	    case PyArray_CHAR:
	      ptrd=(void*)malloc(sizeof(char)*imax*jmax*kmax);       break;
	    case PyArray_STRING:
	      ptrd=(void*)malloc(sizeof(char)*imax*jmax*kmax);       break;
	    default:
	      ptrd=NULL;
	      break;
	  }
	  if (ptrd == NULL)
	  {
	    PyErr_SetString(ADFErrorObject,
			    "Cannot allocate memory for C array transpose");
	    return NULL;
	  }
	  localptr=1;
	  for (i=0;i<imax;i++)
          {
	    for (j=0;j<jmax;j++)
	    {
	      for (k=0;k<kmax;k++)
	      {
		cx=k+j*kmax+i*jmax*kmax;
		fx=i+j*imax+k*imax*jmax;
		ptrs=PyArray_GETPTR3(oarray,i,j,k);
		switch (PyArray_TYPE(oarray))
		{
		case PyArray_FLOAT:
		  ((float*)ptrd)[fx]=*(float*)ptrs;break;
		case PyArray_DOUBLE:
		  ((double*)ptrd)[fx]=*(double*)ptrs;break;
		case PyArray_LONG:
		  ((long*)ptrd)[fx]=*(long*)ptrs;break;
		case PyArray_INT:
		  ((int*)ptrd)[fx]=*(int*)ptrs;break;
		case PyArray_CHAR:
		  ((char*)ptrd)[fx]=*(char*)ptrs;break;
		case PyArray_STRING:
		  ((char*)ptrd)[fx]=*(char*)ptrs;break;
		default:
		  break;
		}
	      }
	    }
	  }
	}
      }
      else
      {
	PyErr_SetString(ADFErrorObject,
			"Numpy argument should have dims in [1,2,3]");
	return NULL;
      }
    }
    else
    {
      PyErr_SetString(ADFErrorObject,
		      "Numpy argument should have NPY_FORTRAN flag");
      return NULL;
    }
  }
  else
  {
    ptrd=PyArray_DATA(oarray);
  }
  if (ptrd)
  {
    ADF__Write_All_Data(node_id, (char*)ptrd, &(self->last_error));
  }
  else
  {
    PyErr_SetString(ADFErrorObject,
		    "actual memory data is NULL");
    return NULL;
  }
  if (localptr)
  {
    free(ptrd);
  }
  Py_INCREF(Py_None);
  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_error_message
*/
static PyObject*
DBADF_errorMessage(DBADFObject *self, PyObject *args)
{
  int  ecode;
  char emsg[MAXERRORSIZE];
  
  if (!PyArg_ParseTuple(args, "i",&ecode))
  {
    PyErr_SetString(ADFErrorObject,
                    "error-message:string=adf.error_message(error-code:integer)");
    return NULL;
  }
  ADF__Error_Message(ecode,emsg);
  
  return PyString_FromString(emsg);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_set_error_state
*/
static PyObject*
DBADF_setEState(DBADFObject *self, PyObject *args)
{
  int  estate;
  
  if (!PyArg_ParseTuple(args, "i",&estate))
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.set_error_state(state:integer)");
    return NULL;
  }
  ADF__Set_Error_State(estate,&(self->last_error));
  Py_INCREF(Py_None);
  
  return Py_None;
}

/* ----------------------------------------------------------------------
 *ADF* ADF_get_error_state
*/
static PyObject*
DBADF_getEState(DBADFObject *self)
{
  int estate;
  
  ADF__Get_Error_State(&estate,&(self->last_error));

  return PyInt_FromLong(estate);
}

/* ----------------------------------------------------------------------
 *ADF* ADF_flush_to_disk
*/
static PyObject*
DBADF_flush(DBADFObject *self, PyObject *args)
{
  double id;

  if (!PyArg_ParseTuple(args, "d",&id))
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.flush_to_disk(node-id:double)");
    return NULL;
  }
  ADF__Flush_to_Disk(id,&(self->last_error));
  Py_INCREF(Py_None);

  return Py_None;
}

/* --------------------------------------------------------------------- 
 *ADF* ADF_Number_of_Children
*/
static PyObject *
DBADF_getChildrenNum(DBADFObject *self, PyObject *args)
{
  double parent;
  int    c;
  
  if (!PyArg_ParseTuple(args, "d",&parent))
  {
    PyErr_SetString(ADFErrorObject,
                    "number-of-children:integer=adf.number_of_children(node-id:double)");
    return NULL;
  }
  ADF__Number_of_Children(parent,&c,&(self->last_error));

  return PyInt_FromLong(c);   
}

/* --------------------------------------------------------------------- 
 *ADF* ADF_Get_Node_ID
*/
static PyObject*
DBADF_getNodeId(DBADFObject *self, PyObject *args)
{
  double parent,child;
  char *name;
  PyObject *pt;

  parent=-1;
  child=-1;
  name=NULL;

  if (PyTuple_Check(args))
  {
    pt=PyTuple_GetItem(args,0);
    if (PyFloat_Check(pt))
    {
      parent=PyFloat_AsDouble(pt);
    }
    pt=PyTuple_GetItem(args,1);    
    if (PyString_Check(pt))
    {
      name=PyString_AsString(pt);
    }
  }
/*  if (!PyArg_ParseTuple(args, "ds",&parent,&name))
  {
    PyErr_SetString(ADFErrorObject,
      "node-id:double=adf.get_node_id(parent-id:double,node-name:string)");
    return NULL;
  }
*/
  ADF__Get_Node_ID(parent,name,&child,&(self->last_error));

  return PyFloat_FromDouble(child);
}

/* --------------------------------------------------------------------- 
 *ADF* ADF_Get_Label
*/
static PyObject* 
DBADF_getLabel(DBADFObject *self, PyObject *args)
{
  double  id;
  char    label[ADF_LABEL_LENGTH+1];
  
  if (!PyArg_ParseTuple(args, "d",&id))
  {
    PyErr_SetString(ADFErrorObject,
                    "node-label:string=adf.get_node_label(node-id:double)");
    return NULL;
  }
  ADF__Get_Label(id,label,&(self->last_error));
  
  return PyString_FromString(label);
}

/* ---------------------------------------------------------------------
   Extra methods ***
   - calls to non-adf
   - pattern calls to adf, with some python data preparation
*/

/* ---------------------------------------------------------------------
   - re-open of a CGNS already open base
*/
static DBADFObject *
newDBADFObjectFromAlreadyOpen(double rootid)
{
  DBADFObject *self;

  self = PyObject_New(DBADFObject, &DBADF_Type);
  if (self == NULL) return NULL;
  /* should check error */
  self->root_id=rootid; /* should not be zero */
  self->last_error=-1;
  self->status=OPEN;
  return self;
}

/* --------------------------------------------------------------------- 
*/
static void
DBADF_dealloc(DBADFObject *self)
{
  if ((self->root_id) && (self->status == OPEN))
  {
    DBADF_close(self); /* do not call close on an already open ADF db */
  }
  
  PyObject_Del(self);
}

/* --------------------------------------------------------------------- 
*/
static PyObject *
DBADF_getChildren(DBADFObject *self, PyObject *args)
{
  char   cn[ADF_NAME_LENGTH+1];
  double parent,id;
  int    i,c,num;
  PyObject *l, *sx, *ix, *tx;
  
  if (!PyArg_ParseTuple(args, "d",&parent)) return NULL;
  ADF__Number_of_Children(parent,&num,&(self->last_error));
  
  l=PyList_New(num);
  
  for (i=1;i<=num;i++)
  {
    ADF__Children_Names(parent,i,1,ADF_NAME_LENGTH+1,&c,cn,&(self->last_error));

    tx=PyTuple_New(2);
    sx=PyString_FromString(cn);
    ADF__Get_Node_ID(parent,cn,&id,&(self->last_error));
    ix=PyFloat_FromDouble(id);    
    PyTuple_SET_ITEM(tx, 0, sx);
    PyTuple_SET_ITEM(tx, 1, ix);
    PyList_SET_ITEM(l,i-1,tx); /* CAUTION ARG START FROM 0 FOR PYTHON */
  }
  
  return l;
}

/* --------------------------------------------------------------------- 
*/
static PyObject*
DBADF_getRoot(DBADFObject *self)
{ 
  if ((self->status == OPEN) || (self->status == REOPEN))
  {
    return PyFloat_FromDouble(self->root_id);
  }
  else
  {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

/* --- The object method table ------------------------------------ */
static PyMethodDef DBADF_methods[] = {
 /* --- Pure ADF calls */
 {"database_close",             (PyCFunction)DBADF_close,            METH_VARARGS},
 {"database_delete",            (PyCFunction)DBADF_delete,           METH_VARARGS},
 {"database_get_format",        (PyCFunction)DBADF_getFormat,        METH_VARARGS},
 {"database_set_format",        (PyCFunction)DBADF_setFormat,        METH_VARARGS},
 {"database_garbage_collection",(PyCFunction)DBADF_dbgc,             METH_VARARGS},
 {"database_version",           (PyCFunction)DBADF_dbVersion,        METH_VARARGS},
 {"library_version",            (PyCFunction)DBADF_libVersion,       METH_VARARGS},
 {"get_root_id",                (PyCFunction)DBADF_getRootId,        METH_VARARGS},
 {"create",                     (PyCFunction)DBADF_nodeCreate,       METH_VARARGS},
 {"delete",                     (PyCFunction)DBADF_nodeDelete,       METH_VARARGS},
 {"put_name",                   (PyCFunction)DBADF_setNodeName,      METH_VARARGS},
 {"get_name",                   (PyCFunction)DBADF_getNodeName,      METH_VARARGS},
 {"number_of_children",         (PyCFunction)DBADF_getChildrenNum,   METH_VARARGS},
 {"children_names",             (PyCFunction)DBADF_getChildrenNames, METH_VARARGS},
 {"move_child",                 (PyCFunction)DBADF_nodeMove,         METH_VARARGS},
 {"get_node_id",                (PyCFunction)DBADF_getNodeId,        METH_VARARGS},
 {"get_label",                  (PyCFunction)DBADF_getLabel,         METH_VARARGS},
 {"set_label",                  (PyCFunction)DBADF_setLabel,         METH_VARARGS},
 {"get_data_type",              (PyCFunction)DBADF_getDataType,      METH_VARARGS},
 {"get_number_of_dimensions",   (PyCFunction)DBADF_getNumDim,        METH_VARARGS},
 {"get_dimension_values",       (PyCFunction)DBADF_getDimValues,     METH_VARARGS},
 {"put_dimension_information",  (PyCFunction)DBADF_putDimInfo,       METH_VARARGS},
 {"is_link",                    (PyCFunction)DBADF_isLink,           METH_VARARGS},
 {"link",                       (PyCFunction)DBADF_addLink,          METH_VARARGS},
 {"get_link_path",              (PyCFunction)DBADF_getLink,          METH_VARARGS},
 {"read_data",                  (PyCFunction)DBADF_readData,         METH_VARARGS},
 {"read_all_data",              (PyCFunction)DBADF_readAllData,      METH_VARARGS},
 {"write_data",                 (PyCFunction)DBADF_writeData,        METH_VARARGS},
 {"write_all_data",             (PyCFunction)DBADF_writeAllData,     METH_VARARGS},
 {"error_message",              (PyCFunction)DBADF_errorMessage,     METH_VARARGS},
 {"set_error_state",            (PyCFunction)DBADF_setEState,        METH_VARARGS},
 {"get_error_state",            (PyCFunction)DBADF_getEState,        METH_VARARGS},
 {"flush_to_disk",              (PyCFunction)DBADF_flush,            METH_VARARGS},
 /* --- Other methods --- */
 {"root",               (PyCFunction)DBADF_getRoot,     METH_VARARGS},
 {"children",           (PyCFunction)DBADF_getChildren, METH_VARARGS},
 {NULL,         NULL}           /* sentinel */
};

/* --------------------------------------------------------------------- 
   --- Attributes
       - some attributes are functions calls, related to the stored
         current node or not related to a particuliar context.
*/
static PyObject *
DBADF_getattr(DBADFObject *self, char *name)
{
  if (!strcmp(name, "error"))
  {
    return PyInt_FromLong((long)(self->last_error));
  }
  if (!strcmp(name, "rootid")) /* should use only function or attribute ? */
  {
    return PyFloat_FromDouble(self->root_id);
  }
  return Py_FindMethod(DBADF_methods, (PyObject *)self, name);
}

static int
DBADF_setattr(DBADFObject *self, char *name, PyObject *v)
{
  return 0;
}

statichere PyTypeObject DBADF_Type = {
        /* The ob_type field must be initialized in the module init function
         * to be portable to Windows without using C++. */
        PyObject_HEAD_INIT(NULL)
        0,                      /*ob_size*/
        "DBADF",                        /*tp_name*/
        sizeof(DBADFObject),    /*tp_basicsize*/
        0,                      /*tp_itemsize*/
        /* methods */
        (destructor)DBADF_dealloc, /*tp_dealloc*/
        0,                      /*tp_print*/
        (getattrfunc)DBADF_getattr, /*tp_getattr*/
        (setattrfunc)DBADF_setattr, /*tp_setattr*/
        0,                      /*tp_compare*/
        0,                      /*tp_repr*/
        0,                      /*tp_as_number*/
        0,                      /*tp_as_sequence*/
        0,                      /*tp_as_mapping*/
        0,                      /*tp_hash*/
        0,                      /*tp_call*/
        0,                      /*tp_str*/
        0,                      /*tp_getattro*/
        0,                      /*tp_setattro*/
        0,                      /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT,     /*tp_flags*/
        0,                      /*tp_doc*/
        0,                      /*tp_traverse*/
        0,                      /*tp_clear*/
        0,                      /*tp_richcompare*/
        0,                      /*tp_weaklistoffset*/
        0,                      /*tp_iter*/
        0,                      /*tp_iternext*/
        0,                      /*tp_methods*/
        0,                      /*tp_members*/
        0,                      /*tp_getset*/
        0,                      /*tp_base*/
        0,                      /*tp_dict*/
        0,                      /*tp_descr_get*/
        0,                      /*tp_descr_set*/
        0,                      /*tp_dictoffset*/
        0,                      /*tp_init*/
        0,                      /*tp_alloc*/
        0,                      /*tp_new*/
        0,                      /*tp_free*/
        0,                      /*tp_is_gc*/
};
/* ---------------------------------------------------------------------
   ADF top creation. Can be used from an already opened file
   or from scratch
*/
static PyObject *
adf_new(PyObject *self, PyObject *args)
{
  PyObject *fn;
  char *fs,*ff;
  DBADFObject *rv=NULL;

  if (!PyArg_ParseTuple(args, "O|ss",&fn,&fs,&ff)) return NULL;
  if (PyString_Check(fn))
  {
    rv = newDBADFObject(PyString_AsString(fn),fs,ff);
    if (rv) rv->status=OPEN;
  }
  else if (PyFloat_Check(fn))
  {
    rv= newDBADFObjectFromAlreadyOpen(PyFloat_AsDouble(fn));
    if (rv) rv->status=REOPEN;
  }
  else
  {
    PyErr_SetString(ADFErrorObject,
                    "adf.database_open(node-id:double|name:string,status:string,format:string)");
  }
  
  if ( rv == NULL )   return NULL;
  return (PyObject *)rv;
}

/* ---------------------------------------------------------------------
   We only have the adf object creation
*/
static PyMethodDef adf_methods[] = {
  {"database_open",     adf_new,                METH_VARARGS},
  {NULL,                NULL}           /* sentinel */
};

/* ---------------------------------------------------------------------
   Init the module
   - fill in the dictionnaries
*/   
PyObject *adfmodule_init(void)
{
  PyObject *m, *d, *dr, *s;

  /* Initialize the type of the new type object here; doing it here
   * is required for portability to Windows without requiring C++. */
  DBADF_Type.ob_type = &PyType_Type;
  
  /* Create the module and add the functions */
  m = Py_InitModule("adf", adf_methods);
  import_array();
  
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(m);
  ADFErrorObject = PyErr_NewException("adf.error", NULL, NULL);
  PyDict_SetItemString(d, "error", ADFErrorObject);

#define addConstInDict(xd,xxd,xs) \
s= PyString_FromString(xs);\
PyDict_SetItemString(xxd,xs,s);\
PyDict_SetItemString(xd,xs,s);\
Py_DECREF(s); 

  /* ----------- ADF open status */
  dr = PyDict_New();
  PyDict_SetItemString(d, "ADF_OPENSTATUS", dr);
  addConstInDict(d,dr,"READ_ONLY");
  addConstInDict(d,dr,"OLD");
  addConstInDict(d,dr,"NEW");
  addConstInDict(d,dr,"SCRATCH");
  addConstInDict(d,dr,"UNKNOWN");
  Py_DECREF(dr);
                   
  /* ----------- ADF open formats */
  dr = PyDict_New();
  PyDict_SetItemString(d, "ADF_OPENFORMAT", dr);
  addConstInDict(d,dr,"NATIVE");  
  addConstInDict(d,dr,"IEEE_BIG");  
  addConstInDict(d,dr,"IEEE_LITTLE");  
  addConstInDict(d,dr,"CRAY");  
  Py_DECREF(dr);
  return m;
}

DL_EXPORT(void)
initadf(void)
{
  adfmodule_init();
}
/* --- last line */
