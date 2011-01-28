/* 
#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  ------------------------------------------------------------------------- 
*/

/* DbMIDLEVEL objects */

#include "Python.h"
#define  PY_ARRAY_UNIQUE_SYMBOL CGNS_PY_ARRAY_UNIQUE_SYMBOL
#include "numpy/oldnumeric.h"
#include "numpy/ndarrayobject.h"

#include "cgnslib.h"
#include "ADF__.h"

/* 

 *** NUMPY remarks (from Numpy various docs)
 *** NPY_FORTRAN       0x0002
 *** set if array is a contiguous Fortran array: the first index
 *** varies the fastest in memory (strides array is reverse of
 *** C-contiguous array)
 *** Note: all 0-d arrays are CONTIGUOUS and FORTRAN contiguous. If a
 *** 1-d array is CONTIGUOUS it is also FORTRAN contiguous
 *** NPY_OWNDATA       0x0004
 *** If set, the array owns the data: it will be free'd when the array
 *** is deleted.
 *** Always copy the array. Returned arrays are always CONTIGUOUS, ALIGNED,
 *** and WRITEABLE.
 *** NPY_ENSURECOPY    0x0020

 All pyCGNS arrays are (or understood as) Fortran arrays
 If pyCGNS creates an array, it creates a Fortran contiguous array
 If pyCGNS reads an array, it checks wether it is a C or Fortran array.
 In case this array is C then it is translated before being used.

*/


extern void midleveldictionnary_init(PyObject *d);
extern PyObject* adfmodule_init(void);

#define CGNSMAXNAMESIZE    256
#define CGNSMAXDESCSIZE    2048
#define CGNSMAXMESSAGESIZE 256

/* This define is used for large print trace at debug time */
#define CGNS_TRACE_X

static PyObject *MIDLEVELErrorObject;

typedef struct {
  PyObject_HEAD
  int    db_fn;
  double rootid; /* stolen from cgns internals for adf purpose */
  int    last_error;
  int    last_base;
  char   last_error_message[CGNSMAXMESSAGESIZE];
} DbMIDLEVELObject;

static PyTypeObject DbMIDLEVEL_Type;

#define DbMIDLEVELObject_Check(v)	((v)->ob_type == &DbMIDLEVEL_Type)

#if CGNS_VERSION < 3000

double icg_root_id(int f,double *r)
{
  cgns_file *cglocal;
  cgns_file *cgsave;
  
  cgsave = cg; /* cg is global */
  cglocal=cgi_get_file(f); /* side effects in there */
  cg=cgsave;
  *r=cglocal->rootid;
  
  if (cglocal == 0) return 1;
  return 0;
}

#endif

extern int cgio_path_add (const char *path);

/* ---------------------------------------------------------------------- */
static DbMIDLEVELObject *
newDbMIDLEVELObject(char* name, int mode)
{
  double rid;
  DbMIDLEVELObject *self;

  self = PyObject_New(DbMIDLEVELObject, &DbMIDLEVEL_Type);
  if (self == NULL) return NULL;
  ADF_Set_Error_State(1,&(self->last_error));
  self->last_error=cg_open(name,mode,&(self->db_fn));
  strcpy(self->last_error_message,cg_get_error());

  if (self->last_error)
  { 
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    return NULL;
  }

#if CGNS_VERSION < 3000
  self->last_error=icg_root_id(self->db_fn,&rid);
#else
  self->last_error=cg_root_id(self->db_fn,&rid);
#endif
  if (self->last_error)
  { 
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    return NULL;
  }
  self->rootid=rid;

  return self; 
}

/* DbMIDLEVEL methods ================================================== */
/* Close db */
static PyObject*
DbMIDLEVEL_close(DbMIDLEVELObject *self)
{
  self->last_error=cg_close(self->db_fn);
  strcpy(self->last_error_message,cg_get_error());
#ifdef CGNS_TRACE
  printf("## close file [%d] err:[%s]\n",self->db_fn,self->last_error_message);
  fflush(stdout);
#endif
  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* Close db at deletion */
static void
DbMIDLEVEL_dealloc(DbMIDLEVELObject *self)
{
  DbMIDLEVEL_close(self);
#ifdef CGNS_TRACE
  printf("## deallocate\n");fflush(stdout);
#endif  
  PyObject_Del(self);
}

/* ---------------------------------------------------------------------- */
/* lastError:
   returns tuple (error code, error message)
*/
static PyObject *
DbMIDLEVEL_lastError(DbMIDLEVELObject *self, PyObject *args)
{
  PyObject *tp, *ie, *se;

  tp = PyTuple_New(2);
  ie=PyInt_FromLong((long)(self->last_error));
  se=PyString_FromString(self->last_error_message);
  
  PyTuple_SET_ITEM(tp, 0, ie);
  PyTuple_SET_ITEM(tp, 1, se);  

  return tp;
}

/* ---------------------------------------------------------------------- */
/* deleteNode: cg_delete_node                                             */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_deleteNode(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;

  if (!PyArg_ParseTuple(args, "s",&name)) return NULL;

  self->last_error=cg_delete_node(name);
  strcpy(self->last_error_message,cg_get_error());
    
  Py_INCREF(Py_None);
  return Py_None;
}


/* ---------------------------------------------------------------------- */
/* goto: cg_goto
         base index
         set flag (0 means returns the id but do not set as current node)
         list of tuples (label, index)
   returns self
*/
static PyObject *
DbMIDLEVEL_goto(DbMIDLEVELObject *self, PyObject *args)
{
  int b,i,n,s;
  PyObject *path, *tpl;
  char     *label[20]; /* 20 is the GCNS magic value... */
  int       index[20];
  double    id=0;

  if (!PyArg_ParseTuple(args, "iiO",&b,&s,&path)) return NULL;
  if (!PyList_Check(path))
  {
    if ((PyTuple_Check(path)) && (PyTuple_Size(path) == 0))
    {
      n=0;
#ifdef CGNS_TRACE
      printf("## null path...\n");
#endif      
    }
    else
    {
      PyErr_SetString(MIDLEVELErrorObject,
                      "Bad 2nd arg: should be list of tuples");
    return NULL;
    }
  }
  else
  {
    n=PyList_Size(path);
  }
  
  if (n != 0)
  {
    for (i=0; i<n; i++)
    {
      tpl = PyList_GetItem(path,i);
      if (!PyTuple_Check(tpl))
      {
        PyErr_SetString(MIDLEVELErrorObject,
                        "Bad 2nd arg item: should be tuples");
        return NULL;
      }
    }
    /* Go on with args */
#ifdef CGNS_TRACE
    printf("## path...\n");
#endif      
    for (i=0; i<n ; i++)
    {
      tpl =PyList_GetItem(path,i);
      label[i]=PyString_AsString(PyTuple_GetItem(tpl, 0));
      index[i]=PyInt_AsLong(PyTuple_GetItem(tpl, 1));
#ifdef CGNS_TRACE
      printf("## Label [%s] Index [%d] (%d)\n",label[i],index[i],i);
      fflush(stdout);
#endif      
    }
  }

  /* here we have the label and index arrays ready to pass to cg_goto */
  /* caution: n can be zero */
  if (s)
  {
#ifdef CGNS_TRACE
    printf("## switch goto length [%d]\n",n);
#endif      
    switch (n)
    {
      case 0:   self->last_error=cg_goto(self->db_fn,b,"end"); break;
      case 1:   self->last_error=cg_goto(self->db_fn,b,label[0],index[0]
                                         ,"end"); break;
      case 2:   self->last_error=cg_goto(self->db_fn,b,label[0],index[0],
                                    label[1],index[1],"end"); break;
        
      case 3:   self->last_error=cg_goto(self->db_fn,b,label[0],index[0],
                                    label[1],index[1],
                                    label[2],index[2],"end"); break;
        
      case 4:   self->last_error=cg_goto(self->db_fn,b,label[0],index[0],
                                        label[1],index[1],
                                        label[2],index[2],
                                        label[3],index[3],
                                        "end"); break;
        
      case 5:   self->last_error=cg_goto(self->db_fn,b,label[0],index[0],
                                        label[1],index[1],
                                        label[2],index[2],
                                        label[3],index[3],
                                        label[4],index[4],
                                        "end"); break;
        
      case 6:   self->last_error=cg_goto(self->db_fn,b,label[0],index[0],
                                        label[1],index[1],
                                        label[2],index[2],
                                        label[3],index[3],
                                        label[4],index[4],
                                        label[5],index[5],
                                        "end"); break;
        
      case 7:   self->last_error=cg_goto(self->db_fn,b,label[0],index[0],
                                        label[1],index[1],
                                        label[2],index[2],
                                        label[3],index[3],
                                        label[4],index[4],
                                        label[5],index[5],
                                        label[6],index[6],
                                        "end"); break;
        
      case 8:   self->last_error=cg_goto(self->db_fn,b,label[0],index[0],
                                        label[1],index[1],
                                        label[2],index[2],
                                        label[3],index[3],
                                        label[4],index[4],
                                        label[5],index[5],
                                        label[6],index[6],
                                        label[7],index[7],
                                        "end"); break;
        
      case 9:   self->last_error=cg_goto(self->db_fn,b,label[0],index[0],
                                        label[1],index[1],
                                        label[2],index[2],
                                        label[3],index[3],
                                        label[4],index[4],
                                        label[5],index[5],
                                        label[6],index[6],
                                        label[7],index[7],
                                        label[8],index[8],
                                        "end"); break;
        
      case 10:  self->last_error=cg_goto(self->db_fn,b,label[0],index[0],
                                        label[1],index[1],
                                        label[2],index[2],
                                        label[3],index[3],
                                        label[4],index[4],
                                        label[5],index[5],
                                        label[6],index[6],
                                        label[7],index[7],
                                        label[8],index[8],
                                        label[9],index[9],
                                        "end"); break;
      default:
        PyErr_SetString(MIDLEVELErrorObject,"path length too long");
        return NULL;
    }
    id=1.;
  }
  else
  {
    PyErr_SetString(MIDLEVELErrorObject,"goto flag no longer implemented");
    return NULL;
  }

  /* if the goto fails, we return NULL */
  if (self->last_error)
  {
     PyErr_SetString(MIDLEVELErrorObject,"goto leads to bad path");
     return NULL;
  }
  self->last_base=b;
  return PyFloat_FromDouble((double)id);
}

/* ---------------------------------------------------------------------- */
/* TODO - if you are volunteer, go ahead ;) */
static PyObject *
DbMIDLEVEL_coordPartialWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_parentDataPartWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_sectionPartWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_elementsPartRead(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_elementDataPartSize(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_fieldPartWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_ConnOneToOneAverageRead(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_ConnOneToOneAverageWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_ConnOneToOnePeriodicRead(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_ConnOneToOnePeriodicWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_ConnReadShort(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_ConnWriteShort(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_bocoBCDatasetWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_bocoBCDatasetRead(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_bocoBCDatasetInfo(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_gorel(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_gopath(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_golist(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_where(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_freeData(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_arrayReadAs(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_nDescriptor(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_nUnits(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_unitsFullRead(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_unitsFullWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_nExponents(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_exponentsFullRead(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_exponentsFullWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_ptsetInfo(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_ptsetWrite(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}
static PyObject *
DbMIDLEVEL_ptsetRead(DbMIDLEVELObject *self, PyObject *args)
{
    PyErr_SetString(MIDLEVELErrorObject,"Not implemented yet (sorry)");
    return NULL;
}

/* ---------------------------------------------------------------------- */
/* baseWrite: cg_base_write
              base name (string),
              cell dimension (int),
              physical dimension (int)
   returns base index number
*/
static PyObject *
DbMIDLEVEL_baseWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int bs_fn, cdim, pdim;
  char *name;

  if (!PyArg_ParseTuple(args, "sii",&name,&cdim,&pdim)) return NULL;
  self->last_error=cg_base_write(self->db_fn,name,cdim,pdim,&bs_fn);
  if (self->last_error)
  {
     strcpy(self->last_error_message,cg_get_error());
     PyErr_SetString(MIDLEVELErrorObject,"cannot add a new base");
     return NULL;
  }
#ifdef CGNS_TRACE
  printf("## add new base [%s]\n",name);fflush(stdout);
#endif  
  return PyInt_FromLong(bs_fn);
}

/* ---------------------------------------------------------------------- */
/* baseRead: cg_base_read
             index (int)
   returns tuple (base index, base name, cell dim, physical dim)
*/
static PyObject *
DbMIDLEVEL_baseRead(DbMIDLEVELObject *self, PyObject *args)
{
  int       idx=0, cdim=0, pdim=0;
  char     *name;
  PyObject *tp, *ix, *xn, *cd, *pd;

  if (!PyArg_ParseTuple(args, "i",&idx)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_base_read(self->db_fn,idx,name,&cdim,&pdim);
  strcpy(self->last_error_message,cg_get_error());
  tp = PyTuple_New(4);
  
  ix=PyInt_FromLong((long)idx);  /* base index */
  xn=PyString_FromString(name);  /* base name  */
  cd=PyInt_FromLong((long)cdim); /* cell dim   */
  pd=PyInt_FromLong((long)pdim); /* phys dim   */

  PyTuple_SET_ITEM(tp, 0, ix);
  PyTuple_SET_ITEM(tp, 1, xn);
  PyTuple_SET_ITEM(tp, 2, cd);
  PyTuple_SET_ITEM(tp, 3, pd);

  free(name);
  
  return tp;
}

static int
getBaseDim(DbMIDLEVELObject *self, int b)
{
  int   cdim, pdim;
  char  name[CGNSMAXNAMESIZE];

  if (!b)
  {
    b=self->last_base;
  }
  self->last_error=cg_base_read(self->db_fn,b,name,&cdim,&pdim);
  strcpy(self->last_error_message,cg_get_error());
  
  return cdim;
}

/* ---------------------------------------------------------------------- */
/* baseId: cg_base_id (not documented)
           index (int)
   returns id (double)
*/
static PyObject *
DbMIDLEVEL_baseId(DbMIDLEVELObject *self, PyObject *args)
{
  int    idx;
  double id;
  
  if (!PyArg_ParseTuple(args, "i",&idx)) return NULL;
  self->last_error=cg_base_id(self->db_fn,idx,&id);
  strcpy(self->last_error_message,cg_get_error());
  return PyFloat_FromDouble(id);
}

/* ---------------------------------------------------------------------- */
/* simTypeRead: cg_simulation_type_read
           base index (int)
   returns type of simulation
*/
static PyObject *
DbMIDLEVEL_simTypeRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b;
  CG_SimulationType_t st;

  if (!PyArg_ParseTuple(args, "i",&b)) return NULL;
  self->last_error=cg_simulation_type_read(self->db_fn,b,&st);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(st);
}

/* ---------------------------------------------------------------------- */
/* simTypeWrite: cg_simulation_type_write
           base index (int)
           simulation type (int)
   returns None
*/
static PyObject *
DbMIDLEVEL_simTypeWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b;
  CG_SimulationType_t st;
  
  if (!PyArg_ParseTuple(args, "ii",&b,&st)) return NULL;
  self->last_error=cg_simulation_type_write(self->db_fn,b,(CG_SimulationType_t)st);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(st);
}

/* ---------------------------------------------------------------------- */
/* governingRead: cg_governing_read
   returns type of government (Dictature, Republican, Democrat, Anarchoidal...)
*/
static PyObject *
DbMIDLEVEL_governingRead(DbMIDLEVELObject *self, PyObject *args)
{
  CG_GoverningEquationsType_t g;

  self->last_error=cg_governing_read(&g);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(g);
}

/* ---------------------------------------------------------------------- */
/* governingWrite: cg_governing_write
           gov type (int)
   returns None
*/
static PyObject *
DbMIDLEVEL_governingWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int g;

  if (!PyArg_ParseTuple(args, "i",&g)) return NULL;
  self->last_error=cg_governing_write((CG_GoverningEquationsType_t)g);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(g);
}

/* ---------------------------------------------------------------------- */
/* modelRead: cg_model_read
           model label (string)
   returns tuple (lable, type of model)
*/
static PyObject *
DbMIDLEVEL_modelRead(DbMIDLEVELObject *self, PyObject *args)
{
  CG_ModelType_t   g;
  char *name;
  PyObject *xr;
  
  if (!PyArg_ParseTuple(args, "s",&name)) return NULL;
  self->last_error=cg_model_read(name,&g);
  strcpy(self->last_error_message,cg_get_error());

  xr= Py_BuildValue("(si)",name,g);
  return xr;
}

/* ---------------------------------------------------------------------- */
/* modelWrite: cg_model_write
           model label (string)
           model type (int)
   returns None
*/
static PyObject *
DbMIDLEVEL_modelWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int   g;
  char *name;

  if (!PyArg_ParseTuple(args, "si",&name,&g)) return NULL;
  self->last_error=cg_model_write(name,(CG_ModelType_t)g);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* diffusionRead: cg_diffusion_read
   returns type of diffusion
*/
static PyObject *
DbMIDLEVEL_diffusionRead(DbMIDLEVELObject *self, PyObject *args)
{
  int df[6]={0,0,0,0,0,0};
  PyObject *tp;
  
  self->last_error=cg_diffusion_read(df);
  strcpy(self->last_error_message,cg_get_error());
  tp = Py_BuildValue("(iiiiii)",df[0],df[1],df[2],df[3],df[4],df[5]);

  return tp;
}

/* ---------------------------------------------------------------------- */
/* diffusionWrite: cg_diffusion_write
           diffusion type (int)
   returns None
*/
static PyObject *
DbMIDLEVEL_diffusionWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int df[6]={0,0,0,0,0,0};

  if (!PyArg_ParseTuple(args, "(iiiiii)",&df[0],&df[1],&df[2],&df[3],&df[4],&df[5])) return NULL;
  self->last_error=cg_diffusion_write(df);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* gridlocationRead: cg_gridlocation_read
   returns type of grid location
*/
static PyObject *
DbMIDLEVEL_gridlocationRead(DbMIDLEVELObject *self, PyObject *args)
{
  CG_GridLocation_t gt;

  self->last_error=cg_gridlocation_read(&gt);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(gt);
}

/* ---------------------------------------------------------------------- */
/* gridlocationWrite: cg_gridlocation_write
           gridlocation type (int)
   returns arg type
*/
static PyObject *
DbMIDLEVEL_gridlocationWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int gt;

  if (!PyArg_ParseTuple(args, "i",&gt)) return NULL;
  self->last_error=cg_gridlocation_write((CG_GridLocation_t)gt);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(gt);
}

/* ---------------------------------------------------------------------- */
/* rotatingRead: cg_rotating_read                                         */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_rotatingRead(DbMIDLEVELObject *self, PyObject *args)
{
  float *rv,*rc;
  PyObject *tp0,*tp1,*tp2;
  int dim,i;
  
  dim=getBaseDim(self,0);
  rv=(float*) malloc(sizeof(float)*dim);
  rc=(float*) malloc(sizeof(float)*dim);
  self->last_error=cg_rotating_read(rv,rc);
  strcpy(self->last_error_message,cg_get_error());

  tp0 = PyTuple_New(2);
  tp1 = PyTuple_New(dim);
  tp2 = PyTuple_New(dim);
  
  for (i=0;i<dim;i++)
  {
    PyTuple_SET_ITEM(tp1, i, PyFloat_FromDouble(rv[i]));
    PyTuple_SET_ITEM(tp2, i, PyFloat_FromDouble(rc[i]));
  }
  PyTuple_SET_ITEM(tp0, 0, tp1);
  PyTuple_SET_ITEM(tp0, 1, tp2);
  
  return tp0;
}

/* ---------------------------------------------------------------------- */
/* rotatingWrite: cg_rotating_write                                       */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_rotatingWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int dim,i;
  PyObject *rv,*rc;
  float *arv,*arc;
  
  if (!PyArg_ParseTuple(args,"OO",&rv,&rc)) return NULL;

  dim=getBaseDim(self,0);
  /* an error can occur here, if the current base has not be set by
     a goto. The number may be incorrect, and the diagnosis says
     the Physical dimension doesn't match. The problem then is goto,
     no physical dimension
  */
  arv=(float*) malloc(sizeof(float)*dim);
  arc=(float*) malloc(sizeof(float)*dim);
  if (   PyTuple_Check(rv)
      && PyTuple_Check(rc)
      &&(PyTuple_Size(rv)==dim)
      &&(PyTuple_Size(rc)==dim))
  {
    for (i=0;i<dim;i++)
    {
      arv[i]=PyFloat_AsDouble(PyTuple_GetItem(rv,i));
      arc[i]=PyFloat_AsDouble(PyTuple_GetItem(rc,i));
    }
  }
  else
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "Bad arg: should be tuples of base physical dimensions");
    free(arv);
    free(arc);
    return NULL;
  }
  
  self->last_error=cg_rotating_write(arv,arc);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* axisymRead: cg_axisym_read                                             */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_axiSymRead(DbMIDLEVELObject *self, PyObject *args)
{
  float *rv,*rc;
  PyObject *tp0,*tp1,*tp2;
  int dim,i,b;
  
  if (!PyArg_ParseTuple(args, "i",&b)) return NULL;
  dim=getBaseDim(self,b); 
  rv=(float*) malloc(sizeof(float)*dim);
  rc=(float*) malloc(sizeof(float)*dim);
  self->last_error=cg_axisym_read(self->db_fn,b,rv,rc);
  strcpy(self->last_error_message,cg_get_error());

  if (self->last_error)
  {
    /*    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
	  return NULL; 
    */
    Py_INCREF(Py_None);
    return Py_None;
  }

  tp0 = PyTuple_New(2);
  tp1 = PyTuple_New(dim);
  tp2 = PyTuple_New(dim);
  
  for (i=0;i<dim;i++)
  {
    PyTuple_SET_ITEM(tp1, i, PyFloat_FromDouble(rv[i]));
    PyTuple_SET_ITEM(tp2, i, PyFloat_FromDouble(rc[i]));
  }
  PyTuple_SET_ITEM(tp0, 0, tp1);
  PyTuple_SET_ITEM(tp0, 1, tp2);
  
  return tp0;
}

/* ---------------------------------------------------------------------- */
/* axisymWrite: cg_axisym_write                                           */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_axiSymWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,dim,i;
  PyObject *rv,*rc;
  float *arv,*arc;
  
  if (!PyArg_ParseTuple(args,"iOO",&b,&rv,&rc)) return NULL;

  dim=getBaseDim(self,b);
  arv=(float*) malloc(sizeof(float)*dim);
  arc=(float*) malloc(sizeof(float)*dim);
  if (   PyTuple_Check(rv)
      && PyTuple_Check(rc)
      &&(PyTuple_Size(rv)==dim)
      &&(PyTuple_Size(rc)==dim))
  {
    for (i=0;i<dim;i++)
    {
      arv[i]=PyFloat_AsDouble(PyTuple_GetItem(rv,i));
      arc[i]=PyFloat_AsDouble(PyTuple_GetItem(rc,i));
    }
  }
  else
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "Bad arg: should be tuples of base physical dimensions");
    free(arv);
    free(arc);
    return NULL;
  }
  
  self->last_error=cg_axisym_write(self->db_fn,b,arv,arc);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* gravityRead: cg_gravity_read                                           */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_gravityRead(DbMIDLEVELObject *self, PyObject *args)
{
  float  gv[3]={0,0,0};
  double dgv;
  PyObject *tp0;
  int dim,i,b;
  
  if (!PyArg_ParseTuple(args, "i",&b)) return NULL;
  dim=getBaseDim(self,b); 
  self->last_error=cg_gravity_read(self->db_fn,b,gv);
  strcpy(self->last_error_message,cg_get_error());

  tp0 = PyTuple_New(dim);
  
  for (i=0;i<dim;i++)
  {
    dgv=gv[i];
    PyTuple_SET_ITEM(tp0, i, PyFloat_FromDouble((double)dgv));
  }
  
  return tp0;
}

/* ---------------------------------------------------------------------- */
/* gravityWrite: cg_gravity_write                                         */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_gravityWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,dim,i;
  PyObject *gv;
  double agv[3]={0,0,0}; 
  float fgv[3]={0,0,0};  
  
  if (!PyArg_ParseTuple(args,"iO",&b,&gv)) return NULL;

  dim=getBaseDim(self,b);
  if (PyTuple_Check(gv)&&(PyTuple_Size(gv)==dim))
  {
    if (PyTuple_Size(gv)==1)
    {
      PyArg_ParseTuple(gv,"d",&agv[0]);
    }
    if (PyTuple_Size(gv)==2)
    {
      PyArg_ParseTuple(gv,"dd",&agv[0],&agv[1]);
    }
    if (PyTuple_Size(gv)==3)
    {
      PyArg_ParseTuple(gv,"ddd",&agv[0],&agv[1],&agv[2]);
    }
  }
  else
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "Bad arg: should be tuple of base physical dimensions");
    return NULL;
  }
  for (i=0;i<dim;i++)
  {
    fgv[i]=agv[i];
  }

  self->last_error=cg_gravity_write(self->db_fn,b,fgv);
  strcpy(self->last_error_message,cg_get_error());
  
  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* nzones: cg_nzones
           base index (int)
   returns zone total count
*/
static PyObject *
DbMIDLEVEL_nzones(DbMIDLEVELObject *self, PyObject *args)
{
  int   b, nz;

  if (!PyArg_ParseTuple(args, "i",&b)) return NULL;
  self->last_error=cg_nzones(self->db_fn,b,&nz);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nz);
}

/* ---------------------------------------------------------------------- */
/* ZoneId: cg_zone_id (not documented)
           base index (int)
           zone index (int)
   returns id (double)
*/
static PyObject *
DbMIDLEVEL_zoneId(DbMIDLEVELObject *self, PyObject *args)
{
  int    bx,zx;
  double id;
  
  if (!PyArg_ParseTuple(args, "ii",&bx,&zx)) return NULL;
  self->last_error=cg_zone_id(self->db_fn,bx,zx,&id);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyFloat_FromDouble(id);
}

/* ---------------------------------------------------------------------- */
/* zoneWrite: cg_zone_write
              base index (int)
              zone name (string)
              zone size (list of integers)
              zone type (int)
   returns zone index
*/
static PyObject *
DbMIDLEVEL_zoneWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int       bs_fn, zn_fn, zt;
  char     *name;
  int       zsz[9]; /* max size for zone indices description */

  if (!PyArg_ParseTuple(args, "is(iiiiiiiii)i",\
                        &bs_fn,&name,\
                        &zsz[0],&zsz[1],&zsz[2],&zsz[3],&zsz[4],\
                        &zsz[5],&zsz[6],&zsz[7],&zsz[8],&zt))
  {
    return NULL;
  }

  self->last_error=cg_zone_write(self->db_fn,bs_fn,name,zsz,(CG_ZoneType_t)zt,&zn_fn);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(zn_fn);
}

/* ---------------------------------------------------------------------- */
/* zoneRead: cg_zone_read
             base index (int)
             zone index (int)             
   returns tuple
*/
static PyObject *
DbMIDLEVEL_zoneRead(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;
  char  bname[CGNSMAXNAMESIZE];
  int   ncelldim, nphysdim, nsz;
  int  *zsz;
  int   b, z, ct;
  PyObject *tp, *bx, *zx, *nx, *vx, *tx;

  if (!PyArg_ParseTuple(args, "ii",&b,&z))
  {
    return NULL;
  }

  self->last_error=cg_base_read(self->db_fn, b, bname, &ncelldim, &nphysdim);

  switch (ncelldim) {
  case (2):
  case (3):
    break;
  default:
    ncelldim = 3;
  }

  nsz = 2*ncelldim;
  zsz=(int*) malloc(sizeof(int)*9); /* max size for zone indices description */
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");

  for (ct=0; ct <9; ct++) { zsz[ct]=0; }

  self->last_error=cg_zone_read(self->db_fn,b,z,name,zsz);
  strcpy(self->last_error_message,cg_get_error());

  tp = PyTuple_New(4);

  bx=PyInt_FromLong((long)b);   /* base index */
  zx=PyInt_FromLong((long)z);   /* zone index */
  nx=PyString_FromString(name); /* zone name  */
  
  tx = PyTuple_New(nsz);          /* size tuple */
  for (ct=0; ct <nsz; ct++)
  {
    vx=PyInt_FromLong(zsz[ct]);
    PyTuple_SET_ITEM(tx, ct, vx);
  }
  
  PyTuple_SET_ITEM(tp, 0, bx);
  PyTuple_SET_ITEM(tp, 1, zx);
  PyTuple_SET_ITEM(tp, 2, nx);
  PyTuple_SET_ITEM(tp, 3, tx);

  free(name);
  free(zsz);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* zoneType: cg_zone_type
             zone index (int)
   returns tuple
*/
static PyObject *
DbMIDLEVEL_zoneType(DbMIDLEVELObject *self, PyObject *args)
{
  int b, z;
  CG_ZoneType_t nz;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_zone_type(self->db_fn,b,z,&nz);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nz);
}

/* ---------------------------------------------------------------------- */
/* nConns                                                                 */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_nConns(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,nc;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_nconns(self->db_fn,b,z,&nc);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nc);
}

/* ---------------------------------------------------------------------- */
/* ConnInfo                                                               */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_ConnInfo(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,c,np=0,ndd=0;
  CG_GridLocation_t gl=0;
  CG_GridConnectivityType_t gt=0;
  CG_PointSetType_t pt=0,dpt=0;
  char *nm,*dnm;
  CG_ZoneType_t dzt=0;
  CG_DataType_t ddt=0;
  PyObject *tp;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&c)) return NULL;

  nm =(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  dnm=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);

  strcpy(nm,"");
  strcpy(dnm,"");

  self->last_error=cg_conn_info(self->db_fn,b,z,c,
                                nm,&gl,&gt,&pt,&np,dnm,&dzt,&dpt,&ddt,&ndd);
  strcpy(self->last_error_message,cg_get_error());

  tp= Py_BuildValue("(siiiisiiii)",nm,gl,gt,pt,np,dnm,dzt,dpt,ddt,ndd);
  
  free(nm);
  free(dnm);

  return tp;
}

/* ---------------------------------------------------------------------- */
/* ConnRead                                                               */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_ConnRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,c,np=0,ndd=0,dim=0;
  CG_GridLocation_t gl=0;
  CG_GridConnectivityType_t gt=0;
  CG_PointSetType_t pt=0,dpt=0;
  char *nm,*dnm;
  CG_ZoneType_t dzt=0;
  CG_DataType_t ddt=0;
  PyObject *tp;
  void *ddata;
  int  *pnts;
  npy_intp arddims=0,arcdims=0;
  PyArrayObject *ard,*arc;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&c)) return NULL;

  nm =(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  dnm=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);

  strcpy(nm,"");
  strcpy(dnm,"");

  self->last_error=cg_conn_info(self->db_fn,b,z,c,
                                nm,&gl,&gt,&pt,&np,dnm,&dzt,&dpt,&ddt,&ndd);
  strcpy(self->last_error_message,cg_get_error());
  free(nm);
  free(dnm);

  dim=getBaseDim(self,b);
  pnts =(int*)malloc(sizeof(int) *np*dim);
  ddata=(int*)malloc(sizeof(int) *ndd*dim); 
  self->last_error=cg_conn_read(self->db_fn,b,z,c,pnts,ddt,ddata);
  strcpy(self->last_error_message,cg_get_error());

  /* --- create two arrays, fortran order, data is left to python object. */
  arcdims=np*dim;
  arc=(PyArrayObject*)PyArray_New(&PyArray_Type,1,&arcdims,PyArray_INT,
				  NULL,(void*)pnts,0,
				  NPY_OWNDATA|NPY_FORTRAN, NULL);
  arddims=ndd*dim;
  ard=(PyArrayObject*)PyArray_New(&PyArray_Type,1,&arddims,PyArray_INT,
				  NULL,(void*)ddata,0,
				  NPY_OWNDATA|NPY_FORTRAN, NULL);
  tp=PyTuple_New(2);
  PyTuple_SET_ITEM(tp, 0, (PyObject*)arc);
  PyTuple_SET_ITEM(tp, 1, (PyObject*)ard);  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* ConnWrite                                                              */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_ConnWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,c,np,ndd;
  CG_GridLocation_t gl;
  CG_GridConnectivityType_t gt;
  CG_PointSetType_t pt,dpt;
  char *nm,*dnm;
  CG_ZoneType_t dzt;
  CG_DataType_t ddt;
  PyObject *oard,*oarc;
  PyArrayObject *ardt,*arct;
  PyArrayObject *ard,*arc;
  
  if (!PyArg_ParseTuple(args, "iisiiiiOsiiiiO",
                        &b,&z,&nm,&gl,&gt,&pt,&np,&oarc,&dnm,&dzt,
                        &dpt,&ddt,&ndd,&oard))
  {
    return NULL;
  }
  arct=(PyArrayObject*)oarc;
  arc=(PyArrayObject*)PyArray_CopyFromObject(oarc,
                                             arct->descr->type_num,
                                             arct->nd,
                                             arct->nd);
  ardt=(PyArrayObject*)oard;
  ard=(PyArrayObject*)PyArray_CopyFromObject(oard,
                                             ardt->descr->type_num,
                                             ardt->nd,
                                             ardt->nd);
  self->last_error=cg_conn_write(self->db_fn,b,z,nm,
                                 (CG_GridLocation_t)gl,
                                 (CG_GridConnectivityType_t)gt,
                                 (CG_PointSetType_t)pt,
                                 np,
                                 (void*)(arc->data),
                                 dnm,
                                 (CG_ZoneType_t)dzt,
                                 (CG_PointSetType_t)dpt,
                                 (CG_DataType_t)ddt,
                                 ndd,
                                 (void*)(ard->data),
                                 &c);
  if (self->last_error)
  {
     strcpy(self->last_error_message,cg_get_error());
     PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
     return NULL;
  }

  return PyInt_FromLong(c);
}

/* ---------------------------------------------------------------------- */
/* ConnId                                                                 */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_ConnId(DbMIDLEVELObject *self, PyObject *args)
{
  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* ConnAverageRead                                                        */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_ConnAverageRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,c;
  CG_AverageInterfaceType_t at=0;

  if (!PyArg_ParseTuple(args,"iii",&b,&z,&c)) return NULL;
  
  self->last_error=cg_conn_average_read(self->db_fn,b,z,c,&at);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong((int)at);;
}

/* ---------------------------------------------------------------------- */
/* ConnAverageWrite                                                       */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_ConnAverageWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,c,at;
  
  if (!PyArg_ParseTuple(args,"iiii",&b,&z,&c,&at)) return NULL;
  
  self->last_error=cg_conn_average_write(self->db_fn,b,z,c,
                                         (CG_AverageInterfaceType_t)at);
  strcpy(self->last_error_message,cg_get_error());
  
  Py_INCREF(Py_None);
  return Py_None;
}


/* ---------------------------------------------------------------------- */
/* ConnPeriodicRead                                                       */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_ConnPeriodicRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,c,dim,i;
  float *rc,*ra,*tt;
  PyObject *tp0,*tp1,*tp2,*tp3;
  
  if (!PyArg_ParseTuple(args,"iii",&b,&z,&c)) return NULL;
  dim=getBaseDim(self,b);
  rc=(float*) malloc(sizeof(float)*dim);
  ra=(float*) malloc(sizeof(float)*dim);
  tt=(float*) malloc(sizeof(float)*dim);

  self->last_error=cg_conn_periodic_read(self->db_fn,b,z,c,rc,ra,tt);
  strcpy(self->last_error_message,cg_get_error());

  if (! self->last_error)
  {
    tp0 = PyTuple_New(3);
    tp1 = PyTuple_New(dim);
    tp2 = PyTuple_New(dim);
    tp3 = PyTuple_New(dim);
    
    for (i=0;i<dim;i++)
    {
      PyTuple_SET_ITEM(tp1, i, PyFloat_FromDouble(rc[i]));
      PyTuple_SET_ITEM(tp2, i, PyFloat_FromDouble(ra[i]));
      PyTuple_SET_ITEM(tp3, i, PyFloat_FromDouble(tt[i]));
    }

    PyTuple_SET_ITEM(tp0, 0, tp1);
    PyTuple_SET_ITEM(tp0, 1, tp2);
    PyTuple_SET_ITEM(tp0, 2, tp3);
  }
  else
  {
    Py_INCREF(Py_None);
    tp0=Py_None;
  }

  free(rc);
  free(ra);
  free(tt);
  
  return tp0;
}

/* ---------------------------------------------------------------------- */
/* ConnPeriodicWrite                                                      */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_ConnPeriodicWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,c,dim,i;
  PyObject *rc,*ra,*tt;
  float *ara,*arc,*att;
  
  if (!PyArg_ParseTuple(args,"iiiOOO",&b,&z,&c,&rc,&ra,&tt)) return NULL;

  dim=getBaseDim(self,b);
  ara=(float*) malloc(sizeof(float)*dim);
  arc=(float*) malloc(sizeof(float)*dim);
  att=(float*) malloc(sizeof(float)*dim);
  if (   PyTuple_Check(rc)
      && PyTuple_Check(ra)
      && PyTuple_Check(tt)
      &&(PyTuple_Size(rc)==dim)
      &&(PyTuple_Size(ra)==dim)
      &&(PyTuple_Size(tt)==dim))
  {
    for (i=0;i<dim;i++)
    {
      arc[i]=PyFloat_AsDouble(PyTuple_GetItem(rc,i));
      ara[i]=PyFloat_AsDouble(PyTuple_GetItem(ra,i));
      att[i]=PyFloat_AsDouble(PyTuple_GetItem(tt,i));
    }
  }
  else
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "Bad arg: should be tuples of base physical dimensions");
    free(ara);
    free(att);    
    free(arc);
    return NULL;
  }
  
  self->last_error=cg_conn_periodic_write(self->db_fn,b,z,c,arc,ara,att);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* nHoles: cg_nholes
          base index (int)
          zone index (int)
   returns holes number (int)
*/
static PyObject *
DbMIDLEVEL_nHoles(DbMIDLEVELObject *self, PyObject *args)
{
  int b, z, nh;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_nholes(self->db_fn,b,z,&nh);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nh);
}

/* ---------------------------------------------------------------------- */
/* bocoId: cg_hole_id
           base (int)
           zone (int)
           hole (int)
   returns adf id (double)
*/
static PyObject *
DbMIDLEVEL_holeId(DbMIDLEVELObject *self, PyObject *args)
{
  int    b,z,hid;
  double id;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&hid)) return NULL;
  self->last_error=cg_hole_id(self->db_fn,b,z,hid,&id);
  strcpy(self->last_error_message,cg_get_error());
  return PyFloat_FromDouble(id);
}

/* ---------------------------------------------------------------------- */
/* holeInfo: cg_hole_info
     base id (int)
     zone id (int)
     hole id (int)
   returns (name, glocation, pst, nptss, npts)
*/
static PyObject *
DbMIDLEVEL_holeInfo(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,h,npts=0,nptss=0;
  char *name;
  PyObject *tp,*xt,*xs,*xpst,*xnptss,*xnpts;
  CG_GridLocation_t gl=0;
  CG_PointSetType_t pst=0;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&h))
  {
    return NULL;
  }
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_hole_info(self->db_fn,b,z,h,name,&gl,&pst,&nptss,&npts);
  strcpy(self->last_error_message,cg_get_error());

  tp=PyTuple_New(5);

  xs=PyString_FromString(name);
  xt=PyInt_FromLong(gl);
  xpst=PyInt_FromLong(pst);
  xnptss=PyInt_FromLong(nptss);
  xnpts=PyInt_FromLong(npts);

  PyTuple_SET_ITEM(tp, 0, xs);
  PyTuple_SET_ITEM(tp, 1, xt);
  PyTuple_SET_ITEM(tp, 2, xpst);
  PyTuple_SET_ITEM(tp, 3, xnptss);
  PyTuple_SET_ITEM(tp, 4, xnpts);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* holeRead: cg_hole_read
     base id (int)
     zone id (int)
     hole id (int)
   returns point-list:array  (depends on point set type)
*/
static PyObject *
DbMIDLEVEL_holeRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,h,npts,nptss,dim;
  npy_intp tsize;
  CG_PointSetType_t pst;
  CG_GridLocation_t gl;
  int  *pts;
  char *name;
  PyArrayObject *pta;  /* array->data SHOULD be allocated, cgns COPIES */
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&h))
  {
    return NULL;
  }
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_hole_info(self->db_fn,b,z,h,name,&gl,&pst,&nptss,&npts);

  if (! strcmp(name,""))
  {
    pta=(PyArrayObject*)Py_None;
  }
  else
  {
    dim=getBaseDim(self,b);
    tsize=dim*nptss*npts;
    pts=(int*)malloc(sizeof(int)*tsize);

    self->last_error=cg_hole_read(self->db_fn,b,z,h,pts);
    strcpy(self->last_error_message,cg_get_error());

    pta=(PyArrayObject*)PyArray_New(&PyArray_Type,1,&tsize,
				    PyArray_INT, NULL, (void*)pts, 0, 
				    NPY_OWNDATA | NPY_FORTRAN, NULL);
  }

  free(name);
  
  return (PyObject*)pta;
}

/* ---------------------------------------------------------------------- */
/* holeWrite: cg_hole_write 
     base id   (int)
     zone id   (int)
     name      (string)
     glocation (int)
     psettype  (int)
     ptlist    (array of ints)
   returns hole index (int)
*/
static PyObject *
DbMIDLEVEL_holeWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,gl,pst,hid,nptsets,npnts,dim;
  char *name;
  PyObject *ptlist;
  PyArrayObject *aa;

  if (!PyArg_ParseTuple(args, "iisiiO",&b,&z,&name,&gl,&pst,&ptlist))
  {
    return NULL;
  }
  if (!PyArray_Check(ptlist))
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "Bad data: should be an array");
    return NULL;
  }
  /* this stuff is quite complicated :(

     PointSetType | Number of Point sets | Number of points in a set
     PointRange   | N                    | 2
     PointList    | 1                    | N

     Then the data array should be seen as a N*2 or a 1*N array !
     Thanks to Numeric Python, we have functions...
  */
  dim=getBaseDim(self,b);
  if (pst == CG_PointRange)
  {
    nptsets=PyArray_Size(ptlist)/dim;
    npnts=2;
  }
  else
  {
    nptsets=1;
    npnts=PyArray_Size(ptlist)/dim;
  }
  aa=(PyArrayObject*)ptlist;
  if (! (aa->descr->type_num == PyArray_INT))
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "Bad data: should be an array of ints");
    return NULL;
  }
  self->last_error=cg_hole_write(self->db_fn,b,z,name,(CG_GridLocation_t)gl,
                                 (CG_PointSetType_t)pst,
                                 nptsets,npnts,(void*)(aa->data),&hid);
  strcpy(self->last_error_message,cg_get_error());

  return PyInt_FromLong(hid);
}

/* ---------------------------------------------------------------------- */
/* nBoco: cg_nbocos
          base index (int)
          zone index (int)
   returns bocos number (int)
*/
static PyObject *
DbMIDLEVEL_nBoco(DbMIDLEVELObject *self, PyObject *args)
{
  int b, z, nbc;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_nbocos(self->db_fn,b,z,&nbc);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nbc);
}

/* ---------------------------------------------------------------------- */
/* bocoInfo: cg_boco_info
     base id (int)
     zone id (int)
     bc   id (int)
   returns (name, type, psettype, npts, normidx, datatype, normflag, ndataset)
           with normidx 
*/
static PyObject *
DbMIDLEVEL_bocoInfo(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,bc,npt=0,ndst=0,nlf=0;
  char *name;
  int  nidx[3]={0,0,0};
  PyObject *tp,*xt,*xs,*xpst,*xnpt,*xndst,*xdt,*xnlf,*xidx;
  CG_BCType_t bct=0;
  CG_PointSetType_t pst=0;
  CG_DataType_t dt=0;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&bc))
  {
    return NULL;
  }
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");

  self->last_error=cg_boco_info(self->db_fn,b,z,bc,name,
                                &bct,&pst,&npt,nidx,&nlf,&dt,&ndst);
  strcpy(self->last_error_message,cg_get_error());

  tp=PyTuple_New(8);
  xs=PyString_FromString(name);
  xt=PyInt_FromLong(bct);
  xpst=PyInt_FromLong(pst);
  xnpt=PyInt_FromLong(npt);
  xndst=PyInt_FromLong(ndst);
  xdt=PyInt_FromLong(dt);
  xnlf=PyInt_FromLong(nlf);

  xidx= Py_BuildValue("(iii)",nidx[0],nidx[1],nidx[2]);

  PyTuple_SET_ITEM(tp, 0, xs);
  PyTuple_SET_ITEM(tp, 1, xt);
  PyTuple_SET_ITEM(tp, 2, xpst);
  PyTuple_SET_ITEM(tp, 3, xnpt);
  PyTuple_SET_ITEM(tp, 4, xidx);
  PyTuple_SET_ITEM(tp, 5, xdt);
  PyTuple_SET_ITEM(tp, 6, xnlf);
  PyTuple_SET_ITEM(tp, 7, xndst);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* bocoRead: cg_boco_read
     base id (int)
     zone id (int)
     bc   id (int)
   returns ( point-list:array, normal-list:array )
*/
static PyObject *
DbMIDLEVEL_bocoRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,bc,npt=0,ndst=0,nlf=0,at=0;
  npy_intp dnl[2]={0,0};
  PyObject *tp;
  CG_BCType_t bct=0;
  CG_PointSetType_t pst=0;
  CG_DataType_t dt=0;
  void *nml;
  int  *pts;
  char *name;
  int  nidx[3]={0,0,0};
  PyArrayObject *ptn;  /* array->data SHOULD be allocated, cgns COPIES */
  PyArrayObject *pta;  /* array->data SHOULD be allocated, cgns COPIES */
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&bc))
  {
    return NULL;
  }
  nml=NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_boco_info(self->db_fn, /* file handler */
                                b,           /* base id */
                                z,           /* zone id */
                                bc,          /* bc id */
                                name,        /* = name */
                                &bct,        /* = type of BC  */
                                &pst,        /* = type of Point set  */
                                &npt,        /* = number of points */
                                nidx,        /* = normal index */
                                &nlf,        /* = physical_dim*size_of_patch */
                                &dt,         /* = data type of next array */
                                &ndst);      /* = number of datasets below */

  /* number of points is given by boco_info
     - should be 2 for PointRange
     - and any value for PointList
  */
  pts = (int*) malloc(sizeof(int)*npt*getBaseDim(self,b));
  /* the array to return for normal list has the size:
     physical_dimension*size_of_patch*sizeof(datatype)
     but bocoinfo is returning physical_dimension*size_of_patch into
     normallist flag (nlf)
  */

  /* there is a normal list */
  if (nlf)
  {
    switch(dt)
    {
      case CG_RealDouble:
        at=PyArray_DOUBLE;
        nml=(double*) malloc(sizeof(double)*nlf);
        break;
      case CG_RealSingle:
        at=PyArray_FLOAT;
        nml=(float*) malloc(sizeof(float)*nlf);
        break;
      /* should not happen, nlf is 0 when datatype is null
      case DataTypeNull:
        at = NULL;
        break;
      */
      default:
        PyErr_SetString(MIDLEVELErrorObject,"bad type for normal list vector");
        free(name);
        free(pts);
        return NULL;
    }
    self->last_error=cg_boco_read(self->db_fn,b,z,bc,pts,nml);
    strcpy(self->last_error_message,cg_get_error());
    dnl[0]=npt; // FIXME 01/2009: should be transposed numpy handles fortran
    dnl[1]=getBaseDim(self,b);
    ptn=(PyArrayObject*)PyArray_New(&PyArray_Type, 2, dnl, at,NULL, (void*)nml,
				    0, NPY_OWNDATA | NPY_FORTRAN, NULL);
    }
  /* there is NO normal list */
  else
  {
    self->last_error=cg_boco_read(self->db_fn,b,z,bc,pts,NULL);
    strcpy(self->last_error_message,cg_get_error());
    ptn=(PyArrayObject*)Py_None;
  }

  if (strcmp(name,""))
  {
    dnl[0]=npt;
    dnl[1]=getBaseDim(self,b);
    pta=(PyArrayObject*)PyArray_New(&PyArray_Type, 2, dnl,
				    PyArray_INT, NULL, (void*)pts, 0, 
				    NPY_OWNDATA | NPY_FORTRAN, NULL);
  }
  else
  {
    pta=(PyArrayObject*)Py_None;
  }

  tp = PyTuple_New(2);
  PyTuple_SET_ITEM(tp,0,(PyObject*)pta);
  PyTuple_SET_ITEM(tp,1,(PyObject*)ptn);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* bocoWrite: cg_boco_write 
     base id  (int)
     zone id  (int)
     bcname   (string)
     bctype   (int)
     psettype (int)
     ptlist   (tuple of int)
   returns bc index (int)
*/
static PyObject *
DbMIDLEVEL_bocoWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,bct,pst,ptsz,bcid,i;
  char *bcname;
  PyObject *ptlist,*ix;
  int *apts;

  bcid=0;
  if (!PyArg_ParseTuple(args, "iisiiO",&b,&z,&bcname,&bct,&pst,&ptlist))
  {
    return NULL;
  }
  if (PyList_Check(ptlist))
  {
    if (pst == CG_PointRange) 
    {
      ptsz=2;
      if ( ptsz!= PyList_Size(ptlist))
      {
        PyErr_SetString(MIDLEVELErrorObject,
                        "Bad PointRange: should be list of two points");
        return NULL;
      }
    }
    else
    {
      ptsz=PyList_Size(ptlist);
    }
  }
  else
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "Bad list: should be list of tuple (int int int)");
    return NULL;
  }
  apts=(int *)malloc(sizeof(int)*ptsz*3); /* force 3D ! */
  for (i=0; i<ptsz; i++)
  {
    ix=PyList_GetItem(ptlist,i);
    if ( !(PyTuple_Check(ix) && (PyTuple_Size(ix))) )
    {
      free(apts);
      PyErr_SetString(MIDLEVELErrorObject,
                      "Bad tuple: should be list of tuple (int int int)");
      return NULL;
    }
    apts[i*3+0]=(int)PyInt_AsLong(PyTuple_GetItem(ix,0));
    apts[i*3+1]=(int)PyInt_AsLong(PyTuple_GetItem(ix,1));
    apts[i*3+2]=(int)PyInt_AsLong(PyTuple_GetItem(ix,2));
    /* printf("[%d,%d,%d]",apts[i*3+0],apts[i*3+1],apts[i*3+2]);  */
  }
  self->last_error=cg_boco_write(self->db_fn,b,z,bcname,
                                 (CG_BCType_t)bct,(CG_PointSetType_t)pst,
                                 ptsz,apts,&bcid);
  strcpy(self->last_error_message,cg_get_error());

  free(apts);
  return PyInt_FromLong(bcid);
}

/* ---------------------------------------------------------------------- */
/* bocoId: cg_boco_id
           base (int)
           zone (int)
           bc   (int)
   returns adf id (double)
*/
static PyObject *
DbMIDLEVEL_bocoId(DbMIDLEVELObject *self, PyObject *args)
{
  int    b,z,bcid;
  double id;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&bcid)) return NULL;
  self->last_error=cg_boco_id(self->db_fn,b,z,bcid,&id);
  strcpy(self->last_error_message,cg_get_error());
  return PyFloat_FromDouble(id);
}

/* ---------------------------------------------------------------------- */
/* bocoNormalWrite: cg_boco_normal_write
     base  (int)
     zone  (int)
     bcid  (int)
     nidx  (tuple (int int int))
     flag  (int)                            
     dtype (int)
     nlist (list of (double double double)*)
   returns None
*/
static PyObject *
DbMIDLEVEL_bocoNormalWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,bc,flg,dt;
  int nix[3]={0,0,0};
  PyObject *nlist;
  PyArrayObject *array,*aa;
  
  if (!PyArg_ParseTuple(args, "iii(iii)iiO",
                        &b,&z,&bc,&nix[0],&nix[1],&nix[2],&flg,&dt,&nlist))
  {
    return NULL;
  }
  if (!PyArray_Check(nlist))
  {
    PyErr_SetString(MIDLEVELErrorObject,"Should be an array of vectors");
    return NULL;
  }
  aa=(PyArrayObject*)nlist;
  array=(PyArrayObject*)PyArray_CopyFromObject(nlist,
                                               aa->descr->type_num,
                                               aa->nd,
                                               aa->nd);

  self->last_error=cg_boco_normal_write(self->db_fn,b,z,bc,
                                        nix,
                                        flg,
                                        (CG_DataType_t)dt,
                                        (void*)(array->data));
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* bocoDatasetWrite: cg_dataset_write
      base   (int)
      zone   (int)
      bc     (int)
      dname  (string)
      dtype  (int)
   returns bcdataset index (int)
*/
static PyObject *
DbMIDLEVEL_bocoDatasetWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,bc,dt,ix;
  char *dname;
  
  if (!PyArg_ParseTuple(args, "iiisi",&b,&z,&bc,&dname,&dt))
  {
    return NULL;
  }
  self->last_error=cg_dataset_write(self->db_fn,b,z,bc,dname,(CG_BCType_t)dt,&ix);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(ix);
}

/* ---------------------------------------------------------------------- */
/* bocoDatasetRead: cg_dataset_read
      base,zone,bc,bcdataset (int)*
   returns tuple (name, type, dirflag, neuflag)
*/
static PyObject *
DbMIDLEVEL_bocoDatasetRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,bc,dst,df,nf;
  char *name;
  CG_BCType_t dt;
  PyObject *tp;
  
  if (!PyArg_ParseTuple(args, "iiii",&b,&z,&bc,&dst))
  {
    return NULL;
  }
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  df=0;
  self->last_error=cg_dataset_read(self->db_fn,b,z,bc,dst,name,&dt,&df,&nf);
  strcpy(self->last_error_message,cg_get_error());
  tp = Py_BuildValue("(siii)",name,dt,df,nf);
  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* bocoDataWrite: cg_bcdata_write
      base,zone,bc,bcdataset (int)*
      dtype (int)
   returns None
*/
static PyObject *
DbMIDLEVEL_bocoDataWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,bc,dst,dt;

  if (!PyArg_ParseTuple(args, "iiiii",&b,&z,&bc,&dst,&dt))
  {
    return NULL;
  }
  self->last_error=cg_bcdata_write(self->db_fn,b,z,bc,dst,(CG_BCDataType_t)dt);
  strcpy(self->last_error_message,cg_get_error());
  
  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* nFamilies: cg_nfamilies
     base  (int)
   returns number of families (int)
*/
static PyObject *
DbMIDLEVEL_nFamilies(DbMIDLEVELObject *self, PyObject *args)
{
  int b,nf;
  
  if (!PyArg_ParseTuple(args, "i",&b)) return NULL;
  self->last_error=cg_nfamilies(self->db_fn,b,&nf);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nf);
}

/* ---------------------------------------------------------------------- */
/* familyRead: cg_family_read
      base   (int)
      family (int)
   returns (name, nfamilybc, ngeo)
*/
static PyObject *
DbMIDLEVEL_familyRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,f,nbc,ng;
  char *name;
  PyObject *tp;
  
  if (!PyArg_ParseTuple(args, "ii",&b,&f)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_family_read(self->db_fn,b,f,name,&nbc,&ng);
  strcpy(self->last_error_message,cg_get_error());
  tp= Py_BuildValue("(sii)",name,nbc,ng);
  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* familyWrite: cg_family_write
      base    (int)
      name    (string)
   returns family id
*/
static PyObject *
DbMIDLEVEL_familyWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,ix;
  char *name;

  if (!PyArg_ParseTuple(args, "is",&b,&name)) return NULL;
  self->last_error=cg_family_write(self->db_fn,b,name,&ix);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(ix);
}

/* ---------------------------------------------------------------------- */
/* familyNameRead: cg_famname_read
   returns      name   (string)
*/
static PyObject *
DbMIDLEVEL_familyNameRead(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;
  PyObject *r;
  
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_famname_read(name);
  strcpy(self->last_error_message,cg_get_error());

  r=PyString_FromString(name);  
  free(name);
  
  return r;
}

/* ---------------------------------------------------------------------- */
/* familyNameWrite: cg_famname_write
      name (string)
   returns None
*/
static PyObject *
DbMIDLEVEL_familyNameWrite(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;

  if (!PyArg_ParseTuple(args, "s",&name)) return NULL;
  self->last_error=cg_famname_write(name);
  strcpy(self->last_error_message,cg_get_error());
  
  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* familyBocoRead: cg_fambc_read
     base    (int)
     family  (int)
     bc      (int)
   returns (name, type)
*/
static PyObject *
DbMIDLEVEL_familyBocoRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,f,bc;
  CG_BCType_t ft;
  char *name;
  PyObject *tp;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&f,&bc)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_fambc_read(self->db_fn,b,f,bc,name,&ft);
  strcpy(self->last_error_message,cg_get_error());

  tp = PyTuple_New(2);
  PyTuple_SET_ITEM(tp,0,PyString_FromString(name));
  PyTuple_SET_ITEM(tp,1,PyInt_FromLong(ft));
  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* familyBocoWrite: cg_fambc_write
     base    (int)
     family  (int)
     name    (string)
     type    (int)
   returns bc index (int)
*/
static PyObject *
DbMIDLEVEL_familyBocoWrite(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;
  int b,f,bct,ix;
  
  if (!PyArg_ParseTuple(args, "iisi",&b,&f,&name,&bct)) return NULL;
  self->last_error=cg_fambc_write(self->db_fn,b,f,name,(CG_BCType_t)bct,&ix);
  strcpy(self->last_error_message,cg_get_error());

  return PyInt_FromLong(ix);
}

/* ---------------------------------------------------------------------- */
/* geoRead: cg_geo_read
      base   (int)
      family (int)
      geo    (int)
   returns (geoname, filename, CAD system, number-of-parts)
*/
static PyObject *
DbMIDLEVEL_geoRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,f,g,np;
  char *gname,*fname,*cname;
  PyObject *tp;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&f,&g)) return NULL;
  /* is this the right size ? */
  gname=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  cname=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);

  self->last_error=cg_geo_read(self->db_fn,b,f,g,gname,&fname,cname,&np);
  strcpy(self->last_error_message,cg_get_error());
  tp= Py_BuildValue("(sssi)",gname,fname,cname,np);
  free(gname);
  free(fname);
  free(cname);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* geoWrite: cg_geo_write
      base    (int)
      family  (int)
      geoname (string)
      filename(string)
      CADname (string)
   returns geo id
*/
static PyObject *
DbMIDLEVEL_geoWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int   b,f,ix;
  char *gname,*fname,*cname;

  if (!PyArg_ParseTuple(args, "iisss",&b,&f,&gname,&fname,&cname)) return NULL;
  self->last_error=cg_geo_write(self->db_fn,b,f,gname,fname,cname,&ix);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(ix);
}

/* ---------------------------------------------------------------------- */
/* partRead: cg_part_read
      base   (int)
      family (int)
      geo    (int)
      part   (int)      
   returns part-name
*/
static PyObject *
DbMIDLEVEL_partRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,f,g,p;
  char *pname;
  PyObject *n;
  
  if (!PyArg_ParseTuple(args, "iiii",&b,&f,&g,&p)) return NULL;

  /* is this the right size ? */
  pname=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);

  self->last_error=cg_part_read(self->db_fn,b,f,g,p,pname);
  strcpy(self->last_error_message,cg_get_error());
  n=PyString_FromString(pname);

  free(pname);
  
  return n;
}

/* ---------------------------------------------------------------------- */
/* partWrite: cg_part_write
      base    (int)
      family  (int)
      geo     (int)      
      partname(string)
   returns part id
*/
static PyObject *
DbMIDLEVEL_partWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int   b,f,g,ix;
  char *pname;

  if (!PyArg_ParseTuple(args, "iiis",&b,&f,&g,&pname)) return NULL;
  self->last_error=cg_part_write(self->db_fn,b,f,g,pname,&ix);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(ix);
}

/* ---------------------------------------------------------------------- */
/* convergenceWrite: cg_convergence_write
                     number iteration (int)
                     node name (string)
   returns None
*/
static PyObject *
DbMIDLEVEL_convergenceWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int   it;
  char *name;

  if (!PyArg_ParseTuple(args, "is",&it,&name)) return NULL;
  self->last_error=cg_convergence_write(it,name);
#ifdef CGNS_TRACE
  printf("## convergence write [%d][%s]\n",it,name);
#endif  
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* convergenceRead: cg_convergence_read
   returns tuple (it number:int, node name:string)
*/
static PyObject *
DbMIDLEVEL_convergenceRead(DbMIDLEVELObject *self)
{
  int       idx;
  char     *name;
  PyObject *tp, *xn, *xi;

  idx=0;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  
  self->last_error=cg_convergence_read(&idx,&name);
  strcpy(self->last_error_message,cg_get_error());
  tp = PyTuple_New(2);
  
  xi=PyInt_FromLong(idx);
  xn=PyString_FromString(name);

  PyTuple_SET_ITEM(tp, 0, xi);
  PyTuple_SET_ITEM(tp, 1, xn);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* descriptorWrite: cg_descriptor_write
                    node name (string),
                    text (string)
   returns self
*/
static PyObject *
DbMIDLEVEL_descriptorWrite(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;
  char *txt;

  if (!PyArg_ParseTuple(args, "ss",&name,&txt)) return NULL;
  self->last_error=cg_descriptor_write(name,txt);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* descriptorRead: cg_descriptor_read
                   descriptor index (int)
   returns tuple (node name, text)
*/
static PyObject *
DbMIDLEVEL_descriptorRead(DbMIDLEVELObject *self, PyObject *args)
{
  int       idx;
  char     *name;
  char     *txt;
  PyObject *tp, *xn, *xt;

  /* txt is allocated into cgnslib.c */
  /* txt =(char*) malloc(sizeof(char)*CGNSMAXDESCSIZE); */
  
  if (!PyArg_ParseTuple(args, "i",&idx)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_descriptor_read(idx,name,&txt);
  strcpy(self->last_error_message,cg_get_error());
  tp = PyTuple_New(2);
  
  xn=PyString_FromString(name);
  xt=PyString_FromString(txt);

  PyTuple_SET_ITEM(tp, 0, xn);
  PyTuple_SET_ITEM(tp, 1, xt);

  free(txt);
  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* biterRead: cg_biter_read
     base  (int)
   returns (name, niter)
*/
static PyObject *
DbMIDLEVEL_biterRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,it;
  char *name;
  PyObject *tp,*xi,*xn;
  
  if (!PyArg_ParseTuple(args, "i",&b)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_biter_read(self->db_fn,b,name,&it);
  strcpy(self->last_error_message,cg_get_error());
  tp = PyTuple_New(2);
  
  xi=PyInt_FromLong(it);
  xn=PyString_FromString(name);

  PyTuple_SET_ITEM(tp, 0, xn);
  PyTuple_SET_ITEM(tp, 1, xi);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* biterWrite: cg_biter_write
     base  (int)
     name  (string)
     iter  (int)
   returns none
*/
static PyObject *
DbMIDLEVEL_biterWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,i;
  char *name;
  
  if (!PyArg_ParseTuple(args, "isi",&b,&name,&i)) return NULL;
  self->last_error=cg_biter_write(self->db_fn,b,name,i);
  strcpy(self->last_error_message,cg_get_error());
 
  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* ziterRead: cg_ziter_read
     base  (int)
     zone  (int)
   returns name (string)
*/
static PyObject *
DbMIDLEVEL_ziterRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z;
  char *name;
  PyObject *nn;
  
  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_ziter_read(self->db_fn,b,z,name);
  strcpy(self->last_error_message,cg_get_error());
  nn=PyString_FromString(name);
  free(name);
  
  return nn;
}

/* ---------------------------------------------------------------------- */
/* ziterWrite: cg_ziter_write
     base  (int)
     zone  (int)
     name  (string)
   returns none
*/
static PyObject *
DbMIDLEVEL_ziterWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z;
  char *name;
  
  if (!PyArg_ParseTuple(args, "iis",&b,&z,&name)) return NULL;
  self->last_error=cg_ziter_write(self->db_fn,b,z,name);
  strcpy(self->last_error_message,cg_get_error());
 
  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* linkWrite: cg_link_write
              source node name (string),
              destination file (string)
              destination node name (string)              
   returns self
*/
static PyObject *
DbMIDLEVEL_linkWrite(DbMIDLEVELObject *self, PyObject *args)
{
  char *sname;
  char *dname;
  char *dfile;

  if (!PyArg_ParseTuple(args, "sss",&sname,&dfile,&dname)) return NULL;
  self->last_error=cg_link_write(sname,dfile,dname);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* linkRead: cg_link_read
   returns tuple (elsewhere file name, elsewhere node name)
*/
static PyObject *
DbMIDLEVEL_linkRead(DbMIDLEVELObject *self, PyObject *args)
{
  char     *dname;
  char     *dfile;
  PyObject *tp, *xn, *xt;

  /* names are allocated into cgnslib.c */
  self->last_error=cg_link_read(&dfile,&dname);
  strcpy(self->last_error_message,cg_get_error());
  tp = PyTuple_New(2);
  
  xn=PyString_FromString(dfile);
  xt=PyString_FromString(dname);

  PyTuple_SET_ITEM(tp, 0, xn);
  PyTuple_SET_ITEM(tp, 1, xt);

  free(dname);  /* copies are now handled by Python (thanks) */
  free(dfile);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* isLink: cg_is_link
   returns integer (0 if not a link)
*/
static PyObject *
DbMIDLEVEL_isLink(DbMIDLEVELObject *self, PyObject *args)
{
  int       p;

  self->last_error=cg_is_link(&p);
  strcpy(self->last_error_message,cg_get_error());

  return PyInt_FromLong(p);
}

/* ---------------------------------------------------------------------- */
/* discreteWrite: cg_discrete_write
          base index (int)
          zone index (int)
          node name (string),
   returns  discrete index (int)
*/
static PyObject *
DbMIDLEVEL_discreteWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int   b,z,idx;
  char *name;

  if (!PyArg_ParseTuple(args, "iis",&b,&z,&name)) return NULL;
  self->last_error=cg_discrete_write(self->db_fn,b,z,name,&idx);
  strcpy(self->last_error_message,cg_get_error());

  return PyInt_FromLong(idx);
}

/* ---------------------------------------------------------------------- */
/* discreteRead: cg_discrete_read
          base index (int)
          zone index (int)
          discrete index (int)
   returns discrete name
*/
static PyObject *
DbMIDLEVEL_discreteRead(DbMIDLEVELObject *self, PyObject *args)
{
  int       b,z,idx;
  char     *name;
  PyObject *xn;

  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&idx)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_discrete_read(self->db_fn,b,z,idx,name);
  strcpy(self->last_error_message,cg_get_error());
  
  xn=PyString_FromString(name);
  free(name);
  
  return xn;
}

/* ---------------------------------------------------------------------- */
/* ndiscrete: cg_ndiscrete
          base index (int)
          zone index (int)
   returns discrete data total count
*/
static PyObject *
DbMIDLEVEL_nDiscrete(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,nd;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_ndiscrete(self->db_fn,b,z,&nd);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nd);
}

/* ---------------------------------------------------------------------- */
/* nRigidMotions: cg_n_rigid_motions                                      */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_nRigidMotions(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,nr;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_n_rigid_motions(self->db_fn,b,z,&nr);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nr);
}

/* ---------------------------------------------------------------------- */
/* nArbitraryMotions: cg_n_arbitrary_motions                              */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_nArbitraryMotions(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,nr=0;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_n_arbitrary_motions(self->db_fn,b,z,&nr);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nr);
}

/* ---------------------------------------------------------------------- */
/* rigidMotionRead: cg_rigid_motion_read                                  */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_rigidMotionRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,r;
  char *name;
  CG_RigidGridMotionType_t rt=0;
  PyObject *tp;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&r))
  {
    return NULL;
  }
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_rigid_motion_read(self->db_fn,b,z,r,name,&rt);
  strcpy(self->last_error_message,cg_get_error());

  tp = Py_BuildValue("(si)",name,rt);
  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* arbitraryMotionRead: cg_arbitrary_motion_read                          */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_arbitraryMotionRead(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,r;
  char *name;
  CG_ArbitraryGridMotionType_t rt=0;
  PyObject *tp;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&r))
  {
    return NULL;
  }
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_arbitrary_motion_read(self->db_fn,b,z,r,name,&rt);
  strcpy(self->last_error_message,cg_get_error());

  tp = Py_BuildValue("(si)",name,rt);
  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* rigidMotionWrite: cg_rigid_motion_write                                */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_rigidMotionWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,dt,ix;
  char *name;
  
  if (!PyArg_ParseTuple(args, "iisi",&b,&z,&name,&dt))
  {
    return NULL;
  }
  self->last_error=cg_rigid_motion_write(self->db_fn,b,z,name,
                                         (CG_RigidGridMotionType_t)dt,&ix);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(ix);
}

/* ---------------------------------------------------------------------- */
/* arbitraryMotionWrite: cg_arbitrary_motion_write                        */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_arbitraryMotionWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,dt,ix;
  char *name;
  
  if (!PyArg_ParseTuple(args, "iisi",&b,&z,&name,&dt))
  {
    return NULL;
  }
  self->last_error=cg_arbitrary_motion_write(self->db_fn,b,z,name,
                                             (CG_ArbitraryGridMotionType_t)dt,
                                             &ix);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(ix);
}

/* ---------------------------------------------------------------------- */
/* integralWrite: cg_integral_write
          node name (string),
   returns  index (int)
*/
static PyObject *
DbMIDLEVEL_integralWrite(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;

  if (!PyArg_ParseTuple(args, "s",&name)) return NULL;
  self->last_error=cg_integral_write(name);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* integralRead: cg_integral_read
          integral index (int)
   returns  (idx:int, node name:string)
*/
static PyObject *
DbMIDLEVEL_integralRead(DbMIDLEVELObject *self, PyObject *args)
{
  int       idx;
  char     *name;
  PyObject *tp, *xn, *xi;

  
  if (!PyArg_ParseTuple(args, "i",&idx)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_integral_read(idx,name);
  strcpy(self->last_error_message,cg_get_error());
  
  tp = PyTuple_New(2);

  xi=PyInt_FromLong(idx);
  xn=PyString_FromString(name);

  PyTuple_SET_ITEM(tp, 0, xi);
  PyTuple_SET_ITEM(tp, 1, xn);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* nintegrals: cg_nintegrals
   returns integral total count
*/
static PyObject *
DbMIDLEVEL_nIntegrals(DbMIDLEVELObject *self, PyObject *args)
{
  int nd;
  
  self->last_error=cg_nintegrals(&nd);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nd);
}

/* ---------------------------------------------------------------------- */
/* stateWrite: cg_state_write
          description (string),
   returns  none
*/
static PyObject *
DbMIDLEVEL_stateWrite(DbMIDLEVELObject *self, PyObject *args)
{
  char *txt;

  if (!PyArg_ParseTuple(args, "s",&txt)) return NULL;
  self->last_error=cg_state_write(txt);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* stateRead: cg_state_read
   returns  state description
*/
static PyObject *
DbMIDLEVEL_stateRead(DbMIDLEVELObject *self, PyObject *args)
{
  char     *txt;
  PyObject *xn;

  /* txt=(char*) malloc(sizeof(char)*CGNSMAXDESCSIZE); */
  
  self->last_error=cg_state_read(&txt);
  strcpy(self->last_error_message,cg_get_error());
  
  xn=PyString_FromString(txt);

  free(txt);
  
  return xn;
}

/* ---------------------------------------------------------------------- */
/* nsols: cg_nsols
          base index (int)
          zone index (int)
   returns coords total count
*/
static PyObject *
DbMIDLEVEL_nsols(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,nc=0;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_nsols(self->db_fn,b,z,&nc);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nc);
}

/* ---------------------------------------------------------------------- */
/* solid: cg_sol_id (not documented)
           base index (int)
           zone index (int)
           sol index (int)
   returns id (double)
*/
static PyObject *
DbMIDLEVEL_solId(DbMIDLEVELObject *self, PyObject *args)
{
  int    bx,zx,sx;
  double id;
  
  if (!PyArg_ParseTuple(args, "iii",&bx,&zx,&sx)) return NULL;
  self->last_error=cg_sol_id(self->db_fn,bx,zx,sx,&id);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyFloat_FromDouble(id);
}

/* ---------------------------------------------------------------------- */
/* solwrite: cg_sol_write
             base index (int)
             zone index (int)
             solution name (string)
             location (int)
   returns sol index
*/
static PyObject *
DbMIDLEVEL_solWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int            b,z,sloc,s; 
  char          *name;

  if (!PyArg_ParseTuple(args, "iisi",&b,&z,&name,&sloc))
  {
    PyErr_SetString(MIDLEVELErrorObject,"bad argument");
    return NULL;
  }
  self->last_error=cg_sol_write(self->db_fn,b,z,name,(CG_GridLocation_t)sloc,&s);

  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(s);
}

/* ---------------------------------------------------------------------- */
/* solinfo: cg_sol_info
              base index (int)
              zone index (int)
              solution index (int)
   returns tuple (name, gridlocation)
*/
static PyObject *
DbMIDLEVEL_solInfo(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,s;
  char *name;
  PyObject *tp, *xd, *xs;
  CG_GridLocation_t sloc=0;
  

  if (!PyArg_ParseTuple(args, "iii",&b,&z,&s)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_sol_info(self->db_fn,b,z,s,name,&sloc);
  strcpy(self->last_error_message,cg_get_error());

  tp=PyTuple_New(2);

  xd=PyInt_FromLong(sloc);
  xs=PyString_FromString(name);
  
  PyTuple_SET_ITEM(tp, 0, xd);
  PyTuple_SET_ITEM(tp, 1, xs);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* ncoords: cg_ncoords
            base index (int)
            zone index (int)
   returns coords total count
*/
static PyObject *
DbMIDLEVEL_ncoords(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z, nc;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_ncoords(self->db_fn,b,z,&nc);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nc);
}

/* ---------------------------------------------------------------------- */
/* coordid: cg_coord_id (not documented)
           base index (int)
           zone index (int)
           coord index (int)
   returns id (double)
*/
static PyObject *
DbMIDLEVEL_coordId(DbMIDLEVELObject *self, PyObject *args)
{
  int    bx,zx,cx;
  double id;
  
  if (!PyArg_ParseTuple(args, "iii",&bx,&zx,&cx)) return NULL;
  self->last_error=cg_coord_id(self->db_fn,bx,zx,cx,&id);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyFloat_FromDouble(id);
}

/* ---------------------------------------------------------------------- */
/* coordinfo: cg_coordinfo
              base index (int)
              zone index (int)
              coord index (int)
   returns tuple (datatype, nodename)
*/
static PyObject *
DbMIDLEVEL_coordInfo(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,c;
  char *name;
  PyObject *tp, *xd, *xs;
  CG_DataType_t dt;
  


  if (!PyArg_ParseTuple(args, "iii",&b,&z,&c)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_coord_info(self->db_fn,b,z,c,&dt,name);
  strcpy(self->last_error_message,cg_get_error());

  tp=PyTuple_New(2);

  xd=PyInt_FromLong(dt);
  xs=PyString_FromString(name);
  
  PyTuple_SET_ITEM(tp, 0, xd);
  PyTuple_SET_ITEM(tp, 1, xs);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* coordWrite: cg_coord_write
               base index (int)
               zone index (int)
               data type (int)
               node name (string)
               array (Numpy array object)
   returns coord index
*/
static PyObject *
DbMIDLEVEL_coordWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int            b,z,dt,c; 
  char          *name,*fdata;
  PyObject      *ararg;
  PyArrayObject *swparray;

  if (!PyArg_ParseTuple(args, "iiisO",&b,&z,&dt,&name,&ararg))
  {
    PyErr_SetString(MIDLEVELErrorObject,"invalid argument");
    return NULL;
  }
  if (!PyArray_Check(ararg))
  {
    PyErr_SetString(MIDLEVELErrorObject,"argument is not a valid array");
    return NULL;
  }

  /* If array is not fortran contiguous, this allocates a new data zone */
  if (! PyArray_ISFORTRAN(ararg))
  { 
    swparray=NULL;
    fdata = ((PyArrayObject*)ararg)->data; 
  }
  else
  {
    swparray=(PyArrayObject*)PyArray_CopyAndTranspose(ararg); 
    fdata = swparray->data; 
  }

  self->last_error=cg_coord_write(self->db_fn,b,z,
                                  (CG_DataType_t)dt,name,fdata,&c);

  /* Free new data zone if created by SwapAxes */
  if (swparray)
  { 
    Py_DECREF(swparray);
  }

  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(c);
}

/* ---------------------------------------------------------------------- */
/* coordRead: cg_coord_read
              base index (int)
              zone index (int)
              node name (string)
              data type (int)
              minrange (3 int tuple)
              maxrange (3 int tuple)
   returns array
*/
static PyObject *
DbMIDLEVEL_coordRead(DbMIDLEVELObject *self, PyObject *args)
{
  char *coordname;
  char  bname[CGNSMAXNAMESIZE];
  char  sname[CGNSMAXNAMESIZE];
  char  zonename[CGNSMAXNAMESIZE];
  int   zsz[9];
  int   ncelldim, nphysdim;
  int   imin[3]={1,1,1}; /* or {0,0,0} ? */
  int   imax[3];
  npy_intp   imax_[3];
  int   b,z,at,sizetot,maxnc,nc,readmode;
  CG_DataType_t dt;
  void *mzone;
#ifdef CGNS_TRACE
  int i;
#endif    
  
  PyArrayObject *array;  /* array->data SHOULD be allocated, cgns COPIES */

  if (!PyArg_ParseTuple(args, "iisi",&b,&z,&coordname,&readmode))
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "argument list is (base, zone, name, mode)");
    return NULL;
  }
  self->last_error=cg_base_read(self->db_fn, b, bname, &ncelldim, &nphysdim);
  self->last_error=cg_zone_read(self->db_fn,b,z,zonename,zsz);

  switch (ncelldim) {
  case 1:
    imax[0]=zsz[0];
    imax[1]=0;
    imax[2]=0;
    imax_[0]=zsz[0];
    imax_[1]=0;
    imax_[2]=0;
    sizetot=imax[0];
    break;
  case 2:
    imax[0]=zsz[0];
    imax[1]=zsz[1];
    imax[2]=0;
    imax_[0]=zsz[0];
    imax_[1]=zsz[1];
    imax_[2]=0;
    sizetot=imax[0]*imax[1];
    break;
  case 3:
    imax[0]=zsz[0];
    imax[1]=zsz[1];
    imax[2]=zsz[2];
    imax_[0]=zsz[0];
    imax_[1]=zsz[1];
    imax_[2]=zsz[2];
    if   (imax[2]) {sizetot=imax[0]*imax[1]*imax[2];}
    else           {sizetot=imax[0]*imax[1];}
    break;
  default:
    PyErr_SetString(MIDLEVELErrorObject,"bad ncell dim");
    return NULL;
  }

  /* do NOT de-allocate, this is shared by PyArray (not a copy in there) */
  mzone=(void*)malloc(sizeof(double)*sizetot);

  self->last_error=cg_ncoords(self->db_fn,b,z,&maxnc);
  for (nc=1;nc<maxnc;nc++)
  {
    dt=-1; /* used as a flag to detect if zone name is found */
    self->last_error=cg_coord_info(self->db_fn,b,z,nc,&dt,sname);
    if (!strcmp(sname,coordname)) 
    {
      break;
    }
  }
  if (dt == -1)
  {
    PyErr_SetString(MIDLEVELErrorObject,"coordinate name not found in zone");
    return NULL;
  }
  /*  dt=CG_RealDouble; TODO: should use coord_info there, force double */
  switch(dt)
  {
    case CG_RealDouble: at=PyArray_DOUBLE; break;
    case CG_RealSingle: at=PyArray_FLOAT;  break;
    case CG_Integer:    at=PyArray_INT;    break;
    default:
      PyErr_SetString(MIDLEVELErrorObject,"bad data type in CGNS zone");
    return NULL;
  }
  self->last_error=cg_coord_read(self->db_fn,b,z,coordname,
                                 (CG_DataType_t)dt,imin,imax,mzone);
  if (self->last_error)
  {
    strcpy(self->last_error_message,cg_get_error());
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    return NULL;
  }
  if (readmode)
  {
    imax[0]=zsz[2]; /* Force to C-like indices ?? */
    imax[1]=zsz[1];
    imax[2]=zsz[0];
  }
  array=(PyArrayObject*)PyArray_New(&PyArray_Type, 3, imax_, 
				    at, NULL, (void*)mzone, 0, 
				    NPY_OWNDATA | NPY_FORTRAN, NULL);
  /*
  array=(PyArrayObject*)PyArray_FromDimsAndData(3,imax,at,(char*)mzone);
  array->strides[0] = array->descr->elsize;
  array->strides[1] = array->strides[0] * array->dimensions[0];
  array->strides[2] = array->strides[1] * array->dimensions[1];
  array->flags |= NPY_OWNDATA;
  array->flags |= NPY_FORTRAN;
  */

  return (PyObject*)array;
}

/* ---------------------------------------------------------------------- */
/* nSections: cg_nsections                                                */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_nsections(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z;
  int ns=0;
  
  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_nsections(self->db_fn,b,z,&ns);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(ns);
}

/* ---------------------------------------------------------------------- */
/* npe: cg_npe                                                            */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_npe(DbMIDLEVELObject *self, PyObject *args)
{
  int et;
  int np=0;
  
  if (!PyArg_ParseTuple(args, "i",&et)) return NULL;
  self->last_error=cg_npe((CG_ElementType_t)et,&np);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(np);
}

/* ---------------------------------------------------------------------- */
/* npe: cg_elementDataSize                                                */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_elementDataSize(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,s;
  int es=0;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&s)) return NULL;
  self->last_error=cg_ElementDataSize(self->db_fn,b,z,s,&es);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(es);
}

/* ---------------------------------------------------------------------- */
/* sectionRead: cg_section_read                                           */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_sectionRead(DbMIDLEVELObject *self, PyObject *args)
{
  int   b,z,s;
  int   start,end,nbndry,pflag;
  char *name;
  CG_ElementType_t type;
  PyObject *xr;

  start=end=nbndry=pflag=0;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&s)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_section_read(self->db_fn,b,z,s,name,
                                   &type,&start,&end,&nbndry,&pflag);
  strcpy(self->last_error_message,cg_get_error());

  xr = Py_BuildValue("(siiiii)",name,type,start,end,nbndry,pflag);
  free(name);
  
  return xr;
}

/* ---------------------------------------------------------------------- */
/* sectionWrite: cg_section_write                                         */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_sectionWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int   b,z,s;
  int   start,end,nbndry;
  char *name;
  int   type;
  PyObject *ar;
  PyArrayObject *array;

  if (!PyArg_ParseTuple(args, "iisiiiiO",
                        &b,&z,&name,&type,&start,&end,&nbndry,&ar))
  {
    return NULL;
  }
  array=(PyArrayObject*)ar;
  self->last_error=cg_section_write(self->db_fn,b,z,
                                    name,(CG_ElementType_t)type,start,end,nbndry,
                                    (void*)(array->data),&s);
  return PyInt_FromLong(s);
}

/* ---------------------------------------------------------------------- */
/* parentDataWrite: cg_parent_data_write                                  */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_parentDataWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int   b,z,s;
  PyObject *ar;
  PyArrayObject *par;

  if (!PyArg_ParseTuple(args, "iiiO",&b,&z,&s,&ar))
  {
    return NULL;
  }
  par=(PyArrayObject*)ar;
  self->last_error=cg_parent_data_write(self->db_fn,b,z,s,(void*)(par->data));

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* elementsRead: cg_elements_read                                         */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_elementsRead(DbMIDLEVELObject *self, PyObject *args)
{
  int   b,z,s,earsize_;
  npy_intp earsize,nbelt;
  PyObject      *tp;
  PyArrayObject *ear,*par;
  int *eart, *part;
  int   start,end,nbndry,pflag;
  char *name;
  CG_ElementType_t type;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&s))
  {
    return NULL;
  }
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  self->last_error=cg_ElementDataSize(self->db_fn,b,z,s,&earsize_);
  if (self->last_error)
  {
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    free(name);
    return NULL;
  }

  earsize=earsize_; /* can you hear me with this ? */  
  self->last_error=cg_section_read(self->db_fn,b,z,s,name,
                                   &type,&start,&end,&nbndry,&pflag);
  if (self->last_error)
  {
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    free(name);
    return NULL;
  }
  
  nbelt=(end-start+1)*4; /* hum, copied from cgnslib.c */
  
  eart=(int*)  malloc(sizeof(int) *earsize);
  part=(int*)  malloc(sizeof(int) *nbelt);
  
  self->last_error=cg_elements_read(self->db_fn,b,z,s,eart,part);
  if (self->last_error)
  {
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    free(eart);
    free(part);
    free(name);
    return NULL;
  }

  ear=(PyArrayObject*)PyArray_New(&PyArray_Type, 1, &earsize,
				  PyArray_INT, NULL, (void*)eart, 0, 
				  NPY_OWNDATA | NPY_FORTRAN, NULL);
  par=(PyArrayObject*)PyArray_New(&PyArray_Type, 1, &nbelt,
				  PyArray_INT, NULL, (void*)part, 0, 
				  NPY_OWNDATA | NPY_FORTRAN, NULL);
  free(name);
  
  tp=PyTuple_New(2);
  PyTuple_SET_ITEM(tp, 0, (PyObject*)ear);
  PyTuple_SET_ITEM(tp, 1, (PyObject*)par);

  return tp;
}

/* ---------------------------------------------------------------------- */
/* gridWrite: cg_grid_write
          base index (int)
          zone index (int)
          node name (string),
   returns  grid index (int)
*/
static PyObject *
DbMIDLEVEL_gridWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int   b,z,idx;
  char *name;

  if (!PyArg_ParseTuple(args, "iis",&b,&z,&name)) return NULL;
  self->last_error=cg_grid_write(self->db_fn,b,z,name,&idx);
  strcpy(self->last_error_message,cg_get_error());

  return PyInt_FromLong(idx);
}

/* ---------------------------------------------------------------------- */
/* gridRead: cg_grid_read
          base index (int)
          zone index (int)
          grid index (int)
   returns grid name
*/
static PyObject *
DbMIDLEVEL_gridRead(DbMIDLEVELObject *self, PyObject *args)
{
  int       b,z,idx;
  char     *name;
  PyObject *xn;

  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&idx)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_grid_read(self->db_fn,b,z,idx,name);
  strcpy(self->last_error_message,cg_get_error());
  
  xn=PyString_FromString(name);
  free(name);
  
  return xn;
}

/* ---------------------------------------------------------------------- */
/* ngrids: cg_ngrids
          base index (int)
          zone index (int)
   returns grids total count
*/
static PyObject *
DbMIDLEVEL_ngrids(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,nd;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_ngrids(self->db_fn,b,z,&nd);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nd);
}

/* ---------------------------------------------------------------------- */
/* nfields: cg_nfields
            base index (int)
            zone index (int)
            solution index (int)            
   returns fields total count
*/
static PyObject *
DbMIDLEVEL_nfields(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,s,nf=0;

  if (!PyArg_ParseTuple(args, "iii",&b,&z,&s)) return NULL;
  self->last_error=cg_nfields(self->db_fn,b,z,s,&nf);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nf);
}

/* ---------------------------------------------------------------------- */
/* fieldid: cg_field_id (not documented)
           base index (int)
           zone index (int)
           solution index (int)           
           field index (int)
   returns id (double)
*/
static PyObject *
DbMIDLEVEL_fieldId(DbMIDLEVELObject *self, PyObject *args)
{
  int    bx,zx,sx,fx;
  double id;
  
  if (!PyArg_ParseTuple(args, "iiii",&bx,&zx,&sx,&fx)) return NULL;
  self->last_error=cg_field_id(self->db_fn,bx,zx,sx,fx,&id);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyFloat_FromDouble(id);
}

/* ---------------------------------------------------------------------- */
/* fieldinfo: cg_fieldinfo
              base index (int)
              zone index (int)
              solution index (int)
              field index (int)
   returns tuple (datatype, nodename)
*/
static PyObject *
DbMIDLEVEL_fieldInfo(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,s,f;
  char *name;
  PyObject *tp, *xd, *xs;
  CG_DataType_t dt=0;
  

  if (!PyArg_ParseTuple(args, "iiii",&b,&z,&s,&f)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_field_info(self->db_fn,b,z,s,f,&dt,name);
  strcpy(self->last_error_message,cg_get_error());

  tp=PyTuple_New(2);

  xd=PyInt_FromLong(dt);
  xs=PyString_FromString(name);
  
  PyTuple_SET_ITEM(tp, 0, xd);
  PyTuple_SET_ITEM(tp, 1, xs);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* fieldWrite: cg_field_write
               base index (int)
               zone index (int)
               solution index (int)
               node name (string)
               data type (int)
               array (Numpy array object)
   returns field index
*/
static PyObject *
DbMIDLEVEL_fieldWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int            b,z,s,dt,f; 
  char          *name;
  PyObject      *ararg;
  char			*fdata;
  PyArrayObject *array;

  if (!PyArg_ParseTuple(args, "iiisiO",&b,&z,&s,&name,&dt,&ararg))
  {
    PyErr_SetString(MIDLEVELErrorObject,"argument object is not a PyArray");
    return NULL;
  }

  /* If array is not fortran contiguous, this allocates a new data zone */
  array=(PyArrayObject*)ararg;
  fdata = array->data;
  /*
  if (is_fortran_contiguous(array)) fdata = array->data;
  else fdata = numpy_2_fortran(array);
  */

  self->last_error=cg_field_write(self->db_fn,b,z,s,
                                  (CG_DataType_t)dt,name,fdata,&f);

  /* Free new data zone if created by numpy_2_fortran */
  if (fdata != array->data) free(fdata);

  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(f);
}

/* ---------------------------------------------------------------------- */
/* fieldRead: cg_field_read
              base index (int)
              zone index (int)
              solution index (int)
              node name (string)
              data type (int)
              minrange (3 int tuple)
              maxrange (3 int tuple)
   returns array
*/
static PyObject *
DbMIDLEVEL_fieldRead(DbMIDLEVELObject *self, PyObject *args)
{
  char *fieldname;
  int   imin[3]={0,0,0};
  int   imax[3]={0,0,0};
  npy_intp irng[3]={0,0,0};
  /* int   drng[3]={0,0,0}; */ /* reverse indices */
  int   b,z,s,dt,at,sizetot,nd,ndim;
  void *mzone;
  PyObject *rmin,*rmax;
  
  PyArrayObject *array;  /* array->data SHOULD be allocated, cgns COPIES */

  if (!PyArg_ParseTuple(args, "iiisiOO",
                        &b,               /* base id */
                        &z,               /* zone id  */
                        &s,               /* solution id */
                        &fieldname,       /* filed name */
                        &dt,              /* data type */
                        &rmin,            /* range min (size of physical dim) */
                        &rmax))           /* range max " */
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "argument list is (base, zone, solution, name, ...)");
    return NULL;
  }
  ndim=getBaseDim(self,b);
  sizetot=1;
  for (nd=0; nd < ndim; nd++)
  {
    imin[nd]=(int)PyInt_AsLong(PyTuple_GetItem(rmin,nd));
    imax[nd]=(int)PyInt_AsLong(PyTuple_GetItem(rmax,nd));
    irng[nd]=imax[nd]-imin[nd]+1;
    sizetot*=imax[nd]-imin[nd]+1;
  }
  switch(dt)
  {
    /* do NOT de-allocate, this is shared by PyArray (not a copy in there) */
    case CG_RealDouble:
      at=PyArray_DOUBLE;
      mzone=(void*)malloc(sizeof(double)*sizetot);
     break;
    case CG_RealSingle:
      at=PyArray_FLOAT;
      mzone=(void*)malloc(sizeof(float)*sizetot);
      break;
    case CG_Integer:
      at=PyArray_INT;
      mzone=(void*)malloc(sizeof(int)*sizetot);
      break;
    default:
      PyErr_SetString(MIDLEVELErrorObject,"bad data type in CGNS field");
    return NULL;
  }
  self->last_error=cg_field_read(self->db_fn,b,z,s,fieldname,
                                 (CG_DataType_t)dt,imin,imax,mzone);
  strcpy(self->last_error_message,cg_get_error());
  if (self->last_error)
  {
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    return NULL;
  }
  /* reverse indices
  drng[0]=irng[2];
  drng[1]=irng[1];
  drng[2]=irng[0];
  */
  
  array=(PyArrayObject*)PyArray_New(&PyArray_Type, ndim, irng, 
				    at, NULL, (void*)mzone, 0, 
				    NPY_OWNDATA | NPY_FORTRAN, NULL);

  /*
  array=(PyArrayObject*)PyArray_FromDimsAndData(ndim,irng,at,(char*)mzone);
  array->strides[0] = array->descr->elsize;
  array->strides[1] = array->strides[0] * array->dimensions[0];
  array->strides[2] = array->strides[1] * array->dimensions[1];
  array->flags |= NPY_OWNDATA;
  array->flags |= NPY_FORTRAN;
  */
  
  return (PyObject*)array;
}

/* ---------------------------------------------------------------------- */
/* OnrToOneId: cg_1to1_id (not documented)
           base (int)
           zone (int)
           1to1 (int)
   returns id (double)
*/
static PyObject *
DbMIDLEVEL_1to1Id(DbMIDLEVELObject *self, PyObject *args)
{
  int    b,z,idx;
  double id;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&idx)) return NULL;
  self->last_error=cg_1to1_id(self->db_fn,b,z,idx,&id);
  return PyFloat_FromDouble(id);
}

/* ---------------------------------------------------------------------- */
/* OneToOneWrite: cg_1to1_write
          base index (int)
          zone index (int)
          name (string)
          donor (string)
          window range (tuple 6*int)
          window donor range (tuple 6*int)
          transform (tuple 3*int)
   returns  1to1 index (int)
*/
static PyObject *
DbMIDLEVEL_1to1Write(DbMIDLEVELObject *self, PyObject *args)
{
  char     *name, *donor;
  int b,z,xwr[6],xwdr[6],xtsf[3],idx;
  
  if (!PyArg_ParseTuple(args, "iiss(iiiiii)(iiiiii)(iii)",\
                        &b,&z,&name,&donor,
                        &xwr[0],&xwr[1],&xwr[2],&xwr[3],&xwr[4],&xwr[5],
                        &xwdr[0],&xwdr[1],&xwdr[2],&xwdr[3],&xwdr[4],&xwdr[5],
                        &xtsf[0],&xtsf[1],&xtsf[2]))
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "argument list is (base, zone, name, donor, window range, window donor range, transform)");
    return NULL;
  }
  self->last_error=cg_1to1_write(self->db_fn,b,z,name,donor,xwr,xwdr,xtsf,&idx);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(idx);
}

/* ---------------------------------------------------------------------- */
/* OneToOneRead: cg_1to1_read
          base index (int)
          zone index (int)
          1to1 index (int)
   returns tuple (name:string, donor:string, window:range,
                  window:donor-range, tuple:[int,int,int] transform)
   a window tuple is [imin:int,jmin:int,kmin:int,imax:int,jmax:int,kmax:int]
*/
static PyObject *
DbMIDLEVEL_1to1Read(DbMIDLEVELObject *self, PyObject *args)
{
  int       b,z,idx;
  char     *name, *donor;
  int       wr[6] ={0,0,0,0,0,0};
  int       wdr[6]={0,0,0,0,0,0};
  int       tsf[3]={0,0,0};
  PyObject *xn,*xd,*tp,*tp_wr,*tp_wdr,*tp_tsf;

  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&idx)) return NULL;
  name= (char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  donor=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  strcpy(donor,"");
  self->last_error=cg_1to1_read(self->db_fn,b,z,idx,name,donor,wr,wdr,tsf);
  strcpy(self->last_error_message,cg_get_error());
  
  tp     = PyTuple_New(5); /* complete tuple */
  tp_wr  = Py_BuildValue("(iiiiii)",wr[0],wr[1],wr[2],wr[3],wr[4],wr[5]);
  tp_wdr = Py_BuildValue("(iiiiii)",wdr[0],wdr[1],wdr[2],wdr[3],wdr[4],wdr[5]);
  tp_tsf = Py_BuildValue("(iii)",tsf[0],tsf[1],tsf[2]);

  xn=PyString_FromString(name);
  xd=PyString_FromString(donor);
  free(name);
  free(donor);
  
  PyTuple_SET_ITEM(tp, 0, xn);
  PyTuple_SET_ITEM(tp, 1, xd);
  PyTuple_SET_ITEM(tp, 2, tp_wr);
  PyTuple_SET_ITEM(tp, 3, tp_wdr);
  PyTuple_SET_ITEM(tp, 4, tp_tsf);

  return tp;
}

/* ---------------------------------------------------------------------- */
/* OneToOneReadGlobal: cg_1to1_read_global
          base index (int)
   returns list of tuple (name:string, zone:string,donor:string, window:range,
                  window:donor-range, tuple:[int,int,int] transform)
   a window tuple is [imin:int,jmin:int,kmin:int,imax:int,jmax:int,kmax:int]
*/
static PyObject *
DbMIDLEVEL_1to1ReadGlobal(DbMIDLEVELObject *self, PyObject *args)
{
  int       b,cxnum,cx;
  char    **gname,**gz,**gzd;
  int     **gw,**gwd,**gt;

  PyObject *xlst,*xcn,*xcd,*xcz,*tp,*tp_wr,*tp_wdr,*tp_tsf;

  if (!PyArg_ParseTuple(args, "i",&b)) return NULL;
  self->last_error=cg_n1to1_global(self->db_fn,b,&cxnum);
  strcpy(self->last_error_message,cg_get_error());

  if (cxnum && self->last_error)
  {
    /* allocations -> get the number of connectivities and allocate for all */
    gname = (char**) malloc(sizeof(char)*CGNSMAXNAMESIZE*cxnum);
    gz    = (char**) malloc(sizeof(char)*CGNSMAXNAMESIZE*cxnum);
    gzd   = (char**) malloc(sizeof(char)*CGNSMAXNAMESIZE*cxnum);
    gw    = (int**)  malloc(sizeof(int) *CGNSMAXNAMESIZE*cxnum);
    gwd   = (int**)  malloc(sizeof(int) *CGNSMAXNAMESIZE*cxnum);
    gt    = (int**)  malloc(sizeof(int) *CGNSMAXNAMESIZE*cxnum);
    
    self->last_error=cg_1to1_read_global(self->db_fn,b,gname,gz,gzd,gw,gwd,gt);
    strcpy(self->last_error_message,cg_get_error());
    
    xlst = PyList_New(cxnum); /* complete result list */
    for (cx=0;cx<cxnum;cx++)
    {
      tp    =PyTuple_New(6); /* complete tuple */
      tp_wr =Py_BuildValue("(iiiiii)",
                           gw[cx][0],gw[cx][1],
                           gw[cx][2],gw[cx][3],
                           gw[cx][4],gw[cx][5]);
      tp_wdr=Py_BuildValue("(iiiiii)",
                           gwd[cx][0],gwd[cx][1],
                           gwd[cx][2],gwd[cx][3],
                           gwd[cx][4],gwd[cx][5]);
      tp_tsf=Py_BuildValue("(iii)",
                           gt[cx][0],gt[cx][1],gt[cx][2]);
      
      xcn=PyString_FromString(gname[cx]);
      xcz=PyString_FromString(gz[cx]);
      xcd=PyString_FromString(gzd[cx]);
      
      PyTuple_SET_ITEM(tp, 0, xcn);    /* conn. name   */
      PyTuple_SET_ITEM(tp, 1, xcz);    /* zone  name   */
      PyTuple_SET_ITEM(tp, 2, xcd);    /* donor name   */
      PyTuple_SET_ITEM(tp, 2, tp_wr);  /* window       */
      PyTuple_SET_ITEM(tp, 3, tp_wdr); /* donor window */
      PyTuple_SET_ITEM(tp, 4, tp_tsf); /* transform    */
      
      PyList_SET_ITEM(xlst, cx, tp);
    }
    
    free(gw);
    free(gwd);
    free(gt);
    free(gzd);
    free(gz);
    free(gname);
  }
  else
  {
    xlst = PyList_New(0);
  }
  
  return xlst;
}

/* ---------------------------------------------------------------------- */
/* n1to1: cg_n1to1
          base index (int)
          zone index (int)
   returns connectivity 1 to 1 nodes count
*/
static PyObject *
DbMIDLEVEL_n1to1(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,nd;

  if (!PyArg_ParseTuple(args, "ii",&b,&z)) return NULL;
  self->last_error=cg_n1to1(self->db_fn,b,z,&nd);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nd);
}

/* ---------------------------------------------------------------------- */
/* n1to1Global: cg_n1to1_global
          base index (int)
   returns connectivity 1 to 1 nodes count for whole base
*/
static PyObject *
DbMIDLEVEL_n1to1Global(DbMIDLEVELObject *self, PyObject *args)
{
  int b,nd;

  if (!PyArg_ParseTuple(args, "i",&b)) return NULL;
  self->last_error=cg_n1to1_global(self->db_fn,b,&nd);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(nd);
}

/* ---------------------------------------------------------------------- */
/* bcWallFunctionWrite: cg_bc_wallfunction_write                          */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_bcWallFunctionWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int b,z,bc,t;
  
  if (!PyArg_ParseTuple(args, "iiii",&b,&z,&bc,&t)) return NULL;
  self->last_error=cg_bc_wallfunction_write(self->db_fn,b,z,bc,(CG_WallFunctionType_t)t);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* bcWallFunctionRead: cg_bc_wallfunction_read                            */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_bcWallFunctionRead(DbMIDLEVELObject *self, PyObject *args)
{
  CG_WallFunctionType_t t=0;
  int b,z,bc;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&bc)) return NULL;
  self->last_error=cg_bc_wallfunction_read(self->db_fn,b,z,bc,&t);
  strcpy(self->last_error_message,cg_get_error());

  return PyInt_FromLong(t);
}

/* ---------------------------------------------------------------------- */
/* bcAreaWrite: cg_bc_area_write                                          */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_bcAreaWrite(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;
  double s;
  int b,z,a,t;
  
  if (!PyArg_ParseTuple(args, "iiiifs",&b,&z,&a,&t,&s,&name)) return NULL;
  self->last_error=cg_bc_area_write(self->db_fn,b,z,a,(CG_AreaType_t)t,s,name);
  strcpy(self->last_error_message,cg_get_error());
  
  Py_INCREF(Py_None);
  return Py_None;
}


/* ---------------------------------------------------------------------- */
/* bcAreaRead: cg_bc_area_read                                            */
/* ---------------------------------------------------------------------- */
static PyObject *
DbMIDLEVEL_bcAreaRead(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;
  float s;
  int b,z,a;
  CG_AreaType_t t;
  PyObject *xr;
  
  if (!PyArg_ParseTuple(args, "iii",&b,&z,&a)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_bc_area_read(self->db_fn,b,z,a,&t,&s,name);
  strcpy(self->last_error_message,cg_get_error());

  xr = Py_BuildValue("(sif)",name,t,s);
  free(name);
  
  return xr;
}


/* ---------------------------------------------------------------------- */
/* rindWrite: cg_rind_write
   returns tuple 3*int
*/
static PyObject *
DbMIDLEVEL_rindWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int r[6]={0,0,0,0,0,0};
  
  if (!PyArg_ParseTuple(args, "(iiiiii)",&r[0],&r[1],&r[2],&r[3],&r[4],&r[5]))
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "Arg is a tuple of 6 ints, even if dimension is not 3D.");
    return NULL;
  }
  self->last_error=cg_rind_write(r);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* rindRead: cg_rind_read
   returns tuple 3*int
*/
static PyObject *
DbMIDLEVEL_rindRead(DbMIDLEVELObject *self, PyObject *args)
{
  int r[6]={0,0,0,0,0,0};
  int dim;
  PyObject *xr;
  
  self->last_error=cg_rind_read(r);
  strcpy(self->last_error_message,cg_get_error());

  dim=getBaseDim(self,0);
  xr=NULL;

  if (dim == 1)
  {
    xr = Py_BuildValue("(ii)",r[0],r[1]);
  }
  if (dim == 2)
  {
    xr = Py_BuildValue("(iiii)",r[0],r[1],r[2],r[3]);
  }
  if (dim == 3)
  {
    xr = Py_BuildValue("(iiiiii)",r[0],r[1],r[2],r[3],r[4],r[5]);
  }
  
  return xr;
}

/* ---------------------------------------------------------------------- */
/* dataclassWrite: cg_dataclass_write
   returns tuple 3*int
*/
static PyObject *
DbMIDLEVEL_dataclassWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int r;
  
  if (!PyArg_ParseTuple(args, "i",&r)) return NULL;
  self->last_error=cg_dataclass_write((CG_DataClass_t)r);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* dataclassRead: cg_dataclass_read
   returns tuple 3*int
*/
static PyObject *
DbMIDLEVEL_dataclassRead(DbMIDLEVELObject *self, PyObject *args)
{
  CG_DataClass_t r;
  
  self->last_error=cg_dataclass_read(&r);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(r);
}

/* ---------------------------------------------------------------------- */
/* ordinalRead: cg_ordinal_read
   returns int
*/
static PyObject *
DbMIDLEVEL_ordinalRead(DbMIDLEVELObject *self, PyObject *args)
{
  int r;
  
  self->last_error=cg_ordinal_read(&r);
  strcpy(self->last_error_message,cg_get_error());
  
  return PyInt_FromLong(r);
}

/* ---------------------------------------------------------------------- */
/* ordinalWrite: cg_ordinal_write
      ordinal value (int)
   returns none
*/
static PyObject *
DbMIDLEVEL_ordinalWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int t;
  
  if (!PyArg_ParseTuple(args, "i",&t)) return NULL;
  self->last_error=cg_ordinal_write(t);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* equationsetWrite: cg_equationset_write
      equation dimension (int)
   returns none
*/
static PyObject *
DbMIDLEVEL_equationsetWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int t; /* equation dimension */
  
  if (!PyArg_ParseTuple(args, "i",&t)) return NULL;
  self->last_error=cg_equationset_write(t);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* equationsetRead: cg_equationset_read
   returns tuple 5*int
*/
static PyObject *
DbMIDLEVEL_equationsetRead(DbMIDLEVELObject *self, PyObject *args)
{
  int a,b,c,d,e,f,g;
  PyObject *xr;

  a=b=c=d=e=f=g=0;
  
  self->last_error=cg_equationset_read(&a,&b,&c,&d,&e,&f,&g);
  strcpy(self->last_error_message,cg_get_error());
  
  xr = Py_BuildValue("(iiiiiii)",a,b,c,d,e,f,g);
  return xr;
}

/* ---------------------------------------------------------------------- */
/* equationsetChemistryRead: cg_equationset_chemistry_read
   returns tuple 2*int
*/
static PyObject *
DbMIDLEVEL_equationsetChemistryRead(DbMIDLEVELObject *self, PyObject *args)
{
  int a,b;
  PyObject *xr;

  a=b=0;
  
  self->last_error=cg_equationset_chemistry_read(&a,&b);
  strcpy(self->last_error_message,cg_get_error());
  
  xr = Py_BuildValue("(ii)",a,b);
  return xr;
}

/* ---------------------------------------------------------------------- */
/* equationsetElecetromagneticRead: cg_equationset_electromagn_read
   returns tuple 3*int
*/
static PyObject *
DbMIDLEVEL_equationsetElecRead(DbMIDLEVELObject *self, PyObject *args)
{
  int a,b,c;
  PyObject *xr;

  a=b=c=0;
  
  self->last_error=cg_equationset_elecmagn_read(&a,&b,&c);
  strcpy(self->last_error_message,cg_get_error());
  
  xr = Py_BuildValue("(iii)",a,b,c);
  return xr;
}

/* ---------------------------------------------------------------------- */
/* exponentsWrite: cg_exponents_write
     dataclass:         CG_RealSingle or CG_RealDouble
     exponents:         float/double depending on dataclass *5
   returns nothing
*/
static PyObject *
DbMIDLEVEL_exponentsWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int dt;
  PyObject *ocs;
  double ds[5]={0.0,0.0,0.0,0.0,0.0};
  float  fs[5]={0.0,0.0,0.0,0.0,0.0};
  
  if (!PyArg_ParseTuple(args, "iO",&dt,&ocs)) return NULL;
  if (!PyTuple_Check(ocs))
  {
    PyErr_SetString(MIDLEVELErrorObject,"Bad 2nd arg item: should be tuple");
    return NULL;
  }
  if (PyTuple_Size(ocs) != 5)
  {
    PyErr_SetString(MIDLEVELErrorObject,"Bad 2nd arg size: should be 5");
    return NULL;
  }
  if (dt == CG_RealDouble)
  {
    if (!PyArg_ParseTuple(args, "i(ddddd)",
                          &dt,&ds[0],&ds[1],&ds[2],&ds[3],&ds[4])) return NULL;
    
    self->last_error=cg_exponents_write((CG_DataType_t)dt,(void*)ds);
    strcpy(self->last_error_message,cg_get_error());
  }
  else if (dt == CG_RealSingle)
  {
    if (!PyArg_ParseTuple(args, "i(fffff)",
                          &dt,&fs[0],&fs[1],&fs[2],&fs[3],&fs[4])) return NULL;
    self->last_error=cg_exponents_write((CG_DataType_t)dt,(void*)&fs);
    strcpy(self->last_error_message,cg_get_error());
  }
  else
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "First arg should be CG_RealDouble or CG_RealSingle");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* exponentsInfo: cg_exponents_info
   returns int (DataType_t)
*/
static PyObject *
DbMIDLEVEL_exponentsInfo(DbMIDLEVELObject *self, PyObject *args)
{
  PyObject *xr;
  CG_DataType_t dt;

  self->last_error=cg_exponents_info(&dt);
  strcpy(self->last_error_message,cg_get_error());

  xr=PyInt_FromLong(dt);
  return xr;
}

/* ---------------------------------------------------------------------- */
/* exponentsRead: cg_exponents_read
   returns (double, double, double, double, double)
*/
static PyObject *
DbMIDLEVEL_exponentsRead(DbMIDLEVELObject *self, PyObject *args)
{
  PyObject *xr;
  double cd[5]={0.0,0.0,0.0,0.0,0.0};
  float  cf[5]={0.0,0.0,0.0,0.0,0.0};
  CG_DataType_t dt;
  
  self->last_error=cg_exponents_info(&dt);
  strcpy(self->last_error_message,cg_get_error());
  if (dt == CG_RealDouble)
  {
    self->last_error=cg_exponents_read(cd);
    strcpy(self->last_error_message,cg_get_error());
    xr = Py_BuildValue("(ddddd)",cd[0],cd[1],cd[2],cd[3],cd[4]);
  }
  else
  {
    self->last_error=cg_exponents_read(cf);
    strcpy(self->last_error_message,cg_get_error());
    xr = Py_BuildValue("(fffff)",cf[0],cf[1],cf[2],cf[3],cf[4]);
  }
  
  return xr;
}

/* ---------------------------------------------------------------------- */
/* conversionWrite: cg_conversion_write
     dataclass:         CG_RealSingle or CG_RealDouble
     conversionscale:   float/double depending on dataclass
     conversionoffset:  "
   returns nothing
*/
static PyObject *
DbMIDLEVEL_conversionWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int dt;
  PyObject *ocs;
  double ds[2]={0.0,0.0};
  float  fs[2]={0.0,0.0};
  
  if (!PyArg_ParseTuple(args, "iO",&dt,&ocs)) return NULL;
  if (dt == CG_RealDouble)
  {
    if (!PyArg_ParseTuple(args, "i(dd)",&dt,&ds[0],&ds[1])) return NULL;
    self->last_error=cg_conversion_write((CG_DataType_t)dt,(void*)ds);
    strcpy(self->last_error_message,cg_get_error());
  }
  else if (dt == CG_RealSingle)
  {
    if (!PyArg_ParseTuple(args, "i(ff)",&dt,&fs[0],&fs[1])) return NULL;
    self->last_error=cg_conversion_write((CG_DataType_t)dt,(void*)&fs);
    strcpy(self->last_error_message,cg_get_error());
  }
  else
  {
    PyErr_SetString(MIDLEVELErrorObject,
                    "First arg should be CG_RealDouble or CG_RealSingle");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* conversionInfo: cg_conversion_info
   returns int (DataType_t)
*/
static PyObject *
DbMIDLEVEL_conversionInfo(DbMIDLEVELObject *self, PyObject *args)
{
  PyObject *xr;
  CG_DataType_t dt;

  self->last_error=cg_conversion_info(&dt);
  strcpy(self->last_error_message,cg_get_error());

  xr=PyInt_FromLong(dt);
  return xr;
}

/* ---------------------------------------------------------------------- */
/* conversionRead: cg_conversion_read
   returns (double, double)
*/
static PyObject *
DbMIDLEVEL_conversionRead(DbMIDLEVELObject *self, PyObject *args)
{
  PyObject *xr;
  double cd[2]={0.0,0.0};
  float  cf[2]={0.0,0.0};
  CG_DataType_t dt;
  
  self->last_error=cg_conversion_info(&dt);
  strcpy(self->last_error_message,cg_get_error());
  if (dt == CG_RealDouble)
  {
    self->last_error=cg_conversion_read(cd);
    strcpy(self->last_error_message,cg_get_error());
    xr = Py_BuildValue("(dd)",cd[0],cd[1]);
  }
  else
  {
    self->last_error=cg_conversion_read(cf);
    strcpy(self->last_error_message,cg_get_error());
    xr = Py_BuildValue("(ff)",cf[0],cf[1]);
  }
  
  return xr;
}

/* ---------------------------------------------------------------------- */
/* UnitsWrite: cg_units_write
   returns tuple 3*int
*/
static PyObject *
DbMIDLEVEL_unitsWrite(DbMIDLEVELObject *self, PyObject *args)
{
  int m,l,t,k,a;
  
  if (!PyArg_ParseTuple(args, "iiiii",&m,&l,&t,&k,&a)) return NULL;
  self->last_error=cg_units_write((CG_MassUnits_t)m,
                                  (CG_LengthUnits_t)l,
                                  (CG_TimeUnits_t)t,
                                  (CG_TemperatureUnits_t)k,
                                  (CG_AngleUnits_t)a);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* unitsRead: cg_units_read
   returns tuple 5*int
*/
static PyObject *
DbMIDLEVEL_unitsRead(DbMIDLEVELObject *self, PyObject *args)
{
  PyObject *xr;
  CG_TimeUnits_t t;
  CG_LengthUnits_t l;
  CG_MassUnits_t m;
  CG_TemperatureUnits_t k;
  CG_AngleUnits_t a;
  
  self->last_error=cg_units_read(&m,&l,&t,&k,&a);
  strcpy(self->last_error_message,cg_get_error());
  
  xr = Py_BuildValue("(iiiii)",m,l,t,k,a);
  return xr;
}

/* ---------------------------------------------------------------------- */
/* nuserdata: cg_nuser_data
   returns number of CG_UserDefinedData_t children under the current node
*/
static PyObject *
DbMIDLEVEL_nuserdata(DbMIDLEVELObject *self, PyObject *args)
{
  int n=0;

  self->last_error=cg_nuser_data(&n);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(n);
}

/* ---------------------------------------------------------------------- */
/* userdataRead: cg_user_data_read
                 user data index (int)
   returns (idx:int. node name:string)
*/
static PyObject *
DbMIDLEVEL_userdataRead(DbMIDLEVELObject *self, PyObject *args)
{
  int       idx;
  char     *name;
  PyObject *tp, *xn, *xi;

  
  if (!PyArg_ParseTuple(args, "i",&idx)) return NULL;
  name=(char*) malloc(sizeof(char)*CGNSMAXNAMESIZE);
  strcpy(name,"");
  self->last_error=cg_user_data_read(idx,name);
  strcpy(self->last_error_message,cg_get_error());
  
  tp = PyTuple_New(2);

  xi=PyInt_FromLong(idx);
  xn=PyString_FromString(name);

  PyTuple_SET_ITEM(tp, 0, xi);
  PyTuple_SET_ITEM(tp, 1, xn);

  free(name);
  
  return tp;
}

/* ---------------------------------------------------------------------- */
/* userdataWrite: cg_user_data_write
                  node name (string)
   returns index (int)
*/
static PyObject *
DbMIDLEVEL_userdataWrite(DbMIDLEVELObject *self, PyObject *args)
{
  char *name;

  if (!PyArg_ParseTuple(args, "s",&name)) return NULL;
  self->last_error=cg_user_data_write(name);
  strcpy(self->last_error_message,cg_get_error());

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* narrays: cg_narrays
   returns number of CG_DataArray_t children under the current node
*/
static PyObject *
DbMIDLEVEL_narrays(DbMIDLEVELObject *self, PyObject *args)
{
  int n=0;

  self->last_error=cg_narrays(&n);
  strcpy(self->last_error_message,cg_get_error());
  return PyInt_FromLong(n);
}

/* ---------------------------------------------------------------------- */
/* arrayRead: cg_array_read
              child id (int)
   returns array of data
*/
static PyObject *
DbMIDLEVEL_arrayRead(DbMIDLEVELObject *self, PyObject *args)
{
  PyArrayObject *array;  /* array->data SHOULD be allocated, cgns COPIES */
  void *mzone;
  int   c,dd,sizetot,at;
  CG_DataType_t dt;
  npy_intp   dv[3]={0,0,0};
  int  dv_[3]={0,0,0}; 
  char  name[CGNSMAXNAMESIZE];
  
  if (!PyArg_ParseTuple(args, "i",&c))
  {
    PyErr_SetString(MIDLEVELErrorObject,"argument is (child-id integer)");
    return NULL;
  }

  self->last_error=cg_array_info(c,name,&dt,&dd,dv_);
  if (self->last_error)
  {
    strcpy(self->last_error_message,cg_get_error());
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    return NULL;
  }
  
  switch (dd) {
  case (1):
    dv[0]=dv_[0];
    sizetot=dv[0];
    break;
  case (2):
    dv[0]=dv_[0];
    dv[1]=dv_[1];
    if   (dv[1]) {sizetot=dv[0]*dv[1];}  
    else         {sizetot=dv[0];}
    break;
  case (3):
  default:
    dv[0]=dv_[0];
    dv[1]=dv_[1];
    dv[2]=dv_[2];
    if   (dv[2]) {sizetot=dv[0]*dv[1]*dv[2];}
    if   (dv[1]) {sizetot=dv[0]*dv[1]*dv[2];}  
    else         {sizetot=dv[0];}
  }
  /* do NOT de-allocate, this is shared by PyArray (not a copy in there) */
  switch(dt)
  {
    case CG_RealDouble:
      at=PyArray_DOUBLE;
      mzone=(void*)malloc(sizeof(double)*sizetot);
      break;
    case CG_RealSingle:
      at=PyArray_FLOAT;
      mzone=(void*)malloc(sizeof(float)*sizetot);
      break;
    case CG_Integer:
      at=PyArray_INT;
      mzone=(void*)malloc(sizeof(int)*sizetot);
      break;
    case CG_Character:
      at=PyArray_UBYTE;
      mzone=(void*)malloc(sizeof(char)*sizetot);
      break;      
    default:
      PyErr_SetString(MIDLEVELErrorObject,"bad element type in DataArray");
    return NULL;
  }
  
#ifdef CGNS_TRACE
  printf("## array read [%s] type[%d] dim[%d](%dx%dx%d)=%d\n",
         name,(CG_DataType_t)dt,dd,dv[0],dv[1],dv[2],sizetot);fflush(stdout);
  printf("## last error: %s\n",cg_get_error());
#endif  
  self->last_error=cg_array_read(c,mzone);
  strcpy(self->last_error_message,cg_get_error());
  
  array=(PyArrayObject*)PyArray_New(&PyArray_Type, dd, dv, 
				    at, NULL, (void*)mzone, 0, 
				    NPY_OWNDATA | NPY_FORTRAN, NULL);

  return (PyObject*)array;
}

/* ---------------------------------------------------------------------- */
/* arrayWrite: cg_array_write
               name (string)
               datatype (int)
               datadim (int)
               datavector (2 or 3 tuple)
               array (Numpy array object)
   returns none
*/
static PyObject *
DbMIDLEVEL_arrayWrite(DbMIDLEVELObject *self, PyObject *args)
{
  char          *name;
  int            dd,dt,sizetot;
  int            ddim[3]={0,0,0};
  PyObject      *dv,*ar;
  PyArrayObject *array;
  
  if (!PyArg_ParseTuple(args, "siiOO",&name,&dt,&dd,&dv,&ar))
  {
    PyErr_SetString(MIDLEVELErrorObject,"bad argument list");
    return NULL;
  }
  if (!PyTuple_Check(dv))
  {
    PyErr_SetString(MIDLEVELErrorObject,"Bad 4th arg: should be tuple");
    return NULL;
  }
  if (!PyArray_Check(ar))
  {
    PyErr_SetString(MIDLEVELErrorObject,"Bad 5th arg: should be numarray array");
    return NULL;
  }
  ddim[0]=PyInt_AsLong(PyTuple_GetItem(dv,0));
  ddim[1]=0;
  ddim[2]=0;
  sizetot=ddim[0];
  if (dd > 1)
  {
    ddim[1]=PyInt_AsLong(PyTuple_GetItem(dv,1));
    sizetot*=ddim[1];
  }
  if (dd > 2)
  {
    ddim[2]=PyInt_AsLong(PyTuple_GetItem(dv,2));
    sizetot*=ddim[2];
  }

#ifdef CGNS_TRACE
  printf("## array write [%s] type[%d] dim[%d](%dx%dx%d)=%d\n",
         name,(CG_DataType_t)dt,dd,ddim[0],ddim[1],ddim[2],sizetot);
  fflush(stdout);
  printf("## last error: %s\n",cg_get_error());
#endif  

  array=(PyArrayObject*)ar;
  self->last_error=cg_array_write(name,(CG_DataType_t)dt,dd,ddim
                                  ,(void*)(array->data));

  if (self->last_error)
  {
    strcpy(self->last_error_message,cg_get_error());
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

/* ---------------------------------------------------------------------- */
/* arrayinfo: cg_arrayinfo
              child id (int)
   returns tuple (name, data-type,dada-dim, data-dim-vector)
*/
static PyObject *
DbMIDLEVEL_arrayInfo(DbMIDLEVELObject *self, PyObject *args)
{
  int   c,dd;
  CG_DataType_t  dt;
  int   dv[3]={0,0,0};
  char  name[CGNSMAXNAMESIZE];
  PyObject *tpl;
  
  if (!PyArg_ParseTuple(args, "i",&c))
  {
    PyErr_SetString(MIDLEVELErrorObject,"argument is (child-id integer)");
    return NULL;
  }

  self->last_error=cg_array_info(c,name,&dt,&dd,dv);
  if (self->last_error)
  {
    strcpy(self->last_error_message,cg_get_error());
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    return NULL;
  }

  if (dd == 3)
  {
    tpl = Py_BuildValue("(sii(iii))",name,dt,dd,dv[0],dv[1],dv[2]);
  }
  else if (dd == 2)
  {
    tpl = Py_BuildValue("(sii(ii))",name,dt,dd,dv[0],dv[1]);
  }
  else 
  {
    tpl = Py_BuildValue("(sii(i))",name,dt,dd,dv[0]);
  }

  return tpl;
}

/* ---------------------------------------------------------------------- */
/* mapping between Python class and mid-level calls
   a leading * means implemented
   a leading ~ means to be implemented
   lines with a # means I won't implement it, I think this doesn't fit
   with the Python interface I'd like to define. However, any demand about
   an implementation of a # function will be taken into account and actually
   done

   ***** THE NAMES YOU CAN READ HERE MAY BE OVERRIDEN BY OTHER NAMES IN
         THE PYTHON LAYER.

*/
static PyMethodDef DbMIDLEVEL_methods[] = {
/*
 * cg_open    (class constructor)
 * cg_close
 * cg_version (attribute)
 * cg_root_id (attribute)
 * cg_configure (MOVED TO MODULE LEVEL)
 * cg_is_cgns (MOVED TO MODULE LEVEL)
 */
{"close",	        (PyCFunction)DbMIDLEVEL_close,	     METH_VARARGS},

/*
 * cg_equationset_read
 * cg_equationset_write
 * cg_equationset_chemistry_read
 * cg_governing_read
 * cg_governing_write
 * cg_diffusion_read
 * cg_diffusion_write
 * cg_model_read
 * cg_model_write
 */
{"equationsetRead", (PyCFunction)DbMIDLEVEL_equationsetRead,  METH_VARARGS},
{"equationsetChemistryRead",
           (PyCFunction)DbMIDLEVEL_equationsetChemistryRead,  METH_VARARGS},
{"equationsetElectromagneticRead",
           (PyCFunction)DbMIDLEVEL_equationsetElecRead,  METH_VARARGS},
{"equationsetWrite",(PyCFunction)DbMIDLEVEL_equationsetWrite, METH_VARARGS},
{"governingRead",   (PyCFunction)DbMIDLEVEL_governingRead,    METH_VARARGS},
{"governingWrite",  (PyCFunction)DbMIDLEVEL_governingWrite,   METH_VARARGS},
{"diffusionRead",   (PyCFunction)DbMIDLEVEL_diffusionRead,    METH_VARARGS},
{"diffusionWrite",  (PyCFunction)DbMIDLEVEL_diffusionWrite,   METH_VARARGS},
{"modelRead",       (PyCFunction)DbMIDLEVEL_modelRead,        METH_VARARGS},
{"modelWrite",      (PyCFunction)DbMIDLEVEL_modelWrite,       METH_VARARGS},

/*
 * cg_nbases     (attribute)
 * cg_base_read
 * cg_base_id
 * cg_base_write
 */
{"baseWrite",	   (PyCFunction)DbMIDLEVEL_baseWrite,	     METH_VARARGS},
{"baseRead",	   (PyCFunction)DbMIDLEVEL_baseRead,	     METH_VARARGS},
{"baseId",	   (PyCFunction)DbMIDLEVEL_baseId,	     METH_VARARGS},

/*
 * cg_nzones
 * cg_zone_read
 * cg_zone_type
 * cg_zone_id
 * cg_zone_write
 */
{"nzones",	   (PyCFunction)DbMIDLEVEL_nzones,	     METH_VARARGS},
{"zoneRead",	   (PyCFunction)DbMIDLEVEL_zoneRead,	     METH_VARARGS},
{"zoneType",	   (PyCFunction)DbMIDLEVEL_zoneType,	     METH_VARARGS},
{"zoneId",	   (PyCFunction)DbMIDLEVEL_zoneId,	     METH_VARARGS},  
{"zoneWrite",	   (PyCFunction)DbMIDLEVEL_zoneWrite,	     METH_VARARGS},    

/*  
 * cg_ngrids
 * cg_grid_read
 * cg_grid_write
 */
{"ngrids",     (PyCFunction)DbMIDLEVEL_ngrids,	     METH_VARARGS},
{"gridRead",   (PyCFunction)DbMIDLEVEL_gridRead,     METH_VARARGS}, 
{"gridWrite",  (PyCFunction)DbMIDLEVEL_gridWrite,    METH_VARARGS}, 

/*  
 * cg_ncoords
 * cg_coord_info
 * cg_coord_read
 * cg_coord_id     
 * cg_coord_write
 */
{"ncoords",	   (PyCFunction)DbMIDLEVEL_ncoords,	     METH_VARARGS},
{"coordInfo",	   (PyCFunction)DbMIDLEVEL_coordInfo,	     METH_VARARGS},
{"coordRead",	   (PyCFunction)DbMIDLEVEL_coordRead,	     METH_VARARGS},
{"coordId",	   (PyCFunction)DbMIDLEVEL_coordId,	     METH_VARARGS},
{"coordWrite",	   (PyCFunction)DbMIDLEVEL_coordWrite,	     METH_VARARGS},    
{"coordPartialWrite",(PyCFunction)DbMIDLEVEL_coordPartialWrite,METH_VARARGS},  

/*
 * cg_nsections
 * cg_section_read
 * cg_section_write
 * cg_elements_read
 * cg_parent_data_write
 * cg_npe
 * cg_ElementDataSize
 */
{"nsections",	   (PyCFunction)DbMIDLEVEL_nsections,	     METH_VARARGS},
{"sectionRead",	   (PyCFunction)DbMIDLEVEL_sectionRead,	     METH_VARARGS},
{"sectionWrite",   (PyCFunction)DbMIDLEVEL_sectionWrite,     METH_VARARGS},
{"elementsRead",   (PyCFunction)DbMIDLEVEL_elementsRead,     METH_VARARGS},
{"parentDataWrite",(PyCFunction)DbMIDLEVEL_parentDataWrite,  METH_VARARGS},
{"npe",            (PyCFunction)DbMIDLEVEL_npe,              METH_VARARGS},
{"elementDataSize",(PyCFunction)DbMIDLEVEL_elementDataSize,  METH_VARARGS},
{"sectionPartialWrite",(PyCFunction)DbMIDLEVEL_sectionPartWrite,METH_VARARGS},
{"parentDataPartialWrite",(PyCFunction)DbMIDLEVEL_parentDataPartWrite, METH_VARARGS},
{"elementsPartialRead",   (PyCFunction)DbMIDLEVEL_elementsPartRead,    METH_VARARGS},
{"elementDataPartialSize",(PyCFunction)DbMIDLEVEL_elementDataPartSize, METH_VARARGS},
   
/*  
 * cg_nsols
 * cg_sol_info
 * cg_sol_id
 * cg_sol_write
 */
{"nsols",	   (PyCFunction)DbMIDLEVEL_nsols,	     METH_VARARGS},
{"solInfo",	   (PyCFunction)DbMIDLEVEL_solInfo,	     METH_VARARGS},
{"solId",	   (PyCFunction)DbMIDLEVEL_solId,	     METH_VARARGS},
{"solWrite",	   (PyCFunction)DbMIDLEVEL_solWrite,	     METH_VARARGS},    

/*  
 * cg_nfields
 * cg_field_info
 * cg_field_read
 * cg_field_id
 * cg_field_write
 */
{"nfields",	   (PyCFunction)DbMIDLEVEL_nfields,	     METH_VARARGS},
{"fieldInfo",	   (PyCFunction)DbMIDLEVEL_fieldInfo,	     METH_VARARGS},
{"fieldRead",	   (PyCFunction)DbMIDLEVEL_fieldRead,	     METH_VARARGS},
{"fieldId",	   (PyCFunction)DbMIDLEVEL_fieldId,	     METH_VARARGS},
{"fieldWrite",	   (PyCFunction)DbMIDLEVEL_fieldWrite,	     METH_VARARGS},    
{"fieldPartialWrite",(PyCFunction)DbMIDLEVEL_fieldPartWrite, METH_VARARGS},    

/*  
 * cg_n1to1
 * cg_1to1_read
 * cg_1to1_id
 * cg_1to1_write
 */
{"nOneToOne",	  (PyCFunction)DbMIDLEVEL_n1to1,	     METH_VARARGS},
{"oneToOneRead",  (PyCFunction)DbMIDLEVEL_1to1Read,	     METH_VARARGS},
{"oneToOneId",	  (PyCFunction)DbMIDLEVEL_1to1Id,	     METH_VARARGS},
{"oneToOneWrite", (PyCFunction)DbMIDLEVEL_1to1Write,	     METH_VARARGS},

/*
 * cg_n1to1_global
 * cg_1to1_read_global
 */
{"nOneToOneGlobal",     (PyCFunction)DbMIDLEVEL_n1to1Global,	 METH_VARARGS},
{"OneToOneReadGlobal",  (PyCFunction)DbMIDLEVEL_1to1ReadGlobal,  METH_VARARGS},

/*
 * cg_nconns
 * cg_conn_info
 * cg_conn_read
 * cg_conn_id
 * cg_conn_write

 * cg_conn_average_read
 * cg_conn_average_write
 * cg_conn_periodic_read
 * cg_conn_periodic_write
 */
{"nConns",           (PyCFunction)DbMIDLEVEL_nConns,	        METH_VARARGS},
{"ConnInfo",         (PyCFunction)DbMIDLEVEL_ConnInfo,          METH_VARARGS},
{"ConnRead",         (PyCFunction)DbMIDLEVEL_ConnRead,          METH_VARARGS},
{"ConnId",           (PyCFunction)DbMIDLEVEL_ConnId,            METH_VARARGS},
{"ConnWrite",        (PyCFunction)DbMIDLEVEL_ConnWrite,         METH_VARARGS},
{"ConnAverageRead",  (PyCFunction)DbMIDLEVEL_ConnAverageRead,   METH_VARARGS},
{"ConnAverageWrite", (PyCFunction)DbMIDLEVEL_ConnAverageWrite,  METH_VARARGS},
{"ConnOneToOneAverageRead",  (PyCFunction)DbMIDLEVEL_ConnOneToOneAverageRead,   METH_VARARGS},
{"ConnOneToOneAverageWrite", (PyCFunction)DbMIDLEVEL_ConnOneToOneAverageWrite,  METH_VARARGS},
{"ConnPeriodicRead", (PyCFunction)DbMIDLEVEL_ConnPeriodicRead,  METH_VARARGS},
{"ConnPeriodicWrite",(PyCFunction)DbMIDLEVEL_ConnPeriodicWrite, METH_VARARGS},
{"ConnOneToOnePeriodicRead", (PyCFunction)DbMIDLEVEL_ConnOneToOnePeriodicRead,  METH_VARARGS},
{"ConnOneToOnePeriodicWrite",(PyCFunction)DbMIDLEVEL_ConnOneToOnePeriodicWrite, METH_VARARGS},
{"ConnReadShort",    (PyCFunction)DbMIDLEVEL_ConnReadShort,     METH_VARARGS},
{"ConnWriteShort",   (PyCFunction)DbMIDLEVEL_ConnWriteShort,    METH_VARARGS},

/*  
 * cg_nholes
 * cg_hole_info
 * cg_hole_read
 * cg_hole_id
 * cg_hole_write
 */
{"nHoles",       (PyCFunction)DbMIDLEVEL_nHoles,	 METH_VARARGS},
{"holeInfo",     (PyCFunction)DbMIDLEVEL_holeInfo,	 METH_VARARGS},
{"holeRead",     (PyCFunction)DbMIDLEVEL_holeRead,	 METH_VARARGS},
{"holeWrite",    (PyCFunction)DbMIDLEVEL_holeWrite,	 METH_VARARGS},
{"holeId",       (PyCFunction)DbMIDLEVEL_holeId,	 METH_VARARGS},
  
/*  
 * cg_nbocos
 * cg_boco_info
 * cg_boco_read
 * cg_boco_id
 * cg_boco_write
 * cg_boco_normal_write
 * cg_dataset_read
 * cg_dataset_write
 * cg_bcdata_write
 */
{"nBoco",           (PyCFunction)DbMIDLEVEL_nBoco,            METH_VARARGS},
{"bocoInfo",        (PyCFunction)DbMIDLEVEL_bocoInfo,         METH_VARARGS},
{"bocoRead",        (PyCFunction)DbMIDLEVEL_bocoRead,         METH_VARARGS},
{"bocoWrite",       (PyCFunction)DbMIDLEVEL_bocoWrite,        METH_VARARGS},
{"bocoId",          (PyCFunction)DbMIDLEVEL_bocoId,           METH_VARARGS},
{"bocoNormalWrite", (PyCFunction)DbMIDLEVEL_bocoNormalWrite,  METH_VARARGS},
{"bocoDatasetWrite",(PyCFunction)DbMIDLEVEL_bocoDatasetWrite, METH_VARARGS},
{"bocoDatasetRead", (PyCFunction)DbMIDLEVEL_bocoDatasetRead,  METH_VARARGS},

{"bocoLocalDatasetWrite",(PyCFunction)DbMIDLEVEL_bocoBCDatasetWrite, METH_VARARGS},
{"bocoLocalDatasetRead", (PyCFunction)DbMIDLEVEL_bocoBCDatasetRead,  METH_VARARGS},
{"bocoLodalDatasetInfo", (PyCFunction)DbMIDLEVEL_bocoBCDatasetInfo,  METH_VARARGS},

{"bocoDataWrite",   (PyCFunction)DbMIDLEVEL_bocoDataWrite,    METH_VARARGS},

/*  
 * cg_nfamilies
 * cg_family_read
 * cg_family_write
 * cg_famname_read
 * cg_famname_write
 * cg_fambc_read
 * cg_fambc_write
 */
{"nFamilies",       (PyCFunction)DbMIDLEVEL_nFamilies,       METH_VARARGS},
{"familyRead",      (PyCFunction)DbMIDLEVEL_familyRead,      METH_VARARGS},
{"familyWrite",     (PyCFunction)DbMIDLEVEL_familyWrite,     METH_VARARGS},
{"familyNameWrite", (PyCFunction)DbMIDLEVEL_familyNameWrite, METH_VARARGS},
{"familyNameRead",  (PyCFunction)DbMIDLEVEL_familyNameRead,  METH_VARARGS},
{"familyBocoRead",  (PyCFunction)DbMIDLEVEL_familyBocoRead,  METH_VARARGS},
{"familyBocoWrite", (PyCFunction)DbMIDLEVEL_familyBocoWrite, METH_VARARGS},

/*
 * cg_geo_read
 * cg_geo_write

 * cg_part_read
 * cg_part_write
 */
{"geoRead",	   (PyCFunction)DbMIDLEVEL_geoRead,	     METH_VARARGS},
{"geoWrite",	   (PyCFunction)DbMIDLEVEL_geoWrite,	     METH_VARARGS},
{"partRead",	   (PyCFunction)DbMIDLEVEL_partRead,	     METH_VARARGS},
{"partWrite",	   (PyCFunction)DbMIDLEVEL_partWrite,	     METH_VARARGS},

/*  
 * cg_ndiscrete
 * cg_discrete_read
 * cg_discrete_write
 */
{"nDiscrete",	   (PyCFunction)DbMIDLEVEL_nDiscrete,	     METH_VARARGS},
{"discreteRead",   (PyCFunction)DbMIDLEVEL_discreteRead,     METH_VARARGS}, 
{"discreteWrite",  (PyCFunction)DbMIDLEVEL_discreteWrite,    METH_VARARGS}, 

/*  
 * cg_simulation_type_read
 * cg_simulation_type_write
 */
{"simulationTypeRead", (PyCFunction)DbMIDLEVEL_simTypeRead,  METH_VARARGS},
{"simulationTypeWrite",(PyCFunction)DbMIDLEVEL_simTypeWrite, METH_VARARGS}, 

/*  
 * cg_goto
 */
{"goto",	       (PyCFunction)DbMIDLEVEL_goto,	     METH_VARARGS},
{"gorel",	       (PyCFunction)DbMIDLEVEL_gorel,	     METH_VARARGS},
{"gopath",	       (PyCFunction)DbMIDLEVEL_gopath,	     METH_VARARGS},
{"golist",	       (PyCFunction)DbMIDLEVEL_golist,	     METH_VARARGS},
{"where",	       (PyCFunction)DbMIDLEVEL_where,	     METH_VARARGS},

/*  
 * cg_delete_node
 */
{"deleteNode",	       (PyCFunction)DbMIDLEVEL_deleteNode,   METH_VARARGS},

/*  
 * cg_free
 */
{"freeData",	       (PyCFunction)DbMIDLEVEL_freeData,   METH_VARARGS},

/*  
 * cg_convergence_read
 * cg_convergence_write
 */
{"convergenceRead",    (PyCFunction)DbMIDLEVEL_convergenceRead, METH_VARARGS},
{"convergenceWrite",   (PyCFunction)DbMIDLEVEL_convergenceWrite,METH_VARARGS},

/*  
 * cg_state_read
 * cg_state_write
 */
{"stateRead",    (PyCFunction)DbMIDLEVEL_stateRead, METH_VARARGS},
{"stateWrite",   (PyCFunction)DbMIDLEVEL_stateWrite,METH_VARARGS},

/*  
 * cg_narrays
 * cg_array_info
 * cg_array_read
 ~ cg_array_read_as
 * cg_array_write
 */
{"nArrays",	  (PyCFunction)DbMIDLEVEL_narrays,	     METH_VARARGS},
{"arrayInfo",	  (PyCFunction)DbMIDLEVEL_arrayInfo,	     METH_VARARGS},
{"arrayRead",	  (PyCFunction)DbMIDLEVEL_arrayRead,	     METH_VARARGS},
{"arrayReadAs",	  (PyCFunction)DbMIDLEVEL_arrayReadAs,	     METH_VARARGS},
{"arrayWrite",    (PyCFunction)DbMIDLEVEL_arrayWrite,	     METH_VARARGS},    

/*  
 * cg_biter_read
 * cg_biter_write
 * cg_ziter_read
 * cg_ziter_write
 */
{"biterRead",	  (PyCFunction)DbMIDLEVEL_biterRead,	     METH_VARARGS},
{"biterWrite",	  (PyCFunction)DbMIDLEVEL_biterWrite,	     METH_VARARGS},
{"ziterRead",	  (PyCFunction)DbMIDLEVEL_ziterRead,	     METH_VARARGS},
{"ziterWrite",	  (PyCFunction)DbMIDLEVEL_ziterWrite,	     METH_VARARGS},

/*  
 * cg_nintegrals
 * cg_integral_read
 * cg_integral_write
 */
{"nIntegrals",	   (PyCFunction)DbMIDLEVEL_nIntegrals,	     METH_VARARGS},
{"integralRead",   (PyCFunction)DbMIDLEVEL_integralRead,     METH_VARARGS}, 
{"integralWrite",  (PyCFunction)DbMIDLEVEL_integralWrite,    METH_VARARGS}, 

/*  
 * cg_rind_read
 * cg_rind_write
 */
{"rindRead",   (PyCFunction)DbMIDLEVEL_rindRead,     METH_VARARGS},
{"rindWrite",  (PyCFunction)DbMIDLEVEL_rindWrite,    METH_VARARGS},

/*  
 * cg_link_write
 * cg_is_link
 * cg_link_read
 */
{"linkWrite", (PyCFunction)DbMIDLEVEL_linkWrite, METH_VARARGS},
{"linkRead",  (PyCFunction)DbMIDLEVEL_linkRead,  METH_VARARGS},    
{"isLink",    (PyCFunction)DbMIDLEVEL_isLink,    METH_VARARGS},    

/*  
 * cg_ndescriptors   
 * cg_descriptor_read
 * cg_descriptor_write
 */
{"nDescriptor", (PyCFunction)DbMIDLEVEL_nDescriptor,  METH_VARARGS},
{"descriptorRead", (PyCFunction)DbMIDLEVEL_descriptorRead,  METH_VARARGS},
{"descriptorWrite",(PyCFunction)DbMIDLEVEL_descriptorWrite, METH_VARARGS},    

/*  
 * cg_nuser_data
 * cg_user_data_read
 * cg_user_data_write
 */
{"nUserdata",     (PyCFunction)DbMIDLEVEL_nuserdata,      METH_VARARGS},
{"userdataRead",  (PyCFunction)DbMIDLEVEL_userdataRead,   METH_VARARGS},
{"userdataWrite", (PyCFunction)DbMIDLEVEL_userdataWrite,  METH_VARARGS},
/*  
 * cg_units_read
 * cg_units_write
 */
{"nUnits",    (PyCFunction)DbMIDLEVEL_nUnits,  METH_VARARGS},
{"unitsRead", (PyCFunction)DbMIDLEVEL_unitsRead,  METH_VARARGS},
{"unitsWrite",(PyCFunction)DbMIDLEVEL_unitsWrite, METH_VARARGS},    
{"unitsFullRead", (PyCFunction)DbMIDLEVEL_unitsFullRead,  METH_VARARGS},
{"unitsFullWrite",(PyCFunction)DbMIDLEVEL_unitsFullWrite, METH_VARARGS},    

/*  
 * cg_dataclass_read
 * cg_dataclass_write
 */
{"dataclassRead", (PyCFunction)DbMIDLEVEL_dataclassRead,  METH_VARARGS},
{"dataclassWrite",(PyCFunction)DbMIDLEVEL_dataclassWrite, METH_VARARGS},    

/*
 * cg_exponents_info
 * cg_exponents_read
 * cg_exponents_write
 */
{"nExponents", (PyCFunction)DbMIDLEVEL_nExponents,  METH_VARARGS},
{"exponentsInfo", (PyCFunction)DbMIDLEVEL_exponentsInfo,  METH_VARARGS},
{"exponentsRead", (PyCFunction)DbMIDLEVEL_exponentsRead,  METH_VARARGS},
{"exponentsWrite",(PyCFunction)DbMIDLEVEL_exponentsWrite, METH_VARARGS},    
{"exponentsFullRead", (PyCFunction)DbMIDLEVEL_exponentsFullRead, METH_VARARGS},
{"exponentsFullWrite",(PyCFunction)DbMIDLEVEL_exponentsFullWrite,METH_VARARGS},

/*  
 * cg_conversion_info
 * cg_conversion_read
 * cg_conversion_write
 */
{"conversionInfo", (PyCFunction)DbMIDLEVEL_conversionInfo,  METH_VARARGS},
{"conversionRead", (PyCFunction)DbMIDLEVEL_conversionRead,  METH_VARARGS},
{"conversionWrite",(PyCFunction)DbMIDLEVEL_conversionWrite, METH_VARARGS},    
  
/*  
 * cg_ordinal_read
 * cg_ordinal_write
 */
{"ordinalRead", (PyCFunction)DbMIDLEVEL_ordinalRead,  METH_VARARGS},
{"ordinalWrite",(PyCFunction)DbMIDLEVEL_ordinalWrite, METH_VARARGS},    

/*  
 * cg_ptset_info
 * cg_ptset_write
 * cg_ptset_read
 */
{"ptsetInfo", (PyCFunction)DbMIDLEVEL_ptsetInfo,  METH_VARARGS},
{"ptsetWrite",(PyCFunction)DbMIDLEVEL_ptsetWrite, METH_VARARGS},
{"ptsetRead", (PyCFunction)DbMIDLEVEL_ptsetRead,  METH_VARARGS},

/*  
 * cg_gridlocation_read
 * cg_gridlocation_write
 */
{"gridlocationRead", (PyCFunction)DbMIDLEVEL_gridlocationRead,  METH_VARARGS},
{"gridlocationWrite",(PyCFunction)DbMIDLEVEL_gridlocationWrite, METH_VARARGS},

/*  
 * cg_n_rigid_motions
 * cg_rigid_motion_read
 * cg_rigid_motion_write
 * cg_n_arbitrary_motions
 * cg_arbitrary_motion_read
 * cg_arbitrary_motion_write
 */
{"nRigidMotions",    (PyCFunction)DbMIDLEVEL_nRigidMotions,     METH_VARARGS},
{"rigidMotionRead",  (PyCFunction)DbMIDLEVEL_rigidMotionRead,   METH_VARARGS},
{"rigidMotionWrite", (PyCFunction)DbMIDLEVEL_rigidMotionWrite,  METH_VARARGS},
{"nArbitraryMotions",(PyCFunction)DbMIDLEVEL_nArbitraryMotions, METH_VARARGS},
{"arbitraryMotionRead",
                   (PyCFunction)DbMIDLEVEL_arbitraryMotionRead, METH_VARARGS},
{"arbitraryMotionWrite",
                   (PyCFunction)DbMIDLEVEL_arbitraryMotionWrite,METH_VARARGS},

/*  
 * cg_bc_area_read
 * cg_bc_area_write
 * cg_bc_wallfunction_read
 * cg_bc_wallfunction_write
 */
{"bcAreaRead",    (PyCFunction)DbMIDLEVEL_bcAreaRead,     METH_VARARGS},
{"bcAreaWrite",   (PyCFunction)DbMIDLEVEL_bcAreaWrite,    METH_VARARGS},
{"bcWallFunctionRead",
              (PyCFunction)DbMIDLEVEL_bcWallFunctionRead, METH_VARARGS},
{"bcWallFunctionWrite",
              (PyCFunction)DbMIDLEVEL_bcWallFunctionWrite,METH_VARARGS},

/*  
 * cg_axisym_read
 * cg_axisym_write
 * cg_rotating_read
 * cg_rotating_write
 */
{"axisymRead",   (PyCFunction)DbMIDLEVEL_axiSymRead,    METH_VARARGS},
{"axisymWrite",  (PyCFunction)DbMIDLEVEL_axiSymWrite,   METH_VARARGS},
{"rotatingRead", (PyCFunction)DbMIDLEVEL_rotatingRead,  METH_VARARGS},
{"rotatingWrite",(PyCFunction)DbMIDLEVEL_rotatingWrite, METH_VARARGS},

/*  
 * cg_gravity_read
 * cg_gravity_write
 */
{"gravityRead",   (PyCFunction)DbMIDLEVEL_gravityRead,    METH_VARARGS},
{"gravityWrite",  (PyCFunction)DbMIDLEVEL_gravityWrite,   METH_VARARGS},
  
/*  
 * cg_get_error
 # cg_error_exit  
 # cg_error_print
*/
{"lastError",	   (PyCFunction)DbMIDLEVEL_lastError,	     METH_VARARGS},

{NULL,		NULL}		/* sentinel */
};

/* ===================================================== */
static PyObject *
DbMIDLEVEL_getattr(DbMIDLEVELObject *self, char *name)
{
  if (!strcmp(name, "version"))
  {
    float f;
    double d;
    self->last_error=cg_version(self->db_fn,&f);
    d=f;
    strcpy(self->last_error_message,cg_get_error());
    return PyFloat_FromDouble(d);
  }
  if (!strcmp(name, "nbases"))
  {
    int  i;
    long l;
    self->last_error=cg_nbases(self->db_fn,&i);
    l=i;
    strcpy(self->last_error_message,cg_get_error());
    return PyInt_FromLong(l);
  }
  if (!strcmp(name, "ndescriptors"))
  {
    int i;
    long l;
    self->last_error=cg_ndescriptors(&i);
    l=i;
    strcpy(self->last_error_message,cg_get_error());
    return PyInt_FromLong(l);
  }
  if (!strcmp(name, "rootid"))
  {
    /* test on ia64 i8 */
    float f;
    double d;
    f=self->rootid;
    d=f;
    return PyFloat_FromDouble(d);
  }
  return Py_FindMethod(DbMIDLEVEL_methods, (PyObject *)self, name);
}

/* --------------------------------------------------------------------- */
static PyTypeObject DbMIDLEVEL_Type = {
  /* The ob_type field must be initialized in the module init function
   * to be portable to Windows without using C++. */
  PyObject_HEAD_INIT(NULL)
  /*ob_size*/          0,			 
  /*tp_name*/          "DbMIDLEVEL",		
  /*tp_basicsize*/     sizeof(DbMIDLEVELObject),	
  /*tp_itemsize*/      0,			
  /* methods */
  /*tp_dealloc*/       (destructor)DbMIDLEVEL_dealloc, 
  /*tp_print*/         0,			
  /*tp_getattr*/       (getattrfunc)DbMIDLEVEL_getattr, 
  /*tp_setattr*/       0,                      
  /*tp_compare*/       0,			
  /*tp_repr*/          0,			
  /*tp_as_number*/     0,			
  /*tp_as_sequence*/   0,			
  /*tp_as_mapping*/    0,			
  /*tp_hash*/          0,			
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
/* --------------------------------------------------------------------- */
static PyObject *
midlevel_new(PyObject *self, PyObject *args)
{
  DbMIDLEVELObject *rv;
  char	       *name;            
  int           mode = CG_MODE_READ; /* cgnslib.h value */

  if (!PyArg_ParseTuple(args, "si",&name,&mode)) return NULL;
  rv = newDbMIDLEVELObject(name,mode);

  if ( rv == NULL ) return NULL;
  Py_INCREF(rv);
  return (PyObject *)rv;
}


#if CGNS_VERSION >= 3000

/* --------------------------------------------------------------------- */
static PyObject *
midlevel_is_cgns(PyObject *self, PyObject *args)
{
  char *name;            
  int   error;
  int   ftype;

  if (!PyArg_ParseTuple(args, "s",&name)) return NULL;
  error = cg_is_cgns(name,&ftype);

  if (error != CG_OK) 
  { 
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    return NULL;
  }

  return PyInt_FromLong(1);
}

/* ---------------------------------------------------------------------- */
/* addnuserdata: cg_nuser_data
   returns number of CG_UserDefinedData_t children under the current node
*/
static PyObject *
midlevel_path_add(PyObject *self, PyObject *args)
{
  char *path;
  int error;

  if (!PyArg_ParseTuple(args, "s",&path)) return NULL;

  error=cgio_path_add(path);
  if ( error ) return NULL;

  Py_INCREF(Py_None);
  return Py_None;
}

/* --------------------------------------------------------------------- */
static PyObject *
midlevel_configure(PyObject *self, PyObject *args)
{
  PyObject *value;
  int       option,error;
  void     *pvalue;

  if (!PyArg_ParseTuple(args, "iO",&option,&value)) return NULL;

  switch (option)
  {
    case CG_CONFIG_ERROR:
      PyErr_SetString(MIDLEVELErrorObject,
                      "Bad cg_configure option: callback not implemented");
      return NULL;
    case CG_CONFIG_COMPRESS:
      pvalue=(void*)PyInt_AsLong(value);
      error=cg_configure(option,pvalue);
      break;
    case CG_CONFIG_SET_PATH:
    case CG_CONFIG_ADD_PATH:
      pvalue=(void*)PyString_AsString(value);
      error=cg_configure(option,pvalue);
      break;
    default:
      PyErr_SetString(MIDLEVELErrorObject,
                      "Bad cg_configure option: enumerate unknown");
      return NULL;
  }

  if ( error ) return NULL;
  /*  { 
    PyErr_SetString(MIDLEVELErrorObject,cg_get_error());
    return NULL;
    }*/
  Py_INCREF(Py_None); 
  return Py_None;
}
#endif

/* --------------------------------------------------------------------- */
static PyObject *
midlevel_mode(PyObject *self, PyObject *args)
{
  int mode;

  if (!PyArg_ParseTuple(args, "i",&mode)) return NULL;

  Py_INCREF(Py_None); 
  return Py_None;
}

/* --------------------------------------------------------------------- */
/* List of functions defined in the module */
static PyMethodDef midlevel_methods[] = {
  {"connect",		midlevel_new,		METH_VARARGS},
  {"mode",		midlevel_mode,		METH_VARARGS},  
#if CGNS_VERSION >= 3000
  {"configure",		midlevel_configure,	METH_VARARGS},  
  {"is_cgns",		midlevel_is_cgns,	METH_VARARGS},  
  {"path_add",	        midlevel_path_add,	METH_VARARGS},
#endif
  {NULL,		NULL}		/* sentinel */
};


/* --------------------------------------------------------------------- */
/* Initialization function for the module (*must* be called initmidlevel) */
DL_EXPORT(void)
init_mll(void)
{
  PyObject *m, *d, *a;
  
  /* Initialize the type of the new type object here; doing it here
   * is required for portability to Windows without requiring C++. */
  DbMIDLEVEL_Type.ob_type = &PyType_Type;
  
  /* Create the module and add the functions */
  m = Py_InitModule("_mll", midlevel_methods);

  /* is also done into adfmodule. Should work... */
  import_array();
  a = adfmodule_init();
  
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(m);
  PyDict_SetItemString(d,"_adf",a);

  MIDLEVELErrorObject = PyString_FromString("_mll.error");
  PyDict_SetItemString(d, "error", MIDLEVELErrorObject);

  midleveldictionnary_init(d); /* set many constants into dicts */

}

/* --------------------------------------------------------------------- */
/* last line */
