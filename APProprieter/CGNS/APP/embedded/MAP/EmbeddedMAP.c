/* 
   ------------------------------------------------------------ 
   How to use the MAP translation in your C application code to
   produce/use Python CGNS trees.
   ------------------------------------------------------------ 
   To compile in the CGNS/MAP/test directory, use the following command:

h5cc -I.. -I/home_local/eucass/tools/include/python2.6 -c EmbeddedMAP.c

*/

#include "SIDStoPython.h"
#include "numpy/arrayobject.h"

PyObject *loadHDF5andGetPythonObject(char *filename)
{
  char *path;    /* The path you want to load, in other words the sub-tree.
                    If you set a NULL value, the load gets the whole tree. */
  int flags;     /* The flags are used to drive the load when it encounters
                    links or other specific stuff. */
  int threshold; /* Actual DataArray_t are load only if they are below this
                    value. A 0 stands for no limit. */
  int depth;     /* The parse depth, i.e. the max level of children you want.
                    A value of 999 clearly sets no limit. */

  flags     = S2P_FFOLLOWLINKS|S2P_FTRACE;
  threshold = 0;
  depth     = 999;
  path      = NULL;

  return s2p_loadAsHDF(filename,flags,threshold,depth,path);
}

int main(int argc, char **argv)
{
  PyObject *cgnstree;

  Py_Initialize();
  import_array(); /* Crash if not there */
  cgnstree=loadHDF5andGetPythonObject("./T0.cgns");
  Py_Finalize();
}

