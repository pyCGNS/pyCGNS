/*
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
*/
#include "cgnslib.h"
#include <stdlib.h>

int cgu_1to1_read_global(int fn, int B,
			 char **connectname, char **zonename,
			 char **donorname,
			 cgsize_t **range, cgsize_t **donor_range,
			 int **transform)
{
  int cnum,n;

  cg_n1to1_global(fn,B,&cnum);

  /* create table for pointer storage */
  connectname=(char**)malloc((sizeof(char*)*cnum));
  zonename=(char**)malloc((sizeof(char*)*cnum));
  donorname=(char**)malloc((sizeof(char*)*cnum));
  
  range=(cgsize_t **)malloc(sizeof(cgsize_t *)*cnum);
  donor_range=(cgsize_t **)malloc(sizeof(cgsize_t *)*cnum);
  transform=(int **)malloc(sizeof(cgsize_t *)*cnum);

  /*create contiguous memory zone, first pointer has all memory  */
  connectname[0]=(char*)malloc(sizeof(char)*33*cnum);
  zonename[0]=(char*)malloc(sizeof(char)*33*cnum);
  donorname[0]=(char*)malloc(sizeof(char)*33*cnum);

  range[0]=(cgsize_t*)malloc(sizeof(cgsize_t)*6*cnum);
  donor_range[0]=(cgsize_t*)malloc(sizeof(cgsize_t)*6*cnum);
  transform[0]=(char*)malloc(sizeof(int)*3*cnum);

  /* redistribute memory pointers */  
  for (n=1;n<cnum;n++)
  {
    connectname[n]=connectname[0]+(sizeof(char)*33*n);
    zonename[n]=zonename[0]+(sizeof(char)*33*n);
    donorname[n]=donorname[0]+(sizeof(char)*33*n);

    range[n]=range[0]+(sizeof(cgsize_t)*33*n);
    donor_range[n]=donor_range[0]+(sizeof(cgsize_t)*33*n);
    transform[n]=transform[0]+(sizeof(int)*33*n);
  }

  /* fill data */
  cg_1to1_read_global(fn,B,
		      connectname,zonename,donorname,
		      range,donor_range,transform);

}
