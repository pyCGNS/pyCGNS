/* ======================================================================
 * CHLone - CGNS HDF5 LIBRARY only node edition
 * See license.txt in the root directory of this source release
 * ====================================================================== 
*/
#include "l3.h"

/* ------------------------------------------------------------------------- */
#ifdef CHLONE_HAS_PTHREAD

#define L3M_MXINIT( ctxt ) \
pthread_mutex_init(&(ctxt->g_mutex),NULL);

#define L3M_MXLOCK( ctxt ) \
if (L3M_HASFLAG(ctxt,L3F_WITHMUTEX))\
{pthread_mutex_lock(&(ctxt->g_mutex));}

#define L3M_MXUNLOCK( ctxt ) \
if (L3M_HASFLAG(ctxt,L3F_WITHMUTEX))\
{pthread_mutex_unlock(&(ctxt->g_mutex));}

#define L3M_MXDESTROY( ctxt ) \
pthread_mutex_destroy(&(ctxt->g_mutex));

#else

#define L3M_MXINIT( ctxt )

#define L3M_MXLOCK( ctxt ) 

#define L3M_MXUNLOCK( ctxt ) 

#define L3M_MXDESTROY( ctxt )

#endif

/* ------------------------------------------------------------------------- */
#ifdef CHLONE_TRACK_TIME
#define L3M_TIMESTAMP( ctxt, msg ) \
{times(&(((L3_Cursor_t*)ctxt)->time));	\
printf("# L3 :: ");\
printf("U:[%.4d] S:[%.4d]\n",\
((L3_Cursor_t*)ctxt)->time.tms_utime,\
((L3_Cursor_t*)ctxt)->time.tms_stime);\
printf msg;fflush(stdout);}
#else
#define L3M_TIMESTAMP( ctxt, msg )
#endif

/* ------------------------------------------------------------------------- */
#define L3M_ECHECKID(ctxt,id,ret)			\
if (H5Iis_valid(id) != 1){ L3M_MXUNLOCK( ctxt ); return ret;}

#define L3M_ECHECKL3NODE(ctxt,id,ret) \
if (id == NULL){ L3M_MXUNLOCK( ctxt ); return ret;}

#define  L3M_CHECK_CTXT_OR_DIE(ctxt,rval)\
if (     (ctxt == NULL)\
     ||  (((L3_Cursor_t*)ctxt)->file_id<0)\
     ||  (((((L3_Cursor_t*)ctxt)->last_error)!=CHL_NOERROR)\
          &&(L3M_HASFLAG(ctxt,L3F_SKIPONERROR)))\
   )\
{ if (ctxt!=NULL) {L3M_MXUNLOCK( ctxt );} return rval ;}

/* ------------------------------------------------------------------------- */
#define L3T_MT "MT"
#define L3T_LK "LK"
#define L3T_B1 "B1"
#define L3T_C1 "C1"
#define L3T_I4 "I4"
#define L3T_I8 "I8"
#define L3T_U4 "U4"
#define L3T_U8 "U8"
#define L3T_R4 "R4"
#define L3T_R8 "R8"
#define L3T_X4 "X4"
#define L3T_X8 "X8"

/* HDF5 Compound names used for complex value */
#define CMPLX_REAL_NAME "r"
#define CMPLX_IMAG_NAME "i"

/* string type cache */
#define L3T_STR_NAME  2
#define L3T_STR_LABEL 1
#define L3T_STR_DTYPE 0

/* HDF5 compact storage limit */
#define CGNS_64KB (64 * 1024)


/* ------------------------------------------------------------------------- */
char *get_link_search_path(L3_Cursor_t* ctxt,int index);
int del_link_search_path(L3_Cursor_t* ctxt,char *path);
int add_link_search_path(L3_Cursor_t* ctxt,char *path);
int get_file_in_search_path(L3_Cursor_t *ctxt, char *file);

/* --- last line */
