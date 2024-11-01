/* ======================================================================
 * CHLone - CGNS HDF5 LIBRARY only node edition
 * See license.txt in the root directory of this source release
 * ====================================================================== */
#include "l3.h"

typedef struct 
{
  int   error;
  int   code;
  char *msg;
} CHL_error_t;

/* ------------------------------------------------------------------------- */
static CHL_error_t __CHL_errorTable[]= 
{
#include "l3error.h"
  {0000, 0, "Unknown error code"}
};
/* ------------------------------------------------------------------------- */
int CHL_setMessage(L3_Cursor_t* ctxt,char *msg)
{
  int maxbuffsize;

  if (ctxt!=NULL)
  {
    maxbuffsize=strlen(msg)+strlen(ctxt->ebuff);
    if (maxbuffsize>L3C_MAX_SIZE_BUFFER) {return 0;}
    strcpy(ctxt->ebuff+ctxt->ebuffptr,msg);
    ctxt->ebuffptr+=strlen(msg);
    return 1;
  }
  return 0;
}
/* ------------------------------------------------------------------------- */
void CHL_setError_aux(L3_Cursor_t* ctxt,int err, va_list arg) 
{
  int  n;
  char localbuff[512]; /* bet ! */

  if (ctxt==NULL) { return; }
  n=0;
  if (err<0){return;}
  strcpy(localbuff,"# ### ");
  while ((__CHL_errorTable[n].error)&&(__CHL_errorTable[n].error!=err)){n++;}
  sprintf(localbuff+strlen(localbuff),"E:%.4d ",__CHL_errorTable[n].error);
  vsprintf(localbuff+strlen(localbuff),__CHL_errorTable[n].msg,arg);
  strcat(localbuff+strlen(localbuff),"\n");

  CHL_setMessage(ctxt,localbuff);
  ctxt->last_error=err;
}
/* ------------------------------------------------------------------------- */
void CHL_setError(L3_Cursor_t* ctxt,int err, ...) 
{
  va_list arg;
  va_start(arg, err);
  CHL_setError_aux(ctxt,err,arg);
  va_end(arg);
}
/* ------------------------------------------------------------------------- */
void CHL_printError(L3_Cursor_t *ctxt)
{
  if (ctxt==NULL)
  {
    printf("# CHLone error stack: [NULL context]\n");
    return;
  }
  if (ctxt->ebuffptr==0)
  {
    if (L3M_HASFLAG(ctxt,L3F_DEBUG))
    {
      printf("# CHLone error stack: [NO ERROR]\n");
    }
  }
  else
  {
    printf("# CHLone error stack:\n%s",ctxt->ebuff);
    printf("# \n");
  }
}
/* ------------------------------------------------------------------------- */
int CHL_getError(L3_Cursor_t *ctxt)
{
  if (ctxt!=NULL)
  {
    return ctxt->last_error;
  }
  return 3100;
}
/* ------------------------------------------------------------------------- */
char *CHL_getMessage(L3_Cursor_t *ctxt)
{
  if (ctxt!=NULL)
  {
    return ctxt->ebuff;
  }
  return '\0';
}
/* --- last line ----------------------------------------------------------- */
