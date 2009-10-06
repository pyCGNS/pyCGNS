/* ======================================================================
 * CHLone - CFD General Notation System -                                
 * ONERA/DSNA/ELSA - poinot@onera.fr                                     
 * $Rev: 12 $ $Date: 2008-05-30 17:32:00 +0200 (Fri, 30 May 2008) $
 * See file license.txt in the root directory of this distribution 
 * ====================================================================== */

/* All messages including errors, all interface levels. */
#include "CHLone_l3.h"
#include "CHLone_SIDS.h"

typedef struct 
{
  int   error;
  int   code;
  char *msg;
} CHL_error_t;
/* ------------------------------------------------------------------------- */
static CHL_error_t __CHL_errorTable[]= 
{
  {2000,CG_ERROR,"[Name] attribute not found while DataArray_t creation"},
  {2001,CG_ERROR,"[DataType] attribute not found while DataArray_t creation"},
  {2002,CG_ERROR,"[DataDimension] attribute not found while DataArray_t creation"},
  {2003,CG_ERROR,"[DataPointer] attribute not found while DataArray_t creation"},
  {2010,CG_ERROR,"[Name] attribute not found while Zone_t creation"},
  {2011,CG_ERROR,"[ZoneType] attribute not found while Zone_t creation"},
  {2012,CG_ERROR,"[ZoneDimensionSize] attribute not found while Zone_t creation"},
  {2013,CG_ERROR,"[ZoneDimension] attribute not found while Zone_t creation"},
  {3000,CG_ERROR,"File '%s' not found (read only mode)"},
  {3001,CG_ERROR,"File '%s' not found (modify mode)"},
  {3002,CG_ERROR,"Create file '%s' fails"},
  {3003,CG_ERROR,"Update file '%s' fails"},
  {3004,CG_ERROR,"Open file '%s' as read only fails"},
  {3010,CG_ERROR,"Unknown file mode (integer=%d)"},
  {3011,CG_ERROR,"Property list fails for 'file'"},
  {3012,CG_ERROR,"Property list fails for 'link'"},
  {3020,CG_ERROR,"No data to get in this node"},
  {3021,CG_ERROR,"No data to set in this node"},
  {3022,CG_ERROR,"Create root fails while trying H5Gopen2"},
  {3023,CG_ERROR,"Create root fails while trying to add HDF5 MotherNode"},
  {3024,CG_ERROR,"Create root fails while trying to add Root Node"},
  {3025,CG_ERROR,"Create root fails while trying to set MT"},
  {3026,CG_ERROR,"Create root fails while trying to set format"},
  {3027,CG_ERROR,"Create root fails while trying to set version"},
  {3030,CG_ERROR,"Bad nodeCreate: cannot create node [%s]"},  
  {3031,CG_ERROR,"Bad nodeCreate: cannot add name attribute [%s]"},  
  {3032,CG_ERROR,"Bad nodeCreate: cannot add 'label' attribute"},  
  {3033,CG_ERROR,"Bad nodeCreate: cannot add 'datatype' attribute"},  
  {3034,CG_ERROR,"Bad nodeCreate: cannot add data"},  
  {3035,CG_ERROR,"Bad nodeCreate: empty data in non-MT node"},  
  {3036,CG_ERROR,"Bad nodeCreate: cannot add flags"},  
  {3040,CG_ERROR,"Bad nodeMove: source/destination is a link"},
  {3041,CG_ERROR,"Bad nodeMove: cannot change name attribute"},
  {3050,CG_ERROR,"Bad nodeUpdate: cannot update a link"},
  {3051,CG_ERROR,"Bad nodeUpdate: bad node id"},
  {3052,CG_ERROR,"Bad nodeUpdate: cannot change name on [%s]"},
  {3053,CG_ERROR,"Bad nodeUpdate: cannot update label on [%s]"},
  {3054,CG_ERROR,"Bad nodeUpdate: cannot update dtype on [%s]"},
  {3055,CG_ERROR,"Bad nodeUpdate: cannot update data on [%s]"},
  {3056,CG_ERROR,"Bad nodeUpdate: cannot update flags on [%s]"},
  {3060,CG_ERROR,"Bad nodeLink: parent already is a link"},
  {3061,CG_ERROR,"Bad nodeLink: cannot create node [%s]"},
  {3062,CG_ERROR,"Bad nodeLink: cannot add name attribute [%s]"},
  {3063,CG_ERROR,"Bad nodeLink: cannot set LK type"},
  {3064,CG_ERROR,"Bad nodeLink: cannot set empty label"},
  {3070,CG_ERROR,"Bad nodeDelete: bad parent node id"},
  {3080,CG_ERROR,"Bad nodeFind: path too long (%d chars)"},
  {3081,CG_ERROR,"Bad nodeFind: bad parent node id"},
  {3090,CG_ERROR,"Bad nodeRetrieve: bad node id"},
  {0000, 0, "Unknown error code"}
};
/* ------------------------------------------------------------------------- */
int CHL_setMessage(L3_Cursor_t* ctxt,char *msg)
{
  int maxbuffsize;

  maxbuffsize=strlen(msg)+strlen(ctxt->ebuff);
  if (maxbuffsize>L3_MAX_SIZE_BUFFER) {return 0;}
  strcpy(ctxt->ebuff+ctxt->ebuffptr,msg);
  ctxt->ebuffptr+=strlen(msg);

  return 1;
}
/* ------------------------------------------------------------------------- */
void CHL_setError(L3_Cursor_t* ctxt,int err, ...) 
{
  int  n;
  char localbuff[512]; /* bet ! */
  va_list arg;

  n=0;
  if (err<0){return;}
  va_start(arg, err);
  while ((__CHL_errorTable[n].error)&&(__CHL_errorTable[n].error!=err)){n++;}
  vsprintf(localbuff,__CHL_errorTable[n].msg,arg);
  va_end(arg);

  CHL_setMessage(ctxt,localbuff);
  ctxt->last_error=err;
}

/* --- last line ----------------------------------------------------------- */
