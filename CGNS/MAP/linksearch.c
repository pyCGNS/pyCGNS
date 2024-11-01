/* ======================================================================
 * CHLone - CGNS HDF5 LIBRARY only node edition
 * See license.txt in the root directory of this source release
 * ====================================================================== */
#include "l3P.h"

/* ------------------------------------------------------------------------- */
int __checkFile(char *path)
{
  struct stat *s;
  int r;

  s=(struct stat *)malloc(sizeof(struct stat));
  r=stat(path,s);
  free(s);
  if (r)
  {
    return -1;
  }
  return 0;
}

/* ------------------------------------------------------------------------- */
int get_file_in_search_path(L3_Cursor_t *ctxt, char *file)
{
  L3_PathList_t *current;
  int idx;
  char *buf;

  if (file==NULL)
  {
    return -1;
  }

  idx=0;
  current=ctxt->pathlist;

  while (current != NULL)
  {
    buf=(char*)malloc(strlen(current->path)+strlen(file)+2);
    strcpy(buf,current->path);
#ifdef CHLONE_ON_WINDOWS
    strcat(buf,"\\");
#else
    strcat(buf,"/");
#endif
    strcat(buf,file);
    if (!__checkFile(buf))
    {
      break;
    }
    free(buf);
    current=current->next;
    idx++;
  }
  if (current==NULL)
  {
    idx=-1;
  }
  return idx;
}

/* ------------------------------------------------------------------------- */
int CHL_getFileInSearchPath(L3_Cursor_t *ctxt, char *file)
{
  int idx;

  if (file==NULL)
  {
    return -1;
  }

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);
  L3M_MXLOCK(ctxt);

  idx=get_file_in_search_path(ctxt,file);

  L3M_MXUNLOCK(ctxt);
  return idx;
}

/* ------------------------------------------------------------------------- */
int add_link_search_path(L3_Cursor_t* ctxt,char *path)
{
  L3_PathList_t *current;
  int idx;

  if (path==NULL)
  {
    return -1;
  }

  idx=0;

  if (ctxt->pathlist == NULL)
  {
    ctxt->pathlist=(L3_PathList_t*)malloc(sizeof(L3_PathList_t));

    ctxt->pathlist->path=(char*)malloc(strlen(path)+1);
    ctxt->pathlist->next=NULL;
    strcpy(ctxt->pathlist->path,path);
  }
  else
  {
    idx=1;
    current=ctxt->pathlist;
    while ((current->next!=NULL) && (strcmp(current->path,path)))
    {
      current=current->next;
      idx++;
    }
    if (current->next==NULL)
    {
      current->next=(L3_PathList_t*)malloc(sizeof(L3_PathList_t));
      current->next->path=(char*)malloc(strlen(path)+1);
      current->next->next=NULL;
      strcpy(current->next->path,path);
    }
    else
    {
      idx--;
    }
  }
  return idx;
}

/* ------------------------------------------------------------------------- */
int CHL_addLinkSearchPath(L3_Cursor_t* ctxt,char *path)
{
  int idx;

  if (path==NULL)
  {
    return -1;
  }

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);
  L3M_MXLOCK(ctxt);
  
  idx=add_link_search_path(ctxt,path);

  L3M_MXUNLOCK(ctxt);
  return idx;
}

/* ------------------------------------------------------------------------- */
int del_link_search_path(L3_Cursor_t* ctxt,char *path)
{
  L3_PathList_t *current;

  if (ctxt->pathlist == NULL)
  {
    return 0;
  }
  current=ctxt->pathlist;
  while ((current->next!=NULL) && (!strcmp(current->path,path)))
  {
    current=current->next;
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
int CHL_freeLinkSearchPath(L3_Cursor_t* ctxt)
{
  L3_PathList_t *current,*next;

  if (ctxt->pathlist == NULL)
  {
    return 0;
  }
  current=ctxt->pathlist;
  while (current!=NULL)
  {
    next=current->next;
    current->next=NULL;
    if (current->path!=NULL)
    {
      free(current->path);
    }
    free(current);
    current=next;
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
int CHL_delLinkSearchPath(L3_Cursor_t* ctxt,char *path)
{
  int ret;

  L3M_CHECK_CTXT_OR_DIE(ctxt,0);
  L3M_MXLOCK(ctxt);

  ret=del_link_search_path(ctxt,path);

  L3M_MXUNLOCK(ctxt);
  return ret;
}

/* ------------------------------------------------------------------------- */
char *get_link_search_path(L3_Cursor_t* ctxt,int index)
{
  char *ret=NULL;
  L3_PathList_t *current;

  if (ctxt->pathlist == NULL)
  {
    ret=NULL;
  }
  current=ctxt->pathlist;
  while ((current->next!=NULL) && (index>0))
  {
    index--;
    current=current->next;
  }
  if (index==0)
  {
    ret=current->path;
  }
  else
  {
    ret=NULL;
  }
  
  return ret;
}
/* ------------------------------------------------------------------------- */
char  *CHL_getLinkSearchPath(L3_Cursor_t* ctxt,int index)
{
  char *ret;
  static char CHL_EMPTY[1];
  
  CHL_EMPTY[0]='\0';

  L3M_CHECK_CTXT_OR_DIE(ctxt,CHL_EMPTY);
  L3M_MXLOCK(ctxt);

  ret=get_link_search_path(ctxt,index);
  L3M_MXUNLOCK(ctxt);

  if (ret==NULL)
  {
    return CHL_EMPTY;
  }

  return ret;
}
/* --- last line ----------------------------------------------------------- */
