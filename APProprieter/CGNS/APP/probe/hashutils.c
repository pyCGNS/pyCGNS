/*===========================================================================*
  HASH TABLE (hash function FNV)
  http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
 *===========================================================================*/

#undef PROBE_DEBUG 

#ifdef PROBE_DEBUG
#include <stdio.h>
#endif
#include <stdlib.h>
#include <string.h>
#include "hashutils.h"

/* ------------------------------------------------------------------------  */
unsigned long __f_hash(int *key,int hsize)
{
  unsigned long hash = 5381;
  int i,c;

  for (i=0;i<4;i++)
  {
    c=key[i];
    hash=hash*33+c;
    hash=((hash<<5)+hash)+c;
  }
  return hash%hsize;
}
/* ------------------------------------------------------------------------  */
int __f_hashLookUp(__f_entry_t *table[],int *key,int hsize,int ins,int sec, 
		   int *i)
{
  __f_entry_t *e;
  int t;

  *i=__f_hash(key,hsize);
  e=table[*i];
  t=4*sizeof(int);
  
  if (e!=NULL)
  {
    while ((e->next!=NULL)&&memcmp(e->k,key,t)){e=e->next;}
    if (!memcmp(e->k,key,t))
    {
      if (ins&&((e->sec)==sec)){e->ext=0;}
      return e->face;
    }
  }
  return 0;
}
/* ------------------------------------------------------------------------  */
int __f_hashLookUpExt(__f_entry_t *table[],int *key,int hsize,int sec)
{
  int i,t;
  __f_entry_t *e;

  i=__f_hash(key,hsize);
  e=table[i];
  t=4*sizeof(int);
  
  if (e!=NULL)
  {
    while ((e->next!=NULL)&&memcmp(e->k,key,t)){e=e->next;}
    if (!memcmp(e->k,key,t)){return e->ext;}
  }
  return -1;
}
/* ------------------------------------------------------------------------  */
void __f_hashInsert(__f_entry_t *table[],int *key,int face,int hsize,int sec)
{
  int i,t,rface;
  __f_entry_t *e;

  rface=__f_hashLookUp(table,key,hsize,0,sec,&i);

  if (rface!=0){return;}

  e=table[i];
  t=4*sizeof(int);

  if (e!=NULL)
  {
    while (e->next!=NULL){e=e->next;}
    e->next=(__f_entry_t*)malloc(sizeof(__f_entry_t));
    e=e->next;
  }
  else
  {
    table[i]=(__f_entry_t*)malloc(sizeof(__f_entry_t));
    e=table[i];
  }
  memcpy(e->k,key,t);
  e->face=face;
  e->sec=sec;
  e->ext=1;
  e->next=NULL;
}
/* ------------------------------------------------------------------------  */
__f_entry_t *newHashTable(int s)
{
  __f_entry_t *ret=NULL;

  ret=(__f_entry_t*)malloc(sizeof(__f_entry_t)*s);
  return ret;
}
/* ------------------------------------------------------------------------  */
void freeHashTable(__f_entry_t *table[],int hsize)
{
  __f_entry_t *e,*p;
  int i;

  for (i=0;i<hsize;i++)
  {
    e=table[i];
    p=e;
    if (e!=NULL)
    {
      while (e->next!=NULL)
      {
#ifdef PROBE_DEBUG
        if (e->ext){printf("EXT FACE [%.8d] %d\n",e->face,e->sec);}
	else {printf("INT FACE [%.8d] %d\n",e->face,e->sec);}
#endif
        e=e->next;
        free(p);
	p=e;
      }
      free(p);
    }
  }
  free(table);
}
/* ------------------------------------------------------------------------  */
#define SORTPOINTSANDFILLBUFF(buff,ix1,ix2,ix3,ix4) \
buff[0]=(ix1<ix2);\
buff[1]=(ix1<ix3);\
buff[2]=(ix2<ix3);\
buff[3]=ix4;\
if      ( buff[0]&& buff[1]&& buff[2]){buff[0]=ix1;buff[1]=ix2;buff[2]=ix3;}\
else if ( buff[0]&& buff[1]&&!buff[2]){buff[0]=ix1;buff[1]=ix3;buff[2]=ix2;}\
else if (!buff[0]&& buff[1]&& buff[2]){buff[0]=ix2;buff[1]=ix1;buff[2]=ix3;}\
else if (!buff[0]&&!buff[1]&& buff[2]){buff[0]=ix2;buff[1]=ix3;buff[2]=ix1;}\
else if (!buff[0]&&!buff[1]&&!buff[2]){buff[0]=ix3;buff[1]=ix2;buff[2]=ix1;}\
else if ( buff[0]&&!buff[1]&&!buff[2]){buff[0]=ix3;buff[1]=ix1;buff[2]=ix2;}\
if (ix4<=buff[0])\
{buff[3]=buff[2];buff[2]=buff[1];buff[1]=buff[0];buff[0]=ix4;}\
if ((ix4>buff[0])&&(ix4<=buff[1]))\
{buff[3]=buff[2];buff[2]=buff[1];buff[1]=ix4;}\
if ((ix4>buff[1])&&(ix4<=buff[2]))\
{buff[3]=buff[2];buff[2]=ix4;}\
if (ix4>buff[2])\
{buff[3]=ix4;}

/* ------------------------------------------------------------------------  */
/* use only 3 points for each face, manage collision on 4th point            */
void addHashEntry(__f_entry_t *table[],int tsize,
		  int p1,int p2,int p3,int p4,int face,int sec)
{
  int buff[4];

  SORTPOINTSANDFILLBUFF(buff,p1,p2,p3,p4);
#ifdef PROBE_DEBUG
  printf("FACE [%.8d,%.8d,%.8d,%.8d] INS %.8d\n",
	 buff[0],buff[1],buff[2],buff[3],face);
#endif
  __f_hashInsert(table,buff,face,tsize,sec);
}
/* ------------------------------------------------------------------------  */
int fetchHashEntry(__f_entry_t *table[],int tsize,
		   int p1,int p2,int p3,int p4,int sec)
{
  int buff[4],r,i;

  SORTPOINTSANDFILLBUFF(buff,p1,p2,p3,p4);
  r=__f_hashLookUp(table,buff,tsize,(sec==-1)?0:1,sec,&i);
#ifdef PROBE_DEBUG
  if (r) {printf("FACE [%.8d,%.8d,%.8d,%.8d] RET %.8d SEC %d\n",
		 buff[0],buff[1],buff[2],buff[3],r,sec);}
#endif
  return r;
}
/* ------------------------------------------------------------------------  */
int exteriorHashEntry(__f_entry_t *table[],int tsize,
		      int p1,int p2,int p3,int p4,int sec)
{
  int buff[4],r;

  SORTPOINTSANDFILLBUFF(buff,p1,p2,p3,p4);
  r=__f_hashLookUpExt(table,buff,tsize,sec);
  return r;
}
/* ------------------------------------------------------------------------  */
