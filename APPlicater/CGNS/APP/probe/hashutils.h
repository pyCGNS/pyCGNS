/*
# ----------------------------------------------------------------------
# fetch - (c) Onera The French Aerospace Lab - http://www.onera.fr
#         See Copyright notice in README.txt file
# ----------------------------------------------------------------------
*/
/* please read hasutils.c for doc and functions args */

typedef struct __f_entry_t
{
  struct __f_entry_t *next;
  int k[4];
  int face;
  int sec;
  short int ext;
} __f_entry_t;

typedef __f_entry_t ** F_entry_ptr;

__f_entry_t *newHashTable(int s);
void freeHashTable(__f_entry_t *t[],int);
void addHashEntry(__f_entry_t *t[],int,int,int,int,int,int,int);
int fetchHashEntry(__f_entry_t *t[],int,int,int,int,int,int);
int exteriorHashEntry(__f_entry_t *t[],int,int,int,int,int,int);

/* --- last line */
