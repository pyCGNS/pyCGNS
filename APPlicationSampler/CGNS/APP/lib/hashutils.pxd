
cdef extern from "hashutils.h":
  ctypedef struct __f_entry_t:
      pass
  
  ctypedef __f_entry_t* F_entry_ptr
  
  F_entry_ptr newHashTable(int s) nogil
  
  void freeHashTable(F_entry_ptr,int) nogil
  void addHashEntry(F_entry_ptr tab,int tsz,
                    int p1,int p2,int p3,int p4,int idx,int sec) nogil
  int  fetchHashEntry(F_entry_ptr tab,int tsz,
                      int p1,int p2,int p3,int p4,int sec) nogil
  int exteriorHashEntry(F_entry_ptr tab,int,int,int,int,int,int) nogil
