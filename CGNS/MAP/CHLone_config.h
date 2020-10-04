/* ======================================================================
 * CHLone - CGNS HDF5 LIBRARY only node edition
 * See license.txt in the root directory of this source release
 * ====================================================================== */
/*
   configuration file
*/
#define CHLONE_ON_WINDOWS 0
#define CHLONE_H5CONF_STD 1
#define CHLONE_H5CONF_64  0
#define CHLONE_H5CONF_UP  1

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#if (CHLONE_ON_WINDOWS == 1)
#include <io.h>
#else
#include <unistd.h>
#undef CHLONE_ON_WINDOWS
#endif

#undef H5_NO_DEPRECATED_SYMBOLS 

#include "hdf5.h"

/* Manage public conf header wrt actual distribution changes */
#if (CHLONE_H5CONF_STD==1)
#if (CHLONE_H5CONF_UP==1)
#include "H5pubconf.h"
#else
#include "h5pubconf.h"
#endif
#endif

#if (CHLONE_H5CONF_64==1)
#if (CHLONE_H5CONF_UP==1)
#include "H5pubconf-64.h"
#else
#include "h5pubconf-64.h"
#endif
#endif

#ifdef H5_HAVE_THREADSAFE
#define CHLONE_HAS_PTHREAD 1
#else
#define CHLONE_HAS_PTHREAD 0
#endif

#if (CHLONE_HAS_PTHREAD == 1)
#include <pthread.h>
#else
#undef CHLONE_HAS_PTHREAD
#endif

/* Flag indicating regexp is used for paths */
#define CHLONE_HAS_REGEXP 1

/* Flag indicating you have CHLone inside */
#define CHLONE_IMPLEMENTATION 1

/* Trace and debug using raw printf to stdout */
#define CHLONE_PRINTF_TRACE 0

/* DEBUG ONLY */
#ifdef CHLONE_TRACK_TIME
#if (CHLONE_ON_WINDOWS == 1)
#include <sys/utime.h>
#else
#include <sys/times.h>
#include <sys/time.h>
#include <sys/resource.h>
extern rusage_snapshot(char *tag);
#endif
#endif

/* C DIALECT */
#if (CHLONE_ON_WINDOWS == 1)
#define CHL_INLINE __inline
#undef CHLONE_HAS_REGEXP
#else
#define CHL_INLINE static inline
#endif

/* trick for cython include (don't remember why!) */
#ifdef FAKE__FLAG
#include "CGNS/MAP/SIDStoPython.h"
#endif

/* windows stat/access */
#if (CHLONE_ON_WINDOWS == 1)
#define CHL_STAT(a,b) _stat(a,b)
#define CHL_ACCESS_READ(a) _access(a, 2) || _access(a, 6)
#else
#define CHL_STAT(a,b) stat(a,b)
#define CHL_ACCESS_READ(a) access(a, R_OK)
#endif

/* --- last line */
