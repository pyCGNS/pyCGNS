/* ======================================================================
 * CHLone - CGNS HDF5 LIBRARY only node edition
 * See license.txt in the root directory of this source release
 * ====================================================================== */
#ifndef __SHA256_H__
#define __SHA256_H__

#ifdef _MSC_VER
typedef signed   __int8   int8_t;
typedef signed   __int16  int16_t;
typedef signed   __int32  int32_t;
typedef unsigned __int8   uint8_t;
typedef unsigned __int16  uint16_t;
typedef unsigned __int32  uint32_t;
typedef signed   __int64  int64_t;
typedef unsigned __int64  uint64_t;

#else
#include <stdint.h>
#endif

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
  uint32_t total[2];
  uint32_t state[8];
  uint8_t buffer[64];
} sha256_t;

void sha256_starts(sha256_t *ctx);
void sha256_update(sha256_t *ctx,const uint8_t *input, uint32_t length);
void sha256_finish(sha256_t *ctx,uint8_t digest[32]);

#endif
/* --- last line -------------------------------------------------------- */
