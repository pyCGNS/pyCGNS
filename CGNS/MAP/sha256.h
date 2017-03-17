/* ======================================================================
 * CHLone - CGNS HDF5 LIBRARY only node edition
 * See license.txt in the root directory of this source release
 * ====================================================================== */
#ifndef __SHA256_H__
#define __SHA256_H__

#include <stdint.h>
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
