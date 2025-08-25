#ifndef __API_VTA_V_H__
#define __API_VTA_V_H__
#include <stdint.h>
#include <stddef.h>
int vta_status(void);

ervp_mop_wait_fx_custom_t Tiled_block_gemm(ervp_mop_mapping_custom_t *mop_mapping, const ErvpMatrixInfo *a, const ErvpMatrixInfo *b, const ErvpMatrixInfo *bb, ErvpMatrixInfo *c);
#endif
