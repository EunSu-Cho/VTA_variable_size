// GEMM with Row Major Order
#include <stdint.h>
#include <stddef.h>
#include "platform_info.h"
#include "xvta_hw.h"

#include "ervp_printf.h"
#include "ervp_profiling.h"
#include "ervp_printf_section.h"
#include "ervp_variable_allocation.h"
#include "ervp_matrix.h"
#include "ervp_matrix_op_sw.h"
#include "ervp_matrix_op_sw_custom.h"
#include "ervp_special_matrix_op.h"
#include "ervp_matrix_datatype_define.h"
#include "ervp_assert.h"
#include "ervp_matrix_op_custom.h"

#include "ervp_tensor.h"
#include "core_dependent.h"
#include "matrix_config.h"
#include "gen_inst_uop.h"

//========================== VTA_config =========================== //
#define VTA_DATA __attribute__ ((aligned(0x1000)))
#define VTA_BLOCK_SIZE 16
#define BUFFER_SIZE (VTA_BLOCK_SIZE*VTA_BLOCK_SIZE) //16*16

//===================== Instructions and uops ===================== //
uint64_t insns[] NOTCACHED_DATA VTA_DATA = {
  LSINST_LO(0,0,0,0,0,0),                   LSINST_HI(1,VTA_BLOCK_SIZE,VTA_BLOCK_SIZE,0,0,0,0), // LOAD uop to uop buffer
  LSINST_LO(0,4,3,0,0,0),                   LSINST_HI(1,VTA_BLOCK_SIZE,VTA_BLOCK_SIZE,0,0,0,0), // LOAD space to accumulate buffer  
  LSINST_LO(0,2,1,0,0,0),                   LSINST_HI(1,1,1,0,0,0,0),                           // LOAD right_array to wgt buffer. 
  LSINST_LO(0,8,2,0,0,0),                   LSINST_HI(1,VTA_BLOCK_SIZE,VTA_BLOCK_SIZE,0,0,0,0), // LOAD left_array to inp buffer
  GMINST_LO(2,9,0,0,VTA_BLOCK_SIZE,1,1,0),  GMINST_HI(0,0,0,0,0,0),                             // GEMM
  LSINST_LO(1,5,4,0,0,0),                   LSINST_HI(1,VTA_BLOCK_SIZE,VTA_BLOCK_SIZE,0,0,0,0), // STORE
  FNINST_LO(3,2,0),                         FNINST_HI(0)                                        //fINISH
}; 
uint32_t uops[] NOTCACHED_DATA VTA_DATA = {
  UOP(0,0,0),//acc,inp,wgt  
  UOP(1,1,0),
  UOP(2,2,0),
  UOP(3,3,0),
  UOP(4,4,0), 
  UOP(5,5,0),
  UOP(6,6,0),
  UOP(7,7,0),
  UOP(8,8,0),
  UOP(9,9,0),
  UOP(10,10,0),
  UOP(11,11,0),
  UOP(12,12,0),
  UOP(13,13,0),
  UOP(14,14,0),
  UOP(15,15,0)
};

int vta_status(void)
{
  return REG32(XVTA_CONTROL_ADDR_AP_CTRL);
}

static uint32_t left_buffer[BUFFER_SIZE] NOTCACHED_DATA VTA_DATA;
static uint32_t right_buffer[BUFFER_SIZE] NOTCACHED_DATA VTA_DATA;
static uint32_t bias_buffer[BUFFER_SIZE] NOTCACHED_DATA VTA_DATA;
static uint32_t output_buffer[BUFFER_SIZE] NOTCACHED_DATA VTA_DATA;

static ErvpMatrixInfo left_buffer_info;
static ErvpMatrixInfo right_buffer_info;
static ErvpMatrixInfo bias_buffer_info;
static ErvpMatrixInfo output_buffer_info;

static void setup_vta_var()
{
  // Data signal of insn_count
  REG32(XVTA_CONTROL_ADDR_INSN_COUNT_DATA) = sizeof(insns)/16;
  // Data signal of insns
  REG64(XVTA_CONTROL_ADDR_INSNS_DATA)   = insns;
  // Data signal of uops
  REG32(XVTA_CONTROL_ADDR_UOPS_DATA)    = uops;
  // Data signal of inputs
  REG32(XVTA_CONTROL_ADDR_INPUTS_DATA)  = left_buffer_info.addr;
  // Data signal of weights
  REG32(XVTA_CONTROL_ADDR_WEIGHTS_DATA) = right_buffer_info.addr;
  // Data signal of biases
  REG32(XVTA_CONTROL_ADDR_BIASES_DATA)  = bias_buffer_info.addr;
  // Data signal of outputs
  REG32(XVTA_CONTROL_ADDR_OUTPUTS_DATA) = output_buffer_info.addr;
}

static void __attribute__ ((constructor)) construct_vta()
{
  matrix_generate_info(MATRIX_DATATYPE_SINT32,VTA_BLOCK_SIZE,VTA_BLOCK_SIZE,left_buffer,&left_buffer_info);
  matrix_generate_info(MATRIX_DATATYPE_SINT32,VTA_BLOCK_SIZE,VTA_BLOCK_SIZE,right_buffer,&right_buffer_info);
  matrix_generate_info(MATRIX_DATATYPE_SINT32,VTA_BLOCK_SIZE,VTA_BLOCK_SIZE,bias_buffer,&bias_buffer_info);
  matrix_generate_info(MATRIX_DATATYPE_SINT32,VTA_BLOCK_SIZE,VTA_BLOCK_SIZE,output_buffer,&output_buffer_info);
  setup_vta_var();
}

// =========================== GEMM with Row Major Order =========================== //
// Function to perform matrix multiplication in row-major order(GUSTAVSON) using VTA.
// a: left matrix, b: right matrix, bb: bias matrix, c: output matrix
// data flow : matrix -> 16*16 buffer -> VTA
// If you want speed up, remove step "16*16 buffer" and directly map address of matrix to VTA.
// ================================================================================= //
ervp_mop_wait_fx_custom_t matrix_mult_vta_F(
                            ervp_mop_mapping_custom_t *mop_mapping, 
                            const ErvpMatrixInfo *a, 
                            const ErvpMatrixInfo *b, 
                            const ErvpMatrixInfo *bb, 
                            ErvpMatrixInfo *c
                        )
{

    int m = (a->num_row)/VTA_BLOCK_SIZE;
    int k = (a->num_col)/VTA_BLOCK_SIZE;
    int n = (b->num_col)/VTA_BLOCK_SIZE;

    int mr = (a->num_row)%VTA_BLOCK_SIZE;
    int kr = (a->num_col)%VTA_BLOCK_SIZE;
    int nr = (b->num_col)%VTA_BLOCK_SIZE;

    int iter_m = m + (mr ? 1 : 0);
    int iter_k = k + (kr ? 1 : 0);
    int iter_n = n + (nr ? 1 : 0);

    ErvpMatrixInfo *left_info = &left_buffer_info;
    ErvpMatrixInfo *right_info = &right_buffer_info;
    ErvpMatrixInfo *bias_info = &bias_buffer_info;
    ErvpMatrixInfo *output_info = &output_buffer_info;

    for(int i=0; i < iter_m; i++) { // M
        for(int j=0; j < iter_n; j++) { // N
            for(int l=0; l < iter_k; l++) { // K
                //input a copy
                if(l==k || i==m){//remaining part exists in bottom or right
                    matrix_zero_sw(left_info);
                    matrix_copy_partition_sw(a, left_info, i*VTA_BLOCK_SIZE, l*VTA_BLOCK_SIZE, 0, 0, i==m ? mr : VTA_BLOCK_SIZE, l==k ? kr : VTA_BLOCK_SIZE, 0);
                }
                else{//no remaining part, 16x16 full matrix
                    matrix_copy_partition_sw(a, left_info, i*VTA_BLOCK_SIZE, l*VTA_BLOCK_SIZE, 0, 0, VTA_BLOCK_SIZE, VTA_BLOCK_SIZE, 0);
                }

                //input b copy
                if(l==k || j==n){//remaining part exists in bottom or right
                    matrix_zero_sw(right_info);
                    matrix_copy_transpose_partition_sw(b, right_info, l*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, 0, 0, l==k ? kr : VTA_BLOCK_SIZE, j==n ? nr : VTA_BLOCK_SIZE, 0);
                }
                else{//no remaining part, 16x16 full matrix
                    matrix_copy_transpose_partition_sw(b, right_info, l*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, 0, 0, VTA_BLOCK_SIZE, VTA_BLOCK_SIZE, 0);
                }

                //bias copy
                if(l == 0){//real bias
                    matrix_zero_sw(bias_info);
                    matrix_copy_partition_sw(bb, bias_info, i*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, 0, 0, i==m ? mr : VTA_BLOCK_SIZE, j==n ? nr : VTA_BLOCK_SIZE, 0);
                }

                #ifdef CACHING_ALL
                    flush_cache();
                #endif

                REG32(XVTA_CONTROL_ADDR_AP_CTRL) = 0x0;
                REG32(XVTA_CONTROL_ADDR_AP_CTRL) = 0x1;   // ap_start

                while(1){
                    int ap_done = vta_status() & 0x0002;
                    if(ap_done){
                    break;
                    }
                }

                // store output
                if(l == iter_k - 1 || iter_k == 1){//k loop last iteration, store output to real result c
                    matrix_copy_partition_sw(output_info, c, 0, 0, i*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, i==m ? mr : VTA_BLOCK_SIZE ,j==n ? nr : VTA_BLOCK_SIZE, 0);
                }
            }
        }
    }
    return NULL;
}