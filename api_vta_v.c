#include "platform_info.h"
#include "xvta_hw.h"
#include "ervp_printf.h"
#include "ervp_profiling.h"
#include "ervp_printf_section.h"
#include "ervp_variable_allocation.h"
#include "ervp_mmio_util.h"
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

#define VTA_DATA __attribute__ ((aligned(0x1000)))
#define VTA_BLOCK_SIZE 16

uint64_t insns_f[12+(((FG_LEF_R/VTA_BLOCK_SIZE) + ((FG_LEF_R%VTA_BLOCK_SIZE) ? 1 : 0))*2)] VTA_DATA ;
        // 12+left array 행방향 블록 수. 12+M*2

uint32_t uops[16*((FG_LEF_R/VTA_BLOCK_SIZE) + ((FG_LEF_R%VTA_BLOCK_SIZE) ? 1 : 0))] VTA_DATA ;

int vta_status(void)
{
    return REG32(XVTA_CONTROL_ADDR_AP_CTRL);
}

#define BUFFER_SIZE_INP (16*16*((FG_LEF_R/VTA_BLOCK_SIZE) + ((FG_LEF_R%VTA_BLOCK_SIZE) ? 1 : 0))*((FG_LEF_C/VTA_BLOCK_SIZE) + ((FG_LEF_C%VTA_BLOCK_SIZE) ? 1 : 0)))
#define BUFFER_SIZE_WGT (16*16*((FG_RIG_R/VTA_BLOCK_SIZE) + ((FG_RIG_R%VTA_BLOCK_SIZE) ? 1 : 0))*((FG_RIG_C/VTA_BLOCK_SIZE) + ((FG_RIG_C%VTA_BLOCK_SIZE) ? 1 : 0)))
#define BUFFER_SIZE_ACC (16*16*((FG_LEF_R/VTA_BLOCK_SIZE) + ((FG_LEF_R%VTA_BLOCK_SIZE) ? 1 : 0))*((FG_RIG_C/VTA_BLOCK_SIZE) + ((FG_RIG_C%VTA_BLOCK_SIZE) ? 1 : 0))) 

static uint32_t left_buffer[BUFFER_SIZE_INP] VTA_DATA;
static uint32_t right_buffer[BUFFER_SIZE_WGT] VTA_DATA;
static uint32_t bias_buffer[BUFFER_SIZE_ACC] VTA_DATA;
static uint32_t output_buffer[BUFFER_SIZE_ACC] VTA_DATA;

static ErvpMatrixInfo left_buffer_info;
static ErvpMatrixInfo right_buffer_info;
static ErvpMatrixInfo bias_buffer_info;
static ErvpMatrixInfo output_buffer_info;

static void setup_vta_var()
{
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
{//static void __attribute__ ((constructor))는 프로그램 실행 시 자동으로 호출되는 함수. 
  //GCC에서 제공하는 기능으로, main 호출 전에 실행됨.
    matrix_generate_info(MATRIX_DATATYPE_SINT32,16*((FG_LEF_R/VTA_BLOCK_SIZE) + ((FG_LEF_R%VTA_BLOCK_SIZE) ? 1 : 0))*((FG_LEF_C/VTA_BLOCK_SIZE) + ((FG_LEF_C%VTA_BLOCK_SIZE) ? 1 : 0)),16,left_buffer,&left_buffer_info);
    matrix_generate_info(MATRIX_DATATYPE_SINT32,16*((FG_RIG_R/VTA_BLOCK_SIZE) + ((FG_RIG_R%VTA_BLOCK_SIZE) ? 1 : 0))*((FG_RIG_C/VTA_BLOCK_SIZE) + ((FG_RIG_C%VTA_BLOCK_SIZE) ? 1 : 0)),16,right_buffer,&right_buffer_info);
    matrix_generate_info(MATRIX_DATATYPE_SINT32,16*((FG_LEF_R/VTA_BLOCK_SIZE) + ((FG_LEF_R%VTA_BLOCK_SIZE) ? 1 : 0))*((FG_RIG_C/VTA_BLOCK_SIZE) + ((FG_RIG_C%VTA_BLOCK_SIZE) ? 1 : 0)),16,bias_buffer,&bias_buffer_info);//
    matrix_generate_info(MATRIX_DATATYPE_SINT32,16*((FG_LEF_R/VTA_BLOCK_SIZE) + ((FG_LEF_R%VTA_BLOCK_SIZE) ? 1 : 0))*((FG_RIG_C/VTA_BLOCK_SIZE) + ((FG_RIG_C%VTA_BLOCK_SIZE) ? 1 : 0)),16,output_buffer,&output_buffer_info);
    setup_vta_var();
}
void setup_insnsf_uops(int M, int N, int K){
    //////////////////////////////////////////////
    /////////field to define instruction./////////
    ///////// [LO & HI] is 1 instruction.///////// 
    //////////////////////////////////////////////
    
    insns_f[0]  = LSINST_LO(0,0,0,0,0,0);
    insns_f[1]  = LSINST_HI(1,M*VTA_BLOCK_SIZE,1,0,0,0,0);  //uop
    insns_f[2]  = LSINST_LO(0,4,3,0,0,0);
    insns_f[3]  = LSINST_HI(1,M*N*VTA_BLOCK_SIZE,1,0,0,0,0);//acc
    insns_f[4]  = LSINST_LO(0,2,1,0,0,0);
    insns_f[5]  = LSINST_HI(1,N*K,1,0,0,0,0);//wgt
    insns_f[6]  = LSINST_LO(0,8,2,0,0,0);
    insns_f[7]  = LSINST_HI(1,K*M*VTA_BLOCK_SIZE,1,0,0,0,0);//inp

    // 건너뛰지 않고 읽기 ? x_stride가 1이라는 의미.
    // uop -> M*VTA_BLOCK_SIZE 만큼의 line 수를 1개씩 건너뛰지 않고 읽어옴.
    // acc -> M*N*VTA_BLOCK_SIZE(출력 크기랑 동일) 만큼의 line 수를 1개씩 건너뛰지 않고 읽어옴.
    // wgt -> n_wgt_blocks 만큼의 16*16 블록을 1개씩 건너뛰지 않고 읽어옴.
    // inp -> n_rows(K*M*VTA_BLOCK_SIZE) 만큼의 line 수를 1개씩 건너뛰지 않고 읽어옴.
    for(int i=0; i<M; i++){
        insns_f[8+(i*2)]  = GMINST_LO(2,(i == 0 ? (M == 1 ? 9 : 1) : (i == M-1 ?  8 : 4)),0,16*i,16+(16*i),N,K,0);
        ///////////////////////////////////////////
        // [GEMM INSTRUCTION LOWER BITS SETTING] //
        ///////////////////////////////////////////
        // 1. [opcode] : 2
        // 2. [dependency flag setting] :
        //////////////////////////////////////////////////////////////       
        // if(처음 GEMM 명령){
        //     if(필요한 GEMM 명령이 오직 1개){//M == 1
        //         dept_flag = 9; // 1001 
        //         // -> pop_prev(앞의 load 완료되었는지 확인)
        //         // -> push_next(gemm 종료 후 store가 꺼낼 수 있게 함)
        //     }
        //     else
        //         dept_flag = 1;  // 0001
        //         // -> pop_prev(앞의 load 완료되었는지 확인)
        // }
        // else{// 중간 명령 or 마지막 명령
        //     if(마지막 GEMM 명령){//M - 1
        //         dept_flag = 8; // 1000
        //         // -> push_next(gemm 종료 후 store가 꺼낼 수 있게 함)
        //     }
        //     else{// 중간 명령
        //         dept_flag = 4; // 0010
        //         // -> 이어나가게 함. 
        //     }
        // }
        //////////////////////////////////////////////////////////////
        // 3. [uop begin]   : 16* i
        // 4. [uop end]     : 16 + (16 * i)
        // 5. [lpext0]      : N
        // 6. [lpext1]      : K
        insns_f[8+(i*2)+1]= GMINST_HI(16,0,0,16,K,1);
        ///////////////////////////////////////////
        // [GEMM INSTRUCTION UPPER BITS SETTING] //
        ///////////////////////////////////////////
        // 1. [accidx0]     : 16
        // 2. [accidx1]     : 0
        // 3. [inpidx0]     : 0
        // 4. [inpidx1]     : 16
        // 5. [wgtidx0]     : K
        // 6. [wgtidx1]     : 1
        
        //////////////////////////////////////////////
        /////////field to define uops        ///////// 
        //////////////////////////////////////////////
        uops[16*i]   = UOP((i*16*N) + 0,  (i*(16*K)) + 0,  0); //acc,inp,wgt
        uops[16*i+1] = UOP((i*16*N) + 1,  (i*(16*K)) + 1,  0);
        uops[16*i+2] = UOP((i*16*N) + 2,  (i*(16*K)) + 2,  0);
        uops[16*i+3] = UOP((i*16*N) + 3,  (i*(16*K)) + 3,  0);
        uops[16*i+4] = UOP((i*16*N) + 4,  (i*(16*K)) + 4,  0);
        uops[16*i+5] = UOP((i*16*N) + 5,  (i*(16*K)) + 5,  0);
        uops[16*i+6] = UOP((i*16*N) + 6,  (i*(16*K)) + 6,  0);
        uops[16*i+7] = UOP((i*16*N) + 7,  (i*(16*K)) + 7,  0);
        uops[16*i+8] = UOP((i*16*N) + 8,  (i*(16*K)) + 8,  0);
        uops[16*i+9] = UOP((i*16*N) + 9,  (i*(16*K)) + 9,  0);
        uops[16*i+10]= UOP((i*16*N) + 10, (i*(16*K)) + 10, 0);
        uops[16*i+11]= UOP((i*16*N) + 11, (i*(16*K)) + 11, 0);
        uops[16*i+12]= UOP((i*16*N) + 12, (i*(16*K)) + 12, 0);
        uops[16*i+13]= UOP((i*16*N) + 13, (i*(16*K)) + 13, 0);
        uops[16*i+14]= UOP((i*16*N) + 14, (i*(16*K)) + 14, 0);
        uops[16*i+15]= UOP((i*16*N) + 15, (i*(16*K)) + 15, 0);
    }
    //////////////////////////////////////////////
    /////////field to define uops        /////////
    ///////// [LO & HI] is 1 instruction.///////// 
    //////////////////////////////////////////////
    insns_f[8+(2*M)] = LSINST_LO(1,5,4,0,0,0);
    insns_f[8+(2*M)+1] = LSINST_HI(1,M*N*VTA_BLOCK_SIZE,1,0,0,0,0);
    insns_f[8+(2*M)+2] = FNINST_LO(3,2,0);
    insns_f[8+(2*M)+3] = FNINST_HI(0);
}

void setup_ins_and_uops()
{
    // Data signal of insn_count
    REG32(XVTA_CONTROL_ADDR_INSN_COUNT_DATA) = sizeof(insns_f)/16;
    // Data signal of insns
    REG64(XVTA_CONTROL_ADDR_INSNS_DATA)   = insns_f;
    // Data signal of uops
    REG32(XVTA_CONTROL_ADDR_UOPS_DATA)    = uops;
}

ervp_mop_wait_fx_custom_t Tiled_block_gemm(ervp_mop_mapping_custom_t *mop_mapping, const ErvpMatrixInfo *a, const ErvpMatrixInfo *b, const ErvpMatrixInfo *bb, ErvpMatrixInfo *c)
{
    //////////////////////////// Blocked Matrix Multiplication ////////////////////////////
    // a -> left matrix    b -> right matrix    bb -> bias matrix    c -> output matrix  //
    // 1. 행렬의 size 계산 및 인자 계산
    // 2. INSTRUCTION, UOP 설정
    // 3. INP, WGT, ACC, OUT Flat버퍼 값 할당 및 zero padding
    // 4. VTA 실행
    // 5. VTA 결과(OUT 버퍼)를 C로 복사.
    ////////////////////////////////////////////////////////////////////////////////////////
    int LR = (a->num_row)/VTA_BLOCK_SIZE; // input left row
    int LC = (a->num_col)/VTA_BLOCK_SIZE; // input left col
    int RR = (b->num_row)/VTA_BLOCK_SIZE; // input right row
    int RC = (b->num_col)/VTA_BLOCK_SIZE; // input right col

    int LR_R = (a->num_row)%VTA_BLOCK_SIZE; // input left row remain
    int LC_R = (a->num_col)%VTA_BLOCK_SIZE; // input left col remain
    int RR_R = (b->num_row)%VTA_BLOCK_SIZE; // input right row remain
    int RC_R = (b->num_col)%VTA_BLOCK_SIZE; // input right col remain

    int N = RC+(RC_R ? 1 : 0);
    int M = LR+(LR_R ? 1 : 0);
    int K = LC+(LC_R ? 1 : 0);

    int n_rows = K*M*VTA_BLOCK_SIZE;
    int n_wgt_blocks = N*K;

    setup_insnsf_uops(M, N, K);
    setup_ins_and_uops();

    ErvpMatrixInfo *left_info = &left_buffer_info;
    ErvpMatrixInfo *right_info = &right_buffer_info;
    ErvpMatrixInfo *bias_info = &bias_buffer_info;
    ErvpMatrixInfo *output_info = &output_buffer_info;//수정?
    //matrix_zero대신, remain 있는 부분만 matrix_zero_sw_custom로 0으로 초기화.
    for(int i=0; i<(N > M ? N : M); i++){ 
        for(int j=0; j<K; j++){
            if(i<M){
                if(j==LC || i==LR){//input matrix 복사. 남는 부분.
                    matrix_zero_sw_custom(left_info, (i*K+j)*VTA_BLOCK_SIZE, 0, VTA_BLOCK_SIZE, VTA_BLOCK_SIZE);
                    matrix_copy_partition_sw(a, left_info, i*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, ((i*K)+j)*VTA_BLOCK_SIZE, 0, i==LR ? LR_R : VTA_BLOCK_SIZE, j==LC ? LC_R : VTA_BLOCK_SIZE, 0);
                }
                else{//input matrix 복사. 16*16 full block.
                    matrix_copy_partition_sw(a, left_info, i*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, ((i*K)+j)*VTA_BLOCK_SIZE, 0, VTA_BLOCK_SIZE, VTA_BLOCK_SIZE, 0);
                }
            }
            
            if(i<N){
                if(j==RR || i==RC){//weight matrix 복사. 남는 부분.
                    matrix_zero_sw_custom(right_info, (i*K+j)*VTA_BLOCK_SIZE, 0, VTA_BLOCK_SIZE, VTA_BLOCK_SIZE);
                    matrix_copy_transpose_partition_sw(b, right_info, j*VTA_BLOCK_SIZE, i*VTA_BLOCK_SIZE, ((i*K)+j)*VTA_BLOCK_SIZE, 0, j==RR ? RR_R : VTA_BLOCK_SIZE, i==RC ? RC_R : VTA_BLOCK_SIZE, 0);
                }
                else{//weight matrix 복사. 16*16 full block.
                    matrix_copy_transpose_partition_sw(b, right_info, j*VTA_BLOCK_SIZE, i*VTA_BLOCK_SIZE, ((i*K)+j)*VTA_BLOCK_SIZE, 0, VTA_BLOCK_SIZE, VTA_BLOCK_SIZE, 0);
                } 
            }  
        }
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if(j==RC || i==LR){//remain 있는 부분
                matrix_zero_sw_custom(bias_info, (j+(i*N))*VTA_BLOCK_SIZE, 0, VTA_BLOCK_SIZE, VTA_BLOCK_SIZE);
                matrix_copy_partition_sw(bb, bias_info, i*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, ((i*N)+j)*VTA_BLOCK_SIZE, 0, i==LR ? LR_R : VTA_BLOCK_SIZE, j==RC ? RC_R : VTA_BLOCK_SIZE, 0);
            }
            else{//full block
                matrix_copy_partition_sw(bb, bias_info, i*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, ((i*N)+j)*VTA_BLOCK_SIZE, 0, VTA_BLOCK_SIZE, VTA_BLOCK_SIZE, 0);
            }
        }
    }
#ifdef CACHING_ALL
flush_cache();
#endif
    REG32(XVTA_CONTROL_ADDR_AP_CTRL) = 0x0;
    REG32(XVTA_CONTROL_ADDR_AP_CTRL) = 0x1;   // ap_start

    while(1)
    {
        int ap_done = vta_status() & 0x0002;
        if(ap_done){
            break;
        }
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){//flatten array(output_info) to c
            if(j==RC || i==LR){//remain 있는 부분
                matrix_copy_partition_sw(output_info, c, ((i*N)+j)*VTA_BLOCK_SIZE, 0, i*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, i==LR ? LR_R : VTA_BLOCK_SIZE, j==RC ? RC_R : VTA_BLOCK_SIZE, 0);
            }
            else{//full block
                matrix_copy_partition_sw(output_info, c, ((i*N)+j)*VTA_BLOCK_SIZE, 0, i*VTA_BLOCK_SIZE, j*VTA_BLOCK_SIZE, VTA_BLOCK_SIZE, VTA_BLOCK_SIZE, 0);
            }
        }
    }
    return NULL;
}
