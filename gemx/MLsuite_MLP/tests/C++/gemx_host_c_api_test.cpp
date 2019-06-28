/**********
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
* **********/
/**
 *  @brief Simple GEMM example of C++ API client interaction with GEMMX linear algebra accelerator on Xilinx FPGA 
 *
 *  $DateTime: 2019/01/18 08:31:34 $
 *  $Author: Xilinx $
 */

// Prerequisites:
//  - Compiled GEMX engine to bitstream accelerator kernel gemx.xclbin file
// Example command for Compiling this application:
// g++ -g -O0 -std=c++11 -DTEST_SDX=1 -D FLOW_HLS_CSIM -D GEMX_dataType=short -D GEMX_dataEqIntType=short -D GEMX_ddrWidth=32 -D GEMX_argInstrWidth=1 -D GEMX_numInstr=16 -D GEMX_argPipeline=2 -D GEMX_numKernels=1 -D GEMX_gemmMBlocks=4 -D GEMX_gemmKBlocks=4 -D GEMX_gemmNBlocks=4 -D GEMX_splitMesh=1 -D GEMX_keepMacBits=1 -D GEMX_macBits=48 -D GEMX_XdataType=int32_t  -D GEMX_XddrWidth=16 -I$XILINX_XRT/include -I ./C++/src -L$XILINX_XRT/lib -lz -lxilinxopencl -lstdc++ -lrt -pthread  tests/C++/gemx_host_c_api_test.cpp C++/src/gemx_host_c_api.cpp C++/src/xcl2/xcl2.cpp -o ./gemx_host_c_api_test.exe 2>&1 | tee log
// Example command for running this application
// ./gemx_host_c_api_test.exe ../../out_sw_emu/gemx.xclbin dev_buf_size 128 128 128 128 128 128 128 1 0 A05 B05 C05 X05


#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <stdio.h>  // fgets for popen
#include <assert.h>

#include "gemx_host_c_api.h"
#include "gemx_util.h"

using namespace std;
using namespace gemx;

bool checkDim(unsigned int p_Val, unsigned int p_Mod, unsigned int p_Min) {
    bool l_ok = true;
    if (p_Val % p_Mod != 0) {
        cerr << "ERROR: value " << p_Val << " must be multiple of " << p_Mod << "\n";
        l_ok = false;
    }
    if (p_Val < p_Min) {
        cerr << "ERROR: value " << p_Val << " must be at least " << p_Min << "\n";
        l_ok = false;
    }
    return(l_ok);
}

int main(int argc, char **argv)
{
    //############  UI and GEMM problem size  ############
    if (argc < 16) {
        cerr << "Usage:\n"
                <<  "  gemm.exe <path/gemx.xclbin> [M K N  LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX]\n"
                <<  "  For each instruction, [M K N  LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX] could not be missing\n"
                <<  "  Examples:\n"
                <<  "    gemm_test.exe   out_hw/gemx.xclbin  32 128 256  256 256 256 256 1 0 A B C X\n";
        exit(2);
    }
    if(((argc - 3) % 13 != 0) && argc > 13) {
        cerr << "For each instruction, [M K N LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX] could not be missing\n";
        exit(2);
    }
    unsigned int l_argIdx = 1;
    string l_xclbinFile(argv[l_argIdx++]);
                unsigned int l_progBufSz = atoi(argv[l_argIdx++]);
    const float  l_TolRel = 1e-3,  l_TolAbs = 1e-5;

    unsigned int l_instrCount = ((argc-3)/13>1)?((argc-3)/13):1; //number of instructions

    if(l_instrCount > 15){
        cerr << "  Too many instructions at same time\n";
        exit(2);
    }

    unsigned int l_ddrW = GEMX_ddrWidth;
    unsigned int l_m[l_instrCount];
    unsigned int l_k[l_instrCount];
    unsigned int l_n[l_instrCount];

    unsigned int l_lda[l_instrCount];
    unsigned int l_ldb[l_instrCount];
    unsigned int l_ldc[l_instrCount];
    unsigned int l_ldx[l_instrCount];
    
    int32_t l_postScaleVal[l_instrCount];
    int32_t l_postScaleShift[l_instrCount];
    int32_t l_postScale[l_instrCount];

    vector<char*> l_handleA, l_handleB, l_handleC, l_handleX;

    printf("GEMX-gemm C++ API example using accelerator image \n",
            l_xclbinFile);
    unsigned long int l_Ops[l_instrCount]; //operations carried out by each kernel
    l_argIdx = 3;
    for(int index = 0; index<l_instrCount; index++){
        l_m[index] = atoi(argv[l_argIdx++]);
        l_k[index] = atoi(argv[l_argIdx++]);
        l_n[index] = atoi(argv[l_argIdx++]);
        
        l_lda[index] = atoi(argv[l_argIdx++]);
        l_ldb[index] = atoi(argv[l_argIdx++]);
        l_ldc[index] = atoi(argv[l_argIdx++]);
        l_ldx[index] = atoi(argv[l_argIdx++]);
        
        l_postScaleVal[index] = atoi(argv[l_argIdx++]);
        l_postScaleShift[index] = atoi(argv[l_argIdx++]);
        l_postScale[index] = (l_postScaleVal[index] << 8) | (l_postScaleShift[index] & 0x000000ff);

        char* l_handle;
        l_handle = argv[l_argIdx++];
        l_handleA.push_back(l_handle);
        l_handle = argv[l_argIdx++];
        l_handleB.push_back(l_handle);
        l_handle = argv[l_argIdx++];
        l_handleC.push_back(l_handle);
        l_handle = argv[l_argIdx++];
        l_handleX.push_back(l_handle);

        assert(l_lda[index] >= l_k[index]);
        assert(l_ldb[index] >= l_n[index]);
        assert(l_ldc[index] >= l_n[index]);
        if (! (
                checkDim(l_m[index], l_ddrW*GEMX_gemmMBlocks, l_ddrW*GEMX_gemmMBlocks) &&
                checkDim(l_k[index], l_ddrW*GEMX_gemmKBlocks, l_ddrW*GEMX_gemmKBlocks) &&
                checkDim(l_n[index], l_ddrW*GEMX_gemmNBlocks, l_ddrW*GEMX_gemmNBlocks) &&
                checkDim(l_lda[index], l_ddrW, l_k[index]) &&
                checkDim(l_ldb[index], l_ddrW, l_n[index]) &&
                checkDim(l_ldc[index], l_ddrW, l_n[index])
        )) {
            return EXIT_FAILURE;
        }
    }
    

    //GEMMHost<string> gemm_host( l_xclbinFile, "gemxKernel_0",0);
    MakeStrGEMMHost(l_xclbinFile.c_str(), 1);
    AllocProgBuf(l_progBufSz,0);

    vector < Mat<GEMX_dataType> > A_vec, B_vec, C_vec;
    vector < Mat<int> > X_vec;
    for(int index = 0; index<l_instrCount; index++){
        unsigned int l_aSize = l_m[index] * l_lda[index] * sizeof(GEMX_dataType);
        unsigned int l_bSize = l_k[index] * l_ldb[index] * sizeof(GEMX_dataType);
        unsigned int l_xSize = l_m[index] * l_ldx[index] * sizeof(int);
        unsigned int l_cSize = l_m[index] * l_ldc[index] * sizeof(GEMX_dataType);

        GEMX_dataType* l_matA_addr = (GEMX_dataType*) AddDevBuf(l_handleA[index], l_aSize, 0);
        GEMX_dataType* l_matB_addr = (GEMX_dataType*) AddDevBuf(l_handleB[index],l_bSize, 0);
        int* l_matX_addr = (int*) AddDevBuf(l_handleX[index],l_xSize, 0);
        GEMX_dataType* l_matC_addr = (GEMX_dataType*) AddDevBuf(l_handleC[index],l_cSize, 0);
        Mat<GEMX_dataType> l_matA(l_m[index], l_n[index], l_lda[index], l_matA_addr );
        Mat<GEMX_dataType> l_matB(l_k[index], l_n[index], l_ldb[index], l_matB_addr );
        Mat<int> l_matX(l_m[index], l_n[index], l_ldx[index], l_matX_addr );
        Mat<GEMX_dataType> l_matC(l_m[index], l_n[index], l_ldc[index], l_matC_addr );

        A_vec.push_back(l_matA);
        B_vec.push_back(l_matB);
        X_vec.push_back(l_matX);
        C_vec.push_back(l_matC);
        l_matA.fillModRange(-100, 0);
        cout << l_matA.rows() << " " << l_matA.cols() << " " << l_matA.ld() << " " << l_matA.buf_sz() << endl;
        SendDevBuf(l_handleA[index], false, 0);
        l_matB.fillModRange(-10, 10);
        SendDevBuf(l_handleB[index], false, 0);
        l_matX.fillModRange(-100,100);
        SendDevBuf(l_handleX[index], false, 0);
        SendDevBuf(l_handleC[index], false, 0);
        AddGEMMDevOp(l_handleA[index], l_handleB[index], l_handleC[index], l_handleX[index], l_m[index], l_k[index], l_n[index], l_postScaleVal[index],l_postScaleShift[index], 0);
    }

    ExecuteDev(true, 0);

    // Check result
    for(int index = 0; index<l_instrCount; index++){
        cout << "Checking instr " << index << ": " << l_handleA[index] << " " << l_handleB[index] << " " << l_handleC[index] << endl;
        GEMX_dataType *l_matC_cpu_addr = (GEMX_dataType*)malloc(l_m[index]*l_ldc[index]*sizeof(GEMX_dataType));
        Mat<GEMX_dataType> l_matC_cpu (l_m[index], l_n[index], l_ldc[index], l_matC_cpu_addr);

        void* l_ptr;
        l_ptr = GetDevBuf(l_handleC[index], 0, true);

        l_matC_cpu.multiplyAddScale(A_vec[index], B_vec[index], X_vec[index], l_postScaleVal[index], l_postScaleShift[index]);
        bool l_res =l_matC_cpu.cmp(l_TolRel, l_TolAbs, C_vec[index]);
        if (l_res) {
                cout << "INFO: Test pass." << endl;
        }
        free(l_matC_cpu_addr);
        //cpu_result[l_handleC[index]] = l_matC_cpu;
    }

    return EXIT_SUCCESS;
}



