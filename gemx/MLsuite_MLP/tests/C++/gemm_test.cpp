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
// g++ -g -O0 -std=c++11 -D FLOW_HLS_CSIM -I ./C++/src/ -D GEMX_dataType=short -D GEMX_ddrWidth=32 -D GEMX_gemmMBlocks=4 -D GEMX_gemmKBlocks=4 -D GEMX_gemmNBlocks=4 -I$XILINX_XRT/include -L$XILINX_XRT/lib  -lz -lxilinxopencl -lstdc++ -lrt -pthread -Wl,--rpath=$XILINX_XRT/lib tests/C++/gemm_test.cpp C++/src/xcl2/xcl2.cpp -o ./gemm_test.exe -lxilinxopencl 2>&1 | tee log
// Example command for running this application
// ./gemm_test.exe ../out_sw_emu/gemx.xclbin 2097152 128 128 128 128 128 128 128 1 0 A05 B05 C05 X05

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <stdio.h>  // fgets for popen

#include "gemm_host.h"

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
            <<  "  gemm_test.exe <path/gemx.xclbin> dev_buf_size_in_bytes [M K N  LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX]\n"
            <<  "  For each instruction, [M K N  LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX] could not be missing\n"
            <<  "  Examples:\n"
            <<  "    ./gemm_test.exe   ...path-to-xclbin/gemx.xclbin 2097152  256 256 256  256 256 256 256 1 0 A B C X\n";
        exit(2);
    }
    if(((argc - 3) % 13 != 0) && argc > 13) {
        cerr << "For each instruction, [M K N LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX] could not be missing\n";
        exit(2);
    }
    unsigned int l_argIdx = 1;
    string l_xclbinFile(argv[l_argIdx++]);
    unsigned int l_buf_sz = atoi(argv[l_argIdx++]);
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

    string l_handleA[l_instrCount], l_handleB[l_instrCount], l_handleC[l_instrCount], l_handleX[l_instrCount];

    printf("GEMX-gemm C++ API example using library memory allocation \n",
            l_xclbinFile.c_str());
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

        l_handleA[index] = argv[l_argIdx++];
        l_handleB[index] = argv[l_argIdx++];
        l_handleC[index] = argv[l_argIdx++];
        l_handleX[index] = argv[l_argIdx++];

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


    GEMMHost<string> gemm_host( l_xclbinFile, "gemxKernel_0");
    gemm_host.AllocProgBuf(l_buf_sz);

    vector < Mat<GEMX_dataType> > A_vec, B_vec, C_vec;
    vector < Mat<int> > X_vec;
    for(int index = 0; index<l_instrCount; index++){
        unsigned int l_aSize = l_m[index] * l_lda[index] * sizeof(GEMX_dataType);
        unsigned int l_bSize = l_k[index] * l_ldb[index] * sizeof(GEMX_dataType);
        unsigned int l_xSize = l_m[index] * l_ldx[index] * sizeof(int);
        unsigned int l_cSize = l_m[index] * l_ldc[index] * sizeof(GEMX_dataType);

        GEMX_dataType* l_matA_addr = (GEMX_dataType*) gemm_host.AddDevBuf(l_handleA[index], l_aSize);
        GEMX_dataType* l_matB_addr = (GEMX_dataType*) gemm_host.AddDevBuf(l_handleB[index], l_bSize);
        int* l_matX_addr = (int*) gemm_host.AddDevBuf(l_handleX[index], l_xSize);
        GEMX_dataType* l_matC_addr = (GEMX_dataType*) gemm_host.AddDevBuf(l_handleC[index], l_cSize);

        cout<<"l_matA_addr "<<l_matA_addr<<" l_matB_addr "<<l_matB_addr<<" l_matX_addr "<<l_matX_addr<<" l_matC_addr"<<l_matC_addr<<"\n";

        Mat<GEMX_dataType> l_matA(l_m[index], l_n[index], l_lda[index], l_matA_addr );
        Mat<GEMX_dataType> l_matB(l_k[index], l_n[index], l_ldb[index], l_matB_addr );
        Mat<int> l_matX(l_m[index], l_n[index], l_ldx[index], l_matX_addr );
        Mat<GEMX_dataType> l_matC(l_m[index], l_n[index], l_ldc[index], l_matC_addr );

        A_vec.push_back(l_matA);
        B_vec.push_back(l_matB);
        X_vec.push_back(l_matX);
        C_vec.push_back(l_matC);
        l_matA.fillModRange(-100, 0);
        //l_matA.fillModRange(-3, 3);
        //l_matA.fillModRange(-10, 10);
        //l_matA.fillMod(100, index);
        cout << l_matA.rows() << " " << l_matA.cols() << " " << l_matA.ld() << " " << l_matA.buf_sz() << endl;
        gemm_host.SendDevBuf(l_handleA[index], false);
        l_matB.fillModRange(-10, 10);
        //l_matB.fillMod(100, index);
        gemm_host.SendDevBuf(l_handleB[index], false);
        l_matX.fillModRange(-100,100);
        gemm_host.SendDevBuf(l_handleX[index], false );

        gemm_host.SendDevBuf(l_handleC[index], false);
        gemm_host.AddGEMMDevOp ( l_handleA[index], l_handleB[index], l_handleC[index], l_handleX[index], l_m[index], l_k[index], l_n[index],
                l_lda[index],l_ldb[index],l_ldc[index],l_ldx[index],l_postScaleVal[index],l_postScaleShift[index]);
    }

    gemm_host.ExecuteDev();

    // Check result
    for(int index = 0; index<l_instrCount; index++){
        cout << "Checking instr " << index << ": " << l_handleA[index] << " " << l_handleB[index] << " " << l_handleC[index] << endl;
        GEMX_dataType *l_matC_cpu_addr = (GEMX_dataType*)malloc(l_m[index]*l_ldc[index]*sizeof(GEMX_dataType));
        Mat<GEMX_dataType> l_matC_cpu (l_m[index], l_n[index], l_ldc[index], l_matC_cpu_addr);
        l_matC_cpu.multiplyAddScale(A_vec[index], B_vec[index], X_vec[index], l_postScaleVal[index],l_postScaleShift[index]);
        void* l_ptr;
        l_ptr = gemm_host.GetDevBuf(l_handleA[index], true);
        cout << "Read A back successful" << endl;
        l_ptr = gemm_host.GetDevBuf(l_handleB[index], true);
        cout << "Read B back successful" << endl;
        l_ptr = gemm_host.GetDevBuf(l_handleX[index], true);
        cout << "Read X back successful" << endl;
        l_ptr = gemm_host.GetDevBuf(l_handleC[index], true, true);
        gemm_host.Wait();

        bool l_res =l_matC_cpu.cmp(l_TolRel, l_TolAbs, C_vec[index]);
        if (l_res) {
            cout << "INFO: Test pass." << endl;
        }
        free(l_matC_cpu_addr);
        //cpu_result[l_handleC[index]] = l_matC_cpu;
    }

    return EXIT_SUCCESS;
}



