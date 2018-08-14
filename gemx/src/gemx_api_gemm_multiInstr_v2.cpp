/**********
 * Copyright (c) 2017, Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * **********/
/**
 *  @brief Simple GEMM example of C++ API client interaction with GEMMX linear algebra accelerator on Xilinx FPGA 
 *
 *  $DateTime: 2017/08/18 08:31:34 $
 *  $Author: jzejda $
 */

// Prerequisites:
//  - Boost installation (edit the Makefile with your boost path)
//  - Compiled C++ to bitstream accelerator kernel
//     - use "make run_hw"
//     - or get a pre-compiled copy of the out_hw/gemx.xclbin)
// Compile and run this API example with 4GEMM FPGA image on F1:
//   export s= 32
//   make GEMX_ddrWidth=$s GEMX_argInstrWidth=`expr 32/$s` GEMX_gemmMBlocks=8 GEMX_gemmKBlocks=8 GEMX_gemmNBlocks=8 GEMX_numKernels=4 out_host/gemx_api_gemm_multiInstr.exe
//   gemx_api_gemm_multiInstr.exe gemx.awsxclbin 512 512 512 512 512 512 A1 B1 C1 512 512 512 512 512 512 C1 B2 C2
//   it will run two matrix multiplications C1=A1*B1 C2=C1*B2 on each GEMM kernel, and calculate the performance.

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <stdio.h>  // fgets for popen

#include "host/gemm_host.h"

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
    if (argc < 15) {
        std::cerr << "Usage:\n"
                <<  "  gemx_api_gemm_multiInstr.exe <path/gemx.xclbin> [M K N  LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX]\n"
                <<  "  For each instruction, [M K N  LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX] could not be missing\n"
                <<  "  Examples:\n"
                <<  "    gemx_api_gemm_multiInstr_v2.exe   out_hw/gemx.xclbin  32 128 256  256 256 256 256 1 0 A B C X\n";
        exit(2);
    }
    if(((argc - 2) % 13 != 0) && argc > 13) {
        cerr << "For each instruction, [M K N LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX] could not be missing\n";
        exit(2);
    }
    unsigned int l_argIdx = 1;
    string l_xclbinFile(argv[l_argIdx]);
    const float  l_TolRel = 1e-3,  l_TolAbs = 1e-5;

    unsigned int l_instrCount = ((argc-2)/13>1)?((argc-2)/13):1; //number of instructions

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

    printf("GEMX-gemm C++ API example using accelerator image \n",
            l_xclbinFile.c_str());
    unsigned long int l_Ops[l_instrCount]; //operations carried out by each kernel


    
        l_argIdx = 2;
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
    

    GEMMHost<std::string> gemm_host( l_xclbinFile, "gemxKernel_0", "kcu1500");

    vector < shared_ptr< Mat<GEMX_dataType> > > A_vec, B_vec, C_vec;
    vector < shared_ptr< Mat<int> > > X_vec;
    for(int index = 0; index<l_instrCount; index++){
        shared_ptr< Mat<GEMX_dataType> > l_matC = shared_ptr<  Mat<GEMX_dataType> > ( new Mat<GEMX_dataType> ( l_m[index], l_n[index], l_ldc[index] ) );
        shared_ptr< Mat<int> > l_matX = shared_ptr<  Mat<int>> ( new  Mat<int> (l_m[index], l_n[index], l_ldx[index] ) );
        shared_ptr< Mat<GEMX_dataType> > l_matA = shared_ptr<  Mat<GEMX_dataType>> ( new  Mat<GEMX_dataType> (l_m[index], l_k[index], l_lda[index] ) );
        shared_ptr< Mat<GEMX_dataType> > l_matB = shared_ptr<  Mat<GEMX_dataType>> ( new  Mat<GEMX_dataType> ( l_k[index], l_n[index], l_ldb[index]));

        A_vec.push_back(l_matA);
        B_vec.push_back(l_matB);
        X_vec.push_back(l_matX);
        gemm_host.SendToFPGA(l_handleC[index], l_matC->data(), l_matC->buf_sz());
        C_vec.push_back(l_matC);
        l_matX->fillModRange(-100,100);
        gemm_host.SendToFPGA(l_handleX[index], l_matX->data(), l_matX->buf_sz() );

        //l_matB->fillModRange(-3, 3);
        l_matB->fillModRange(-10, 10);
        //l_matB->fillMod(100, index);
        gemm_host.SendToFPGA(l_handleB[index], l_matB->data(), l_matB->buf_sz());
        l_matA->fillModRange(-100, 0);
        //l_matA->fillModRange(-3, 3);
        //l_matA->fillModRange(-10, 10);
        //l_matA->fillMod(100, index);
        cout << l_matA->rows() << " " << l_matA->cols() << " " << l_matA->ld() << " " << l_matA->buf_sz() << endl;
        gemm_host.SendToFPGA(l_handleA[index], l_matA->data(), l_matA->buf_sz());

        gemm_host.AddGEMMOp ( l_handleA[index], l_handleB[index], l_handleC[index], l_handleX[index], l_m[index], l_k[index], l_n[index],
			      l_lda[index],l_ldb[index],l_ldc[index],l_ldx[index],l_postScaleVal[index],l_postScaleShift[index]);
    }

    gemm_host.Execute();

    // Check result
    for(int index = 0; index<l_instrCount; index++){
        std::cout << "Checking instr " << index << ": " << l_handleA[index] << " " << l_handleB[index] << " " << l_handleC[index] << std::endl;
        Mat<GEMX_dataType> l_matC_cpu ( l_m[index], l_n[index], l_ldc[index]);

        gemm_host.GetMat(l_handleC[index],true);
        gemm_host.GetMat(l_handleA[index],true);
        gemm_host.GetMat(l_handleB[index],true);
        gemm_host.GetMat(l_handleX[index],true);

        l_matC_cpu.multiplyAddScale(*A_vec[index], *B_vec[index], *X_vec[index], l_postScaleVal[index],l_postScaleShift[index]);
        l_matC_cpu.cmp(l_TolRel, l_TolAbs, *C_vec[index]);
        //cpu_result[l_handleC[index]] = l_matC_cpu;
    }

    return EXIT_SUCCESS;
}



