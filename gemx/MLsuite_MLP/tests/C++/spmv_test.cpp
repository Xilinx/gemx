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


// Example command for Compiling this application:
//g++ -g -O0 -std=c++11 -D FLOW_HLS_CSIM -I ./C++/src/ -D GEMX_dataType=float -D GEMX_ddrWidth=16 -D GEMX_spmvWidth=8 -D  GEMX_spmvMacGroups=12 -D GEMX_spmvNumCblocks=1024 -D GEMX_spmvColAddIdxBits=2 -D GEMX_spmvkVectorBlocks=2048 -D GEMX_runSpmv=1 -I$XILINX_XRT/include -L$XILINX_XRT/lib -lboost_iostreams -lz -lxilinxopencl -lstdc++ -lrt -pthread -Wl,--rpath=$XILINX_XRT/lib tests/C++/spmv_test.cpp C++/src/xcl2/xcl2.cpp -o ./spmv_test.exe -lxilinxopencl

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include "spmv_host.h"

using namespace std;
using namespace gemx;


template <typename t_FloatType>
void
spmv_ref(vector<MtxRow> l_rows , Mat<t_FloatType> &p_b, Mat<t_FloatType> &p_c, bool p_usePrelu) {
    t_FloatType l_val = 0;
    for (MtxRow &l_row : l_rows) {
        unsigned int row = l_row.getRow(),
                     col = l_row.getCol();
        l_val = l_row.getVal();
        p_c.getVal(row, 0) += l_val * p_b.getVal(col, 0);
    }
    if (p_usePrelu) {
        for (unsigned int i=0; i<p_c.rows(); ++i) {
            t_FloatType l_valRow = p_c.getVal(i,0);
            p_c.getVal(i,0) = (l_valRow<0)? 0: l_valRow;
        }
    }
    return;
}

bool
check(
        unsigned int &p_M,  // The check() modifies the dimensions when loading from a file
        unsigned int &p_K,
        unsigned int &p_Nnz,
        MtxFile &p_MtxFile
     ) {
    bool ok = true;        
    // m_C
    const unsigned int l_mEdge = GEMX_spmvWidth * GEMX_spmvMacGroups;
    // m_B
    const unsigned int l_kEdge = GEMX_ddrWidth;
    unsigned int t_mVectorBlocks = (1 << (16 - GEMX_spmvColAddIdxBits)) / GEMX_spmvWidth  / GEMX_spmvMacGroups / GEMX_ddrWidth;
    const unsigned int l_bBlockSize = GEMX_spmvWidth * GEMX_spmvkVectorBlocks * GEMX_ddrWidth;
    const unsigned int l_cBlockSize = GEMX_spmvWidth * GEMX_spmvMacGroups * t_mVectorBlocks * GEMX_ddrWidth;        
    if (!p_MtxFile.good() && (p_MtxFile.fileName() != "none")) {
        cerr << "ERROR: spmv  mtxFile " << p_MtxFile.fileName()
            << " must exist or use none for auto-generated data"
            << "\n";
        ok = false;
    }       
    // Use the file only
    if (p_MtxFile.good()) {
        if ((p_M != 0) || (p_K != 0) || (p_Nnz != 0)) {
            std::cerr << "ERROR: spmv  M, K, Nnz must be 0 when using mtx file: "
                << "  M " << p_M
                << "  K " << p_K
                << "  Nnz " << p_Nnz
                << "\n";
        }
        p_M = p_MtxFile.rows();
        p_K = p_MtxFile.cols();
        p_Nnz = p_MtxFile.nnz();

    } else {
        if (p_Nnz % GEMX_spmvWidth != 0) {
            std::cout << "INFO: spmv  Nnz " << p_Nnz << " must be multiple of GEMX_spmvWidth "
                << GEMX_spmvWidth << "\n";
            while(p_Nnz % GEMX_spmvWidth != 0){
                std::cout << "INFO: add one to given number of non-zero element\n";
                p_Nnz++;
            }
        }
        if (p_M % l_mEdge != 0) {
            std::cout << "INFO: spmv  M dimension " << p_M << " must be multiple of "
                << l_mEdge << "\n";
            while(p_M % l_mEdge != 0){
                std::cout << "INFO: add one to given m\n";
                p_M++;
            }    
        }
        if (p_K % l_kEdge != 0) {
            std::cout << "INFO: spmv  K dimension " << p_K << " must be multiple of "
                << l_kEdge << "\n";
            while(p_K % l_kEdge != 0){
                std::cout << "INFO: add one to given K\n";
                p_K++;
            }
        }  
    }

    if (p_Nnz == 0) {
        std::cerr << "ERROR: spmv  Nnz must be non-0, it is " << p_Nnz << "\n";
        ok = false;
    }
    unsigned int l_bBlocks = (p_K + l_bBlockSize - 1) / l_bBlockSize;
    unsigned int l_cBlocks = (p_M + l_cBlockSize - 1) / l_cBlockSize;
    unsigned int l_totalBlocks = l_bBlocks * l_cBlocks;
    unsigned int l_maxBlocks = GEMX_spmvNumCblocks;
    if (l_totalBlocks > l_maxBlocks) {
        std::cerr << "ERROR: spmv  total blocks " << l_totalBlocks << " is larger than max supported " << l_maxBlocks << "   Recompile the kernel with larger GEMX_spmvNumCblocks\n";
        ok = false;
    }
    return(ok);
}


int main(int argc, char **argv)
{
    if (argc < 5) {
        std::cerr << "Usage:\n"
            <<  "  spmv_test.exe <path/gemx.xclbin> dev_buf_size_in_bytes [M K Nnz [mtxFile]]\n"
            <<  "  Examples:\n"
            <<  "    spmv_test.exe   out_hw/gemx.xclbin  20971520 96 128 256 none\n"
            <<  "    spmv_test.exe   out_hw/gemx.xclbin  20971520 0 0 0 data/spmv/diag16.mtx\n";
        exit(2);
    }

    unsigned int l_argIdx=1;
    string l_xclbinFile(argv[l_argIdx]);
    unsigned int l_buf_sz = atoi(argv[++l_argIdx]);
    unsigned int l_M = atoi(argv[++l_argIdx]);
    unsigned int l_K = atoi(argv[++l_argIdx]);
    unsigned int l_NNZ = atoi(argv[++l_argIdx]);
    std::string l_mtxFileName("none");
    l_mtxFileName = argv[++l_argIdx];

    MtxFile l_mtxFile(l_mtxFileName);

    if (!check(l_M, l_K, l_NNZ,l_mtxFile)){ 
        return EXIT_FAILURE;
    }

    std::string l_handleA = "A";
    std::string l_handleB = "B";
    std::string l_handleC = "C";

    /////////////////////////////////////////////////////////
    SPMVDevHost<string> spmv_host( l_xclbinFile, "gemxKernel_0");
    // Allocate a contiguous host and device buffer(cl::buffer)
    cout<<"Allocating Contiguous Host side memory of size:"<<l_buf_sz<<endl;
    spmv_host.AllocProgBuf(l_buf_sz);


    typedef SpmvAd<GEMX_dataType> SpmvAdType; 


    unsigned int t_mVectorBlocks = (1 << (16 - GEMX_spmvColAddIdxBits)) / GEMX_spmvWidth  / GEMX_spmvMacGroups / GEMX_ddrWidth;

    unsigned int capacity_Cblocks = GEMX_spmvWidth * GEMX_spmvMacGroups * t_mVectorBlocks * GEMX_ddrWidth;
    unsigned int capacity_Bblocks = GEMX_spmvWidth * GEMX_spmvkVectorBlocks * GEMX_ddrWidth;
    unsigned int l_Cblocks = (l_M + capacity_Cblocks - 1) / capacity_Cblocks;
    unsigned int l_Bblocks = (l_K + capacity_Bblocks - 1) / capacity_Bblocks;        
    unsigned int l_numDescPages = (GEMX_spmvNumCblocks + SpmvAdesc::t_per4k - 1) / SpmvAdesc::t_per4k; 
    unsigned int l_numDescDdrWords = l_numDescPages * 4096 / sizeof(float) / GEMX_ddrWidth;
    unsigned int l_numPaddingDdrWords = GEMX_spmvNumCblocks * 4096 / sizeof(float) / GEMX_ddrWidth;      
    unsigned int l_aSize = (l_numDescDdrWords * GEMX_ddrWidth + l_NNZ * GEMX_ddrWidth / GEMX_spmvWidth + l_numPaddingDdrWords * GEMX_ddrWidth) * sizeof(GEMX_dataType);
    unsigned int l_bSize = l_K *  sizeof(GEMX_dataType);
    unsigned int l_cSize = l_M *  sizeof(GEMX_dataType);


    GEMX_dataType* l_matA_addr = (GEMX_dataType*) spmv_host.AddDevBuf(l_handleA, l_aSize);
    GEMX_dataType* l_matB_addr = (GEMX_dataType*) spmv_host.AddDevBuf(l_handleB, l_bSize);
    GEMX_dataType* l_matC_addr = (GEMX_dataType*) spmv_host.AddDevBuf(l_handleC, l_cSize);

    SpMat<GEMX_dataType,SpmvAdType> l_matA(l_M,l_K,l_NNZ,l_Bblocks,l_Cblocks,l_matA_addr);
    Mat<GEMX_dataType> l_matB(l_K, 1, 1, l_matB_addr);
    Mat<GEMX_dataType> l_matC(l_M, 1, 1, l_matC_addr);

    l_matB.fillModRange(5, 10);
    vector<MtxRow> l_rows;
    if (l_mtxFile.good()) {
        l_matA.fillFromVector(l_mtxFile.getRows(), capacity_Cblocks, capacity_Bblocks, GEMX_spmvWidth);
    } else {
        l_rows = l_matA.fillMod(17, capacity_Cblocks, capacity_Bblocks, GEMX_spmvWidth);
    }

    spmv_host.SendDevBuf(l_handleA, false);
    spmv_host.SendDevBuf(l_handleB, false);
    spmv_host.SendDevBuf(l_handleC, false);
    spmv_host.AddSPMVDevOp(l_handleA, l_handleB, l_handleC, l_M, l_K, l_NNZ, 0, GEMX_spmvNumCblocks, capacity_Cblocks, capacity_Bblocks);
    spmv_host.ExecuteDev();
    spmv_host.GetDevBuf(l_handleC, true, true);


    Mat<GEMX_dataType> l_matC_cpu (l_M, 1, 1);
    l_matC_cpu.fill(0);
    if (l_mtxFile.good()) {
        spmv_ref<GEMX_dataType>(l_mtxFile.getRows(), l_matB, l_matC_cpu, 0);
    } else {
        spmv_ref<GEMX_dataType>(l_rows,l_matB,l_matC_cpu,0);
    }
    
    const float  l_TolRel = 1e-3,  l_TolAbs = 1e-5;
    bool l_res =l_matC.cmp(l_TolRel, l_TolAbs,l_matC_cpu);
    if (l_res) {
            cout << "INFO: Test pass." << endl;
    }


}
