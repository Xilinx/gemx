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
//g++ -g -O0 -std=c++11 -D FLOW_HLS_CSIM -I ./C++/src/ -D GEMX_dataType=float -D GEMX_ddrWidth=16 -D GEMX_uspmvStages=1 -D GEMX_uspmvInterleaves=8 -D GEMX_uspmvNnzVectorBlocks=62500 -D GEMX_uspmvMvectorBlocks=152 -D GEMX_runUspmv=1 -I$XILINX_XRT/include -L$XILINX_XRT/lib -lboost_iostreams -lz -lxilinxopencl -lstdc++ -lrt -pthread -Wl,--rpath=$XILINX_XRT/lib tests/C++/uspmv_test.cpp C++/src/xcl2/xcl2.cpp -o ./uspmv_test.exe -lxilinxopencl

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
#include "uspmv_host.h"


using namespace std;
using namespace gemx;


bool
check(
        unsigned int *p_m,
        unsigned int *p_k,
        unsigned int *p_nnz,
        vector<MtxFile> &p_mtxFile,
        unsigned int numInstr
     ) {
    bool ok = true;
    const unsigned int l_kEdge = GEMX_ddrWidth;
    const unsigned int l_mEdge = GEMX_ddrWidth * GEMX_uspmvInterleaves;
    const unsigned int l_maxNnz = GEMX_ddrWidth * GEMX_uspmvNnzVectorBlocks;
    const unsigned int l_maxM = GEMX_ddrWidth * GEMX_uspmvMvectorBlocks; 

    for (unsigned int i=0; i<numInstr; ++i) { 
        if (!p_mtxFile[i].good() && (p_mtxFile[i].fileName() != "none")) {
            cerr << "ERROR: spmv  mtxFile " << p_mtxFile[i].fileName()
                << " must exist or use none for auto-generated data"
                << "\n";
            ok = false;
        }         
        // Use the file only
        if (p_mtxFile[i].good()) {
            if ((p_m[i] != 0) || (p_k[i] != 0) || (p_nnz[i] != 0)) {
                cerr << "ERROR: spmv  M, K, Nnz must be 0 when using mtx file: "
                    << "  M " << p_m[i]
                    << "  K " << p_k[i]
                    << "  Nnz " << p_nnz[i]
                    << "\n";
            }
            p_m[i] = p_mtxFile[i].rows();
            p_k[i] = p_mtxFile[i].cols();
            p_nnz[i] = p_mtxFile[i].nnz();              
        } 
        if (p_m[i] % l_mEdge != 0) {
            cout << "INFO: spmv  M dimension " << p_m[i] << " must be multiple of "
                << l_mEdge << "\n";
            while(p_m[i] % l_mEdge != 0){
                cout << "INFO: add one to given m\n";
                p_m[i]++;
            }    
        }
        if (p_k[i] % l_kEdge != 0) {
            cout << "INFO: spmv  K dimension " << p_k[i] << " must be multiple of "
                << l_kEdge << "\n";
            while(p_k[i] % l_kEdge != 0){
                cout << "INFO: add one to given K\n";
                p_k[i]++;
            }
        }

        if (p_nnz[i] % GEMX_ddrWidth != 0) {
            std::cout << "INFO: spmv  nnz dimension " << p_nnz[i] << " must be multiple of "
                << GEMX_ddrWidth << "\n";
            while(p_nnz[i] % GEMX_ddrWidth != 0){
                std::cout << "INFO: add one to given nnz\n";
                p_nnz[i]++;
            }
        }   

        if (p_nnz[i] == 0) {
            cerr << "ERROR: spmv  Nnz must be non-0, it is " << p_nnz[i] << "\n";
            ok = false;
        }
        if (p_nnz[i] > l_maxNnz) {
            cerr << "ERROR: spmv Nnz " << p_nnz[i] << "must be less than or equal to " << l_maxNnz << "\n";
            ok = false;
        }
        if (p_m[i] > l_maxM) {
            cerr << "ERROR: spmv M dimension " << p_m[i] << "must be less than or equal to " << l_maxM << "\n";
            ok = false;
        }

        p_mtxFile[i].setIndex(p_m[i],p_k[i],p_nnz[i]);
    }
    return(ok);
}

template<typename t_FloatType, typename t_IdxType, unsigned int t_Stages, unsigned int t_DdrWidth>
void
multiplySingle(UspMat<t_FloatType, t_IdxType> &p_a, vector<t_FloatType> &p_b, unsigned int p_mtxId, vector<t_FloatType> &p_c){
    unsigned int l_mtxBaseIdx = 0;
    for (unsigned int i=0; i<p_mtxId; ++i) {
        l_mtxBaseIdx += p_a.getNnzs(i)*2;
    }
    unsigned int l_nnz = p_a.getNnzs(p_mtxId);
    for (unsigned int i=0; i<l_nnz; ++i) {
        unsigned int l_col = p_a.getIdx(l_mtxBaseIdx*2 + i*2);
        unsigned int l_row = p_a.getIdx(l_mtxBaseIdx*2+ i*2+1);
        t_FloatType l_val = p_a.getVal(l_mtxBaseIdx+l_nnz+i);
        t_FloatType l_b = (p_b[l_col]<0)? (p_a.getPrelu(p_mtxId-1) * p_b[l_col]): p_b[l_col];
        p_c[l_row] += l_val * l_b;
    }
}

template<typename t_FloatType, typename t_IdxType, unsigned int t_Stages, unsigned int t_DdrWidth>
void
spmm_ref(UspMat<t_FloatType, t_IdxType> &p_a, Mat<t_FloatType> &p_b, unsigned int p_numRuns, Mat<t_FloatType> &p_c) {
    //define vector arrays for storing intermediate SPMV results from each stage spmv engine
    array<vector<t_FloatType>, GEMX_uspmvStages> l_immVectors;
    for (unsigned int i=0; i<p_numRuns; ++i) {
        for (unsigned int i=0; i<GEMX_uspmvStages; ++i) {
            unsigned int l_rows = p_a.getRows(i);
            l_immVectors[i].resize(l_rows);
            for (unsigned int j=0; j<l_rows; ++j) {
                l_immVectors[i][j] = 0;
            }
        }    
        //compute multiplication with input p_b to get first intermediate vector
        unsigned int l_nnz = p_a.getNnzs(0);
        for (unsigned int j=0; j<l_nnz; ++j) {
            unsigned int l_col = p_a.getIdx(j*2);
            unsigned int l_row = p_a.getIdx(j*2+1);
            t_FloatType l_val = p_a.getVal(l_nnz +j);
            l_immVectors[0][l_row] += l_val * p_b.getVal(i, l_col);    
        }
        for (unsigned int j=1; j<GEMX_uspmvStages; ++j) {
            multiplySingle<t_FloatType, t_IdxType, t_Stages, t_DdrWidth>(p_a, l_immVectors[j-1], j, l_immVectors[j]);
        }
        //assign l_immVectors[t_Stages-1] to p_c
        for (unsigned int l_cCol=0; l_cCol<p_a.getRows(GEMX_uspmvStages-1); ++l_cCol) {
            p_c.getVal(i,l_cCol) = (l_immVectors[GEMX_uspmvStages-1][l_cCol]<0)? l_immVectors[GEMX_uspmvStages-1][l_cCol] * p_a.getPrelu(GEMX_uspmvStages-1): l_immVectors[GEMX_uspmvStages-1][l_cCol];
        }
    }
}

int main(int argc, char **argv)
{
    //define tolerances for verification when comparing CPU vs. FPGA results
    const float  l_TolRel = 1e-3,  l_TolAbs = 1e-5;
    //############  UI and FCN problem size  ############
    /* Check if the minimum required number of command line arguments
     * are passed which is 7 in case when matrices are read from files.
     */
    if (argc < 12)
    {
        cerr << "Usage:\n"
            <<  "  uspmv_test.exe <path/gemx.xclbin> dev_buf_size_in_bytes [M K Nnz pRelu mtxFile numRuns HandleA HandleB HandleC]\n"
            <<  "  For each instruction, [M K Nnz pRelu mtxFile/none numRuns HandleA HandleB HandleC] could not be missing\n"
            <<  "  Examples:\n"
            <<  "    uspmv_test.exe gemx.xclbin 209715200 0 0 0 1 ../data/spmv/keras_weight_0.mtx 300 A0 B0 C0\n";
        exit(2);
    }

    unsigned int l_argIdx = 1;
    string l_xclbinFile(argv[l_argIdx++]);
    unsigned int l_buf_sz = atoi(argv[l_argIdx++]);

    unsigned int l_instrCount = ((argc-3)/9>1)?((argc-3)/9):1; //number of instructions

    if(l_instrCount > 15)
    {
        cerr << "  Too many instructions at same time, Max allowed number of instruction is 15\n";
        exit(2);
    }

    unsigned int l_m[l_instrCount];
    unsigned int l_k[l_instrCount];
    unsigned int l_nnz[l_instrCount];
    GEMX_dataType l_pRelu[l_instrCount];
    vector<MtxFile> l_mtxFiles;
    unsigned int l_numRuns[l_instrCount];
    string l_handleA[l_instrCount];
    string l_handleB[l_instrCount];
    string l_handleC[l_instrCount];

    for(int index=0;index<l_instrCount;index++){
        l_m[index] = atoi(argv[l_argIdx++]);
        l_k[index] = atoi(argv[l_argIdx++]);
        l_nnz[index] = atoi(argv[l_argIdx++]);      
        l_pRelu[index] = atoi(argv[l_argIdx++]);
        string l_mtxFileName = argv[l_argIdx++];
        MtxFile l_mtxFile(l_mtxFileName);
        l_mtxFiles.push_back(l_mtxFile);
        l_numRuns[index] = atoi(argv[l_argIdx++]);
        l_handleA[index] = argv[l_argIdx++];
        l_handleB[index] = argv[l_argIdx++];
        l_handleC[index] = argv[l_argIdx++];      

    }
    if (!check(l_m, l_k, l_nnz, l_mtxFiles,l_instrCount)){ 
        return EXIT_FAILURE;
    }

    /////////////////////////////////////////////////////////
    USPMVDevHost<string> uspmv_host( l_xclbinFile, "gemxKernel_0");
    // Allocate a contiguous host and device buffer(cl::buffer)
    cout<<"Allocating Contiguous Host side memory of size:"<<l_buf_sz<<endl;
    uspmv_host.AllocProgBuf(l_buf_sz);

    /*
     * Create matrix descriptors, will be used to access input
     * matrices for cpu based GEMM operation and verification
     */
    vector< Mat<GEMX_dataType> > B_vec, C_vec;
    vector< UspMat<GEMX_dataType,uint16_t> > A_vec;

    //Mat<GEMX_dataType> l_matB(l_numRuns[1], l_k[1], l_k[1]);
    // Create and Add instructions for execution
    for(int index = 0; index<l_instrCount; index++)
    {
        // Calculate Matrix Sizes	
        unsigned int t_DoubleDdrWidth = GEMX_ddrWidth*2;
        unsigned int t_StageBlocks = (GEMX_uspmvStages + t_DoubleDdrWidth -1) / t_DoubleDdrWidth;
        unsigned int l_aSize = ((t_StageBlocks * t_DoubleDdrWidth * 3)+1)/2 + l_nnz[index] * 2;

        unsigned int l_bSize = l_k[index] * l_numRuns[index] * sizeof(GEMX_dataType);
        unsigned int l_cSize = l_m[index] * l_numRuns[index] * sizeof(GEMX_dataType);

        // Create Device buffer to host mapping for each matrixc
        GEMX_dataType* l_matA_addr = (GEMX_dataType*) uspmv_host.AddDevBuf(l_handleA[index], l_aSize * sizeof(GEMX_dataType));
        GEMX_dataType* l_matB_addr = (GEMX_dataType*) uspmv_host.AddDevBuf(l_handleB[index], l_bSize);
        GEMX_dataType* l_matC_addr = (GEMX_dataType*) uspmv_host.AddDevBuf(l_handleC[index], l_cSize);
        //Create Matrix descriptors that point to host memory.
        UspMat<GEMX_dataType,uint16_t> l_matA(l_matA_addr, GEMX_ddrWidth, 1);
        Mat<GEMX_dataType> l_matB(l_numRuns[index], l_k[index], l_k[index], l_matB_addr);
        Mat<GEMX_dataType> l_matC(l_numRuns[index], l_m[index], l_m[index] ,l_matC_addr);

        l_matB.fillModRange(5, 10);

        vector<MtxFile> l_mtxFile_tmp;
        l_mtxFile_tmp.push_back(l_mtxFiles[index]);
        l_matA.fillFromMtxFile(l_mtxFile_tmp, l_pRelu);

        // Add all the descriptors to a vector
        A_vec.push_back(l_matA);
        B_vec.push_back(l_matB);
        C_vec.push_back(l_matC);

        uspmv_host.SendDevBuf(l_handleA[index], false);
        uspmv_host.SendDevBuf(l_handleB[index], false);
        uspmv_host.SendDevBuf(l_handleC[index], false);
        uspmv_host.AddUSPMVDevOp(l_handleA[index],l_handleB[index],l_handleC[index],l_numRuns[index]);
    }

    // Execute all the USPMV instruction
    cout<<"Executing : " << l_instrCount << " USPMV instructions ...\n";
    uspmv_host.ExecuteDev();

    for(int index = 0; index<l_instrCount; index++) {
        Mat<GEMX_dataType> l_matC_cpu (l_numRuns[index], l_m[index], l_m[index]);   
        uspmv_host.GetDevBuf(l_handleC[index], true, true);
        uspmv_host.Wait();
        l_matC_cpu.fill(0);
        spmm_ref<GEMX_dataType,uint16_t, GEMX_uspmvStages, GEMX_ddrWidth>(A_vec[index], B_vec[index], l_numRuns[index], l_matC_cpu);
        bool l_res =C_vec[index].cmp(l_TolRel, l_TolAbs,l_matC_cpu);
        if (l_res) {
            cout << "INFO: Test pass." << endl;
        }
    }

    return EXIT_SUCCESS;
}


