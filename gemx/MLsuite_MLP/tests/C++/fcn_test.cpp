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
#include "fcn_host.h"


using namespace std;
using namespace gemx;
bool checkDim(unsigned int p_Val, unsigned int p_Mod, unsigned int p_Min)
{
    bool l_ok = true;
    if (p_Val % p_Mod != 0)
    {
        cerr << "ERROR: value " << p_Val << " must be multiple of " << p_Mod << "\n";
        l_ok = false;
    }
    if (p_Val < p_Min)
    {
        cerr << "ERROR: value " << p_Val << " must be at least " << p_Min << "\n";
        l_ok = false;
    }
    return(l_ok);
}
int main(int argc, char **argv)
{
    //define tolerances for verification when comparing CPU vs. FPGA results
    const float  l_TolRel = 1e-3,  l_TolAbs = 1e-5;
    //############  UI and FCN problem size  ############
    /* Check if the minimum required number of command line arguments
     * are passed which is 7 in case when matrices are read from files.
     */
    if (argc < 7)
    {
        cerr << "Usage:\n"
            <<  "  fcn_test.exe <path/gemx.xclbin> dev_buf_size [M K N  LdA LdB LdC LdX postScaleVal postScaleShift PReluScale PReluAlpha HandleA HandleB HandleC HandleX]\n"
            <<  "  fcn_test.exe <path/gemx.xclbin> dev_buf_size [0 0 0 insFile matAFile matBFile matXFile] \n";
        exit(2);
    }

    unsigned int l_argIdx = 1;
    string l_xclbinFile(argv[l_argIdx++]);
    unsigned int l_buf_sz = atoi(argv[l_argIdx++]);
    unsigned int l_instrCount = 0;
    bool readFromFile = 0;
    bool readFromFiles = 0;

    // Check type if instruction format, requires reading from file
    if(atoi(argv[3]) == 0 && atoi(argv[4]) == 0 && atoi(argv[5]) == 0)
    {
        if((argc-3)% 7 == 0)
        {
            l_instrCount = ((argc-3)/7>1)?((argc-3)/7):1; //number of instructions
            readFromFiles = 1;
        }
        else
        {
            cerr << "  usage: \n"
                << "         fcn_test.exe <path/gemx.xclbin> dev_buf_size 0 0 0 insFile matAFile matBFile matXFile\n";
            exit(2);
        }
    }
    else
    {
        if ((argc - 3) % 15 != 0)
        {
            cerr << "  For each instruction, [M K N  LdA LdB LdC LdX postScaleVal postScaleShift PReluScale PReluAlpha HandleA HandleB HandleC HandleX] could not be missing\n";
            exit(2);
        }
        l_instrCount = ((argc-2)/15>1)?((argc-2)/15):1; //number of instructions
    }
    if(l_instrCount > 15)
    {
        cerr << "  Too many instructions at same time, Max allowed number of instruction is 15\n";
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
    int16_t l_PReluScale[l_instrCount];
    int16_t l_PReluAlpha[l_instrCount];
    int16_t l_PReluVal[l_instrCount];
    string l_handleA[l_instrCount];
    string l_handleB[l_instrCount];
    string l_handleC[l_instrCount];
    string l_handleX[l_instrCount];
    string l_insFileName[l_instrCount];
    string l_matAFileName[l_instrCount];
    string l_matBFileName[l_instrCount];
    string l_matXFileName[l_instrCount];
    cout<<"GEMX-IFNO: FCN Engine C++ test using "<< l_xclbinFile << " xclbin file\n";
    char* execution_mode;
    execution_mode = getenv ("XCL_EMULATION_MODE");
    if(execution_mode!=NULL)
    {
    	cout <<"=================================================================\n";
    	cout <<"GEMX-INFO: The application is running in : " << execution_mode <<" MODE \n";
    	cout <<"=================================================================\n";
    }
    for(int index = 0; index<l_instrCount; index++)
    {
        //parse matrix sizes
        l_m[index] = atoi(argv[l_argIdx++]);
        l_k[index] = atoi(argv[l_argIdx++]);
        l_n[index] = atoi(argv[l_argIdx++]);
        //parse matrix lead dims
        l_lda[index] = atoi(argv[l_argIdx++]);
        l_ldb[index] = atoi(argv[l_argIdx++]);
        l_ldc[index] = atoi(argv[l_argIdx++]);
        l_ldx[index] = atoi(argv[l_argIdx++]);
        //gemm scaling factors
        l_postScaleVal[index] = atoi(argv[l_argIdx++]);
        l_postScaleShift[index] = atoi(argv[l_argIdx++]);
        //relu related scaling factors
        l_PReluScale[index] = atoi(argv[l_argIdx++]);
        l_PReluAlpha[index] = atoi(argv[l_argIdx++]);
        // Matrix handle to be used for indexing dictionary
        l_handleA[index] = argv[l_argIdx++];
        l_handleB[index] = argv[l_argIdx++];
        l_handleC[index] = argv[l_argIdx++];
        l_handleX[index] = argv[l_argIdx++];
        l_postScale[index] = (l_postScaleVal[index] << 8) | (l_postScaleShift[index] & 0x000000ff);
        l_PReluVal[index] = (l_PReluScale[index] << 6) | (l_PReluAlpha[index] & 0x003f);

#ifndef __SYNTHESIS__
        cout<<"GEMX-INFO : Running FCN test with following user provided parameters ....\n";
        cout<<"Memory Image Size="<<l_buf_sz<<endl;
        cout<<"M="<<l_m[index]<<endl;
        cout<<"K="<<l_k[index]<<endl;
        cout<<"N="<<l_n[index]<<endl;
        cout<<"lda="<<l_lda[index]<<endl;
        cout<<"ldb="<<l_ldb[index]<<endl;
        cout<<"ldc="<<l_ldc[index]<<endl;
        cout<<"ldx="<<l_ldx[index]<<endl;
        cout<<"postScaleVal="<<l_postScaleVal[index]<<endl;
        cout<<"postScaleShift="<<l_postScaleShift[index]<<endl;
        cout<<"PReluScale="<<l_PReluScale[index]<<endl;
        cout<<"PReluAlpha="<<l_PReluAlpha[index]<<endl;
        cout<<"l_handleA="<<l_handleA[index]<<endl;
        cout<<"l_handleB="<<l_handleB[index]<<endl;
        cout<<"l_handleC="<<l_handleC[index]<<endl;
        cout<<"l_handleX="<<l_handleX[index]<<endl;
#endif
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
    /////////////////////////////////////////////////////////
    FCNHost<string> fcn_host( l_xclbinFile, "gemxKernel_0");
    // Allocate a contiguous host and device buffer(cl::buffer)
    cout<<"Allocating Contiguous Host side memory of size:"<<l_buf_sz<<endl;
    fcn_host.AllocProgBuf(l_buf_sz);

    /*
     * Create matrix descriptors, will be used to access input
     * matrices for cpu based GEMM operation and verification
     */
    vector< Mat<GEMX_dataType> > A_vec, B_vec, C_vec;
    vector<Mat<int> > X_vec;
    // Create and Add instructions for execution
    for(int index = 0; index<l_instrCount; index++)
    {
        // Calculate Matrix Sizes
        unsigned int l_aSize = l_m[index] * l_lda[index]* sizeof(GEMX_dataType);
        unsigned int l_bSize = l_k[index] * l_ldb[index]* sizeof(GEMX_dataType);
        unsigned int l_xSize = l_m[index] * l_ldx[index] *sizeof(int);//(sizeof(GEMX_XdataType)/sizeof(GEMX_dataType));
        unsigned int l_cSize = l_m[index] * l_ldc[index]* sizeof(GEMX_dataType);
        // Create Device buffer to host mapping for each matrixc
        GEMX_dataType* l_matA_addr = (GEMX_dataType*) fcn_host.AddDevBuf(l_handleA[index], l_aSize);
        GEMX_dataType* l_matB_addr = (GEMX_dataType*) fcn_host.AddDevBuf(l_handleB[index], l_bSize);
        int* l_matX_addr = (int*) fcn_host.AddDevBuf(l_handleX[index],l_xSize);
        GEMX_dataType* l_matC_addr = (GEMX_dataType*) fcn_host.AddDevBuf(l_handleC[index], l_cSize);
        //Create Matrix descriptors that point to host memory.
        Mat<GEMX_dataType> l_matA(l_m[index], l_k[index], l_lda[index],l_matA_addr);
        Mat<GEMX_dataType> l_matB(l_k[index], l_n[index], l_ldb[index],l_matB_addr);
        Mat<int> l_matX(l_m[index], l_n[index], l_ldx[index], l_matX_addr);
        Mat<GEMX_dataType> l_matC(l_m[index], l_n[index], l_ldc[index],l_matC_addr);
        // Add all the descriptors to a vector
        A_vec.push_back(l_matA);
        B_vec.push_back(l_matB);
        X_vec.push_back(l_matX);
        C_vec.push_back(l_matC);
        l_matB.fillModRange(5, 10);
        l_matA.fillMod(5,10);
        l_matX.fillModRange(5, 100);
        fcn_host.SendDevBuf(l_handleB[index], false);
        fcn_host.SendDevBuf(l_handleA[index], false);
        fcn_host.SendDevBuf(l_handleX[index], false);
        fcn_host.SendDevBuf(l_handleC[index], false);
        fcn_host.AddFCNDevOp(
                l_handleA[index],
                l_handleB[index],
                l_handleC[index],
                l_handleX[index],
                l_m[index],
                l_k[index],
                l_n[index],
                l_lda[index],
                l_ldb[index],
                l_ldc[index],
                l_ldx[index],
                l_postScaleVal[index],
                l_postScaleShift[index],
                l_PReluAlpha[index],
                l_PReluVal[index]
                );

    }
    // Execute all the FCN instruction
    cout<<"Executing : " << l_instrCount << " FCN instructions ...\n";
    fcn_host.ExecuteDev();
    // Read processed data back from the device and compare with golden results
    // calculated on the CPU.
    for(int index = 0; index<l_instrCount; index++)
    {
        cout << "Verifying instruction : " << index << " Results : " << l_handleA[index] << " "
            << l_handleB[index] << " " << l_handleX[index] << " "
            << l_handleC[index] << endl;
        GEMX_dataType *l_matC_cpu_addr = (GEMX_dataType*)malloc(l_m[index]*l_ldc[index]*sizeof(GEMX_dataType));
        Mat<GEMX_dataType> l_matC_cpu (l_m[index], l_n[index], l_ldc[index], l_matC_cpu_addr);
        // Copy all the matrices from the device
        void* l_ptr;
        l_ptr = fcn_host.GetDevBuf(l_handleA[index], true);
        cout << "Read A back successful" << endl;
        l_ptr = fcn_host.GetDevBuf(l_handleB[index], true);
        cout << "Read B back successful" << endl;
        l_ptr = fcn_host.GetDevBuf(l_handleX[index], true);
        cout << "Read X back successful" << endl;
        l_ptr = fcn_host.GetDevBuf(l_handleC[index], true, true);
        fcn_host.Wait();
        l_matC_cpu.matMultWithScaleAndPRelu(A_vec[index],B_vec[index],X_vec[index],
                l_postScale[index],l_PReluVal[index]
                );
        bool l_res =l_matC_cpu.cmp(l_TolRel, l_TolAbs, C_vec[index]);
        if (l_res)
        {
            cout << "INFO: Test pass." << endl;
        }
        free(l_matC_cpu_addr);

    }
    return EXIT_SUCCESS;
}

