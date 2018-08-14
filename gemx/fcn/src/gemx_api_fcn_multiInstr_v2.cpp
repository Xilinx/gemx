#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <stdio.h>  // fgets for popen

#include "gemx_host.h"

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
    //############  UI and FCN problem size  ############
    if (argc < 17) {
        std::cerr << "Usage:\n"
                <<  "  gemx_api_fcn_multiInstr.exe <path/gemx.xclbin> [M K N  LdA LdB LdC LdX postScaleVal postScaleShift PReluScale PReluAlpha HandleA HandleB HandleC HandleX]\n";
        exit(2);
    }
    if ((argc - 2) % 15 != 0) {
        std::cerr << "  For each instruction, [M K N  LdA LdB LdC LdX postScaleVal postScaleShift PReluScale PReluAlpha HandleA HandleB HandleC HandleX] could not be missing\n";
        exit(2);
    }
    unsigned int l_argIdx = 1;
    std::string l_xclbinFile(argv[l_argIdx]);

    unsigned int l_instrCount = ((argc-2)/14>1)?((argc-2)/14):1; //number of instructions

    if(l_instrCount > 15){
        std::cerr << "  Too many instructions at same time\n";
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

    int32_t l_bias[l_instrCount];
    int32_t l_postScaleVal[l_instrCount];
    int32_t l_postScaleShift[l_instrCount];
    int32_t l_postScale[l_instrCount];
    int16_t l_PReluScale[l_instrCount];
    int16_t l_PReluAlpha[l_instrCount];
    int16_t l_PReluVal[l_instrCount];
    
    string l_handleA[l_instrCount], l_handleB[l_instrCount], l_handleC[l_instrCount], l_handleX[l_instrCount];

    //unsigned long int l_Ops[l_instrCount]; //operations carried out by each kernel


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

        l_PReluScale[index] = atoi(argv[l_argIdx++]);
        l_PReluAlpha[index] = atoi(argv[l_argIdx++]);
        l_PReluVal[index] = (l_PReluScale[index] << 6) | (l_PReluAlpha[index] & 0x003f);

        l_handleA[index] = argv[l_argIdx++];
        l_handleB[index] = argv[l_argIdx++];
        l_handleC[index] = argv[l_argIdx++];
        l_handleX[index] = argv[l_argIdx++];

        assert(l_lda[index] >= l_k[index]);
        assert(l_ldb[index] >= l_n[index]);
        assert(l_ldc[index] >= l_n[index]);
        assert(l_ldx[index] >= l_n[index]);
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
        //l_Ops[index] = 2ull * l_m[index] * l_n[index] * l_k[index];
    }

    const float  l_TolRel = 1e-3,  l_TolAbs = 1e-5;
    FCNHost<std::string> fcn_host( l_xclbinFile, "gemxKernel_0", "kcu1500");

    /*
    for(int index = 0; index<l_instrCount; index++){
        shared_ptr < Mat<GEMX_dataType> > l_matA = fcn_host.AllocMat(l_handleA[index], l_m[index], l_k[index], l_lda[index]);
        shared_ptr < Mat<GEMX_dataType> > l_matC = fcn_host.AllocMat(l_handleC[index], l_m[index], l_n[index], l_ldc[index], true);
        //l_matA->fillModRange(-100, 100);
        l_matA->fillModRange(-10, 10);
        //l_matA->fillMod(25, index);

        fcn_host.SendToFPGA(l_handleA[index]);
    }


    for(int index = 0; index<l_instrCount; index++){
        shared_ptr < Mat<GEMX_dataType> > l_matB = fcn_host.AllocMat(l_handleB[index], l_k[index], l_n[index], l_ldb[index]);
        //l_matB->fillModRange(-100, 100);
        l_matB->fillModRange(-10, 10);
        //l_matB->fillMod(55, index);
        fcn_host.SendToFPGA(l_handleB[index]);

        fcn_host.AddFCNOp ( l_handleA[index], l_handleB[index], l_handleC[index],l_bias[index],l_postScaleVal[index],l_postScaleShift[index],l_PReluScale[index],l_PReluAlpha[index]);
        //fcn_host.AddFCNOp ( l_handleA[index], l_handleB[index], l_handleC[index],0, 1, 0, 1,0);

        fcn_host.Execute();

        // Compare FPGA result with software FCN
        Mat<GEMX_dataType> l_matC_cpu ( l_m[index], l_n[index], l_ldc[index]);
        shared_ptr < Mat<GEMX_dataType> > l_matC_get = fcn_host.GetMat(l_handleC[index],true);
        shared_ptr < Mat<GEMX_dataType> > l_matA_get = fcn_host.GetMat(l_handleA[index],true);

        //l_matC_cpu.multiply(*l_matA_get, *l_matB);
        l_matC_cpu.matMultWithScaleAndPRelu(*l_matA_get, *l_matB,l_bias[index],l_postScale[index],l_PReluVal[index]);
        l_matC_cpu.cmp(l_TolRel, l_TolAbs, *l_matC_get);
    }
    */
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
        fcn_host.SendToFPGA( l_handleC[index], l_matC->data(), l_matC->buf_sz());
        C_vec.push_back(l_matC);
        l_matX->fillModRange(-100,100);
        fcn_host.SendToFPGA( l_handleX[index], l_matX->data(), l_matX->buf_sz() );

        //l_matB->fillModRange(-3, 3);
        l_matB->fillModRange(-10, 10);
        //l_matB->fillMod(100, index);
        fcn_host.SendToFPGA(l_handleB[index], l_matB->data(), l_matB->buf_sz());
        l_matA->fillModRange(-100, 0);
        //l_matA->fillModRange(-3, 3);
        //l_matA->fillModRange(-10, 10);
        //l_matA->fillMod(100, index);
        cout << l_matA->rows() << " " << l_matA->cols() << " " << l_matA->ld() << " " << l_matA->buf_sz() << endl;
        fcn_host.SendToFPGA(l_handleA[index], l_matA->data(), l_matA->buf_sz());

        fcn_host.AddFCNOp ( l_handleA[index], l_handleB[index], l_handleC[index], l_handleX[index], l_m[index], l_k[index], l_n[index], l_lda[index],l_ldb[index],l_ldc[index],l_ldx[index],l_postScaleVal[index],l_postScaleShift[index],l_PReluScale[index],l_PReluAlpha[index]);
    }

    //std::unordered_map< GEMX_dataType, Mat<GEMX_dataType> > cpu_result;
    fcn_host.Execute();

    // Check result
    for(int index = 0; index<l_instrCount; index++){
        std::cout << "Checking instr " << index << ": " << l_handleA[index] << " " << l_handleB[index] << " " << l_handleC[index] << std::endl;
        //fcn_host.AddFCNOp ( l_handleA[index], l_handleB[index], l_handleC[index],0, 1, 0, 1,0);
        //final_result_handle = l_handleC[index];
        // Compare FPGA result with software FCN
        Mat<GEMX_dataType> l_matC_cpu ( l_m[index], l_n[index], l_ldc[index]);

        fcn_host.GetMat(l_handleC[index],true);
        fcn_host.GetMat(l_handleA[index],true);
        fcn_host.GetMat(l_handleB[index],true);
        fcn_host.GetMat(l_handleX[index],true);

        //l_matC_cpu.multiply(*l_matA_get, *l_matB_get);
        l_matC_cpu.matMultWithScaleAndPRelu(*A_vec[index], *B_vec[index], *X_vec[index], l_postScale[index],l_PReluVal[index]);
        //l_matC_cpu.matMultWithScaleAndPRelu(*l_matA_get, *l_matB_get,l_bias[index],l_postScale[index],l_PReluVal[index]);
        l_matC_cpu.cmp(l_TolRel, l_TolAbs, *C_vec[index]);
        //cpu_result[l_handleC[index]] = l_matC_cpu;
    }

    return EXIT_SUCCESS;
}

