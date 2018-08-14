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

#include "gemx_kernel.h"
#include "gemx_fpga.h"
#include "gemx_gen_bin.h"

//#define VERBOSE 0 

bool checkDim(unsigned int p_Val, unsigned int p_Mod, unsigned int p_Min) {
    bool l_ok = true;
    if (p_Val % p_Mod != 0) {
        std::cerr << "ERROR: value " << p_Val << " must be multiple of " << p_Mod << "\n";
        l_ok = false;
    }
    if (p_Val < p_Min) {
        std::cerr << "ERROR: value " << p_Val << " must be at least " << p_Min << "\n";
        l_ok = false;
    }
    return(l_ok);
}

float getBoardFreqMHz(unsigned int p_BoardId) {
    std::string l_freqCmd = "$XILINX_OPENCL/runtime/bin/xbsak query -d" + std::to_string(p_BoardId);;
    float l_freq = -1;
    char l_lineBuf[256];
    std::shared_ptr<FILE> l_pipe(popen(l_freqCmd.c_str(), "r"), pclose);
    //if (!l_pipe) throw std::runtime_error("ERROR: popen(" + l_freqCmd + ") failed");
    if (!l_pipe) std::cout << ("ERROR: popen(" + l_freqCmd + ") failed");
    bool l_nextLine_isFreq = false;
    while (l_pipe && fgets(l_lineBuf, 256, l_pipe.get()) ) {
        std::string l_line(l_lineBuf);
        //std::cout << "DEBUG: read line " << l_line << std::endl;
        if (l_nextLine_isFreq) {
            std::string l_prefix, l_val, l_mhz;
            std::stringstream l_ss(l_line);
            l_ss >> l_prefix >> l_val >> l_mhz;
            l_freq = std::stof(l_val);
            assert(l_mhz == "MHz");
            break;
        } else if (l_line.find("OCL Frequency:") != std::string::npos) {
            l_nextLine_isFreq = true;
        }
    }
    if (l_freq == -1) {
        //if xbsak does not work, as happens on F1, put the XOCC achieved kernel frequcy here
        l_freq = -1;
        std::cout << "INFO: Failed to get board frequency by xbsak. This is normal for cpu and hw emulation, using -1 MHz for reporting.\n";
    }
    return(l_freq);
}

int main(int argc, char **argv)
{
    //############  UI and GEMM problem size  ############
    if (argc < 15) {
        std::cerr << "Usage:\n"
                <<  "  gemx_api_gemm_multiInstr.exe <path/gemx.xclbin> [M K N  LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX]\n"
                <<  "  For each instruction, [M K N  LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX] could not be missing\n"
                <<  "  Examples:\n"
                <<  "    gemx_api_gemm_multiInstr.exe   out_hw/gemx.xclbin  32 128 256  256 256 256 256 1 0 A B C X\n";
        exit(2);
    }
    if(((argc - 2) % 13 != 0) && argc > 13) {
        std::cerr << "  If enter more than one instructions, for each instruction, [M K N LdA LdB LdC LdX postScaleVal postScaleShift HandleA HandleB HandleC HandleX] could not be missing\n";
        exit(2);
    }
    unsigned int l_argIdx = 1;
    std::string l_xclbinFile(argv[l_argIdx]);

    unsigned int l_instrCount = ((argc-2)/13>1)?((argc-2)/13):1; //number of instructions

    if(l_instrCount > 15){
        std::cerr << "  Too many instructions at same time\n";
        exit(2);
    }

    unsigned int l_ddrW = GEMX_ddrWidth;
    unsigned int l_m[l_instrCount];
    unsigned int l_k[l_instrCount];
    unsigned int l_n[l_instrCount];

    unsigned int l_lda;
    unsigned int l_ldb;
    unsigned int l_ldc;
    unsigned int l_ldx;
    int32_t l_postScaleVal;
    int32_t l_postScaleShift;
    int32_t l_postScale;


    printf("GEMX-gemm C++ API example using accelerator image \n",
            l_xclbinFile.c_str());
    ProgramType l_program[GEMX_numKernels];
    ProgramType l_program_golden;
    //unsigned long int l_Ops[l_instrCount]; //operations carried out by each kernel


    for(int i = 0; i < GEMX_numKernels; i++) {
        l_argIdx = 2;
        for(int index = 0; index<l_instrCount; index++){
            // Row major  C  M rows N cols  =  A  M rows K cols  *  B  K rows N cols
            //   MatType - tensor like type to allocate/store/align memory; you can use your own type instead
            //   Min size is the array edge (e.g., 32 on ku115), see GenGemm::check() for example of arg checking functions
            l_m[index] = atoi(argv[l_argIdx++]);
            l_k[index] = atoi(argv[l_argIdx++]);
            l_n[index] = atoi(argv[l_argIdx++]);
           
            l_lda = atoi(argv[l_argIdx++]);
            l_ldb = atoi(argv[l_argIdx++]);
            l_ldc = atoi(argv[l_argIdx++]);
	    l_ldx = atoi(argv[l_argIdx++]);
	   
	    l_postScaleVal = atoi(argv[l_argIdx++]);
	    l_postScaleShift = atoi(argv[l_argIdx++]);
	    l_postScale = (l_postScaleVal << 8) | (l_postScaleShift & 0x000000ff);
	    
	    std::string l_handleA = argv[l_argIdx++];
            std::string l_handleB = argv[l_argIdx++];
            std::string l_handleC = argv[l_argIdx++];
	    std::string l_handleX = argv[l_argIdx++];

            assert(l_lda >= l_k[index]);
            assert(l_ldb >= l_n[index]);
            assert(l_ldc >= l_n[index]);
	    assert(l_ldx >= l_n[index]);
            if (! (
                    checkDim(l_m[index], l_ddrW*GEMX_gemmMBlocks, l_ddrW*GEMX_gemmMBlocks) &&
                    checkDim(l_k[index], l_ddrW*GEMX_gemmKBlocks, l_ddrW*GEMX_gemmKBlocks) &&
                    checkDim(l_n[index], l_ddrW*GEMX_gemmNBlocks, l_ddrW*GEMX_gemmNBlocks) &&
                    checkDim(l_lda, l_ddrW, l_k[index]) &&
                    checkDim(l_ldb, l_ddrW, l_n[index]) &&
                    checkDim(l_ldc, l_ddrW, l_n[index]) &&
		    checkDim(l_ldx, l_ddrW, l_n[index])
            )) {
                return EXIT_FAILURE;
            }
            
           // l_Ops[index] = 2ull * l_m[index] * l_n[index] * l_k[index] + l_m[index] * l_n[index] * 3;

            // Allocate all pages before getting any address
            bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocX;
            unsigned int l_pageA = l_program[i].allocPages(l_handleA, l_newAllocA, l_m[index] * l_lda);
            unsigned int l_pageB = l_program[i].allocPages(l_handleB, l_newAllocB, l_k[index] * l_ldb);
            unsigned int l_pageC = l_program[i].allocPages(l_handleC, l_newAllocC, l_m[index] * l_ldc);
	    unsigned int l_pageX = l_program[i].allocPages(l_handleX, l_newAllocX, l_m[index] * l_ldx * (sizeof(GEMX_XdataType)/sizeof(GEMX_dataType)));



            // Get addresses where matrices are stored
            MatType l_matA(l_m[index], l_k[index], l_lda, l_program[i].getPageAddr(l_pageA));
            MatType l_matB(l_k[index], l_n[index], l_ldb, l_program[i].getPageAddr(l_pageB));
            XMatType l_matX(l_m[index], l_n[index], l_ldx, (GEMX_XdataType *) l_program[i].getPageAddr(l_pageX));
	    MatType l_matC(l_m[index], l_n[index], l_ldc, l_program[i].getPageAddr(l_pageC));

            /*l_matA.init(p_M, p_K, p_LdA, p_Program.getPageAddr(l_pageA));
	l_matB.init(p_K, p_N, p_LdB, p_Program.getPageAddr(l_pageB));
	l_matC.init(p_M, p_N, p_LdC, p_Program.getPageAddr(l_pageC));
             */

            if (l_newAllocA) {
                l_matA.fillMod(3, index);
            }
            if (l_newAllocB) {
                l_matB.fillMod(2, index);
            }
            
            if (l_newAllocX) {
		l_matX.fillMod(1,0);
	    }

            // Instruction
            GemmArgsType l_gemmArgs(
                    l_pageA, l_pageB, l_pageC,l_pageX,
                    l_m[index], l_k[index], l_n[index],
                    l_lda, l_ldb, l_ldc, l_ldx,
		    l_postScale
            );
            KargsType l_kargs;
            l_kargs.setGemmArgs(l_gemmArgs);
            l_kargs.store(l_program[i].addInstr(), 0);

            std::cout << "Added instruction GEMM " << l_m[index] << " x " << l_k[index] << " x " <<  l_n[index] <<" in kernel " << i << "  \n";
            std::cout << "Added instruction GEMM " << l_handleA << " x " << l_handleB << " x " << l_handleC <<" in kernel " << i << "  \n";

        }
    }
    //golden program
    l_argIdx = 2;
    std::cout << "Calculate golden result on host, for large matrix size, this will take long time.\n" << std::endl;
    if(!getenv("SKIPPED_GOLD_CAL")){
        for(int index = 0; index<l_instrCount; index++){
            l_m[index] = atoi(argv[l_argIdx++]);
            l_k[index] = atoi(argv[l_argIdx++]);
            l_n[index] = atoi(argv[l_argIdx++]);
           
            l_lda = atoi(argv[l_argIdx++]);
            l_ldb = atoi(argv[l_argIdx++]);
            l_ldc = atoi(argv[l_argIdx++]);
	    l_ldx = atoi(argv[l_argIdx++]);
	   
	    l_postScaleVal = atoi(argv[l_argIdx++]);
	    l_postScaleShift = atoi(argv[l_argIdx++]);
	    l_postScale = (l_postScaleVal << 8) | (l_postScaleShift & 0x000000ff);
	    
	    std::string l_handleA = argv[l_argIdx++];
            std::string l_handleB = argv[l_argIdx++];
            std::string l_handleC = argv[l_argIdx++];
	    std::string l_handleX = argv[l_argIdx++];

            // Allocate all pages before getting any address
            bool l_newAllocA, l_newAllocB, l_newAllocC,l_newAllocX;
            unsigned int l_pageA = l_program_golden.allocPages(l_handleA, l_newAllocA, l_m[index] * l_lda);
            unsigned int l_pageB = l_program_golden.allocPages(l_handleB, l_newAllocB, l_k[index] * l_ldb);
            unsigned int l_pageC = l_program_golden.allocPages(l_handleC, l_newAllocC, l_m[index] * l_ldc);
	    unsigned int l_pageX = l_program_golden.allocPages(l_handleX, l_newAllocX, l_m[index] * l_ldx * (sizeof(GEMX_XdataType)/sizeof(GEMX_dataType)));

            // Get addresses where matrices are stored
            MatType l_matA(l_m[index], l_k[index], l_lda, l_program_golden.getPageAddr(l_pageA));
            MatType l_matB(l_k[index], l_n[index], l_ldb, l_program_golden.getPageAddr(l_pageB));
            XMatType l_matX(l_m[index], l_n[index], l_ldx, (GEMX_XdataType *) l_program_golden.getPageAddr(l_pageX));
	    MatType l_matC(l_m[index], l_n[index], l_ldc, l_program_golden.getPageAddr(l_pageC));

            if (l_newAllocA) {
                l_matA.fillMod(3, index);
            }
            if (l_newAllocB) {
                l_matB.fillMod(2, index);
            }
            
            if (l_newAllocX) {
		l_matX.fillMod(1,0);
	    }

            // Instruction
            GemmArgsType l_gemmArgs(
                    l_pageA, l_pageB, l_pageC,l_pageX,
                    l_m[index], l_k[index], l_n[index],
                    l_lda, l_ldb, l_ldc, l_ldx,
		    l_postScale
            );
            KargsType l_kargs;
            l_kargs.setGemmArgs(l_gemmArgs);
            l_kargs.store(l_program_golden.addInstr(), 0);

            l_matC.multiplyAddScale(l_matA, l_matB, l_matX, l_postScale);

            std::cout << "Calculated golden result GEMM"<< l_m[index] << "x" << l_k[index] << "x" <<  l_n[index] <<" on host\n";

        }
    }
    std::string kernelNames[GEMX_numKernels];
    gemx::MemDesc l_memDesc[GEMX_numKernels];

    for (int i=0; i<GEMX_numKernels; ++i) {
        l_memDesc[i] = l_program[i].getMemDesc();
    }

    //############  Runtime reporting Infra  ############
    TimePointType l_tp[10];
    unsigned int l_tpIdx = 0;
    l_tp[l_tpIdx] = std::chrono::high_resolution_clock::now();

    //############  Run FPGA accelerator  ############
    // Init FPGA
    gemx::Fpga l_fpga;

    for (int i=0; i<GEMX_numKernels; ++i){
        kernelNames[i] = "gemxKernel_" + std::to_string(i);
    }
    if (l_fpga.loadXclbin(l_xclbinFile, kernelNames)) {
        std::cout << "INFO: created kernels" << std::endl;
    } else {
        std::cerr << "ERROR: failed to load " + l_xclbinFile + "\n";
        return EXIT_FAILURE;
    }
    showTimeData("loadXclbin", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

    //create buffers for transferring data to FPGA
    if (!l_fpga.createBuffers(l_memDesc)) {
        std::cerr << "ERROR: failed to create buffers for transffering data to FPGA DDR\n";
        return EXIT_FAILURE;
    }
    showTimeData("created buffers", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

    // Transfer data to FPGA
    if (l_fpga.copyToFpga()) {
        (VERBOSE > 0) && std::cout << "INFO: transferred data to FPGA" << std::endl;
    } else {
        std::cerr << "ERROR: failed to copy data to FPGA DDR\n";
        return EXIT_FAILURE;
    }
    showTimeData("copyToFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

    // Gemx kernel ops
    if (l_fpga.callKernels()) {
        (VERBOSE > 0) && std::cout << "INFO: Executed kernel" << std::endl;
    } else {
        std::cerr << "ERROR: failed to call kernels ";
        for (int i=0; i<GEMX_numKernels; ++i) {
            std::cerr << kernelNames[i] << " ";
        }
        std::cerr << "\n";
        return EXIT_FAILURE;
    }
    showTimeData("callKernel", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

    // Transfer data back to host - due to lazy evaluation this is generally wheer the accelerator performs the work
    if (l_fpga.copyFromFpga()) {
        (VERBOSE > 0) && std::cout << "INFO: Transferred data from FPGA" << std::endl;
    } else {
        std::cerr << "ERROR: failed to copy data from FPGA DDR\n";
        return EXIT_FAILURE;
    }
    showTimeData("copyFromFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
    showTimeData("total", l_tp[0], l_tp[l_tpIdx]); l_tpIdx++;
    double l_timeApiInMs = -1;
    showTimeData("subtotalFpga", l_tp[2], l_tp[l_tpIdx], &l_timeApiInMs); l_tpIdx++; // Host->DDR, kernel, DDR->host

    //############  Get the exact kernel time from HW cycle counters on the accelerator  ############
    float l_boardFreqMHz = getBoardFreqMHz(0);
    //unsigned long int l_Ops = 2ull * l_m * l_n * l_k * 2; //operations carried out by each kernel
    KargsType l_kargsRes[GEMX_numKernels];
    KargsOpType l_op;
    gemx::InstrResArgs l_instrRes;
    unsigned long int l_cycleCount;
    unsigned long int l_maxCycleCount[l_instrCount] = {0};
    double l_timeKernelInMs;
    double l_maxTimeKernelInMs[l_instrCount] = {0};
    double l_perfKernelInTops[l_instrCount];
    double l_totalPerfKernelInTops=0;
    double l_perfApiInTops;
    double l_timeMsAt100pctEff;
    double l_timeMsAt100pctEffKernel;
    double l_effKernelPct;
    double l_effApiPct;

    unsigned long int l_total_Op[l_instrCount];
    unsigned long int l_total_Ops = 0;
    unsigned long int l_total_parallel_Op[l_instrCount];
    unsigned long int l_total_parallel_Ops = 0;
    for(int j=0;j<l_instrCount;++j){
      l_total_Op[j] = 2ull * l_m[j] * l_n[j] * l_k[j] + l_m[j] * l_n[j] * 3;
      l_total_Ops += 2ull * l_m[j] * l_n[j] * l_k[j] + l_m[j] * l_n[j] * 3;
      l_total_parallel_Op[j] = 2ull * l_m[j] * l_k[j] * l_n[j];
      l_total_parallel_Ops += 2ull * l_m[j] * l_k[j] * l_n[j];
    }

    for (int i=0; i<GEMX_numKernels; ++i) {
        for(int j=0;j<l_instrCount;++j){ //number of instructions
            l_op = l_kargsRes[i].load(l_program[i].getBaseResAddr(), j * l_kargsRes[i].getInstrWidth());
            assert(l_op == KargsType::OpResult);
            l_instrRes = l_kargsRes[i].getInstrResArgs();
            l_cycleCount = l_instrRes.getDuration();
            std::cout << std::string("cycles in kernel ")<< i << " "<<l_cycleCount <<std::endl;        
            l_maxCycleCount[j] = (l_cycleCount > l_maxCycleCount[j])? l_cycleCount: l_maxCycleCount[j];
            l_timeKernelInMs = l_maxCycleCount[j] / (l_boardFreqMHz * 1e6) * 1e3;
            l_maxTimeKernelInMs[j] = (l_timeKernelInMs > l_maxTimeKernelInMs[j])? l_timeKernelInMs: l_maxTimeKernelInMs[j];
            l_perfKernelInTops[j] = l_total_Op[j] / (l_maxTimeKernelInMs[j] * 1e-3) / 1e12;
            //l_totalPerfKernelInTops += l_perfKernelInTops[i];
      }
    }

    // Show time, Tops in csv format
    std::cout << std::string("DATA_CSV:,DdrWidth,Freq,M,K,N,")
    + "Ops,KernelCycles,"
    + "TimeKernelMs,TimeApiMs,"
    + "EffKernelPct,EffApiPct,"
    + "PerfKernelTops,PerfApiTops\n";
    for(int i=0;i<l_instrCount;++i){
    l_perfApiInTops = (l_total_Ops*GEMX_numKernels) / (l_timeApiInMs * 1e-3) / 1e12;
    l_timeMsAt100pctEff = (l_total_parallel_Ops*GEMX_numKernels) / 2 / GEMX_ddrWidth / GEMX_ddrWidth / (l_boardFreqMHz * 1e6) * 1e3;
    l_timeMsAt100pctEffKernel = l_total_parallel_Op[i] / 2 / GEMX_ddrWidth / GEMX_ddrWidth / (l_boardFreqMHz * 1e6) * 1e3;
    l_effKernelPct = 100 * l_timeMsAt100pctEffKernel / l_maxTimeKernelInMs[i];
    l_effApiPct = 100 * l_timeMsAt100pctEff / l_timeApiInMs;
        std::cout << "DATA_CSV:," <<  GEMX_ddrWidth << "," << l_boardFreqMHz << ","
        << l_m[i]<<","<<l_k[i]<<","<<l_n[i] << ","
        << l_total_Op[i] << "," << l_maxCycleCount[i] << ","
        << l_maxTimeKernelInMs[i] << "," << l_timeApiInMs << ","
        << l_effKernelPct << "," << l_effApiPct << ","
        << l_perfKernelInTops[i] << "," << l_perfApiInTops
        << std::endl;
    }

    //############  Compare tha FPGA results with the reference results  ############
    // Calculate reference C = A * B
    // Since the reference is not needed on the acclerator allocate memory in any way
    if(!getenv("SKIPPED_GOLD_CAL")){
        float  l_TolRel = 1e-3,  l_TolAbs = 1e-5;
        bool l_isLastOp = false;
        bool l_compareOk = true;
        KargsType l_kargs0, l_kargs1;
        unsigned int l_pc = 0;
        GenGemm l_gemm;

        do {
            KargsOpType l_op0 = l_kargs0.load(l_program_golden.getBaseInstrAddr(), l_pc);
            KargsOpType l_op1 = l_kargs1.load(l_program[0].getBaseInstrAddr(), l_pc);
            if (l_op1 == KargsType::OpResult) {
                break;
            }
            assert(l_op0 == l_op1);
            switch(l_op0) {
                case KargsType::OpGemm: {
                    GemmArgsType l_gemmArgs = l_kargs0.getGemmArgs();
                    bool l_opOk = l_gemm.compare(l_TolRel, l_TolAbs, l_program_golden, l_program[0], l_gemmArgs);
                    l_compareOk = l_compareOk && l_opOk;
                    break;
                }
            }
            l_pc += l_kargs0.getInstrWidth();
        } while(!l_isLastOp);

        if (!l_compareOk) {
	std::cout << "fail\n" << std::endl;
	}else{
	std::cout << "pass\n" << std::endl;
	}
        return l_compareOk ? EXIT_SUCCESS : EXIT_FAILURE;
    }else{
        std::cout << "INFO: skipped gold calculation on host since it may take too long\n" << std::endl;
    }
    return EXIT_SUCCESS;
}



