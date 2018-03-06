
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
 *  @brief Simple SPMV example of C++ API client interaction with SPMV linear algebra accelerator on Xilinx FPGA 
 *
 *  $DateTime: 2017/11/16$
 *  $Author: Yifei $
 */

// Prerequisites:
//  - Boost installation (edit the Makefile with your boost path)
//  - Compiled C++ to bitstream accelerator kernel
//     - use "make run_hw"
//     - or get a pre-compiled copy of the out_hw/gemx.xclbin)
// Compile and run this API example:
//   make out_host/gemx_api_spmv.exe
//   out_host/gemx_api_spmv.exe
//     # No argumens will show help message
// You can also test it with a cpu emulation accelerator kernel (faster to combile, make run_cpu_em)
//   ( setenv XCL_EMULATION_MODE true ; out_host/gemx_api_spmv.exe out_cpu_emu/gemx.xclbin )
 
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
	l_freq = 200.2;
    std::cout << "INFO: Failed to get board frequency by xbsak. This is normal for cpu and hw emulation, using -1 MHz for reporting.\n";
  }
  return(l_freq);
}

int main(int argc, char **argv)
{
  //############  UI and SPMV problem size  ############
  if (argc < 5) {
    std::cerr << "Usage:\n"
              <<  "  gemx_api_spmv.exe <path/gemx.xclbin> [M K Nnz [mtxFile]]\n";
    exit(2);
  }
  unsigned int l_argIdx = 1;
  std::string l_xclbinFile(argv[l_argIdx]);
  // Row major  Vector C  =  Matrix A  M rows K cols  *  Vector B  K rows
  //   MatType - tensor like type to allocate/store/align memory; you can use your own type instead
  //   See GenSpmv::check() for example of arg checking functions
  unsigned int l_ddrW = GEMX_ddrWidth;
  unsigned int l_M,  l_K, l_NNZ;
  l_M = atoi(argv[++l_argIdx]);
  l_K = atoi(argv[++l_argIdx]);
  l_NNZ = atoi(argv[++l_argIdx]);
  //std::string l_mtxFileName = argv[++l_argIdx];
  std::string l_mtxFileName("none");
  if (argc > ++l_argIdx) {l_mtxFileName = argv[l_argIdx];}
  
  
  MtxFile l_mtxFile(l_mtxFileName);
  GenSpmv l_spmv;
  //The check() modifies the dimensions when loading from a matrix file. Please use 0 for l_M, l_K and l_NNZ when provding matrix file
  l_spmv.check(l_M, l_K, l_NNZ, l_mtxFile);
  
  //std::cout << "M = " << l_M << " K = " << l_K << " NNZ = " << l_NNZ << " name =  "<<l_mtxFileName<<"      ";
  
  printf("GEMX-spmv C++ API example using accelerator image \n",
         l_xclbinFile.c_str());
    
  //############  Client code - prepare the spmv problem input  ############
  ProgramType l_program[GEMX_numKernels];  // Holds instructions and controls memory allocation
  std::string l_handleA[GEMX_numKernels];
  std::string l_handleB[GEMX_numKernels];
  std::string l_handleC[GEMX_numKernels];

  bool l_newAllocA[GEMX_numKernels];
  bool l_newAllocB[GEMX_numKernels];
  bool l_newAllocC[GEMX_numKernels];
  
  const unsigned int l_numDescPages = (GEMX_spmvNumCblocks + SpMatType::t_numDescPerPage - 1) / SpMatType::t_numDescPerPage;
  unsigned int l_numDescDdrWords = l_numDescPages * SpMatType::t_numDdrWordsPerPage;
  const unsigned int l_numPaddingPages = GEMX_spmvNumCblocks;
  const unsigned int l_numPaddingDdrWords = l_numPaddingPages * SpMatType::t_numDdrWordsPerPage;
  
  unsigned int l_pageA[GEMX_numKernels];
  unsigned int l_pageB[GEMX_numKernels];
  unsigned int l_pageC[GEMX_numKernels];
  SpMatType l_matA;
  MatType l_matB[GEMX_numKernels];
  MatType l_matC[GEMX_numKernels];
  
  unsigned int l_Cblocks = (l_M + SpmvType::getRowsInCblock() - 1) / SpmvType::getRowsInCblock();
  
  //SpmvArgsType l_spmvArgs[GEMX_numKernels];
  KargsType l_kargs[GEMX_numKernels];

  for (int i=0; i<GEMX_numKernels; ++i) {
	l_handleA[i] = "A"+std::to_string(i);
	l_handleB[i] = "B"+std::to_string(i);
	l_handleC[i] = "C"+std::to_string(i);
  	// Allocate all pages before getting any address

  	l_pageA[i] = l_program[i].allocPages(l_handleA[i], l_newAllocA[i], l_numDescDdrWords * GEMX_ddrWidth +
                                                l_NNZ * GEMX_ddrWidth / GEMX_spmvWidth +
                                                l_numPaddingDdrWords * GEMX_ddrWidth);
  	l_pageB[i] = l_program[i].allocPages(l_handleB[i], l_newAllocB[i], l_K * 1);
  	l_pageC[i] = l_program[i].allocPages(l_handleC[i], l_newAllocC[i], l_M * 1);
  	
	// Get addresses where matrices are stored
	l_matA.init(l_M, l_K, l_NNZ, 0, l_program[i].getPageAddr(l_pageA[i]));
  	l_matB[i].init(l_K, 1, 1, l_program[i].getPageAddr(l_pageB[i]));
  	l_matC[i].init(l_M, 1, 1, l_program[i].getPageAddr(l_pageC[i]));
  	
	// Fill inputs with random data
	if (l_newAllocA[i]) {
	  if (l_mtxFile.good()) {
	    l_matA.fillFromVector(l_mtxFile.getRows());
	  } else {
	    l_matA.fillMod(std::numeric_limits<GEMX_dataType>::max());
	  }
  	}
  	if (l_newAllocB[i]) {
	  l_matB[i].fillMod(9);
  	}
  	
	//############  Compile program for the accelerator with a single SPMV instruction  ############
  	// SpmvArgsType, KargsType - helper types to compile and check the accelerator program
     SpmvArgsType l_spmvArgs(
      l_pageA[i], l_pageB[i], l_pageC[i],
      l_M, l_K, l_NNZ, l_Cblocks, l_numDescPages
    );
    l_kargs[i].setSpmvArgs(l_spmvArgs);
    l_kargs[i].store(l_program[i].addInstr(), 0);
    std::cout << "Added instruction SPMV M = " << l_M << " K = " << l_K << " NNZ = " << l_NNZ << "  \n";
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
  unsigned long int l_Ops = 2ull * l_NNZ;
  unsigned long int theory_cycles = 2 * l_M / 16 + l_K / 16 + l_NNZ / 8;
  double l_effCycles;
  KargsType l_kargsRes[GEMX_numKernels];
  KargsOpType l_op[GEMX_numKernels];
  gemx::InstrResArgs l_instrRes[GEMX_numKernels];
  unsigned long int l_cycleCount[GEMX_numKernels];
  unsigned long int l_maxCycleCount=0;
  double l_timeKernelInMs[GEMX_numKernels];
  double l_maxTimeKernelInMs=0;
  double l_perfKernelInGops[GEMX_numKernels];
  double l_totalPerfKernelInGops=0;
  double l_perfApiInGops;
  double l_timeMsAt100pctEff;
  //double l_effKernelPct;
  double l_effApiPct;

  for (int i=0; i<GEMX_numKernels; ++i) {
    l_op[i] = l_kargsRes[i].load(l_program[i].getBaseResAddr(), 0);
    assert(l_op[i] == KargsType::OpResult);
    l_instrRes[i] = l_kargsRes[i].getInstrResArgs();
    l_cycleCount[i] = l_instrRes[i].getDuration();
    l_maxCycleCount = (l_cycleCount[i] > l_maxCycleCount)? l_cycleCount[i]: l_maxCycleCount;
    l_timeKernelInMs[i] = l_cycleCount[i] / (l_boardFreqMHz * 1e6) * 1e3;
    l_maxTimeKernelInMs = (l_timeKernelInMs[i] > l_maxTimeKernelInMs)? l_timeKernelInMs[i]: l_maxTimeKernelInMs;
    l_perfKernelInGops[i] = l_Ops / (l_timeKernelInMs[i] * 1e-3) / 1e9;
    l_totalPerfKernelInGops += l_perfKernelInGops[i];
  }
  
  l_effCycles = 100 * theory_cycles / l_maxCycleCount;
  l_perfApiInGops = (l_Ops*GEMX_numKernels) / (l_timeApiInMs * 1e-3) / 1e9;
  l_timeMsAt100pctEff = theory_cycles / (l_boardFreqMHz * 1e6) * 1e3;
  //l_effKernelPct = 100 * l_timeMsAt100pctEff / l_maxTimeKernelInMs;
  l_effCycles = 100 * theory_cycles / l_maxCycleCount;
  l_effApiPct = 100 * l_timeMsAt100pctEff / l_timeApiInMs;
  // Show time, Gops in csv format
  std::cout << std::string("DATA_CSV:,DdrWidth,Freq,M,K,NNZ,")
             + "KernelCycles,"
             + "TimeKernelMs,TimeApiMs,"
             + "EffKernelPct,EffApiPct,"
             + "PerfKernelGops,PerfApiGops\n"
            << "DATA_CSV:," <<  GEMX_ddrWidth << "," << l_boardFreqMHz << ","
            << l_M << "," << l_K << "," << l_NNZ << ","
            << l_maxCycleCount << ","
            << l_maxTimeKernelInMs << "," << l_timeApiInMs << ","
            << l_effCycles<<","<<l_effApiPct<<","
            << l_totalPerfKernelInGops << "," << l_perfApiInGops
            << std::endl;

  //############  Compare tha FPGA results with the reference results  ############
  // Calculate reference C = A * B
  // Since the reference is not needed on the acclerator allocate memory in any way
  std::cout << "INFO: Calculating gold values on host ..." << std::endl;
  // C matrix reference calculated on the host - l_matCref
  std::vector<GEMX_dataType> l_CmatAlloc;
  l_CmatAlloc.resize(l_M * 1);
  for (int i=0; i<GEMX_numKernels; ++i) {
  	assert((l_matC[i].rows() == l_M) && (l_matC[i].cols() == 1));
  }
  MatType l_matCref(l_M, 1, 1, l_CmatAlloc.data());
  bool l_calcGold = (l_NNZ < 450000);
  if (l_calcGold) {
    l_matCref.multiplySpmv(l_matA, l_matB[0]);
  } else {
    std::cout << "INFO: skipped gold calculation on host since it may take too long\n";
  }
  // C matrix in the DDR image received from the accelerator - l_matCfpga
  MatType l_matCfpga[GEMX_numKernels];
  for (int i=0; i<GEMX_numKernels; ++i) {
	l_matCfpga[i].init(l_M, 1, 1, l_program[i].getPageAddr(l_pageC[i]));  
  }
  
  // Compare FPGA and the reference
  if (l_calcGold) {
    float  l_TolRel = 1e-3,  l_TolAbs = 1e-5;
    bool l_ok;
    bool l_pass = true;
    for (int i=0; i<GEMX_numKernels; ++i) {
	l_ok = l_matCfpga[i].cmp(l_TolRel, l_TolAbs, l_matCref);
    	std::cout << "INFO: accelerator kernel " << i << " result " << (l_ok ? "MATCHES" : "does NOT match") << " the reference\n";
    	std::cout << "INFO: status " << (l_ok ? "PASS" : "FAIL") << "\n";
        if (!l_ok){
        	l_pass = false;
        }
    }
    return l_pass ? EXIT_SUCCESS : EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
  return EXIT_SUCCESS;
}