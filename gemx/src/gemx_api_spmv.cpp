
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
 *  @brief Simple SPMV example of C++ API client interaction with SPMV linear algebra accelerator on Xilinx FPGA 
 *
 *  $DateTime: 2017/11/16$
 *  $Author: Yifei $
 */

// Prerequisites:
//  - Compiled C++ to bitstream accelerator kernel
//     - use "make run_hw"
//     - or get a pre-compiled copy of the out_hw/gemx.xclbin)
// Compile and run this API example:
//   make out_host/gemx_api_spmv.exe
//   out_host/gemx_api_spmv.exe
//     # No argumens will show help message
// You can also test it with a cpu emulation accelerator kernel (faster to combile, make run_cpu_em)
//   ( setenv XCL_EMULATION_MODE true ; out_host/gemx_api_spmv.exe out_cpu_emu/gemx.xclbin )
 
#include "gemx_kernel.h"
#include "gemx_gen_spmv.h"
#include "gemx_api_test.h"


int main(int argc, char **argv)
{
  //############  UI and SPMV problem size  ############
  if (argc < 5) {
    std::cerr << "Usage:\n"
              <<  "  gemx_api_spmv.exe <path/gemx.xclbin> [M K Nnz [mtxFile]]\n"
              <<  "  Examples:\n"
              <<  "    gemx_api_spmv.exe   out_hw/gemx.xclbin  96 128 256\n"
              <<  "    gemx_api_spmv.exe   out_hw/gemx.xclbin  0 0 0 data/spmv/diag16.mtx\n";
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
  std::string l_mtxFileName("none");
  std::string l_vectorFileName("none");
  if (argc > ++l_argIdx) {l_mtxFileName = argv[l_argIdx];}
  //this is a debug only option, not necessary 
  if (argc > ++l_argIdx) {l_vectorFileName = argv[l_argIdx];}
  
  #if GEMX_useURAM
  MtxFileUram l_mtxFile(l_mtxFileName);
  GenSpmvUram l_spmv;
  #else	 
  MtxFile l_mtxFile(l_mtxFileName);
  GenSpmv l_spmv;
  #endif
  //The check() modifies the dimensions when loading from a matrix file. Please use 0 for l_M, l_K and l_NNZ when provding matrix file
  l_spmv.check(l_M, l_K, l_NNZ, l_mtxFile);
  
  //std::cout << "M = " << l_M << " K = " << l_K << " NNZ = " << l_NNZ << " name =  "<<l_mtxFileName<<"      ";
  
  printf("GEMX-spmv C++ API example using accelerator image \n",
         l_xclbinFile.c_str());
    
  //############  Client code - prepare the spmv problem input  ############
  ProgramType l_program[GEMX_numKernels];  // Holds instructions and controls memory allocation
  ProgramType l_program_golden;
  std::string l_handleA[GEMX_numKernels];
  std::string l_handleB[GEMX_numKernels];
  std::string l_handleC[GEMX_numKernels];
  
  for (int i=0; i<GEMX_numKernels; ++i) {
    l_handleA[i] = "A"+std::to_string(i);
    l_handleB[i] = "B"+std::to_string(i);
    l_handleC[i] = "C"+std::to_string(i);
  } 

  for (int i=0; i<GEMX_numKernels; ++i) {
    #if GEMX_useURAM
    l_spmv.addInstr(l_program[i], l_M, l_K, l_NNZ, l_mtxFile, l_handleA[i], l_handleB[i], l_handleC[i],false);
    #else
    if (l_vectorFileName == "none") {
       l_spmv.addInstr(l_program[i], l_M, l_K, l_NNZ, l_mtxFile, l_handleA[i], l_handleB[i], l_handleC[i], false, false);
    }else{
       l_spmv.addInstrReadVector(l_program[i], l_M, l_K, l_NNZ, l_mtxFile, l_vectorFileName, l_handleA[i], l_handleB[i], l_handleC[i], false, false);
    }
    #endif
  }
  #if GEMX_useURAM
  l_spmv.addInstr(l_program_golden, l_M, l_K, l_NNZ, l_mtxFile, l_handleA[0], l_handleB[0], l_handleC[0], true);
  #else	
  if (l_vectorFileName == "none") {
    l_spmv.addInstr(l_program_golden, l_M, l_K, l_NNZ, l_mtxFile, l_handleA[0], l_handleB[0], l_handleC[0], false, true);
  }else{
    l_spmv.addInstrReadVector(l_program_golden, l_M, l_K, l_NNZ, l_mtxFile, l_vectorFileName, l_handleA[0], l_handleB[0], l_handleC[0], false, true);
  } 
  #endif
    
  //############  Run FPGA accelerator  ############

  double l_timeApiInMs = run_hw_test(l_xclbinFile, l_program);
  
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
  std::string matrixName = l_mtxFileName.substr(l_mtxFileName.find_last_of("/")+1);
  // Show time, Gops in csv format
  std::cout << std::string("DATA_CSV:,DdrWidth,Freq,M,K,NNZ,MatrixName,")
             + "KernelCycles,"
             + "TimeKernelMs,TimeApiMs,"
             + "EffKernelPct,EffApiPct,"
             + "PerfKernelGops,PerfApiGops\n"
            << "DATA_CSV:," <<  GEMX_ddrWidth << "," << l_boardFreqMHz << ","
            << l_M << "," << l_K << "," << l_NNZ << "," << matrixName << ","
            << l_maxCycleCount << ","
            << l_maxTimeKernelInMs << "," << l_timeApiInMs << ","
            << l_effCycles<<","<<l_effApiPct<<","
            << l_totalPerfKernelInGops << "," << l_perfApiInGops
            << std::endl;

  if(!getenv("SKIPPED_GOLD_CAL")){
        float  l_TolRel = 1e-3,  l_TolAbs = 1e-5;
        compareMultiInstrs(l_TolRel, l_TolAbs, l_program_golden, l_program[0]);
    }else{
        std::cout << "INFO: skipped gold calculation on host since it may take too long\n" << std::endl;
    }    
  return EXIT_SUCCESS;
}
