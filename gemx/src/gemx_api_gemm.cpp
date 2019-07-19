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
 *  $DateTime: 2017/08/18 08:31:34 $
 *  $Author: jzejda $
 */

// Prerequisites:
//  - Compiled C++ to bitstream accelerator kernel
//     - use "make run_hw"
//     - or get a pre-compiled copy of the out_hw/gemx.xclbin)
// Compile and run this API example:
//   make out_host/gemx_api_gemm.exe
//   out_host/gemx_api_gemm.exe
//     # No argumens will show help message
// You can also test it with a cpu emulation accelerator kernel (faster to combile, make run_cpu_em)
//   ( setenv XCL_EMULATION_MODE true ; out_host/gemx_api_gemm.exe out_cpu_emu/gemx.xclbin )


#include "gemx_kernel.h"
#include "gemx_gen_gemm.h"
#include "gemx_api_test.h"

int main(int argc, char **argv)
{
  //############  UI and GEMM problem size  ############
  if (argc < 2) {
    std::cerr << "Usage:\n"
              <<  "  gemx_api_gemm.exe <path/gemx.xclbin> [M K N  [LdA LdB LdC LdX postScaleVal postScaleShift] ]\n"
              <<  "  Examples:\n"
              <<  "    gemx_api_gemm.exe   out_hw/gemx.xclbin\n"
              <<  "    gemx_api_gemm.exe   out_hw/gemx.xclbin  256 256 256\n"
              <<  "    gemx_api_gemm.exe   out_hw/gemx.xclbin  256 256 256  256 256 256\n"
              <<  "    gemx_api_gemm.exe   out_hw/gemx.xclbin  256 256 256  256 256 256  256 1 0\n";
    exit(2);
  }
  unsigned int l_argIdx = 1;
  std::string l_xclbinFile(argv[l_argIdx]);

  // Row major  C  M rows N cols  =  A  M rows K cols  *  B  K rows N cols
  //   MatType - tensor like type to allocate/store/align memory; you can use your own type instead
  //   Min size is the array edge (e.g., 32 on ku115), see GenGemm::check() for example of arg checking functions
  unsigned int l_ddrW = GEMX_ddrWidth;
  // the smallest matrices for flow testing
  unsigned int l_M = l_ddrW * GEMX_gemmMBlocks,  l_K = l_ddrW * GEMX_gemmKBlocks,  l_N = l_ddrW * GEMX_gemmNBlocks;  
  if (argc > ++l_argIdx) {l_M = atoi(argv[l_argIdx]);}  
  if (argc > ++l_argIdx) {l_K = atoi(argv[l_argIdx]);}  
  if (argc > ++l_argIdx) {l_N = atoi(argv[l_argIdx]);}  
  unsigned int l_LdA = l_K,  l_LdB = l_N,  l_LdC = l_N, l_LdX = l_N;
  int32_t l_postScaleVal = 1, l_postScaleShift = 0;
  if (argc > ++l_argIdx) {l_LdA = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_LdB = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_LdC = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_LdX = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_postScaleVal = atoi(argv[l_argIdx]);}
  if (argc > ++l_argIdx) {l_postScaleShift= atoi(argv[l_argIdx]);}
  int32_t l_postScale = (l_postScaleVal << 8) | (l_postScaleShift & 0x000000ff);
  
  printf("GEMX-gemm C++ API example using accelerator image \n",
         l_xclbinFile.c_str());
    
  //############  Client code - prepare the gemm problem input  ############
  GenGemm l_gemm;
  
  if (! l_gemm.check(l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX)) {
    return EXIT_FAILURE;
  }  
  
  ProgramType l_program[GEMX_numKernels];  // Holds instructions and controls memory allocation
  
  std::string l_handleA[GEMX_numKernels];
  std::string l_handleB[GEMX_numKernels];
  std::string l_handleC[GEMX_numKernels];
  std::string l_handleX[GEMX_numKernels];

  for (int i=0; i<GEMX_numKernels; ++i) {
      l_handleA[i] = "A"+std::to_string(i);
      l_handleB[i] = "B"+std::to_string(i);
      l_handleC[i] = "C"+std::to_string(i);
      l_handleX[i] = "X"+std::to_string(i);

    l_gemm.addInstr(l_program[i], l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX, l_postScale, l_handleA[i], l_handleB[i], l_handleC[i], l_handleX[i], false);
    std::cout << "In kernel " << i << " ";
    std::cout << "Added instruction GEMM (" << l_M << "x" << l_K <<" * "<< l_K << "x" << l_N << " + " << l_M << "x" << l_N << ") * " << l_postScaleVal <<" >> " << l_postScaleShift <<"\n";
  }
  
  //############  Run FPGA accelerator  ############

  double l_timeApiInMs = run_hw_test(l_xclbinFile, l_program);
 
  //############  Get the exact kernel time from HW cycle counters on the accelerator  ############
  float l_boardFreqMHz = getBoardFreqMHz(l_xclbinFile);
  unsigned long int l_Ops = 2ull * l_M * l_N * l_K + l_M * l_N * 3;
  unsigned long int l_Parallel_Ops = 2ull * l_M * l_N * l_K;
  KargsType l_kargsRes[GEMX_numKernels];
  KargsOpType l_op[GEMX_numKernels];
  gemx::InstrResArgs l_instrRes[GEMX_numKernels];
  unsigned long int l_cycleCount[GEMX_numKernels];
  unsigned long int l_maxCycleCount=0;
  double l_timeKernelInMs[GEMX_numKernels];
  double l_maxTimeKernelInMs=0;
  double l_perfKernelInTops[GEMX_numKernels];
  double l_totalPerfKernelInTops=0;
  double l_perfApiInTops;
  double l_timeMsAt100pctEff;
  double l_effKernelPct;
  double l_effApiPct;

  for (int i=0; i<GEMX_numKernels; ++i) {
      l_op[i] = l_kargsRes[i].load(l_program[i].getBaseResAddr(), 0);
      assert(l_op[i] == KargsType::OpResult);
      l_instrRes[i] = l_kargsRes[i].getInstrResArgs();
      l_cycleCount[i] = l_instrRes[i].getDuration();
      l_maxCycleCount = (l_cycleCount[i] > l_maxCycleCount)? l_cycleCount[i]: l_maxCycleCount;
      l_timeKernelInMs[i] = l_cycleCount[i] / (l_boardFreqMHz * 1e6) * 1e3;
      l_maxTimeKernelInMs = (l_timeKernelInMs[i] > l_maxTimeKernelInMs)? l_timeKernelInMs[i]: l_maxTimeKernelInMs;
      l_perfKernelInTops[i] = l_Ops / (l_timeKernelInMs[i] * 1e-3) / 1e12;
      l_totalPerfKernelInTops += l_perfKernelInTops[i];
  }
  l_perfApiInTops = (l_Ops*GEMX_numKernels) / (l_timeApiInMs * 1e-3) / 1e12;
  l_timeMsAt100pctEff = l_Parallel_Ops / 2 / GEMX_ddrWidth / GEMX_ddrWidth / (l_boardFreqMHz * 1e6) * 1e3;
  l_effKernelPct = (100 * l_timeMsAt100pctEff / l_maxTimeKernelInMs < 100)?(100 * l_timeMsAt100pctEff / l_maxTimeKernelInMs):100;
  l_effApiPct = 100 * l_timeMsAt100pctEff / l_timeApiInMs;
  // Show time, Tops in csv format
  std::cout << std::string("DATA_CSV:,DdrWidth,Freq,M,K,N,")
             + "Ops,KernelCycles,"
             + "TimeKernelMs,TimeApiMs,"
             + "EffKernelPct,EffApiPct,"
             + "PerfKernelTops,PerfApiTops\n"
            << "DATA_CSV:," <<  GEMX_ddrWidth << "," << l_boardFreqMHz << ","
            << l_M << "," << l_K << "," << l_N << "," 
            << l_Ops*GEMX_numKernels << "," << l_maxCycleCount << ","
            << l_maxTimeKernelInMs << "," << l_timeApiInMs << ","
            << l_effKernelPct << "," << l_effApiPct << ","
            << l_totalPerfKernelInTops << "," << l_perfApiInTops
            << std::endl;
  return EXIT_SUCCESS;
}

  
