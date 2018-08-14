
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

#include "gemx_xblas.h"

#define l_calcGold 1
#define performance_check 1

//enum CBLAS_ORDER { CblasRowMajor, CblascolMajor };
//enum CBLAS_TRANSPOSE { CblasNoTrans, CblasTrans };

#ifdef  __cplusplus
extern "C" {
#endif
    static bool hasLoadXclbin = false;
    static gemx::Fpga l_fpga;
    
    
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
	  l_freq = 220.7;
      std::cout << "INFO: Failed to get board frequency by xbsak. This is normal for cpu and hw emulation, using -1 MHz for reporting.\n";
    }
    return(l_freq);
  }
  
  void clearMemory(){
    if(hasLoadXclbin){
      l_fpga.clearAll();
    }else{
      std::cout<<"Error: xclbin hasn't been loaded yet \n";
    }
  }

   void xblas_sgemm(const enum CBLAS_ORDER __Order, const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N, const int __K, const GEMX_dataType __alpha, std::vector<GEMX_dataType>& __A, const int __lda, std::vector<GEMX_dataType>& __B, const int __ldb, GEMX_dataType __beta, std::vector<GEMX_dataType>& __C, const int __ldc){   
    
    if (__Order == CblascolMajor){
      std::cerr << "ERROR: Column major order is not supported." << "\n";
    }
    //############  Client code - prepare the gemm problem input  ############
    ProgramType l_program[GEMX_numKernels];
    std::string l_handleA[GEMX_numKernels];
    std::string l_handleB[GEMX_numKernels];
    std::string l_handleC[GEMX_numKernels];
    std::string l_handleX[GEMX_numKernels];
    //std::string l_handleA("A"), l_handleB("B"), l_handleC("C");
    //bool l_newAllocA, l_newAllocB, l_newAllocC;
    bool l_newAllocA[GEMX_numKernels];
    bool l_newAllocB[GEMX_numKernels];
    bool l_newAllocC[GEMX_numKernels];
    bool l_newAllocX[GEMX_numKernels];
    
    unsigned int l_pageA[GEMX_numKernels];
    unsigned int l_pageB[GEMX_numKernels];
    unsigned int l_pageC[GEMX_numKernels];
    unsigned int l_pageX[GEMX_numKernels];
    
    MatType l_matA[GEMX_numKernels];
    MatType l_matB[GEMX_numKernels];
    MatType l_matC[GEMX_numKernels];
    XMatType l_matX[GEMX_numKernels];
    
    GemmArgsType l_gemmArgs[GEMX_numKernels];
    KargsType l_kargs[GEMX_numKernels];
    int32_t l_postScale;
  // unsigned int l_pageA, l_pageB, l_pageC;
  // MatType l_matA, l_matB, l_matC;
  // GemmArgsType l_gemmArgs;
  // KargsType l_kargs;

    for (int i=0; i<GEMX_numKernels; ++i) {
      l_handleA[i] = "A"+std::to_string(i);
      l_handleB[i] = "B"+std::to_string(i);
      l_handleC[i] = "C"+std::to_string(i);
      l_handleC[i] = "X"+std::to_string(i);
      
      l_pageA[i] = l_program[i].allocPages(l_handleA[i], l_newAllocA[i], __M * __lda);
      l_pageB[i] = l_program[i].allocPages(l_handleB[i], l_newAllocB[i], __K * __ldb);
      l_pageC[i] = l_program[i].allocPages(l_handleC[i], l_newAllocC[i], __M * __ldc);
      l_pageX[i] = l_program[i].allocPages(l_handleX[i], l_newAllocX[i], __M * __ldc * (sizeof(GEMX_XdataType)/sizeof(GEMX_dataType)));
      l_matA[i].init(__M, __K, __lda, l_program[i].getPageAddr(l_pageA[i]));
      l_matB[i].init(__K, __N, __ldb, l_program[i].getPageAddr(l_pageB[i]));
      l_matC[i].init(__M, __N, __ldc, l_program[i].getPageAddr(l_pageC[i]));
      l_matX[i].init(__M, __N, __ldc, (GEMX_XdataType *) l_program[i].getPageAddr(l_pageX[i]));
      for (int row = 0; row < l_matA[i].rows();  ++row) {
	for (int col = 0; col < l_matA[i].cols();  ++col) {
	  GEMX_dataType l_val = __A[row * __M + col];
	  l_matA[i].getVal(row, col) = l_val;
	}
      }
      
      for (int row = 0; row < l_matB[i].rows();  ++row) {
	for (int col = 0; col < l_matB[i].cols();  ++col) {
	  GEMX_dataType l_val = __B[row * __K + col];
	  l_matB[i].getVal(row, col) = l_val;
	}
      }
      if(((int)__alpha == 1) && ((int)__beta != 0)){
	for (int row = 0; row < l_matX[i].rows();  ++row) {
	  for (int col = 0; col < l_matX[i].cols();  ++col) {
	    GEMX_dataType l_val = __C[row * __K + col];
	    l_matX[i].getVal(row, col) = __beta * l_val;
	  }
	}
      }else{
	l_matX[i].fillModRange(0,0);
      }
      
      if(((int)__alpha != 1) && ((int)__beta == 0)){
	l_postScale = (__alpha << 8) | (0 & 0x000000ff);
      }else{
	l_postScale = 256;
      }
      
      l_gemmArgs[i].init(
	  l_pageA[i], l_pageB[i], l_pageC[i],l_pageX[i],
	  __M, __K, __N,
	  __lda, __ldb, __ldc,__ldc,
	  l_postScale	  
      );
      l_kargs[i].setGemmArgs(l_gemmArgs[i]);
      l_kargs[i].store(l_program[i].addInstr(), 0);
    }
  
      
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
    
    std::string kernelNames[GEMX_numKernels];
    
    for (int i=0; i<GEMX_numKernels; ++i){
	  kernelNames[i] = "gemxKernel_" + std::to_string(i);
    }
    //Please set environment variable of the path of gemx.xclbin
    std::string l_xclbinFile = std::getenv("GEMX_XCLBIN_PATH");
    
    if(!hasLoadXclbin){
      if (l_fpga.loadXclbin(l_xclbinFile, kernelNames)) {
	hasLoadXclbin = true;
	std::cout << "INFO: created kernels" << std::endl;
      } else {
	std::cerr << "ERROR: failed to load " + l_xclbinFile + "\n";
      }
      showTimeData("loadXclbin", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
    }
    //create buffers for transferring data to FPGA
    if (!l_fpga.createBuffers(l_memDesc)) {
      std::cerr << "ERROR: failed to create buffers for transffering data to FPGA DDR\n";
    }
    showTimeData("created buffers", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

    // Transfer data to FPGA
    if (l_fpga.copyToFpga()) {
      std::cout << "INFO: transferred data to FPGA" << std::endl;
    } else {
      std::cerr << "ERROR: failed to copy data to FPGA DDR\n";
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
    }
    showTimeData("callKernel", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

    // Transfer data back to host - due to lazy evaluation this is generally wheer the accelerator performs the work
    if (l_fpga.copyFromFpga()) {
      (VERBOSE > 0) && std::cout << "INFO: Transferred data from FPGA" << std::endl;
    } else {
      std::cerr << "ERROR: failed to copy data from FPGA DDR\n";
    }
    showTimeData("copyFromFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
    
    //############  Write the results back to the array of Matrix C  ############
    
    if (((int)__alpha == 1) && ((int)__beta == 0)){
      for (int row = 0; row < l_matC[0].rows();  ++row) {
	for (int col = 0; col < l_matC[0].cols();  ++col) {
	  //C= A*B
	  __C[row * __M + col] = l_matC[0].getVal(row, col);
	}
      }
    }else if(((int)__alpha != 1) && ((int)__beta == 0)){
	//C= alpha*A*B
      for (int row = 0; row < l_matC[0].rows();  ++row) {
	for (int col = 0; col < l_matC[0].cols();  ++col) {
	__C[row * __M + col] = l_matC[0].getVal(row, col); //multiply has done in kernel
	}
      }
    }else if(((int)__alpha == 1) && ((int)__beta != 0)){
	//C= A*B+ beta*C
      for (int row = 0; row < l_matC[0].rows();  ++row) {
	for (int col = 0; col < l_matC[0].cols();  ++col) {
	  __C[row * __M + col] = l_matC[0].getVal(row, col); //addition has done in kernel
	}
      }
    }else{
      for (int row = 0; row < l_matC[0].rows();  ++row) {
	for (int col = 0; col < l_matC[0].cols();  ++col) {
	  //C= alpha*A*B + beta*C
	  __C[row * __M + col] = __alpha * l_matC[0].getVal(row, col) + __beta * __C[row * __M + col];
	}
      }
    }
    
  showTimeData("writeToArray", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
  
  showTimeData("total", l_tp[0], l_tp[l_tpIdx]); l_tpIdx++;
  double l_timeApiInMs = -1;
  showTimeData("subtotalFpga", l_tp[2], l_tp[l_tpIdx], &l_timeApiInMs); l_tpIdx++; // Host->DDR, kernel, DDR->host

    //############  Get the exact kernel time from HW cycle counters on the accelerator  ############
    unsigned long int l_Ops = 2ull * __M * __N * __K;
    if (performance_check) {
    float l_boardFreqMHz = getBoardFreqMHz(0);
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
    l_timeMsAt100pctEff = l_Ops / 2 / GEMX_ddrWidth / GEMX_ddrWidth / (l_boardFreqMHz * 1e6) * 1e3;
    l_effKernelPct = 100 * l_timeMsAt100pctEff / l_maxTimeKernelInMs;
    l_effApiPct = 100 * l_timeMsAt100pctEff / l_timeApiInMs;
    // Show time, Tops in csv format
    std::cout << std::string("DATA_CSV:,DdrWidth,Freq,M,K,N,")
	      + "Ops,KernelCycles,"
	      + "TimeKernelMs,TimeApiMs,"
	      + "EffKernelPct,EffApiPct,"
	      + "PerfKernelTops,PerfApiTops\n"
	      << "DATA_CSV:," <<  GEMX_ddrWidth << "," << l_boardFreqMHz << ","
	      << __M << "," << __K << "," << __N << ","
	      << l_Ops*GEMX_numKernels << "," << l_maxCycleCount << ","
	      << l_maxTimeKernelInMs << "," << l_timeApiInMs << ","
	      << l_effKernelPct << "," << l_effApiPct << ","
	      << l_totalPerfKernelInTops << "," << l_perfApiInTops
	      << std::endl;
  
    }
    //############  Compare tha FPGA results with the reference results  ############
    bool l_calcGold_check = (l_Ops <= 2ull * 2048*2048*2048); //If matrix size is too big, the comparasion will take too long time on the host side, so it will be skipped by default. 
    if (l_calcGold && l_calcGold_check) {
      std::vector<GEMX_dataType> l_CmatAlloc;
      l_CmatAlloc.resize(__M * __ldc);
      
      MatType l_matCref(__M, __N, __ldc, l_CmatAlloc.data());
      if(((int)__alpha == 1) && ((int)__beta != 0)){
	l_matCref.multiplyAddScale(l_matA[0], l_matB[0], l_matX[0], l_postScale);
      }else if(((int)__alpha != 1) && ((int)__beta == 0)){
	l_matCref.multiplyAddScale(l_matA[0], l_matB[0], l_matX[0], l_postScale);
      }else{
	l_matCref.multiply(l_matA[0], l_matB[0]);
      }
      MatType l_matCfpga[GEMX_numKernels];
      for (int i=0; i<GEMX_numKernels; ++i) {
	l_matCfpga[i].init(__M, __N, __ldc, l_program[i].getPageAddr(l_pageC[i]));  
      }
      
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
    } else {
      std::cout << "INFO: skipped gold calculation on host\n";   
    }

  }

#ifdef __cplusplus
}
#endif
