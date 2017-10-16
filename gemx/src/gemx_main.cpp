// Copyright (c) 2017
// Xilinx, Inc.
// All rights reserved.
// $Id: //Rodin/Proj/O/OPENCL_APPS_DEV/src/matrix_mult/gemx/src/gemx_main.cpp#12 $
// 
// No part of this document may be copied, transmitted or
// disclosed in any form or fashion without the express
// written consent of Xilinx, Inc.

/**
 *  @brief Main executable for SDX flow
 *
 *  $DateTime: 2017/10/11 14:33:33 $
 *  $Author: lingl $
 */

// Fast Csim compile
//   make host

// Fast run on board
//  ( gdb --args ./gemx.exe k app.bin app_out.bin )
 
// Fast CPU emu
// ( setenv XCL_EMULATION_MODE true ; ./out_host/gemx_host.exe out_cpu_emu/gemx.xclbin out_host/app.bin out_cpu_emu/app_out.bin )

 
#include <stdio.h>
//#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

#include "gemx_kernel.h"
#if TEST_SDX
  #include "gemx_fpga.h"
#endif

std::ifstream::pos_type getFileSize(std::string p_FileName)
{
  std::ifstream in(p_FileName.c_str(), std::ifstream::ate | std::ifstream::binary);
  return in.tellg(); 
}

std::vector<DdrType>
loadBinFile(std::string p_BinFileName)
{
  std::vector<DdrType> l_memVec;
  // Bin file existence
  std::ifstream l_if(p_BinFileName.c_str(), std::ios::binary);
  if (l_if.is_open()) {
    // Bin file size
    size_t l_binFileSize = getFileSize(p_BinFileName);
    std::cout << "INFO: loading " + p_BinFileName + " of size " << l_binFileSize << "\n";
    assert(l_binFileSize > 0);
    size_t l_binFileSizeInDdrWords = l_binFileSize / sizeof(DdrType);
    assert(l_binFileSize % sizeof(DdrType) == 0);

    // Bin file storage
    //l_memVec.reserve(l_binFileSizeInDdrWords);
    l_memVec.resize(l_binFileSizeInDdrWords);
    DdrType *l_mem = &l_memVec[0];

    // Read the bin file
    l_if.read((char*)l_mem, l_binFileSize);
    if (l_if) {
      std::cout << "INFO: loaded " << l_binFileSize << " bytes from " << p_BinFileName << "\n";
    } else {
      l_memVec.clear();
      std::cout << "ERROR: loaded only " << l_if.gcount() << " bytes from " << p_BinFileName << "\n";
    }
    l_if.close();

    // Debug print the file content
  } else {
    std::cout << "ERROR: failed to open file " + p_BinFileName + "\n";
  }

  return(l_memVec);
}

bool
writeBinFile(std::string p_BinFileName, std::vector<DdrType> &p_MemVec)
{
  bool ok = false;  
  std::ofstream l_of(p_BinFileName.c_str(), std::ios::binary);
  if (l_of.is_open()) {
    size_t l_sizeBytes =  sizeof(DdrType) * p_MemVec.size();
    l_of.write((char*)&p_MemVec[0], l_sizeBytes);
    if (l_of.tellp() == l_sizeBytes) {
      std::cout << "INFO: wrote " << l_sizeBytes << " bytes to " << p_BinFileName << "\n";
      ok = true;
    } else {
      std::cout << "ERROR: wrote only " << l_of.tellp() << " bytes to " << p_BinFileName << "\n";
    }
    l_of.close();
  }
  return(ok);
}

#if TEST_SDX
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePointType;

  void
  showTimeData(std::string p_Task, TimePointType &t1, TimePointType &t2)
  {
    t2 = std::chrono::high_resolution_clock::now();    
    std::chrono::duration<double> l_durationSec = t2 - t1;
    std::cout << "  DATA: time " << p_Task
              << "  " << std::fixed << std::setprecision(6)
              << l_durationSec.count() * 1e3 << " msec\n";
  }
#endif

int main(int argc, char** argv)
{
  if (argc < 4){
    printf("ERROR: passed %d arguments instead of %d, exiting\n",
           argc, 4);
    printf("  Usage:\n    gemx_sdx.exe  gemx.xclbin  app.bin  app_out.bin [kernel_id] [kernelName_id]\n");
    return EXIT_FAILURE;
  }
  
  std::string l_xclbinFile(argv[1]);
  std::string l_binFile(argv[2]);
  std::string l_binFileOut(argv[3]);
  unsigned int l_kernelId = 0;
  unsigned int l_kernelNameId = 0;
  if (argc > 4) {
    l_kernelId = atoi(argv[4]);
    assert(l_kernelId < 4);
  }
  if (argc > 5) {
	l_kernelNameId = atoi(argv[5]);
	assert(l_kernelNameId < 4);
  }
  else {
	l_kernelNameId = l_kernelId;
  }

  printf("GEMX:   %s  %s  %s %s\n",
         argv[0], l_xclbinFile.c_str(), l_binFile.c_str(), l_binFileOut.c_str());
  
  // Load the bin file
  std::vector<DdrType> l_memVec = loadBinFile(l_binFile);
  if (l_memVec.empty()) {
    return EXIT_FAILURE;
  }
  DdrType *l_mem = &l_memVec[0];
  
  
  std::vector<DdrType> l_memVecOut;
  #if TEST_SDX
    #include <chrono>
    TimePointType l_tp[10];
    unsigned int l_tpIdx = 0;
    l_tp[l_tpIdx] = std::chrono::high_resolution_clock::now(); 
    
    // ################# HW run through SDX #################
    // Init FPGA
    gemx::Fpga l_fpga(l_kernelId);
    std::string l_kernelName("gemxKernel_"+std::to_string(l_kernelNameId));
    if (l_fpga.loadXclbinSingleKernel(l_xclbinFile, l_kernelName)) {
      std::cout << "INFO: created kernel" + l_kernelName<< std::endl;
    } else {
      std::cerr << "ERROR: failed to load " + l_xclbinFile + "\n";
      return EXIT_FAILURE;
    }
    showTimeData("loadXclbin", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

    // Transfer data to FPGA
    gemx::MemDesc l_memDesc(l_memVec.size() * sizeof(DdrType) / GEMX_pageSizeBytes, l_memVec.data());
    assert(l_memVec.size() * sizeof(DdrType) % GEMX_pageSizeBytes == 0);
    if (l_fpga.copyToFpgaSingleKernel(l_memDesc)) {
      std::cout << "INFO: transferred data to FPGA" << std::endl;
    } else {
      std::cerr << "ERROR: failed to copy data to FPGA DDR\n";
      return EXIT_FAILURE;
    }
    showTimeData("copyToFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
  
    // Gemx kernel ops
    if (l_fpga.callSingleKernel(l_kernelName)) {
      std::cout << "INFO: Executed kernel" + l_kernelName << std::endl;
    } else {
      std::cerr << "ERROR: failed to call kernel " + l_kernelName + "\n";
      return EXIT_FAILURE;
    }
    showTimeData("callKernel", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

    // Transfer data back to host
    l_memVecOut.resize(l_memVec.size());
    gemx::MemDesc l_memDescOut(l_memDesc.sizePages(), l_memVecOut.data());
    if (l_fpga.copyFromFpgaSingleKernel(l_memDescOut)) {
      std::cout << "INFO: Transferred data from FPGA" << std::endl;
    } else {
      std::cerr << "ERROR: failed to copy data from FPGA DDR\n";
      return EXIT_FAILURE;
    }
    showTimeData("copyFromFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
    showTimeData("total", l_tp[0], l_tp[l_tpIdx]); l_tpIdx++;
    showTimeData("subtotalFpga", l_tp[1], l_tp[l_tpIdx]); l_tpIdx++; // Host->DDR, kernel, DDR->host
    
  #else
    // ################# SW run through HLS #################
    // Gemx kernel ops
    gemxKernel_0(l_mem, l_mem);
    
    l_memVecOut = l_memVec;
  
  #endif
  
  // Write out the received data
  if (!writeBinFile(l_binFileOut, l_memVecOut)) {
      std::cerr << "ERROR: failed to write output file " + l_binFileOut + "\n";
      return EXIT_FAILURE;
    }
  

  return EXIT_SUCCESS;
}

  
