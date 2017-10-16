// Copyright (c) 2017
// Xilinx, Inc.
// All rights reserved.
// $Id: //Rodin/Proj/O/OPENCL_APPS_DEV/src/matrix_mult/gemx/src/gemx_kernel.cpp#12 $
// 
// No part of this document may be copied, transmitted or
// disclosed in any form or fashion without the express
// written consent of Xilinx, Inc.

/**
 *  @brief FPGA SGEMM accelerator kernel
 *
 *  $DateTime: 2017/10/08 10:36:48 $
 *  $Author: lingl $
 */

#include <assert.h>
#include "gemx_kernel.h"

// The extern C still needed - otherwise cpu emu fails
//   prj_sda.exe: symbol lookup error: ./dltmp: undefined symbol: kernelSgemm
#if TEST_SDX
extern "C" {
#endif

void
kernelOpLow(
    DdrType *p_DdrRd,
    DdrType *p_DdrWr,
//    hls::stream<TimeStampType::OpType> &p_Control,
    hls::stream<TimeStampType::TimeType> &p_Time
  ) {
  #pragma HLS INLINE self off
  //GemvType l_gemv;
  GemvM2Mtype l_gemv;
  SpmvType l_spmv;
  GemmType l_gemm;
  TranspType l_transp;
  
  typedef KargsType::OpType KargsOpType;
  typedef gemx::ControlArgs ControlArgsType;
  typedef GemvType::GemvArgsType GemvArgsType;
  typedef GemmType::GemmArgsType GemmArgsType;
  typedef TranspType::TranspArgsType TranspArgsType;
  typedef SpmvType::SpmvArgsType SpmvArgsType;

  ///////////////////////////////////////////////////////////////////////////
  // VLIW op decoding
  ///////////////////////////////////////////////////////////////////////////
  unsigned int l_pc = 0;
  bool l_isLastOp = false;
  static const unsigned int l_tsDepth = TimeStampType::t_FifoDepth;
  
  // Checks for code, result, and data segment sizes
  KargsDdrInstrType l_code[GEMX_numInstr], l_res[GEMX_numInstr];
//#pragma HLS ARRAY_PARTITION variable=l_code complete dim=0
//#pragma HLS ARRAY_PARTITION variable=l_res  complete dim=0
  assert(sizeof(l_code) <= (GEMX_resPage - GEMX_codePage) * GEMX_pageSizeBytes);
  assert(sizeof(l_code) <= (GEMX_dataPage - GEMX_resPage) * GEMX_pageSizeBytes);

  // Prefetch all instructions for more accurate cycle measurements
  for (unsigned int l_pc = 0; l_pc < GEMX_numInstr; ++l_pc) {
    l_code[l_pc].loadFromDdr(p_DdrRd, GEMX_codePage * DdrType::per4k() +
                                      l_pc * KargsType::getInstrWidth());
  }
  
  // Decode and execute
  TimeStampType::TimeType l_tsPrev = 0;
  KargsType l_kargsRes;
  for (unsigned int l_pc = 0; l_pc < GEMX_numInstr; ++l_pc) {
    KargsType l_kargs;
    KargsOpType l_op = l_kargs.loadFromInstr(l_code[l_pc]);
    switch(l_op) {
      case KargsType::OpControl: {
        ControlArgsType l_controlArgs = l_kargs.getControlArgs();
        l_isLastOp = l_controlArgs.getIsLastOp();
        assert(!l_isLastOp || (l_pc == GEMX_numInstr - 1));
        break;
      }
      case KargsType::OpGemv: {
        GemvArgsType l_gemvArgs = l_kargs.getGemvArgs();
        if (GEMX_runGemv)
          l_gemv.runGemv(p_DdrRd, p_DdrWr, l_gemvArgs); 
        break;
      }
      case KargsType::OpGemm: {
        GemmArgsType l_gemmArgs = l_kargs.getGemmArgs();
        if (GEMX_runGemm)
          l_gemm.runGemm(p_DdrRd, p_DdrWr, l_gemmArgs);
        break;
      }
      case KargsType::OpTransp: {
        TranspArgsType l_transpArgs = l_kargs.getTranspArgs();
        if (GEMX_runTransp)
          l_transp.runTransp(p_DdrRd, p_DdrWr, l_transpArgs); 
        break;
      }
      case KargsType::OpSpmv: {
        SpmvArgsType l_spmvArgs = l_kargs.getSpmvArgs();
        if (GEMX_runSpmv)
          l_spmv.runSpmv(p_DdrRd, p_DdrWr, l_spmvArgs);
        break;
      }
      default: {
        assert(false);
      }
    }
    
    // Collect and store cycle count
    TimeStampType::TimeType l_ts = p_Time.read();
    if (l_pc >= l_tsDepth) {
      gemx::InstrResArgs l_instrRes(l_tsPrev, reg(l_ts));
      l_kargsRes.setInstrResArgs(l_instrRes);
      l_kargsRes.storeToInstr(l_res[l_pc - l_tsDepth]);
    }
    l_tsPrev = reg(l_ts);
  }
  
  for(unsigned int l_d = 0; l_d < l_tsDepth; ++l_d) {
    TimeStampType::TimeType l_ts = p_Time.read();
    gemx::InstrResArgs l_instrRes(l_tsPrev, l_ts);
    l_kargsRes.setInstrResArgs(l_instrRes);
    l_kargsRes.storeToInstr(l_res[GEMX_numInstr - l_tsDepth + l_d]);
    l_tsPrev = l_ts;
  }
  
  // Store instruction results in DDR result segment
  for (unsigned int l_pc = 0; l_pc < GEMX_numInstr; ++l_pc) {
    l_res[l_pc].storeToDdr(p_DdrWr, GEMX_resPage * DdrType::per4k() +
                                    l_pc * KargsType::getInstrWidth());
  }
  
}

void
GEMX_EVALUATOR(gemxKernel_, GEMX_kernelId)
  (
    DdrType *p_DdrRd,
    DdrType *p_DdrWr
  )
{
  #pragma HLS INTERFACE m_axi port=p_DdrRd offset=slave bundle=gmemM  num_write_outstanding=16 num_read_outstanding=16 max_write_burst_length=16 max_read_burst_length=16 depth=16 latency=125
  #pragma HLS INTERFACE m_axi port=p_DdrWr offset=slave bundle=gmemM  num_write_outstanding=16 num_read_outstanding=16 max_write_burst_length=16 max_read_burst_length=16 depth=16 latency=125

  #pragma HLS INTERFACE s_axilite port=p_DdrRd bundle=control
  #pragma HLS INTERFACE s_axilite port=p_DdrWr bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  #pragma HLS DATA_PACK variable=p_DdrRd
  #pragma HLS DATA_PACK variable=p_DdrWr

  TimeStampType l_tr;
  //hls::stream<TimeStampType::OpType> l_controlStream;
  hls::stream<TimeStampType::TimeType> l_timeStream;
  //#pragma HLS STREAM   variable=l_controlStream  depth=1
  #pragma HLS STREAM   variable=l_timeStream  depth=1

  # pragma HLS DATAFLOW
  
  l_tr.runTs(/*l_controlStream, */l_timeStream);
  kernelOpLow(p_DdrRd, p_DdrWr, /*l_controlStream,*/ l_timeStream);

}


#ifdef TEST_SDX
} // extern C
#endif

