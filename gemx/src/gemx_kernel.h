// Copyright (c) 2017
// Xilinx, Inc.
// All rights reserved.
// $Id: //Rodin/Proj/O/OPENCL_APPS_DEV/src/matrix_mult/gemx/src/gemx_kernel.h#17 $
// 
// No part of this document may be copied, transmitted or
// disclosed in any form or fashion without the express
// written consent of Xilinx, Inc.

/**
 *  @brief Kernel API header file
 *
 *  $DateTime: 2017/09/07 04:14:42 $
 *  $Author: lingl $
 */

#ifndef GEMX_KERNEL_H
#define GEMX_KERNEL_H

#include "gemx_ddr.h"
#include "gemx_kargs.h"
#include "gemx_gemv.h"
#include "gemx_gemm.h"
#include "gemx_transp.h"
#include "gemx_spmv.h"

// Location of code aand data segements in DDR memory
#define GEMX_codePage 0
#define GEMX_resPage 1
#define GEMX_dataPage 2
// Page and instruction sizes
#define GEMX_pageSizeBytes 4096
#define GEMX_instructionSizeBytes 64

// C storage matches the row index range in cblock; this must be rounded down (hence int division) from the
// row index range so that every stored row is indexable
#define GEMX_spmvmVectorBlocks ((1 << (16 - GEMX_spmvColAddIdxBits)) / GEMX_spmvWidth / GEMX_spmvMacGroups / GEMX_ddrWidth)

// The extern C still needed - otherwise cpu emu fails
//   prj_sda.exe: symbol lookup error: ./dltmp: undefined symbol: kernelSgemm
#ifdef TEST_SDX
extern "C" {
#endif

// DDR interface types
typedef gemx::DdrUtil<
    GEMX_dataType, GEMX_ddrWidth, GEMX_ddrWidth * sizeof(GEMX_dataType) * 8
  > DdrUtilType;
typedef DdrUtilType::DdrWideType DdrType;

// VLIV processing types
typedef gemx::Kargs<
    GEMX_dataType, GEMX_dataEqIntType,
    GEMX_ddrWidth, GEMX_argInstrWidth,
    GEMX_argInstrWidth * GEMX_ddrWidth * sizeof(GEMX_dataType) * 8,
    GEMX_argPipeline
  > KargsType;
typedef KargsType::DdrInstrType KargsDdrInstrType;  // 512 bit wide type across all DDR-width architectures


// Compute engine types
typedef gemx::Gemv<
    GEMX_dataType, GEMX_ddrWidth,
    GEMX_gemvkVectorBlocks, GEMX_gemvmVectorBlocks,
    GEMX_gemvmGroups
  > GemvType;

typedef gemx::GemvM2M<
    GEMX_dataType, GEMX_ddrWidth,
	GEMX_transpBlocks,
    GEMX_gemvmGroups,  GEMX_gemvkVectorBlocks/GEMX_transpBlocks, 
	GEMX_gemvmVectorBlocks
  > GemvM2Mtype;
//typedef gemx::Gemm<
//    GEMX_dataType, GEMX_ddrWidth,
//    GEMX_gemmMeshRows, GEMX_gemmMeshCols, GEMX_gemmMeshDepth
//  > GemmType;
typedef gemx::Gemm<
	GEMX_dataType, GEMX_ddrWidth, GEMX_gemmKBlocks, GEMX_gemmMBlocks, GEMX_gemmNBlocks
> GemmType;

typedef gemx::Transp<
    GEMX_dataType, GEMX_ddrWidth,
    0,
    GEMX_transpBlocks,
    GEMX_gemvmGroups
  > TranspType;
typedef gemx::Spmv<
    GEMX_dataType, GEMX_dataEqIntType,
    GEMX_ddrWidth, GEMX_spmvWidth,
    GEMX_spmvkVectorBlocks, GEMX_spmvmVectorBlocks,
    GEMX_spmvMacGroups,
    GEMX_spmvColAddIdxBits,
    GEMX_spmvNumCblocks,
    GEMX_spmvFloatPerDesc
  > SpmvType;

typedef gemx::TimeStamp<GEMX_numInstr> TimeStampType;

void
gemxKernel_0(
    DdrType *p_DdrRd,
    DdrType *p_DdrWr
  );

#ifdef TEST_SDX
} // extern C
#endif

#endif

