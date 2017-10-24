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
 *  @brief Kernel API header file
 *
 *  $DateTime: 2017/10/24 03:52:34 $
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

