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
 *  @brief Kernel API header file
 *
 *  $DateTime: 2018/03/09 06:16:16 $
 */

#ifndef GEMX_KERNEL_H
#define GEMX_KERNEL_H

#include "gemx_ddr.h"
#include "gemx_kargs.h"
#include "gemx_gemv.h"
#include "gemx_gemm.h"
#include "gemx_transp.h"
#include "gemx_fcn.h"

#if GEMX_useURAM
#include "gemx_spmv_coo.h"
#else
#include "gemx_spmv.h"
#endif 
#include "gemx_uspmv.h"

// Location of code aand data segements in DDR memory
#define GEMX_codePage 0
#define GEMX_resPage 1
#define GEMX_dataPage 2
// Page and instruction sizes
#define GEMX_pageSizeBytes 4096
//#define GEMX_instructionSizeBytes 64

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
#if GEMX_runGemv==1
typedef gemx::Gemv<
    GEMX_dataType, 
    GEMX_ddrWidth,
    GEMX_transpBlocks,
    GEMX_gemvmGroups,  
    GEMX_gemvkVectorBlocks/GEMX_transpBlocks, 
    GEMX_gemvmVectorBlocks/GEMX_gemvmGroups
  > GemvType;
#endif

#if GEMX_runGemm==1
typedef gemx::Gemm<
    GEMX_dataType, 
    GEMX_dataEqIntType, 
    GEMX_XdataType, 
    GEMX_ddrWidth, 
    GEMX_XddrWidth, 
    GEMX_gemmKBlocks, 
    GEMX_gemmMBlocks, 
    GEMX_gemmNBlocks, 
    GEMX_macBits
> GemmType;
#endif

#if GEMX_runTransp==1
typedef gemx::Transp<
    GEMX_dataType, GEMX_ddrWidth,
    GEMX_transpBlocks,
    GEMX_transpBlocks
  > TranspType;
#endif

#if GEMX_runSpmv==1
#if GEMX_useURAM
typedef gemx::SpmvCoo<
    GEMX_dataType,
    GEMX_idxType,
    GEMX_ddrWidth,
    GEMX_nnzBlocks,
    GEMX_spmvKmaxBlocks,
    GEMX_spmvMmaxBlocks,
    GEMX_spmvUramGroups
  > SpmvType;
#else
typedef gemx::Spmv<
    GEMX_dataType, GEMX_dataEqIntType,
    GEMX_ddrWidth, GEMX_spmvWidth,
    GEMX_spmvkVectorBlocks, GEMX_spmvmVectorBlocks,
    GEMX_spmvMacGroups,
    GEMX_spmvColAddIdxBits,
    GEMX_spmvNumCblocks,
    GEMX_spmvFloatPerDesc
  > SpmvType;
#endif
#endif

#if GEMX_runUspmv==1
typedef gemx::Uspmv<
    GEMX_dataType,
    GEMX_idxType,
    GEMX_ddrWidth,
    GEMX_uspmvMvectorBlocks,
    GEMX_uspmvNnzVectorBlocks,
    GEMX_uspmvStages,
    GEMX_uspmvInterleaves
> UspmvType;
#endif

#if GEMX_runFcn==1
typedef gemx::Fcn<
	GEMX_dataType, 
	GEMX_dataEqIntType, 
	GEMX_XdataType, 
	GEMX_ddrWidth, 
	GEMX_XddrWidth,
	GEMX_gemmKBlocks,
	GEMX_gemmMBlocks,
	GEMX_gemmNBlocks,
	GEMX_macBits
> FcnType;
#endif

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

