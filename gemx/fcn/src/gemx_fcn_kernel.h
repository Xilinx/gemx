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
 *  @brief FCN Kernel API header file
 *
 *  $DateTime: 2018/01/08 08:01:21 $
 */

#ifndef GEMX_FCN_KERNEL_H
#define GEMX_FCN_KERNEL_H

#include "gemx_ddr.h"
#include "gemx_fcn_kargs.h"
#include "gemx_fcn.h"

// Location of code aand data segements in DDR memory
#define GEMX_codePage 0
#define GEMX_resPage 1
#define GEMX_dataPage 2
// Page and instruction sizes
#define GEMX_pageSizeBytes 4096
#define GEMX_instructionSizeBytes 64

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
typedef gemx::FcnKargs<
    GEMX_dataType, GEMX_dataEqIntType,
    GEMX_ddrWidth, GEMX_argInstrWidth,
    GEMX_argInstrWidth * GEMX_ddrWidth * sizeof(GEMX_dataType) * 8,
    GEMX_argPipeline
  > KargsType;
typedef KargsType::DdrInstrType KargsDdrInstrType;  // 512 bit wide type across all DDR-width architectures


// Compute engine types

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
