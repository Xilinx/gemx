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
 *  @brief FCN header
 *  a fully connected network implementation
 *
 *  $DateTime: 2018/01/24 14:16:32$
 */
#ifndef GEMX_FCN_H
#define GEMX_FCN_H

#include "gemx_gemm.h"

namespace gemx {
////////////////////////////////////////////////////////////////////////////////
//class FCN (fully connected network)
// add preScale, pRelu and postScal operations to the gemm results
////////////////////////////////////////////////////////////////////////////////

template <
	typename t_FloatType,
	typename t_FloatEqIntType,
	typename t_XDataType, //bias matrix entry type
	unsigned int t_DdrWidth,
	unsigned int t_XDdrWidth,
	unsigned int t_aColMemWords=1,
	unsigned int t_aRowMemWords=1,
	unsigned int t_bColMemWords=1,
	unsigned int t_MacBits=48
>
class Fcn
{
	private:
		static const unsigned int t_debug=0;

	public:
	//type definitions
	typedef typename Gemm<t_FloatType, t_FloatEqIntType, t_XDataType, t_DdrWidth, t_XDdrWidth, t_aColMemWords, t_aRowMemWords, t_bColMemWords, t_MacBits>::DdrWideType DdrWideType;
	typedef typename Gemm<t_FloatType, t_FloatEqIntType, t_XDataType, t_DdrWidth, t_XDdrWidth, t_aColMemWords, t_aRowMemWords, t_bColMemWords, t_MacBits>::DdrStream DdrStream;
	typedef FcnArgs FcnArgsType;

	public:
	void
	FcnScalePRelu(
		DdrStream &p_inS,
		DdrStream &p_outS,
		unsigned int p_aRowBlocks,
		unsigned int p_bColBlocks,
		int16_t p_PReluVal
	) {

			ap_int<16> l_PReluVal = p_PReluVal;
			ap_int<10> l_scaleVal;
			ap_int<6> l_alpha;
			l_scaleVal = l_PReluVal.range(15,6);
			l_alpha = l_PReluVal.range(5,0);

			for (int rowBlock=0; rowBlock < p_aRowBlocks; ++rowBlock) {
				for (int colBlock=0; colBlock < p_bColBlocks; ++colBlock) {
					for (int i=0; i<t_aRowMemWords*t_DdrWidth; ++i) {
						for (int j=0; j<t_bColMemWords; ++j) {
						#pragma HLS PIPELINE
							DdrWideType l_val = p_inS.read();
							#pragma HLS ARRAY_PARTITION variable=l_val complete
							DdrWideType l_valOut;
							#pragma HLS ARRAY_PARTITION variable=l_valOut complete
							for (int w=0; w < t_DdrWidth; ++w){
								t_FloatType l_prePRelu= l_val[w];
								#if GEMX_keepMacBits
								t_FloatType l_postPRelu = (l_prePRelu < 0)? (l_prePRelu *l_scaleVal.to_int()) >> l_alpha.to_int(): l_prePRelu;
                                                                #else
								t_FloatType l_postPRelu = (l_prePRelu < 0)? 0 : l_prePRelu;
                                                                #endif
								l_valOut[w] = l_postPRelu;
							}
							p_outS.write(l_valOut);
						}
					}
				}
			}
	}

	
	void
  FcnBlocks(
		DdrWideType *p_aAddr,
		DdrWideType *p_bAddr,
		DdrWideType *p_cAddr,
		DdrWideType *p_xAddr,
		unsigned int p_aColBlocks,
		unsigned int p_aRowBlocks,
		unsigned int p_bColBlocks,
		unsigned int p_aLd,
		unsigned int p_bLd,
		unsigned int p_cLd,
		unsigned int p_xLd,
		unsigned int p_transpBlocks,
		int32_t p_postScale,
		int16_t p_PReluVal
		) {
		#pragma HLS DATAFLOW

		Gemm<t_FloatType, t_FloatEqIntType, t_XDataType, t_DdrWidth, t_XDdrWidth, t_aColMemWords, t_aRowMemWords, t_bColMemWords, t_MacBits> l_gemm;
		DdrStream  p_C2ScalePRelu;
		DdrStream  p_Cs;

		#pragma HLS data_pack variable=p_Cs
		#pragma HLS data_pack variable=p_C2ScalePRelu

		#pragma HLS STREAM variable=p_C2ScalePRelu depth=4
		#pragma HLS STREAM variable=p_Cs depth=4

		l_gemm.GemmReadAndMult(p_aAddr, p_bAddr, p_xAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, p_xLd, p_transpBlocks, p_postScale, p_C2ScalePRelu);
		FcnScalePRelu(p_C2ScalePRelu, p_Cs, p_aRowBlocks, p_bColBlocks, p_PReluVal);
		l_gemm.GemmWriteDdrStream(p_cAddr, p_Cs, p_aRowBlocks, p_bColBlocks, p_cLd);
	}

	void
	runFcn(
		DdrWideType *p_DdrRd,
		DdrWideType *p_DdrWr,
		FcnArgsType &p_Args
	) {

		
		DdrWideType *l_aAddr = p_DdrRd + p_Args.m_Aoffset * DdrWideType::per4k();
    DdrWideType *l_bAddr = p_DdrRd + p_Args.m_Boffset * DdrWideType::per4k();
		DdrWideType *l_xAddr = p_DdrRd + p_Args.m_Xoffset * DdrWideType::per4k();
    DdrWideType *l_cAddr = p_DdrWr + p_Args.m_Coffset * DdrWideType::per4k();

    const unsigned int l_aColBlocks = p_Args.m_K / (t_DdrWidth * t_aColMemWords);
    const unsigned int l_aRowBlocks = p_Args.m_M / (t_DdrWidth * t_aRowMemWords);
    const unsigned int l_bColBlocks = p_Args.m_N / (t_DdrWidth * t_bColMemWords);
    const unsigned int l_aLd  = p_Args.m_Lda / t_DdrWidth;
    const unsigned int l_bLd  = p_Args.m_Ldb / t_DdrWidth;
    const unsigned int l_cLd  = p_Args.m_Ldc / t_DdrWidth;
		const unsigned int l_xLd 	= p_Args.m_Ldx / t_XDdrWidth;
		int32_t l_postScale = p_Args.m_postScale;
		int16_t l_PReluVal = p_Args.m_PReluVal;

    unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks *t_aRowMemWords;
		FcnBlocks(l_aAddr, l_bAddr, l_cAddr, l_xAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd, l_xLd, l_transpBlocks,
							l_postScale, l_PReluVal);

	}
};

}
#endif
