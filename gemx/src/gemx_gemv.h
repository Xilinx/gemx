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
 *  @brief GEMV based on GEMM-A format input
 *
 *  $DateTime: 2017/10/27 09:54:34 $
 */

#ifndef GEMX_GEMV_H
#define GEMX_GEMV_H

#include "assert.h"
#include "hls_stream.h"
#include "gemx_types.h"
#include "gemx_kargs.h"
#include "gemx_transp.h"

namespace gemx {

////////////////////////////////////////////////////////////////////////////////
//matrix vector multiplication
//matrix is read, multiplied block by block
//   ----------------------------
//   | t_DdrWidth  | t_DdrWidth |
//   | Block 0     |   Block 1  |
//   |             |            |
//   ----------------------------t_colMemWords
//   |             |            |
//   | Block 2     |  Block 3   |
//   |             |            |
//   ----------------------------
//   t_rowMemWords 
////////////////////////////////////////////////////////////////////////////////
template <
	typename t_FloatType,
	unsigned int t_DdrWidth,
	unsigned int t_colMemWords, //number of DDR or memory words in a row of each block
	unsigned int t_rowMemWords, //number of DDR or memory words in a column of each block
	unsigned int t_kVectorBlocks, //GEMV max length of the B vector in t_DdrWidth-wide * t_colMemWords words (Max K)
	unsigned int t_mVectorBlocks //Gemv max length of the C vectore in t_DdrWidth-wid * t_rowMemWords words (Max M)
>
class Gemv
{
public:
	typedef WideType<t_FloatType, t_DdrWidth> DdrWideType;
	typedef hls::stream<DdrWideType> DdrStream;
	typedef hls::stream<unsigned int> ParamStream;
	static const unsigned int t_colBlockLength = t_DdrWidth * t_colMemWords;
	static const unsigned int t_rowBlockLength = t_DdrWidth * t_rowMemWords;
	typedef GemvArgs GemvArgsType;

private:
	//DdrWideType m_B[t_kVectorBlocks][t_colMemWords];
	//DdrWideType m_C[t_mVectorBlocks][t_rowMemWords];
	t_FloatType m_B[t_DdrWidth][t_kVectorBlocks*t_colMemWords];
	t_FloatType m_C[t_DdrWidth][t_mVectorBlocks*t_rowMemWords];

private:
 void loadB(DdrWideType *p_bAddr, unsigned int p_kBlocks) {
        // Load entire B into BRAM
        #pragma HLS ARRAY_PARTITION variable=m_B dim=1 complete
        unsigned int l_addrIdx = 0;
		unsigned int p_memWords = p_kBlocks * t_colMemWords;
        LOOP_GEMV_BLOAD:for(unsigned int l_memWord = 0; l_memWord < p_memWords; ++l_memWord) {
          #pragma HLS pipeline
		  DdrWideType l_b;
		  l_b =  p_bAddr[l_addrIdx];
		  LOOP_COLBLOCK_LOAD:for (int i=0; i<t_DdrWidth; ++i) {
          m_B[i][l_memWord] = l_b[i];
		}
		l_addrIdx++;
       }
      }

void loadC(DdrWideType *p_cAddr, unsigned int p_mBlocks) {
        #pragma HLS ARRAY_PARTITION variable=m_C dim=1 complete
        unsigned int l_addrIdx = 0;
		unsigned int p_memWords = p_mBlocks * t_rowMemWords;

        LOOP_GEMV_CLOAD:for(unsigned int l_memWord = 0; l_memWord < p_memWords; ++l_memWord) {
          #pragma HLS pipeline
		  DdrWideType l_c;
		  l_c = p_cAddr[l_addrIdx];
		  LOOP_ROWBLOCK_LOAD:for (int i=0; i<t_DdrWidth; ++i) {
          m_C[i][l_memWord] = l_c[i];
		}
		l_addrIdx++;
       }
      }

void multA(DdrStream &inStream, unsigned int l_rowBlocks, unsigned int l_colBlocks){
    #pragma HLS ARRAY_PARTITION variable=m_B dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=m_C dim=1 complete

	DdrWideType l_valA;
	unsigned int l_IdxBaseB;
	unsigned int l_IdxBaseC;

	t_FloatType l_B[t_DdrWidth];
	t_FloatType l_C[t_DdrWidth];
#pragma HLS ARRAY_PARTITION variable=l_B complete
#pragma HLS ARRAY_PARTITION variable=l_C complete
#pragma HLS ARRAY_PARTITION variable=l_valA complete

		for (int l_rowBlockCounter=0; l_rowBlockCounter < l_rowBlocks; ++l_rowBlockCounter){
			l_IdxBaseC = l_rowBlockCounter * t_rowMemWords;
			for (int i=0; i<t_DdrWidth; ++i){
				l_C[i] = m_C[i][l_rowBlockCounter];
			}
			for (int l_colBlockCounter=0; l_colBlockCounter < l_colBlocks; ++l_colBlockCounter){
				l_IdxBaseB = l_colBlockCounter * t_colMemWords;
				for (int l_colMemWordCounter = 0; l_colMemWordCounter < t_colMemWords; ++l_colMemWordCounter){
					for (int i=0; i<t_DdrWidth; ++i){
						l_B[i] = m_B[i][l_IdxBaseB+l_colMemWordCounter];
					}
					for (int l_colCounter =0; l_colCounter < t_DdrWidth; ++l_colCounter){
			#pragma HLS PIPELINE II=1
						l_valA = inStream.read();
						for (int i=0; i< t_DdrWidth; ++i){
#pragma HLS UNROLL
							l_C[i] += l_valA[i] * l_B[l_colCounter];
						}
					}
				}
			}
			for (int i=0; i<t_DdrWidth; ++i){
				m_C[i][l_rowBlockCounter] = l_C[i];
			}
		}
}

void  storeC(DdrWideType *p_cAddr, unsigned int p_mBlocks) {
        #pragma HLS ARRAY_PARTITION variable=m_C dim=1 complete
        unsigned int l_addrIdx = 0;
		unsigned int p_memWords = p_mBlocks * t_rowMemWords;

        LOOP_GEMV_CLOAD:for(unsigned int l_memWord = 0; l_memWord < p_memWords; ++l_memWord) {
          #pragma HLS pipeline
		  DdrWideType l_c;
		  LOOP_ROWBLOCK_LOAD:for (int i=0; i<t_DdrWidth; ++i) {
          	l_c[i]=m_C[i][l_memWord] ;
		}
		 p_cAddr[l_addrIdx] = l_c;
		 l_addrIdx++;
       }
 }

public:
void gemv_blocks(DdrWideType *l_aAddr, unsigned int l_srcWordLd, unsigned int l_rowBlocks, unsigned int l_colBlocks, unsigned int numOfBlocks){
#pragma HLS DATAFLOW

	DdrStream aStream("aStream");
	#pragma HLS DATA_PACK variable=aStream
	#pragma HLS STREAM variable=aStream depth=4

	DdrStream aWrStream("aWrStream");
	#pragma HLS DATA_PACK variable=aWrStream
	#pragma HLS STREAM variable=aWrStream depth=t_colMemWords*t_DdrWidth
	
	DdrStream aShuffleStream("aShuffleStream");
	#pragma HLS DATA_PACK variable=aShuffleStream
	#pragma HLS STREAM variable=aShuffleStream depth=4
	
	DdrStream aTranspStream("aTranspStream");
	#pragma HLS DATA_PACK variable=aTranspStream
	#pragma HLS STREAM variable=aTranspStream depth=4
	
	Transp<t_FloatType, t_DdrWidth, t_colMemWords, t_rowMemWords> l_Transp;

	l_Transp.load_matrix(l_aAddr, l_srcWordLd, l_rowBlocks, l_colBlocks, aStream);
	l_Transp.shuffle_input(aStream, aWrStream, numOfBlocks);
	l_Transp.WR_buffer(aWrStream, aShuffleStream, numOfBlocks);
	l_Transp.shuffle_output(aShuffleStream, aTranspStream, numOfBlocks);
	multA(aTranspStream, l_rowBlocks, l_colBlocks);
}
 
	//gemv implementation
	void runGemv (
		DdrWideType *p_DdrRd,
		DdrWideType *p_DdrWr,
		GemvArgsType &p_Args
	){
	#pragma HLS inline off
		assert(t_rowMemWords == 1);
        const unsigned int l_mBlocks = p_Args.m_M / t_rowMemWords / t_DdrWidth;
        assert(l_mBlocks * t_rowMemWords * t_DdrWidth == p_Args.m_M);
        const unsigned int l_kBlocks = p_Args.m_K / t_colMemWords/ t_DdrWidth;
        assert(l_kBlocks * t_colMemWords * t_DdrWidth == p_Args.m_K);
        assert(l_mBlocks <= t_mVectorBlocks);
        assert(l_kBlocks <= t_kVectorBlocks);

		const unsigned int numOfBlocks = l_mBlocks * l_kBlocks;

		// Load entire B into BRAM
		DdrWideType *l_bAddr = p_DdrRd + p_Args.m_Boffset * DdrWideType::per4k();
		loadB(l_bAddr, l_kBlocks);
		// Load C
		DdrWideType *l_cAddr = p_DdrWr + p_Args.m_Coffset * DdrWideType::per4k();
		loadC(l_cAddr, l_mBlocks);

		const unsigned int l_srcWordLd = p_Args.m_Lda / t_DdrWidth;
		assert(l_srcWordLd * t_DdrWidth == p_Args.m_Lda);
		//calculate address for A
		DdrWideType *l_aAddr = p_DdrRd + p_Args.m_Aoffset * DdrWideType::per4k();

		gemv_blocks(l_aAddr, l_srcWordLd, l_mBlocks, l_kBlocks, numOfBlocks);
		
		// Store C
		storeC(l_cAddr, l_mBlocks);
	}
};

} // namespace
#endif

