/**********
Copyright (c) 2017, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
/**
 *  @brief GEMV based on GEMM-A format input
 *
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
class GemvM2M
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
	
	TranspM2M<t_FloatType, t_DdrWidth, t_colMemWords, t_rowMemWords> l_Transp;

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

//    GEMV C vector and matrix A row format as well as processing order:
//    The mGroups are for slow multipliers. Use 10 for fp32; use 1 for 16b.
//       ----------------------------------------------------
//       |    -------------------------------|             |
//       |    |  ----------------            |             |
//       |    |  |              | t_DdrWidth |             |
//       |    |  ----------------            |             |
//       |    |  ----------------            |  t_mGroups  |
//       |    |  |              | t_DdrWidth |             |
//       |    |  ----------------            |             |
//       |    |                              |             |  m-blocks
//       |    -------------------------------|             | ( 1 mblock is a unit 
//       |    |  ----------------            |             |  a unit of processing
//       |    |  |              | t_DdrWidth |             |  per k-index)
//       |    |  ----------------            |             |
//       |    |  ----------------            |  t_mGroups  |
//       |    |  |              | t_DdrWidth |             |
//       |    |  ----------------            |             |
//       |    |                              |             |
//       |    --------------------------------             |
//       |                                                 |
     

template <
    typename t_FloatType,
    unsigned int t_DdrWidth,
    unsigned int t_kVectorBlocks,    // GEMV max length of the B vector in t_DdrWidth-wide * t_mGroups words (max K)
    unsigned int t_mVectorBlocks,    // GEMV max length of the C vector in t_DdrWidth-wide words  (max M)
    unsigned int t_mGroups   // Row groups for higher parallelism when using slow multipliers
  >
class Gemv
{
  public:
    typedef WideType<t_FloatType, t_DdrWidth> DdrWideType;
    typedef hls::stream<DdrWideType> DdrWideStreamType;
    typedef GemvArgs GemvArgsType;

   private:
    DdrWideType m_B[t_kVectorBlocks];
    DdrWideType m_C[t_mVectorBlocks][t_mGroups];

  private:
    
    void
    loadB(DdrWideType *p_bAddr, unsigned int p_kBlocks) {
        // Load entire B into BRAM
        LOOP_GEMV_BLOAD:for(unsigned int l_kBlock = 0; l_kBlock < p_kBlocks; ++l_kBlock) {
          #pragma HLS LOOP_TRIPCOUNT min=1 max=16
          #pragma HLS pipeline
          m_B[l_kBlock] = p_bAddr[l_kBlock];
        }
      }

    void
    loadC(DdrWideType *p_cAddr, unsigned int p_mBlocks) {
        unsigned int l_addrIdx = 0;
        LOOP_GEMV_CLOAD:for(unsigned int l_mBlock = 0; l_mBlock < p_mBlocks; ++l_mBlock) {
          #pragma HLS LOOP_TRIPCOUNT min=1 max=64
          #pragma HLS pipeline
          LOOP_MGROUP_LOAD:for(int mg = 0; mg < t_mGroups; ++mg) {
            m_C[l_mBlock][mg] = p_cAddr[l_addrIdx];
            l_addrIdx++;
          }
        }
      }
      
    void
    storeC(DdrWideType *p_cAddr, unsigned int p_mBlocks) {
        unsigned int l_addrIdx = 0;
        LOOP_GEMV_CSTORE:for(unsigned int l_mBlock = 0; l_mBlock < p_mBlocks; ++l_mBlock) {
          #pragma HLS LOOP_TRIPCOUNT min=1 max=64
          #pragma HLS pipeline
          LOOP_MGROUP_STORE:for(int mg = 0; mg < t_mGroups; ++mg) {
            p_cAddr[l_addrIdx] = m_C[l_mBlock][mg];
            l_addrIdx++;
          }
        }
      }
    
    void
    multA(DdrWideType *p_aAddr, unsigned int p_numWordsA, unsigned int p_mBlocks, unsigned int p_kBlocks) {
      //#pragma HLS inline self off

      DdrWideStreamType l_fifoA;
      #pragma HLS data_pack variable=l_fifoA
      #pragma HLS STREAM   variable=l_fifoA  depth=1
            
      #pragma HLS DATAFLOW

      AREAD:for(int l_idxA = 0; l_idxA < p_numWordsA; ++l_idxA) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32768
        #pragma HLS PIPELINE
        l_fifoA.write(p_aAddr[l_idxA]);
      }
      
      LOOP_MBLOCKS:for(int l_mBlock = 0; l_mBlock < p_mBlocks; ++l_mBlock) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        DdrWideType l_C[t_mGroups];
        //#pragma HLS ARRAY_PARTITION variable=l_C dim=1 COMPLETE
        #pragma HLS ARRAY_PARTITION variable=l_C dim=2 COMPLETE
        #pragma HLS ARRAY_PARTITION variable=m_C dim=3 COMPLETE
        LOOP_MGROUP_COPY_IN:for(int mg = 0; mg < t_mGroups; ++mg) {
          //#pragma HLS PIPELINE
          l_C[mg] = m_C[l_mBlock][mg];
        }
        LOOP_GEMV_KBLOCKS:for(unsigned int l_kBlock = 0; l_kBlock < p_kBlocks; ++l_kBlock) {
          #pragma HLS LOOP_TRIPCOUNT min=1 max=16
          DdrWideType l_B = m_B[l_kBlock];
          #pragma HLS ARRAY_PARTITION variable=l_B
          //std::cout << "    DEBUG runGemv l_B= " << l_B << "\n";

          LOOP_GEMV_K:for(int k = 0; k < t_DdrWidth; ++k) { // HLS latency of this loop should be 32*16 = 512
            #pragma HLS PIPELINE
            const t_FloatType l_Bval = l_B.getVal(k);
            LOOP_GEMV_MGROUP:for(int mg = 0; mg < t_mGroups; ++mg) {
              #pragma HLS UNROLL
              DdrWideType l_valA = l_fifoA.read();
              #pragma HLS ARRAY_PARTITION variable=l_valA
              LOOP_GEMV_MAC_WIDTH:for(int w = 0; w < t_DdrWidth; ++w) {
                #pragma HLS UNROLL
                l_C[mg].getVal(w) += l_valA.getVal(w) * l_Bval;
                //std::cout << "      DEBUG runGemv += a * b  " << l_valA.getVal(w) << " * " << l_Bval << "\n";
              }
            }
          }
        }
        LOOP_MGROUP_STORE:for(int mg = 0; mg < t_mGroups; ++mg) {
          //#pragma HLS PIPELINE
          m_C[l_mBlock][mg] = l_C[mg];
        }
        //std::cout << "    DEBUG runGemv computed l_C[" << l_mBlock << "]=" << l_C << "\n";
      }

    }


  public:
    
    // Simple GEMM-A FORMAT based implementaion 
    // The HLS loop counts are for 2048x512; Total latency should be 2048*512/32 = 32k cycles
    //   plus 16*64+64 for loading B, C, store C so total of ##### 33856  #####
    void runGemv(
        DdrWideType *p_DdrRd,
        DdrWideType *p_DdrWr,
        GemvArgsType &p_Args
      ) {
        //std::cout << "\nrunGemv START M=" << p_Args.m_M << " K=" << p_Args.m_K << "\n";
        #pragma HLS inline off
        const unsigned int l_mBlocks = p_Args.m_M / t_mGroups / t_DdrWidth;
        assert(l_mBlocks * t_mGroups * t_DdrWidth == p_Args.m_M);        
        const unsigned int l_kBlocks = p_Args.m_K / t_DdrWidth;
        assert(l_kBlocks * t_DdrWidth == p_Args.m_K);
        assert(l_mBlocks <= t_mVectorBlocks);
        assert(l_kBlocks <= t_kVectorBlocks);
                
        // Load entire B into BRAM
        DdrWideType *l_bAddr = p_DdrRd + p_Args.m_Boffset * DdrWideType::per4k();
        loadB(l_bAddr, l_kBlocks);
        
        // Load C
        DdrWideType *l_cAddr = p_DdrWr + p_Args.m_Coffset * DdrWideType::per4k();
        loadC(l_cAddr, l_mBlocks);
        
        DdrWideType *l_aAddr = p_DdrRd + p_Args.m_Aoffset * DdrWideType::per4k();
        const unsigned int l_numWordsA = l_mBlocks * t_mGroups * p_Args.m_K;
        multA(l_aAddr, l_numWordsA, l_mBlocks, l_kBlocks);
        
        // Store C
        storeC(l_cAddr, l_mBlocks);
      }
      
};

} // namespace
#endif

