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
 *  @brief GEMM header
 *
 *  $DateTime: 2017/11/10 13:22:30 $
 */

#ifndef GEMX_GEMM_H
#define GEMX_GEMM_H

#include <cassert>
#include "gemx_types.h"
#include "gemx_transp.h"
#include "ap_shift_reg.h"

namespace gemx {

template <
    typename T,
    unsigned int t_Width
  >
class TriangSrl {
  private:
    ap_shift_reg<T, t_Width> m_Sreg[t_Width];
  public:
    WideType<T, t_Width>
    shift(
        WideType<T, t_Width> p_DiagIn
      ) {
        #pragma HLS inline self
        #pragma HLS ARRAY_PARTITION variable=m_Sreg dim=1 complete
        #pragma HLS data_pack variable=p_DiagIn
        WideType<T, t_Width> l_edgeOut;
        #pragma HLS data_pack variable=l_edgeOut
        TRIANGSRL_SHIFT:for (unsigned int i = 0; i < t_Width; ++i) {
          #pragma HLS unroll
          l_edgeOut[i] = m_Sreg[i].shift(p_DiagIn[i], i);
        }
        return(l_edgeOut);
      }
    void
    clear() {
	#pragma HLS inline
	#pragma HLS ARRAY_PARTITION variable=m_Sreg dim=1 complete
        for (unsigned int row = 0; row < t_Width; ++row) {
		#pragma HLS PIPELINE
          for (unsigned int col = 0; col < t_Width; ++col) {
            (void)m_Sreg[row].shift(0, 0);
          }
        }
      }
    void
    print(std::ostream& os) {
        for (unsigned int row = 0; row < t_Width; ++row) {
          for (unsigned int col = 0; col < t_Width; ++col) {
            if (col > row) {
              os << std::setw(GEMX_FLOAT_WIDTH + 2) << "- ";
            } else {
              T l_val = m_Sreg[row].read(col);
              os << std::setw(GEMX_FLOAT_WIDTH) << l_val;
            }
          }
          os << "\n";
        }
      }
};

template <typename T1, unsigned int T2>
std::ostream& operator<<(std::ostream& os, TriangSrl<T1, T2>& p_Val) {
  p_Val.print(os);
  return(os);
}
     
//Gemm class. t_aColMemWords defines number of memwords in the columns of one row of buffer_A. Due to the reusability, the height of buffer_A is only one memwords. For buffer_B, t_aColMemWords defines number of memwords in the rows of one column in buffer_B, t_bColMemWords defines number of memwords in the cols of one row in buffer_B. t_aRowMemWords and t_bColMemWords define the height and width of buffer_C in terms of memwords.
template <
  typename t_FloatType,
  unsigned int t_DdrWidth,
  unsigned int t_aColMemWords=1,
  unsigned int t_aRowMemWords=1,
  unsigned int t_bColMemWords=1
  >
class Gemm
{
  public:
	static const unsigned int t_aMH = t_DdrWidth * t_aRowMemWords;
	static const unsigned int t_bKD = t_DdrWidth * t_aColMemWords; 

    typedef WideType<t_FloatType, t_DdrWidth> DdrWideType;
    typedef TaggedFloat<t_FloatType> TaggedFloatType;
    typedef TaggedWideType<t_FloatType, t_DdrWidth> TaggedWideFloat;
    typedef WideType<TaggedFloatType, t_DdrWidth> TaggedFloatArray; 
    typedef hls::stream<DdrWideType> DdrStream;
    typedef hls::stream<TaggedWideFloat> EdgeStream;

    typedef WideType<t_FloatType, t_DdrWidth/2> HalfDdrWideType;
    typedef ExitTaggedWideType<t_FloatType, t_DdrWidth/2> HalfTaggedDdrWideType;
    typedef TaggedWideType<t_FloatType, t_DdrWidth/2> HalfTaggedWideFloat;
    typedef WideType<TaggedFloatType, t_DdrWidth/2> HalfTaggedFloatArray; 
    typedef hls::stream<HalfTaggedDdrWideType> HalfTaggedDdrStream;
    typedef hls::stream<HalfTaggedWideFloat> HalfEdgeStream;

    typedef GemmArgs GemmArgsType;
  private:
    static const unsigned int t_debug = 0;


  private:
    void
    macStep(
          t_FloatType p_A,
          t_FloatType p_B,
          t_FloatType &p_C,
          t_FloatType &p_Cout,
          bool p_Flush
        ) {
          #pragma HLS inline self
          if (p_Flush) {
            p_Cout = p_C;
          }
          p_C = p_A * p_B + (p_Flush ? 0 : p_C);
        }
      
    ///////////////////////////////////////////////////////////////////////////
    // GEMM AB loader
    ///////////////////////////////////////////////////////////////////////////
    void
    GemmReadAB(
      DdrWideType *l_aAddr,
      DdrWideType *l_bAddr, 
      unsigned int l_aColBlocks, 
      unsigned int l_aRowBlocks, 
      unsigned int l_bColBlocks,
      unsigned int l_aWordLd, 
      unsigned int l_bWordLd,
      DdrStream &p_As,
      DdrStream &p_Bs
     ) {

		DdrWideType l_bufferB[t_bKD][t_bColMemWords];
#if GEMX_useURAM
#pragma HLS ARRAY_RESHAPE variable=l_bufferB block factor=4 dim=3
#pragma HLS ARRAY_PARTITION variable=l_bufferB block factor=8 dim=3
#pragma HLS RESOURCE variable=l_bufferB core=XPM_MEMORY uram
#endif
		DdrWideType l_bufferA[t_aMH][t_aColMemWords];
//if using following pragma to implement l_bufferA, then hw_em fails
//#if GEMX_useURAM
//#pragma HLS ARRAY_RESHAPE variable=l_bufferA block factor=4 dim=3
//#pragma HLS ARRAY_PARTITION variable=l_bufferA block factor=8 dim=3
//#pragma HLS RESOURCE variable=l_bufferA core=XPM_MEMORY uram
//#endif
        unsigned int l_aSrcOffset=0;
		unsigned int l_aRowOffset = 0;
		unsigned int l_aColOffset = 0;

        unsigned int l_bSrcOffset=0;
		unsigned int l_bRowOffset = 0;
		unsigned int l_bColOffset = 0;
		
		for (int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
		  for (int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
			l_bRowOffset = 0;
			l_bColOffset = l_bColBlock * t_bColMemWords;
			for (int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock){
				l_aColOffset = l_aColBlock * t_aColMemWords;
				l_aSrcOffset = l_aRowOffset + l_aColOffset;
				
				l_bSrcOffset = l_bRowOffset + l_bColOffset;
				//read A block, t_DdrWidth(height)*t_aRowMemWordS x t_aColBlocks * t_DdrWidth (width) into l_bufferA
				for (int i=0; i<t_aMH; ++i) {
					//memcpy(&l_bufferA[i][0], &l_aAddr[l_aSrcOffset], t_aBufferWidthBytes);
					for (int j=0; j<t_aColMemWords; ++j) {
					#pragma HLS PIPELINE
						DdrWideType l_word = l_aAddr[l_aSrcOffset+j];
#if GEMX_useURAM
#pragma HLS ARRAY_RESHAPE variable=l_word block factor=4
#pragma HLS ARRAY_PARTITION variable=l_word block factor=8
#endif
						l_bufferA[i][j] = l_word;
					}
					l_aSrcOffset += l_aWordLd;
				}
				//read B Block	
				for (int i=0; i<t_bKD; ++i) {
					//memcpy(&l_bufferB[i][0], &l_bAddr[l_bSrcOffset], t_bBufferWidthBytes);
					for (int j=0; j<t_bColMemWords; ++j) {
					#pragma HLS PIPELINE
						DdrWideType l_word = l_bAddr[l_bSrcOffset+j];
#if GEMX_useURAM
#pragma HLS ARRAY_RESHAPE variable=l_word block factor=4
#pragma HLS ARRAY_PARTITION variable=l_word block factor=8
#endif
						l_bufferB[i][j] = l_word;
					}
					l_bSrcOffset += l_bWordLd;
				}
				
				for (int i=0; i<t_aMH; ++i) {
					for (int j=0; j<t_aColMemWords; ++j) {
					#pragma HLS PIPELINE
						DdrWideType l_word;
#if GEMX_useURAM
#pragma HLS ARRAY_RESHAPE variable=l_word block factor=4
#pragma HLS ARRAY_PARTITION variable=l_word block factor=8
#endif
						l_word = l_bufferA[i][j];
						p_As.write(l_word);
						  #ifndef __SYNTHESIS__
							(t_debug >= 2) && std::cout << "    GemmReadAB Sending A " << l_word << "\n";
						  #endif
					}
				}
				for (int j=0; j<t_bColMemWords; ++j){
					for (int i=0; i<t_bKD; ++i){
					#pragma HLS PIPELINE
						DdrWideType l_word;
#if GEMX_useURAM
#pragma HLS ARRAY_RESHAPE variable=l_word block factor=4
#pragma HLS ARRAY_PARTITION variable=l_word block factor=8
#endif
						l_word = l_bufferB[i][j];
						p_Bs.write(l_word);
						#ifndef __SYNTHESIS__
							(t_debug >= 2) && std::cout << "    GemmReadAB Sending B " << l_word << "\n";
						#endif
					}
				}
				l_bRowOffset += l_bWordLd *t_bKD;
			}
		  }
		  l_aRowOffset += l_aWordLd * t_aMH;
		}
      }

	/*void GemmBufferA(
      unsigned int l_aColBlocks, 
      unsigned int l_aRowBlocks, 
      unsigned int l_bColBlocks,
      DdrStream &p_Ain,
	  DdrStream &p_Aout
	){
	
		for (int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
		  for (int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
			for (int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock){
				//read A block, t_DdrWidth(height)*t_aRowMemWordS x t_aColBlocks * t_DdrWidth (width) into l_bufferA
				for (int i=0; i<t_aMH; ++i) {
					//memcpy(&l_bufferA[i][0], &l_aAddr[l_aSrcOffset], t_aBufferWidthBytes);
					for (int j=0; j<t_aColMemWords; ++j) {
					#pragma HLS PIPELINE
						DdrWideType l_word = p_Ain.read();
						p_Aout.write(l_word);
					#ifndef __SYNTHESIS__
						(t_debug >= 2) && std::cout << "    GemmBufferA Sending A " << l_word << "\n";
				  	#endif
					}
				}
			}
		  }
		}
	}*/

	void GemmBufferB(
      unsigned int l_aColBlocks, 
      unsigned int l_aRowBlocks, 
      unsigned int l_bColBlocks,
      DdrStream &p_Bin,
	  DdrStream &p_Bout) {
	
		DdrWideType l_bufferB[t_bKD][t_bColMemWords];
#if GEMX_useURAM
#pragma HLS ARRAY_RESHAPE variable=l_bufferB block factor=4 dim=3
#pragma HLS ARRAY_PARTITION variable=l_bufferB block factor=8 dim=3
#pragma HLS RESOURCE variable=l_bufferB core=XPM_MEMORY uram
#endif

		for (int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
		  for (int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
			for (int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock){
				//read B Block	
				for (int j=0; j<t_bColMemWords; ++j) {
					//memcpy(&l_bufferB[i][0], &l_bAddr[l_bSrcOffset], t_bBufferWidthBytes);
					for (int i=0; i<t_bKD; ++i) {
					#pragma HLS PIPELINE
						DdrWideType l_word = p_Bin.read();
						l_bufferB[i][j] = l_word;
						p_Bout.write(l_word);
					}
					
				}

				//stream down l_bufferB
				for (int i=1; i<t_aRowMemWords; ++i){
					for (int k=0; k<t_bColMemWords; ++k){
						for (int l=0; l<t_bKD; ++l) {
						#pragma HLS PIPELINE
							DdrWideType l_word = l_bufferB[l][k];
							p_Bout.write(l_word);
						  #ifndef __SYNTHESIS__
							(t_debug >= 2) && std::cout << "    GemmBufferB Sending B " << l_word << "\n";
						  #endif
						}
					}
				}
			}
		  }
		}
	}


    void
    GemmTagAB(DdrStream &p_inAs, DdrStream &p_inBs,  unsigned int p_aColBlocks, unsigned int p_aRowBlocks,unsigned int p_bColBlocks,  EdgeStream &p_outAs, EdgeStream &p_outBs) {

	unsigned int l_aRowBlocks = p_aRowBlocks;
	unsigned int l_aColBlocks = p_aColBlocks;
	unsigned int l_bColBlocks = p_bColBlocks;
	
	for (int rowBlock=0; rowBlock < l_aRowBlocks; ++rowBlock) {
		for (int bColBlock=0; bColBlock < l_bColBlocks; ++bColBlock){
		  for (int aColBlock=0; aColBlock < l_aColBlocks; ++aColBlock) { 
			  for (int i=0; i<t_aRowMemWords; ++i) {
				  for (int k=0; k<t_bColMemWords; ++k) {
					for (int j=0; j<t_bKD; ++j) {
						#pragma HLS PIPELINE
						  DdrWideType l_valA = p_inAs.read();
						  DdrWideType l_valB = p_inBs.read();
						  bool l_flush = (j==0);
						  bool l_exit = false;
						  TaggedWideFloat l_taggedValA(l_valA, l_flush, l_exit);
						  TaggedWideFloat l_taggedValB(l_valB, l_flush, l_exit);
						  #ifndef __SYNTHESIS__
							(t_debug >= 2) && std::cout << "    GemmTagAB Sending A " << l_taggedValA << "\n";
							(t_debug >= 2) && std::cout << "    GemmTagAB Sending B " << l_taggedValB << "\n";
						  #endif
						  p_outAs.write(l_taggedValA);
						  p_outBs.write(l_taggedValB);
						}
					  }
				  }
			  }
			}
    	}
      
        const unsigned int l_flushLen = 3 * t_DdrWidth + 1;
        LOOP_GEMM_FLUSH:for(int i = 0; i < l_flushLen; ++i) {
          bool l_exit = (i == l_flushLen - 1);
          bool l_flush = (i == 0);
          TaggedWideFloat l_taggedValA(0, l_flush, l_exit);
          #pragma HLS data_pack variable=l_taggedValA
          p_outAs.write(l_taggedValA);
          TaggedWideFloat l_taggedValB(0, l_flush, l_exit);
          #pragma HLS data_pack variable=l_taggedValB
          p_outBs.write(l_taggedValB);
        }
    }
 
    ///////////////////////////////////////////////////////////////////////////
    // GemmCalc
    //  GEMM register meshes, triangular queues, flow of data
    //                            
    //              B             
    //              |  b            
    //              V bb           
    //               bbb          
    //              bbbb         C 
    //               |           ^
    //               V           |
    //         a    mmmm       oooo
    //        aa    mmmm       oooo
    // A->   aaa -> mmmm  ->   oooo
    //      aaaa    mmmm       oooo
    //     
    ///////////////////////////////////////////////////////////////////////////
    
    void
    GemmCalc(
        EdgeStream &p_As,
        EdgeStream &p_Bs,
        DdrStream &p_Cs
      ) {
      //#pragma HLS inline region off
      bool l_exit;
      bool l_firstFlushDone = false;
      unsigned int l_step = 0;
      unsigned int l_outCt = 2 * t_DdrWidth; // controls middle stage (Cout to CoutSave strobing)
      unsigned int l_outCt1 = 1 * t_DdrWidth; // controls the output stage (CoutSave shifting)

      WindowRm<TaggedFloatType, t_DdrWidth, t_DdrWidth> l_bwin;
      WindowRm<TaggedFloatType, t_DdrWidth, t_DdrWidth> l_awin;
      WindowRm<t_FloatType, t_DdrWidth, t_DdrWidth> l_cwin, l_cowin, l_cowinSave;
      #pragma HLS ARRAY_PARTITION variable=l_cwin dim=1 complete
      #pragma HLS ARRAY_PARTITION variable=l_cwin dim=2 complete
      #pragma HLS ARRAY_PARTITION variable=l_cowin dim=1 complete
      #pragma HLS ARRAY_PARTITION variable=l_cowin dim=2 complete
      #pragma HLS ARRAY_PARTITION variable=l_cowinSave dim=1 complete
      #pragma HLS ARRAY_PARTITION variable=l_cowinSave dim=2 complete
      TriangSrl<TaggedFloatType, t_DdrWidth> l_Ta;
      TriangSrl<TaggedFloatType, t_DdrWidth> l_Tb;
      //bool l_hlsWaWritingCowin = false;
      
      #ifndef __SYNTHESIS__
      l_Ta.clear();
      l_Tb.clear();
        l_awin.clear();
        l_bwin.clear();
        l_cwin.clear();
        l_cowin.clear();
        l_cowinSave.clear();
      #endif
      
      GEMM_CALC_DO:do {
        l_step++;
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
        #pragma HLS PIPELINE
        
        //#pragma HLS DEPENDENCE variable=l_cowin array inter false
        //#pragma HLS DEPENDENCE variable=l_cowin array intra false
        #pragma HLS DEPENDENCE variable=l_cwin array inter false
        //#pragma HLS DEPENDENCE variable=l_cwin array intra false
        
        #ifndef __SYNTHESIS__
          (t_debug >= 1) && std::cout << "\n" << "  ######### START step  " << l_step << "\n";
        #endif

        TaggedWideFloat l_a = p_As.read();
        TaggedWideFloat l_b = p_Bs.read();
        #pragma HLS data_pack variable=l_a
        #pragma HLS data_pack variable=l_b
        #ifndef __SYNTHESIS__
          (t_debug >= 1) && std::cout << "    GemmCalcT5 Received A " << l_a << "\n";
          (t_debug >= 1) && std::cout << "    GemmCalcT5 Received B " << l_b << "\n";
        #endif
        bool l_exitA = l_a.getExit();
        bool l_exitB = l_b.getExit();
        assert(l_exitA == l_exitB);
        l_exit = l_exitA;
        bool l_flushA = l_a.getFlush();
        bool l_flushB = l_b.getFlush();
        assert(l_flushA == l_flushB);
        bool l_flush = l_flushA;
        
        TaggedFloatArray l_avec = l_a.getVectOfTaggedValues();
        TaggedFloatArray l_bvec = l_b.getVectOfTaggedValues();
        
        TaggedFloatArray l_avec1 = l_Ta.shift(l_avec);
        TaggedFloatArray l_bvec1 = l_Tb.shift(l_bvec);
        
        (void)l_awin.shift_right(l_avec1);
        (void)l_bwin.shift(l_bvec1);
        
        #ifndef __SYNTHESIS__
          (t_debug >= 3) && std::cout << "  Calc before a step  " << l_step << "\n"
            << "  Ta\n" << l_Ta << "\n"
            << "  Tb\n" << l_Tb << "\n"
            << "  A\n" << l_awin << "\n"
            << "  B\n" << l_bwin << "\n"
            << "  C\n" << l_cwin << "\n"
            << "  Cout\n" << l_cowin << "\n"
            << "  CoutSave\n" << l_cowinSave << "\n"
            << "\n";
        #endif


        #ifndef __SYNTHESIS__
          (t_debug >= 1) && std::cout << "    CONTROLS  "
                    << "  l_outCt=" << l_outCt
                    << "  l_outCt1=" << l_outCt1
                    << std::endl;
        #endif
        
        if (l_outCt1 < t_DdrWidth) {
          DdrWideType l_cout = l_cowinSave.unshift();
          //l_hlsWaWritingCowin = true;
          #pragma HLS data_pack variable=l_cout
          //p_Cs.write(TaggedWideFloat(l_cout, false /*flush*/, l_exit));
          p_Cs.write(l_cout);
          #ifndef __SYNTHESIS__
            (t_debug >= 1) && std::cout << "    GemmCalc Sent C " << l_cout << "\n";
          #endif
        } else {
          //l_hlsWaWritingCowin = false;
        }

        if (l_outCt == 2 * t_DdrWidth - 1) {
          #ifndef __SYNTHESIS__
            (t_debug >= 1) && std::cout << "    GemmCalc Strobing l_cowin \n" << l_cowin << "\n";
          #endif
          l_cowinSave = l_cowin;
          l_outCt1 = 0;
        } else {
          l_outCt1++;
        }
                
        if (l_flush) {
          if (l_firstFlushDone) {
            l_outCt = 0;
          } else {
            l_firstFlushDone = true;
          }
        } else {
          l_outCt++;
        }

        GEMM_CALC_ROW:for(unsigned int row = 0; row < t_DdrWidth; ++row) {
          #pragma HLS UNROLL
          TaggedFloatArray l_arow = l_awin[row];
          TaggedFloatArray l_brow = l_bwin[row];
          //DdrWideType &l_crow = l_cwin[row];
          //#pragma HLS ARRAY_PARTITION variable=l_crow dim=1 complete
          #pragma HLS data_pack variable=l_arow
          #pragma HLS data_pack variable=l_brow
          //#pragma HLS data_pack variable=l_crow
          GEMM_CALC_COLS:for(unsigned int i = 0; i < t_DdrWidth; ++i) {
            #pragma HLS UNROLL
            t_FloatType aval = l_arow[i]();
            t_FloatType bval = l_brow[i]();
            bool aflush = l_arow[i].getFlush();
            bool bflush = l_brow[i].getFlush();
            assert(aflush == bflush);
            
            //assert(aflush != l_hlsWaWritingCowin);
            macStep(aval, bval, l_cwin[row][i], l_cowin[row][i], aflush);
          }
          //l_cwin[row] = l_crow;
        }
        
         
        #ifndef __SYNTHESIS__
          (t_debug >= 2) && std::cout << "  Calc after a step  " << l_step << "\n"
            << "  Ta\n" << l_Ta << "\n"
            << "  Tb\n" << l_Tb << "\n"
            << "  A\n" << l_awin << "\n"
            << "  B\n" << l_bwin << "\n"
            << "  C\n" << l_cwin << "\n"
            << "  Cout\n" << l_cowin << "\n"
            << "  CoutSave\n" << l_cowinSave << "\n"
            << "\n";
        #endif
      
     } while (!l_exit);
   }

    ///////////////////////////////////////////////////////////////////////////
    //GemmSplitEdges
    // Gemm split t_DdrWidth Edge stream into two half size edge stream
    ///////////////////////////////////////////////////////////////////////////
    void
	GemmSplitEdges(
		EdgeStream &p_InS,
		HalfEdgeStream &p_Out0S,
		HalfEdgeStream &p_Out1S
	){
		TaggedWideFloat l_in;
		HalfDdrWideType l_val0, l_val1;
		bool l_exit=false;
		bool l_flush=false;
		
		GemmSplitLoop: do{
		#pragma HLS PIPELINE
			l_in = p_InS.read();
			l_flush = l_in.getFlush();
			l_exit = l_in.getExit();
			for (int i=0; i<t_DdrWidth/2; ++i) {
			#pragma HLS UNROLL
				l_val0[i] = l_in[i];
			}
			for (int i=t_DdrWidth/2; i<t_DdrWidth; ++i){
			#pragma HLS UNROLL
				l_val1[i-t_DdrWidth/2] = l_in[i];
			}
			HalfTaggedWideFloat l_out0(l_val0, l_flush, l_exit);
			HalfTaggedWideFloat l_out1(l_val1, l_flush, l_exit);
			#ifndef __SYNTHESIS__
				(t_debug>2) && std::cout << "GemmSplitEdges input: " << l_in << std::endl;
				(t_debug>2) && std::cout << "GemmSplitEdges out0: " << l_out0 << std::endl;
				(t_debug>2) && std::cout << "GemmSplitEdges out1: " << l_out1 << std::endl;
			#endif
			p_Out0S.write(l_out0);
			p_Out1S.write(l_out1);			
		} while(!l_exit);
	}

	///////////////////////////////////////////////////////////////////////////
	//GemmDupEdges	
	// Gemm duplicate HalfEdgeStream
	///////////////////////////////////////////////////////////////////////////	
	void
	GemmDupEdges(
		HalfEdgeStream &p_InS,
		HalfEdgeStream &p_Out0S,
		HalfEdgeStream &p_Out1S
	) {
		bool l_exit=false;
		GemmDupEdgesLoop: do{
		#pragma HLS PIPELINE
			HalfTaggedWideFloat l_in = p_InS.read();
			l_exit = l_in.getExit();
			
			#ifndef __SYNTHESIS__
				(t_debug>2) && std::cout << "GemmDupEdges out0: " << l_in << std::endl;
				(t_debug>2) && std::cout << "GemmDupEdges out1: " << l_in << std::endl;
			#endif
			p_Out0S.write(l_in);
			p_Out1S.write(l_in);
		}while(!l_exit);
	}

	///////////////////////////////////////////////////////////////////////////	
	//GemmMergeDdrS
	// Gemm merge 4 HalfTaggedDdrStream into one DdrStream
	///////////////////////////////////////////////////////////////////////////	
   	void
	GemmMergeDdrS(
		HalfTaggedDdrStream p_InS[2][2],
		DdrStream			&p_OutS
	){
		bool l_exit=false;
		GemmMergeDdrsLoop: do{
		#pragma HLS PIPELINE
			for (int row=0; row<2; ++row){
				DdrWideType l_out;
				for (int k=0; k<t_DdrWidth/2; ++k) {
					for (int col=0; col<2; ++col){
					#pragma HLS UNROLL
						HalfTaggedDdrWideType l_in = p_InS[row][col].read();
						l_exit = l_in.getExit();
						for (int i=0; i<t_DdrWidth/2; ++i) {
							l_out[col*t_DdrWidth/2+i] = l_in[i];
						}
					}	
					if (!l_exit) {
						p_OutS.write(l_out);
					}
				}
			} 
		} while(!l_exit);
	} 
	///////////////////////////////////////////////////////////////////////////
    // GemmCalcHalf
    //  GEMM register meshes, triangular queues, flow of data of t_DdrWidth/2
    //                            
    //              B             
    //              |  b            
    //              V bb           
    //               bbb          
    //              bbbb         C 
    //               |           ^
    //               V           |
    //         a    mmmm       oooo
    //        aa    mmmm       oooo
    // A->   aaa -> mmmm  ->   oooo
    //      aaaa    mmmm       oooo
    //     
    ///////////////////////////////////////////////////////////////////////////
    
    void
    GemmCalcHalf(
        HalfEdgeStream &p_As,
        HalfEdgeStream &p_Bs,
        HalfTaggedDdrStream &p_Cs
      ) {
      //#pragma HLS inline region off
      bool l_exit;
      bool l_firstFlushDone = false;
      unsigned int l_step = 0;
      unsigned int l_outCt = 2 * t_DdrWidth/2; // controls middle stage (Cout to CoutSave strobing)
      unsigned int l_outCt1 = 1 * t_DdrWidth/2; // controls the output stage (CoutSave shifting)

      WindowRm<TaggedFloatType, t_DdrWidth/2, t_DdrWidth/2> l_bwin;
      WindowRm<TaggedFloatType, t_DdrWidth/2, t_DdrWidth/2> l_awin;
      WindowRm<t_FloatType, t_DdrWidth/2, t_DdrWidth/2> l_cwin, l_cowin, l_cowinSave;
      #pragma HLS ARRAY_PARTITION variable=l_cwin dim=1 complete
      #pragma HLS ARRAY_PARTITION variable=l_cwin dim=2 complete
      #pragma HLS ARRAY_PARTITION variable=l_cowin dim=1 complete
      #pragma HLS ARRAY_PARTITION variable=l_cowin dim=2 complete
      #pragma HLS ARRAY_PARTITION variable=l_cowinSave dim=1 complete
      #pragma HLS ARRAY_PARTITION variable=l_cowinSave dim=2 complete
      TriangSrl<TaggedFloatType, t_DdrWidth/2> l_Ta;
      TriangSrl<TaggedFloatType, t_DdrWidth/2> l_Tb;
      //bool l_hlsWaWritingCowin = false;
      
      #ifndef __SYNTHESIS__
      l_Ta.clear();
      l_Tb.clear();
        l_awin.clear();
        l_bwin.clear();
        l_cwin.clear();
        l_cowin.clear();
        l_cowinSave.clear();
      #endif
      
      GEMM_CALC_DO:do {
        l_step++;
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
        #pragma HLS PIPELINE
        
        //#pragma HLS DEPENDENCE variable=l_cowin array inter false
        //#pragma HLS DEPENDENCE variable=l_cowin array intra false
        #pragma HLS DEPENDENCE variable=l_cwin array inter false
        //#pragma HLS DEPENDENCE variable=l_cwin array intra false
        
        #ifndef __SYNTHESIS__
          (t_debug >= 1) && std::cout << "\n" << "  ######### START step  " << l_step << "\n";
        #endif

        HalfTaggedWideFloat l_a = p_As.read();
        HalfTaggedWideFloat l_b = p_Bs.read();
        #pragma HLS data_pack variable=l_a
        #pragma HLS data_pack variable=l_b
        #ifndef __SYNTHESIS__
          (t_debug >= 1) && std::cout << "    GemmCalcT5 Received A " << l_a << "\n";
          (t_debug >= 1) && std::cout << "    GemmCalcT5 Received B " << l_b << "\n";
        #endif
        bool l_exitA = l_a.getExit();
        bool l_exitB = l_b.getExit();
        assert(l_exitA == l_exitB);
        l_exit = l_exitA;
        bool l_flushA = l_a.getFlush();
        bool l_flushB = l_b.getFlush();
        assert(l_flushA == l_flushB);
        bool l_flush = l_flushA;
        
        HalfTaggedFloatArray l_avec = l_a.getVectOfTaggedValues();
        HalfTaggedFloatArray l_bvec = l_b.getVectOfTaggedValues();
        
        HalfTaggedFloatArray l_avec1 = l_Ta.shift(l_avec);
        HalfTaggedFloatArray l_bvec1 = l_Tb.shift(l_bvec);
        
        (void)l_awin.shift_right(l_avec1);
        (void)l_bwin.shift(l_bvec1);
        
        #ifndef __SYNTHESIS__
          (t_debug >= 3) && std::cout << "  Calc before a step  " << l_step << "\n"
            << "  Ta\n" << l_Ta << "\n"
            << "  Tb\n" << l_Tb << "\n"
            << "  A\n" << l_awin << "\n"
            << "  B\n" << l_bwin << "\n"
            << "  C\n" << l_cwin << "\n"
            << "  Cout\n" << l_cowin << "\n"
            << "  CoutSave\n" << l_cowinSave << "\n"
            << "\n";
        #endif


        #ifndef __SYNTHESIS__
          (t_debug >= 1) && std::cout << "    CONTROLS  "
                    << "  l_outCt=" << l_outCt
                    << "  l_outCt1=" << l_outCt1
                    << std::endl;
        #endif
        
		if ((l_outCt1 < t_DdrWidth/2) || (l_exit)) {
		  	HalfDdrWideType l_outVal;

		  	if (l_outCt1 < t_DdrWidth/2) {
          		l_outVal = l_cowinSave.unshift();
			}
          	HalfTaggedDdrWideType l_cout(l_outVal, l_exit);
          	#pragma HLS data_pack variable=l_cout
          	p_Cs.write(l_cout);
          	#ifndef __SYNTHESIS__
            	(t_debug >= 1) && std::cout << "    GemmCalc Sent C " << l_cout << "\n";
          	#endif
        }
        if (l_outCt == 2 * t_DdrWidth/2 - 1) {
          #ifndef __SYNTHESIS__
            (t_debug >= 1) && std::cout << "    GemmCalc Strobing l_cowin \n" << l_cowin << "\n";
          #endif
          l_cowinSave = l_cowin;
          l_outCt1 = 0;
        } else {
          l_outCt1++;
        }
                
        if (l_flush) {
          if (l_firstFlushDone) {
            l_outCt = 0;
          } else {
            l_firstFlushDone = true;
          }
        } else {
          l_outCt++;
        }

        GEMM_CALC_ROW:for(unsigned int row = 0; row < t_DdrWidth/2; ++row) {
          #pragma HLS UNROLL
          HalfTaggedFloatArray l_arow = l_awin[row];
          HalfTaggedFloatArray l_brow = l_bwin[row];
          //DdrWideType &l_crow = l_cwin[row];
          //#pragma HLS ARRAY_PARTITION variable=l_crow dim=1 complete
          #pragma HLS data_pack variable=l_arow
          #pragma HLS data_pack variable=l_brow
          //#pragma HLS data_pack variable=l_crow
          GEMM_CALC_COLS:for(unsigned int i = 0; i < t_DdrWidth/2; ++i) {
            #pragma HLS UNROLL
            t_FloatType aval = l_arow[i]();
            t_FloatType bval = l_brow[i]();
            bool aflush = l_arow[i].getFlush();
            bool bflush = l_brow[i].getFlush();
            assert(aflush == bflush);
            
            //assert(aflush != l_hlsWaWritingCowin);
            macStep(aval, bval, l_cwin[row][i], l_cowin[row][i], aflush);
          }
          //l_cwin[row] = l_crow;
        }
        
         
        #ifndef __SYNTHESIS__
          (t_debug >= 2) && std::cout << "  Calc after a step  " << l_step << "\n"
            << "  Ta\n" << l_Ta << "\n"
            << "  Tb\n" << l_Tb << "\n"
            << "  A\n" << l_awin << "\n"
            << "  B\n" << l_bwin << "\n"
            << "  C\n" << l_cwin << "\n"
            << "  Cout\n" << l_cowin << "\n"
            << "  CoutSave\n" << l_cowinSave << "\n"
            << "\n";
        #endif
      
     } while (!l_exit);
   }
	
	///////////////////////////////////////////////////////////////////////////
	//GemmCalComp
	// Gemm split the original systolic array, triangular shift into 4 and combine the results togeter
	///////////////////////////////////////////////////////////////////////////
	void
	GemmCalcComp(
		EdgeStream &p_As,
		EdgeStream &p_Bs,
		DdrStream &p_Cs
	){
		HalfEdgeStream l_edgesA[2];
		#pragma HLS DATA_PACK variable=l_edgesA
		//#pragma HLS STREAM variable=l_edgesA DEPTH=2

		HalfEdgeStream l_edgesB[2];
		#pragma HLS DATA_PACK variable=l_edgesB
		//#pragma HLS STREAM variable=l_edgesB DEPTH=2

		HalfEdgeStream l_edgesIntA[2][2];
		#pragma HLS DATA_PACK variable=l_edgesIntA
		//#pragma HLS STREAM variable=l_edgesIntA DEPTH=2

		HalfEdgeStream l_edgesIntB[2][2];
		#pragma HLS DATA_PACK variable=l_edgesIntB
		//#pragma HLS STREAM variable=l_edgesIntB DEPTH=2
		
		HalfTaggedDdrStream l_dataS[2][2];
		#pragma HLS DATA_PACK variable=l_dataS
		//#pragma HLS STREAM variable=l_dataS DEPTH=t_DdrWidth

		#pragma HLS DATAFLOW

		GemmSplitEdges(p_As, l_edgesA[0], l_edgesA[1]);
		GemmSplitEdges(p_Bs, l_edgesB[0], l_edgesB[1]);

		for (int row=0; row<2; ++row) {
		#pragma HLS UNROLL
			GemmDupEdges(l_edgesA[row], l_edgesIntA[row][0], l_edgesIntA[row][1]);
		}

		for (int col=0; col<2; ++col){
		#pragma HLS UNROLL
			GemmDupEdges(l_edgesB[col], l_edgesIntB[0][col], l_edgesIntB[1][col]);
		}

		for (int row=0; row<2; ++row) {
		#pragma HLS UNROLL
			for (int col=0; col<2; ++col) {
			#pragma HLS UNROLL
				GemmCalcHalf(l_edgesIntA[row][col], l_edgesIntB[row][col], l_dataS[row][col]);
			}
		}

		GemmMergeDdrS(l_dataS, p_Cs);
	}

    ///////////////////////////////////////////////////////////////////////////
    // GEMM writer
    // 
    ///////////////////////////////////////////////////////////////////////////
    void
    GemmWrite(
      DdrWideType *l_cAddr,
      DdrStream &p_Cs,
      unsigned int l_aColBlocks,
	  unsigned int l_aRowBlocks,
      unsigned int l_bColBlocks,
      unsigned int l_cWordLd
      ) {
		DdrWideType l_bufferC[t_aMH*t_bColMemWords];
		#pragma HLS ARRAY_PARTITION variable=l_bufferC dim=2
#pragma HLS RESOURCE variable=l_bufferC core=RAM_2P_LUTRAM
//#pragma HLS RESOURCE variable=l_bufferC core=RAM_T2P_BRAM
//#pragma HLS DEPENDENCE variable=l_bufferC inter false
        
		unsigned int l_rowOffset = 0;
		unsigned int l_colOffset = 0;
		unsigned int l_dstOffset=0;

		#ifndef __SYNTHESIS__
			(t_debug >= 1) && std::cout << "    GemmWrite Recieved C\n" ;
        #endif
		
			for (int rowBlock=0; rowBlock < l_aRowBlocks; ++rowBlock) {
				for (int colBlock=0; colBlock < l_bColBlocks; ++colBlock) { 
					l_colOffset = colBlock * t_bColMemWords;
					for (int i=0; i<t_aRowMemWords; ++i) {
						for (int j=0; j<t_bColMemWords; ++j) {
							for (int l=0; l<t_DdrWidth; ++l) {
						  #pragma HLS PIPELINE
						  #ifndef __SYNTHESIS__
							  (t_debug >= 1) && std::cout <<"\nl_bufferC[" << i*t_DdrWidth+l <<"][" <<j<<"]="<< "\n";
						  #endif
							  DdrWideType l_val = p_Cs.read();//l_word.getVal();
							  #ifndef __SYNTHESIS__
							  for (int k=0; k < t_DdrWidth; ++k) {
								  (t_debug >= 1) && std::cout << l_val[k] << "  ";
							  }
							#endif
							  l_bufferC[(i*t_DdrWidth+l)*t_bColMemWords+j] = l_val;
							}
						}
					}

					for (int m=1; m<l_aColBlocks; ++m) {
						for (int i=0; i<t_aRowMemWords;++i) {
							unsigned short l_arrBase = i*t_DdrWidth*t_bColMemWords;
							for (int j=0; j<t_bColMemWords; ++j) {
								unsigned short l_arrIdx = l_arrBase+j;
								DdrWideType l_bufferCReg;
								#pragma HLS ARRAY_PARTITION variable=l_bufferCReg complete
								DdrWideType l_val;
								l_bufferCReg = l_bufferC[l_arrIdx];
								for (int l=0; l<t_DdrWidth; ++l){
							  #pragma HLS PIPELINE
							  #ifndef __SYNTHESIS__
								  (t_debug >= 1) && std::cout <<"\nl_bufferC[" << i*t_DdrWidth+l <<"][" <<j<<"] +="<< "\n";
							  #endif
								  l_val = p_Cs.read();
								  #pragma HLS ARRAY_PARTITION variable=l_val complete 
								  for (int k=0; k<t_DdrWidth; ++k) {
									  l_bufferCReg[k] += l_val[k];
								  //l_bufferC[(i*t_DdrWidth+l)*t_bColMemWords+j][k] += l_val[k];
								  #ifndef __SYNTHESIS__
									  (t_debug >= 1) && std::cout << l_val[k] << "  ";
								  #endif
								  }
								  l_bufferC[l_arrIdx] = l_bufferCReg;
								  if (l < t_DdrWidth-1){
								  	l_bufferCReg = l_bufferC[l_arrIdx+t_bColMemWords];
								  }
								  l_arrIdx += t_bColMemWords;
								}
							}
						}
					}
					l_dstOffset = l_rowOffset + l_colOffset;

					#ifndef __SYNTHESIS__
						(t_debug >= 1) && std::cout << "Store l_bufferC to DDR\n ";
          	  	  	#endif
					//write l_bufferC back to DDR
					for (int i=0; i<t_aRowMemWords*t_DdrWidth; ++i) {
						unsigned int memAddr = l_dstOffset;
						unsigned short l_bufBase = i*t_bColMemWords;
						for (int j=0; j<t_bColMemWords; ++j) {
						#pragma HLS PIPELINE
							l_cAddr[memAddr+j] = l_bufferC[l_bufBase+j];
						}
						#ifndef __SYNTHESIS__
							(t_debug >= 1) && std::cout << "l_dstOffset = " << l_dstOffset << "\n";
							for (int j=0; j<t_bColMemWords; ++j){
								(t_debug >= 1) && std::cout << "l_bufferC[" << i <<"][" << j << "]=" << l_bufferC[l_bufBase+j];
							}
							(t_debug >= 1) && std::cout << "\n";
          	  	  	  	#endif
						l_dstOffset += l_cWordLd;	
					}	
				}
				l_rowOffset += l_cWordLd * t_DdrWidth*t_aRowMemWords;
			} 

    }

    //load A and B in t_DdrWidth x t_DdrWidth size blocks, multiply blocks and write results back to memory
    void GemmBlocks(
      DdrWideType *l_aAddr,
      DdrWideType *l_bAddr,
      DdrWideType *l_cAddr,
      unsigned int l_aColBlocks,
      unsigned int l_aRowBlocks,
      unsigned int l_bColBlocks,
      unsigned int l_aLd,
	  unsigned int l_bLd,
	  unsigned int l_cLd,
	  unsigned int l_transpBlocks
    ) {
      #pragma HLS DATAFLOW

      DdrStream  p_As0, p_Bs0; //p_As0_1, 
	  DdrStream  p_As1, p_As1_1, p_As1_2, p_Bs1, p_As2, p_As2_1, p_As2_2, p_As3, p_CEdgeS;
      EdgeStream  p_AEdgeS0, p_BEdgeS0;

      #pragma HLS data_pack variable=p_As0
      #pragma HLS data_pack variable=p_Bs0
      //#pragma HLS data_pack variable=p_As0_1
      #pragma HLS data_pack variable=p_Bs1
      #pragma HLS data_pack variable=p_As1
      #pragma HLS data_pack variable=p_As1_1
      #pragma HLS data_pack variable=p_As1_2
      #pragma HLS data_pack variable=p_Bs1
      #pragma HLS data_pack variable=p_As2_1
      #pragma HLS data_pack variable=p_As2_2
      #pragma HLS data_pack variable=p_As2
      #pragma HLS data_pack variable=p_As3

      #pragma HLS data_pack variable=p_AEdgeS0
      #pragma HLS data_pack variable=p_BEdgeS0
      #pragma HLS data_pack variable=p_CEdgeS

      #pragma HLS STREAM variable=p_As0 depth=t_aColMemWords*t_aMH
      #pragma HLS STREAM variable=p_Bs0 depth=t_bColMemWords*t_bKD
      //#pragma HLS STREAM variable=p_As0_1 depth=4
      #pragma HLS STREAM variable=p_Bs1 depth=4
      #pragma HLS STREAM variable=p_As1 depth=4
      #pragma HLS STREAM variable=p_As1_1 depth=4
      #pragma HLS STREAM variable=p_As1_2 depth=4
      #pragma HLS STREAM variable=p_As2_1 depth=4
      #pragma HLS STREAM variable=p_As2_2 depth=4
      #pragma HLS STREAM variable=p_As2 depth=4
      #pragma HLS STREAM variable=p_As3 depth=4

      #pragma HLS STREAM variable=p_AEdgeS0 depth=4
      #pragma HLS STREAM variable=p_BEdgeS0 depth=4
      //#pragma HLS STREAM variable=p_CEdgeS depth=t_aMH*t_bColMemWords
      #pragma HLS STREAM variable=p_CEdgeS depth=t_DdrWidth*2*t_bColMemWords
	//#pragma HLS RESOURCE variable=p_CEdgeS core=FIFO_LUTRAM

	  Transp<t_FloatType, t_DdrWidth, t_aColMemWords, 1> l_transp;

      GemmReadAB(l_aAddr, l_bAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, p_As0, p_Bs0);
	  //GemmBufferA(l_aColBlocks, l_aRowBlocks, l_bColBlocks, p_As0, p_As0_1);
	  GemmBufferB(l_aColBlocks, l_aRowBlocks, l_bColBlocks, p_Bs0, p_Bs1);
      //l_transp.shuffle_input(p_As0_1, p_As1, l_transpBlocks);
      l_transp.shuffle_input(p_As0, p_As1, l_transpBlocks);
	  l_transp.split(p_As1, p_As1_1, p_As1_2, l_transpBlocks);
      l_transp.WR_bufferWithReuse(p_As1_1, p_As2_1, ((l_transpBlocks/2)+(l_transpBlocks %2)), t_bColMemWords-1);
      l_transp.WR_bufferWithReuse(p_As1_2, p_As2_2, l_transpBlocks/2, t_bColMemWords-1);
	  l_transp.mergeWithReuse(p_As2_1, p_As2_2, p_As2, l_transpBlocks, t_bColMemWords-1);
      l_transp.shuffle_output(p_As2, p_As3, l_transpBlocks*t_bColMemWords);
      GemmTagAB(p_As3, p_Bs1, l_aColBlocks, l_aRowBlocks, l_bColBlocks, p_AEdgeS0, p_BEdgeS0);
      	#if GEMX_splitMesh
		GemmCalcComp(p_AEdgeS0, p_BEdgeS0, p_CEdgeS);
		#else
      	GemmCalc(p_AEdgeS0, p_BEdgeS0, p_CEdgeS);
		#endif
      GemmWrite(l_cAddr, p_CEdgeS, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_cLd);
    }

  public:
    void runGemm(
        DdrWideType *p_DdrRd,
        DdrWideType *p_DdrWr,
        GemmArgsType &p_Args
      ) {
        DdrWideType *l_aAddr = p_DdrRd + p_Args.m_Aoffset * DdrWideType::per4k();
         DdrWideType *l_bAddr = p_DdrRd + p_Args.m_Boffset * DdrWideType::per4k();
        DdrWideType *l_cAddr = p_DdrWr + p_Args.m_Coffset * DdrWideType::per4k();

        const unsigned int l_aColBlocks = p_Args.m_K / (t_DdrWidth * t_aColMemWords);
        const unsigned int l_aRowBlocks = p_Args.m_M / (t_DdrWidth * t_aRowMemWords);
        const unsigned int l_bColBlocks = p_Args.m_N / (t_DdrWidth * t_bColMemWords);
        const unsigned int l_aLd  = p_Args.m_Lda / t_DdrWidth;
        const unsigned int l_bLd  = p_Args.m_Ldb / t_DdrWidth;
        const unsigned int l_cLd  = p_Args.m_Ldc / t_DdrWidth;
		unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks *t_aRowMemWords;

        GemmBlocks(l_aAddr, l_bAddr, l_cAddr, l_aColBlocks, l_aRowBlocks,
                                     l_bColBlocks, l_aLd, l_bLd, l_cLd, l_transpBlocks);
      }
      
};

} // namespace
#endif

