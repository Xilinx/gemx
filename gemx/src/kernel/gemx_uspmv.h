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
 *  @brief Sparse matrix vector multiply  C += A * B
 *  sparse matrix A is stored in Coordinate Formate (COO): 32-bit row, 32-bit col, nnz val
 *	vector B and C will be read from DDR and stored into URAM
 *
 *  $DateTime: 2018/06/06 09:20:31 $
 */

#ifndef GEMX_USPMV_H
#define GEMX_USPMV_H

#include "assert.h"
#include "hls_stream.h"
#include "gemx_types.h"
#include "gemx_kargs.h"

namespace gemx {

template <
	typename t_FloatType,
	typename t_IdxType,
	unsigned int t_NumBanks,
	unsigned int t_Interleaves
>
class UspmvC
{
	private:
		t_FloatType m_ValC;
		t_IdxType m_Row;
	public:
		UspmvC(){}
		UspmvC(t_FloatType p_C, t_IdxType p_row)
			:m_ValC(p_C), m_Row(p_row) {}
		t_IdxType getRow() {
		#pragma HLS INLINE self
			return m_Row;
		}
		void setRow(t_IdxType p_row) {
		#pragma HLS INLINE self
			m_Row=p_row;
		}
		t_FloatType &getC() {
		#pragma HLS INLINE self
			return m_ValC;
		}
		t_IdxType getRowBank() {
		#pragma HLS INLINE self
			return (m_Row % t_NumBanks);
		}
		t_IdxType getRowGroup() {
		#pragma HLS INLINE self
			return ((m_Row / t_NumBanks) % t_Interleaves);
		}
		t_IdxType getRowOffset() {
		#pragma HLS INLINE self
			return m_Row / (t_NumBanks*t_Interleaves);
			}
    void setRowOffsetIntoRow(t_IdxType p_rowOffset) {
		#pragma HLS INLINE self
			m_Row = p_rowOffset;
		}
		void
		print(std::ostream& os) {
			os << std::setw(GEMX_FLOAT_WIDTH) << getRow() << " "
				 << std::setw(GEMX_FLOAT_WIDTH) << getC();
		}
};
template <typename T1, typename T2, unsigned int T3, unsigned int T4>
std::ostream& operator<<(std::ostream& os, UspmvC<T1, T2, T3, T4>& p_val) {
	p_val.print(os);
	return(os);
}

template <
	typename t_FloatType,
	typename t_IdxType,
	unsigned int t_NumBanks,
	unsigned int t_Interleaves
>
class TaggedUspmvC {
	private:
		UspmvC<t_FloatType, t_IdxType, t_NumBanks, t_Interleaves> m_Val;
		bool m_Flush;
		bool m_Exit;
	public:
		TaggedUspmvC() {}
		TaggedUspmvC(UspmvC<t_FloatType, t_IdxType, t_NumBanks, t_Interleaves> p_val, bool p_flush, bool p_exit)
			: m_Val(p_val),
				m_Flush(p_flush),
				m_Exit(p_exit)
		{}
		UspmvC<t_FloatType, t_IdxType, t_NumBanks, t_Interleaves> 
		&getVal(){
		#pragma HLS INLINE self
			return m_Val;
		}
		bool getFlush() {
		#pragma HLS INLINE self
			return m_Flush;
		}
		bool getExit() {
		#pragma HLS INLINE self
			return m_Exit;
		}
		void
		print(std::ostream& os) {
			m_Val.print(os);
			os << " f" << m_Flush << " e" << m_Exit;
		}
};
template <typename T1, typename T2, unsigned int T3, unsigned int T4>
std::ostream& operator<<(std::ostream& os, TaggedUspmvC<T1, T2, T3, T4>& p_val) {
	p_val.print(os);
	return(os);
}
//assume sparse matrix input has been sorted along col indices.     
//assume index type and data type have some number of bits
template <
  typename t_FloatType,
	typename t_IdxType,
  unsigned int t_DdrWidth,       // DDR width in t_FloatType
  unsigned int t_MvectorBlocks,  // controls max size of M, max_M = t_MvectorBlocks * t_DdrWidth
	unsigned int t_NnzVectorBlocks, //controls max size of Nnz, max_Nnz = t_NnzVectorBlocks * t_DdrWidth
	unsigned int t_Stages=1,
	unsigned int t_Interleaves=12 
 >
class Uspmv
{
 	private:
		static const unsigned int t_FloatSize = sizeof(t_FloatType);
		static const unsigned int t_ParamWidth = (t_FloatSize < 4)? t_DdrWidth/ (4/t_FloatSize): t_DdrWidth/(t_FloatSize/4);
		static const unsigned int t_UramWidth = 8 / t_FloatSize;
		static const unsigned int t_IdxSize = sizeof(t_IdxType);
		static const unsigned int t_IdxUramWidth = 8/t_IdxSize;
		static const unsigned int t_NumUramPerDdr = t_DdrWidth / t_UramWidth; //number of URAM slices used to store one data DDR
		static const unsigned int t_IdxNumUramPerDdr = t_NumUramPerDdr / 2; //number of URAM slices used to store one IDX DDR
		static const unsigned int t_DdrWidthMinusOne = t_DdrWidth -1;
		static const unsigned int t_Moffsets = t_MvectorBlocks / t_Interleaves;
		static const unsigned int t_DoubleDdrWidth = t_DdrWidth*2;
		static const unsigned int t_AddrBlocks = (t_Stages + t_DoubleDdrWidth - 1) / t_DoubleDdrWidth;
		static const unsigned int t_StagesPlusOne = t_Stages+1;
  public:
		typedef UspmvC<t_FloatType, t_IdxType, t_DdrWidth, t_Interleaves> UspmvCType;
		typedef TaggedUspmvC<t_FloatType, t_IdxType, t_DdrWidth, t_Interleaves> TaggedUspmvCType;

		typedef WideType<t_IdxType, t_IdxUramWidth> IdxUramType;
		typedef WideType<t_FloatType, t_UramWidth> DataUramType;

		typedef WideType<t_IdxType, t_DoubleDdrWidth> IdxDoubleDdrWideType;
		typedef WideType<t_IdxType, t_DdrWidth> IdxDdrWideType;
		typedef WideType<t_FloatType, t_DdrWidth> DataDdrWideType;
		typedef WideType<UspmvCType, t_DdrWidth> UspmvCDdrWideType;
		typedef TaggedWideType<UspmvCType, t_DdrWidth> TaggedUspmvCDdrWideType;

		typedef WideType<bool, t_DdrWidth> ControlWideType;
    typedef unsigned int ParamType;
		typedef WideType<ParamType, t_ParamWidth> ParamWideType;

		typedef hls::stream<t_FloatType> DataStreamType;
    typedef hls::stream<DataDdrWideType> DataDdrWideStreamType;
		typedef hls::stream<IdxDdrWideType> IdxDdrWideStreamType;
		typedef hls::stream<UspmvCDdrWideType> UspmvCDdrWideStreamType;
		typedef hls::stream<TaggedUspmvCDdrWideType> TaggedUspmvCDdrWideStreamType;

		typedef hls::stream<UspmvCType> UspmvCStreamType;
		typedef hls::stream<TaggedUspmvCType> TaggedUspmvCStreamType;

		typedef hls::stream<bool> ControlStreamType;
		typedef hls::stream<ControlWideType> ControlWideStreamType;
    typedef hls::stream<ParamType> ParamStreamType;

		typedef UspmvArgs UspmvArgsType;

  private:
		IdxUramType m_Acol[t_Stages][t_IdxNumUramPerDdr][t_NnzVectorBlocks];
		IdxUramType m_Arow[t_Stages][t_IdxNumUramPerDdr][t_NnzVectorBlocks];
		DataUramType m_Adata[t_Stages][t_NumUramPerDdr][t_NnzVectorBlocks];
		WideType<ParamType, t_Stages> m_NnzBlocks;
		WideType<t_IdxType, t_Stages> m_Mblocks;
		WideType<t_IdxType, t_Stages> m_Kblocks;
		WideType<t_FloatType, t_Stages> m_Prelus;
		
    static const unsigned int t_Debug = 0;

  private:
	//SPMV middle stage functions
		void
		readCol(
      ParamStreamType &p_inParams,
			DataDdrWideStreamType &p_inDataS, 
      ParamStreamType &p_outParams,
			IdxDdrWideStreamType &p_outIdxS, 
			DataDdrWideStreamType &p_outDataS, 
			unsigned int t_StageId){
      unsigned int p_numRuns = p_inParams.read();
      unsigned int p_nnzBlocks = p_inParams.read();
      unsigned int p_mBlocks = p_inParams.read(); 
      unsigned int p_bBlocks = p_inParams.read();
      p_outParams.write(p_numRuns);
      p_outParams.write(p_nnzBlocks);
      p_outParams.write(p_mBlocks);
      p_outParams.write(p_bBlocks);
      for (unsigned int i=t_StageId+1; i<t_Stages; ++i) {
				for (unsigned int j=0; j<4; ++j) {
					unsigned int l_param = p_inParams.read();
					p_outParams.write(l_param);
				}
      }
			for (unsigned int l_run=0; l_run < p_numRuns; ++l_run) {
      #pragma HLS DATAFLOW
LOOP_READ_COL_URAM:for (unsigned int l_nnzBlocks=0; l_nnzBlocks<p_nnzBlocks; ++l_nnzBlocks) {
        #pragma HLS PIPELINE
				  IdxUramType l_colUram[t_IdxNumUramPerDdr];
				  #pragma HLS ARRAY_PARTITION variable=l_colUram dim=2 complete
				  #pragma HLS ARRAY_PARTITION variable=l_colUram dim=1 complete
				  IdxDdrWideType l_col;
				  #pragma HLS ARRAY_PARTITION variable=l_col dim=1 complete
					for (unsigned int b=0; b<t_IdxNumUramPerDdr; ++b) {
					#pragma HLS unroll
						l_colUram[b] = m_Acol[t_StageId][b][l_nnzBlocks];
						for (unsigned int j=0; j<t_IdxUramWidth; ++j) {
						#pragma HLS unroll
							l_col[b*t_IdxUramWidth+j] = l_colUram[b][j];
						}
					}
          p_outIdxS.write(l_col);
        }

LOOP_FORWARD_B:for (unsigned int l_bBlocks=0; l_bBlocks<p_bBlocks; ++l_bBlocks) {
        #pragma HLS PIPELINE
          DataDdrWideType l_b = p_inDataS.read();
          p_outDataS.write(l_b);
        } 
			}
		}
		
    void
		pairB(
      ParamStreamType &p_inParams,
      IdxDdrWideStreamType &p_inIdxS,
			DataDdrWideStreamType &p_inDataS, 
      ParamStreamType &p_outParams,
			DataDdrWideStreamType &p_outDataS, 
			unsigned int t_StageId){
      unsigned int p_numRuns = p_inParams.read();
      unsigned int p_nnzBlocks = p_inParams.read();
      unsigned int p_mBlocks = p_inParams.read(); 
      unsigned int p_bBlocks = p_inParams.read();
      p_outParams.write(p_numRuns);
      p_outParams.write(p_nnzBlocks);
      p_outParams.write(p_mBlocks);
      p_outParams.write(p_bBlocks);
      for (unsigned int i=t_StageId+1; i<t_Stages; ++i) {
				for (unsigned int j=0; j<4; ++j) {
					unsigned int l_param = p_inParams.read();
					p_outParams.write(l_param);
				}
      }
			for (unsigned int l_run=0; l_run < p_numRuns; ++l_run) {
        IdxDdrWideType l_col;
        #pragma HLS ARRAY_PARTITION variable=l_col dim=1 complete
        DataDdrWideType l_b;
        #pragma HLS ARRAY_PARTITION variable=l_b complete
        DataDdrWideType l_val;
        #pragma HLS ARRAY_PARTITION variable=l_val complete
        IdxDdrWideType l_colOffset;
        #pragma HLS ARRAY_PARTITION variable=l_colOffset complete

        p_inIdxS.read(l_col);
        p_inDataS.read(l_b);
        unsigned int l_nnzBlocks=1;
        unsigned int l_bBlocks=1;
        unsigned int l_outBlocks=0;

        unsigned int l_startCol=0;
        unsigned int l_endCol=t_DdrWidth;

LOOP_PAIRB_MAIN:while (l_outBlocks < p_nnzBlocks) {
        #pragma HLS PIPELINE
          bool l_readCol = l_col[t_DdrWidthMinusOne] < l_endCol;
          for (unsigned int i=0; i<t_DdrWidth; ++i) {
          #pragma HLS UNROLL
            l_colOffset[i] = l_col[i] - l_startCol;
            if ((l_col[i] < l_endCol) && (l_col[i] >= l_startCol)){
               l_val[i] = l_b[l_colOffset[i]];
            } 
          }
          bool l_hasAs = l_nnzBlocks < p_nnzBlocks;
          bool l_hasBs = l_bBlocks < p_bBlocks;
          if (l_readCol) {
            p_outDataS.write(l_val);
            if (l_hasAs){
              p_inIdxS.read(l_col);
              l_nnzBlocks++;
            }
            l_outBlocks++;
          }
          else if (l_hasBs){
            p_inDataS.read(l_b);
            l_startCol = l_endCol;
            l_endCol += t_DdrWidth;
            l_bBlocks++;
          }
        }
LOOP_PAIRB_REST:while (l_bBlocks < p_bBlocks) {
          p_inDataS.read(l_b);
          l_bBlocks++;
        }
			}
		}
		void
		multAB(
			DataDdrWideStreamType &p_inBs, 
			DataDdrWideStreamType &p_outDataS,
      ParamStreamType &p_inParams,
      ParamStreamType &p_outParams,
			unsigned int t_StageId) {
      unsigned int p_numRuns = p_inParams.read();
      unsigned int p_nnzBlocks = p_inParams.read();
      unsigned int p_mBlocks = p_inParams.read(); 
      unsigned int p_bBlocks = p_inParams.read();
      p_outParams.write(p_numRuns);
      p_outParams.write(p_nnzBlocks);
      p_outParams.write(p_mBlocks);
      p_outParams.write(p_bBlocks);
      for (unsigned int i=t_StageId+1; i<t_Stages; ++i) {
				for (unsigned int j=0; j<4; ++j) {
					unsigned int l_param = p_inParams.read();
					p_outParams.write(l_param);
				}
      }
			unsigned int l_run=0;
			while (l_run < p_numRuns) {
				MULTAB_LOOP:for (unsigned int i=0; i<p_nnzBlocks; ++i) {
				#pragma HLS PIPELINE
					DataDdrWideType l_valA;
					#pragma HLS ARRAY_PARTITION variable=l_valA dim=1 complete
					DataUramType l_valUram[t_NumUramPerDdr];
					#pragma HLS ARRAY_PARTITION variable=l_valUram dim=1 complete
					#pragma HLS ARRAY_PARTITION variable=l_valUram dim=2 complete
					DataDdrWideType l_valB;
					#pragma HLS ARRAY_PARTITION variable=l_valB dim=1 complete
					DataDdrWideType l_valOut;
					#pragma HLS ARRAY_PARTITION variable=l_valOut dim=1 complete
					for (unsigned int b=0; b<t_NumUramPerDdr; ++b) {
						l_valUram[b] = m_Adata[t_StageId][b][i];
						for (unsigned int j=0; j<t_UramWidth; ++j) {
							l_valA[b*t_UramWidth+j] = l_valUram[b][j];
						}
					}
					p_inBs.read(l_valB);
					for (unsigned int j=0; j<t_DdrWidth; ++j) {
						l_valOut[j] = l_valA[j] * l_valB[j];
					}
					p_outDataS.write(l_valOut);
				}
				l_run++;
			}
		}
		void
		formC(
			DataDdrWideStreamType &p_inValS, 
			TaggedUspmvCStreamType p_outDataS[t_DdrWidth][t_DdrWidth], 
      ParamStreamType &p_inParams,
      ParamStreamType &p_outParams,
			unsigned int t_StageId) {
      unsigned int p_numRuns = p_inParams.read();
      unsigned int p_nnzBlocks = p_inParams.read();
      unsigned int p_mBlocks = p_inParams.read(); 
      unsigned int p_bBlocks = p_inParams.read();
      p_outParams.write(p_numRuns);
      p_outParams.write(p_nnzBlocks);
      p_outParams.write(p_mBlocks);
      p_outParams.write(p_bBlocks);
      for (unsigned int i=t_StageId+1; i<t_Stages; ++i) {
				for (unsigned int j=0; j<4; ++j) {
					unsigned int l_param = p_inParams.read();
					p_outParams.write(l_param);
				}
      }
			unsigned int l_run=0;
			while (l_run < p_numRuns) {
				UspmvCDdrWideType l_valOut;
				#pragma HLS ARRAY_PARTITION variable=l_valOut dim=1 complete
				//ControlWideType l_cnt;
				//#pragma HLS ARRAY_PARTITION variable=l_cnt dim=1 complete
				FORMC_LOOP:for (unsigned int i=0; i<p_nnzBlocks; ++i) {
				#pragma HLS PIPELINE
					IdxDdrWideType l_idx;
					#pragma HLS ARRAY_PARTITION variable=l_idx dim=1 complete
					IdxUramType l_rowUram[t_IdxNumUramPerDdr];
					#pragma HLS ARRAY_PARTITION variable=l_rowUram dim=1 complete
					#pragma HLS ARRAY_PARTITION variable=l_rowUram dim=2 complete
					DataDdrWideType l_valC;
					#pragma HLS ARRAY_PARTITION variable=l_valC dim=1 complete
					for (unsigned int b=0; b<t_IdxNumUramPerDdr; ++b) {
						l_rowUram[b] = m_Arow[t_StageId][b][i];
						for (unsigned int j=0; j<t_IdxUramWidth; ++j) {
							l_idx[b*t_IdxUramWidth+j] = l_rowUram[b][j];
						}
					}
					l_valC = p_inValS.read();
					for (unsigned int w=0; w<t_DdrWidth; ++w) {
					#pragma HLS UNROLL
						l_valOut[w].getC()=l_valC[w];
						l_valOut[w].setRow(l_idx[w]);
						//l_cnt[w] = (l_valC[w] != 0);
						//if (l_cnt[w]) {
							unsigned int l_rowBank = l_idx[w] % t_DdrWidth;
							TaggedUspmvCType l_taggedVal(l_valOut[w], false, false);
							p_outDataS[w][l_rowBank].write(l_taggedVal);
						//}
					}
				}
				l_run++;
				for (unsigned int w=0; w<t_DdrWidth; ++w) {
				#pragma HLS UNROLL
					for (unsigned int b=0; b<t_DdrWidth; ++b){
					#pragma HLS UNROLL
						UspmvCType l_endEntry(0, 0);
						TaggedUspmvCType l_taggedEndEntry(l_endEntry, true, (l_run == p_numRuns));
						p_outDataS[w][b].write(l_taggedEndEntry);
					}
				}
			}
		}
   
   
		void
		xBarRow(TaggedUspmvCStreamType p_inDataS[t_DdrWidth][t_DdrWidth],
						TaggedUspmvCStreamType p_outDataS[t_DdrWidth],
            ParamStreamType &p_inParams,
            ParamStreamType p_outParams[t_DdrWidth],
            unsigned int t_StageId) {
      unsigned int p_numRuns = p_inParams.read();
      unsigned int p_nnzBlocks = p_inParams.read();
      unsigned int p_mBlocks = p_inParams.read(); 
      unsigned int p_bBlocks = p_inParams.read();
			for (unsigned int w=0; w<t_DdrWidth; ++w) {
			#pragma HLS UNROLL
      	p_outParams[w].write(p_numRuns);
      	p_outParams[w].write(p_nnzBlocks);
      	p_outParams[w].write(p_mBlocks);
      	p_outParams[w].write(p_bBlocks);
			}
      for (unsigned int i=t_StageId+1; i<t_Stages; ++i) {
				for (unsigned int j=0; j<4; ++j) {
					unsigned int l_param = p_inParams.read();
					for (unsigned int w=0; w<t_DdrWidth; ++w) {
					#pragma HLS UNROLL
      			p_outParams[w].write(l_param);
					}
				}
      }
			
			BoolArr<t_DdrWidth> l_exit(false);
			ap_uint<t_DdrWidth> l_numExitBanks[t_DdrWidth];
			#pragma HLS ARRAY_PARTITION variable=l_numExitBanks complete
			#ifndef __SYNTHESIS__
			for (unsigned int i=0; i<t_DdrWidth; ++i) {
				l_numExitBanks[i] = 0;
			}
			#endif
			BoolArr<t_DdrWidth> l_exitRun(true);
			BoolArr<t_DdrWidth> l_isWriteOut(false);
			BoolArr<t_DdrWidth> l_finRun[t_DdrWidth];
			#pragma HLS ARRAY_PARTITION variable=l_finRun dim=0 complete
			ap_uint<t_DdrWidth> l_numFinBanks[t_DdrWidth];
			#pragma HLS ARRAY_PARTITION variable=l_numFinBanks complete
			do {
				for (unsigned int i=0; i<t_DdrWidth; ++i) {
				#pragma HLS UNROLL
					if (l_exitRun[i]) {
						l_finRun[i].Reset();
						l_numFinBanks[i] = 0;
					}
				}
				XBARROW_LOOP:do {
					#pragma HLS PIPELINE
						for (unsigned int l_bank=0; l_bank < t_DdrWidth; ++l_bank){
						#pragma HLS unroll
							unsigned int l_idx=0;
							l_isWriteOut[l_bank] = false;
							for (unsigned int l_w=0; l_w < t_DdrWidth; ++l_w) {
							#pragma HLS unroll
								if (!p_inDataS[l_w][l_bank].empty() && !l_finRun[l_bank][l_w]) {
									l_isWriteOut[l_bank] = true;
									l_idx = l_w;
									break;
								}
							}
							TaggedUspmvCType l_taggedVal;
							if (l_isWriteOut[l_bank]) {
								p_inDataS[l_idx][l_bank].read(l_taggedVal);
								bool l_finRunTmp = l_taggedVal.getFlush();
								bool l_finish	= l_taggedVal.getExit();
								l_finRun[l_bank][l_idx] = l_finRunTmp;
								if (!l_finRunTmp || (l_finRunTmp && l_numFinBanks[l_bank][t_DdrWidth-2] )) {
									p_outDataS[l_bank].write(l_taggedVal);
								}
								if (l_finRunTmp){
									l_numFinBanks[l_bank].range(t_DdrWidth-1, 1) = l_numFinBanks[l_bank].range(t_DdrWidth-2, 0);
									l_numFinBanks[l_bank][0] = 1;
								}
								if (l_finish) {
									l_numExitBanks[l_bank].range(t_DdrWidth-1,1) = l_numExitBanks[l_bank].range(t_DdrWidth-2,0);
									l_numExitBanks[l_bank][0] = 1;
								}
							} 
							l_exitRun[l_bank] = l_numFinBanks[l_bank][t_DdrWidth-1];
							l_exit[l_bank] = l_numExitBanks[l_bank][t_DdrWidth-1];
						}
					} while (l_isWriteOut.Or());
			} while (!l_exit.And());
		}

		void
		rowInterleave(
			TaggedUspmvCStreamType &p_inS, 
			TaggedUspmvCStreamType p_outS[t_Interleaves], 
      ParamStreamType &p_inParams,
      ParamStreamType &p_outParams,
      unsigned int t_StageId
		) {
			for (unsigned int i=0; i<4; ++i) {
				unsigned int l_param = p_inParams.read();
				p_outParams.write(l_param);
			}
      for (unsigned int i=t_StageId+1; i<t_Stages; ++i) {
				for (unsigned int j=0; j<4; ++j) {
					unsigned int l_param = p_inParams.read();
					p_outParams.write(l_param);
				}
      }

			bool l_exit=false;	
			bool l_exitRun=false;
			UspmvCType l_initEntry(0, 0);
			TaggedUspmvCType l_taggedVal(l_initEntry, false, false);
			UspmvCType l_val;
			do {
				bool l_isRead = false;
				ROWINTERLEAVE_LOOP:do {
				#pragma HLS PIPELINE
					l_isRead = false;
					l_exitRun = false;
					l_exit = false;
					if (p_inS.read_nb(l_taggedVal)) {
						//p_inS.read(l_taggedVal);
						l_val = l_taggedVal.getVal();
						t_IdxType l_row = l_val.getRow();
						unsigned int l_rowGroup = l_val.getRowGroup();
						if (l_val.getC() !=0) {
							p_outS[l_rowGroup].write(l_taggedVal);
						}
						l_isRead = true;
					}
					if (l_isRead) {
						l_exitRun = l_taggedVal.getFlush();
				  	l_exit = l_taggedVal.getExit();
					}
				}while (!l_exitRun);
        ROWINTERLEVE_OUPUT_EXIT:for (unsigned int i=0; i<t_Interleaves; ++i) {
				#pragma HLS unroll
        #pragma HLS PIPELINE
					UspmvCType l_endEntry(0, 0);
					TaggedUspmvCType l_taggedEndEntry(l_endEntry, l_exitRun, l_exit);
					p_outS[i].write(l_taggedEndEntry);
				}
        //l_exitRun = false;
			} while (!l_exit);
		}

		void
		accRowBank(
			TaggedUspmvCStreamType p_inS[t_Interleaves],
			DataStreamType &p_outS,
      ParamStreamType &p_inParams,
      ParamStreamType &p_outParams,
			//unsigned int t_BankId,
			unsigned int t_StageId
		){
			unsigned int p_mBlocks; 
			for (unsigned int i=0; i<4; ++i) {
				unsigned int l_param = p_inParams.read();
				if (i==2) {
					p_mBlocks = l_param;
				}
				p_outParams.write(l_param);
			}
      for (unsigned int i=t_StageId+1; i<t_Stages; ++i) {
				for (unsigned int j=0; j<4; ++j) {
					unsigned int l_param = p_inParams.read();
					p_outParams.write(l_param);
				}
      }
			t_FloatType l_cData[t_Interleaves][t_Moffsets];
			#pragma HLS ARRAY_PARTITION variable=l_cData dim=1 complete
			BoolArr<t_Interleaves> l_exit = false;
			do {
				//init m_Cdata with 0s
				ACCROWBANK_LOOP1:for (unsigned int i=0; i<t_Moffsets; ++i) {
        #pragma HLS PIPELINE
          for (unsigned int j=0; j<t_Interleaves; ++j) {
					#pragma HLS unroll
            l_cData[j][i] = 0;
          }
        }
			 	BoolArr<t_Interleaves> l_exitRun(false);
				TaggedUspmvCType l_taggedCVal[t_Interleaves];
        #pragma HLS ARRAY_PARTITION variable=l_taggedCVal complete
				ACCROWBANK_LOOP2:do {
				#pragma HLS PIPELINE II=t_Interleaves
					for (unsigned int g = 0; g < t_Interleaves; ++g) {
					#pragma HLS unroll
						if (!l_exitRun[g]) {
							if (p_inS[g].read_nb(l_taggedCVal[g])) {
								UspmvCType l_cVal = l_taggedCVal[g].getVal();
								l_exitRun[g] = l_taggedCVal[g].getFlush();
								l_exit[g] = l_taggedCVal[g].getExit();
								t_IdxType l_row = l_cVal.getRow();
								unsigned int l_rowOffset = l_cVal.getRowOffset();
								//if (!l_exitRun[g]){
									t_FloatType l_valC = l_cVal.getC();
									l_cData[g][l_rowOffset] += l_valC;
								//}
							}
						}
					}
				} while(!l_exitRun.And());

				//output result vector
				unsigned int l_offsets = p_mBlocks / t_Interleaves;
        //assert(l_offsets*t_Interleaves == p_mBlocks);
        for (unsigned int i=0; i<l_offsets; ++i) {
          for (unsigned int j=0; j<t_Interleaves; ++j) {
          #pragma HLS PIPELINE
            t_FloatType l_val;
            //l_val = m_Cdata[t_StageId][t_BankId][j][i];
            l_val = l_cData[j][i];
            p_outS.write(l_val);
          }
        }
			} while (!l_exit.And());
		}

	void
	mergeC(
		DataStreamType p_inS[t_DdrWidth], 
		DataDdrWideStreamType &p_outS,
    ParamStreamType p_inParams[t_DdrWidth],
    ParamStreamType &p_outParams, 
    unsigned int t_StageId
		){
    unsigned int p_numRuns = p_inParams[0].read();
    unsigned int p_nnzBlocks = p_inParams[0].read();
    unsigned int p_mBlocks = p_inParams[0].read(); 
    unsigned int p_bBlocks = p_inParams[0].read();
    if (t_StageId == (t_Stages-1)) {
      p_outParams.write(p_numRuns);
      p_outParams.write(p_mBlocks);
    }
    for (unsigned int w=1; w<t_DdrWidth; ++w) {
    #pragma HLS UNROLL
      unsigned int l_numRuns = p_inParams[w].read();
      unsigned int l_nnzBlocks = p_inParams[w].read();
      unsigned int l_mBlocks = p_inParams[w].read();
      unsigned int l_kBlocks = p_inParams[w].read();
    }
    for (unsigned int i=t_StageId+1; i<t_Stages; ++i) {
			for (unsigned int j=0; j<4; ++j) {
				unsigned int l_param = p_inParams[0].read();
				p_outParams.write(l_param);
			}
      for (unsigned int w=1; w<t_DdrWidth; ++w) {
      #pragma HLS UNROLL
				for (unsigned int j=0; j<4; ++j) {
					unsigned int l_param = p_inParams[w].read();
				}
      }
    }
		t_FloatType l_cData[t_DdrWidth][t_MvectorBlocks];
		#pragma HLS ARRAY_PARTITION variable=l_cData dim=1 complete
		unsigned int l_counters[t_DdrWidth];
		#pragma HLS ARRAY_PARTITION variable=l_counters complete
		BoolArr<t_DdrWidth> l_contRun(true);

		unsigned int l_run=0;
		while (l_run < p_numRuns) {
			for (unsigned int i=0; i<t_DdrWidth; ++i) {
			#pragma HLS UNROLL
				l_counters[i] = 0; //p_mBlocks;
        l_contRun[i] = true;
			}
			LOOP_MERGC_MAIN:while (l_contRun.Or()) {
				#pragma HLS PIPELINE
					DataDdrWideType l_val;
					#pragma HLS ARRAY_PARTITION variable=l_val complete
					for (unsigned int i=0; i<t_DdrWidth; ++i) {
					#pragma HLS unroll
						if (l_contRun[i]) {
							if (p_inS[i].read_nb(l_val[i])) {
								l_cData[i][l_counters[i]] = l_val[i];
								l_counters[i]++;
							}
						}
						l_contRun[i] = (l_counters[i]<p_mBlocks);
					}
			}
			//stream out C
			LOOP_MERGEC_OUTPUT:for (unsigned int i=0; i<p_mBlocks; ++i) {
			#pragma HLS PIPELINE
				DataDdrWideType l_val;
				#pragma HLS ARRAY_PARTITION variable=l_val complete
				for (unsigned int j=0; j<t_DdrWidth; ++j) {
					l_val[j] = (l_cData[j][i]<0)? l_cData[j][i]*m_Prelus[t_StageId]: l_cData[j][i];
				}
				p_outS.write(l_val);
			}
			
			l_run++;
		}
	}	

public:
		void
		spmvCompute(
			DataDdrWideStreamType &p_inBs, 
			DataDdrWideStreamType &p_outCs,
      ParamStreamType &p_inParams,
      ParamStreamType &p_outParams, 
			unsigned int t_StageId) {
			static const unsigned int t_FifoDeep=16;//16;
			static const unsigned int t_FifoShallow = 2;
			
			IdxDdrWideStreamType l_idx2pairB;
			#pragma HLS data_pack variable=l_idx2pairB
			#pragma HLS stream variable=l_idx2pairB depth=t_FifoShallow
			DataDdrWideStreamType l_data2pairB;
			#pragma HLS data_pack variable=l_data2pairB
			#pragma HLS stream variable=l_data2pairB depth=t_FifoShallow
			DataDdrWideStreamType l_dataB2multAB;
			#pragma HLS data_pack variable=l_dataB2multAB
			#pragma HLS stream variable=l_dataB2multAB depth=t_FifoShallow
			DataDdrWideStreamType l_data2formC;
			#pragma HLS data_pack variable=l_data2formC
			#pragma HLS stream variable=l_data2formC depth=t_FifoShallow
			TaggedUspmvCStreamType l_data2xBarRow[t_DdrWidth][t_DdrWidth];
			#pragma HLS data_pack variable=l_data2xBarRow
			#pragma HLS stream variable=l_data2xBarRow depth=t_FifoShallow

			TaggedUspmvCStreamType l_data2rowInterleave[t_DdrWidth];
			#pragma HLS data_pack variable=l_data2rowInterleave
			#pragma HLS stream variable=l_data2rowInterleave depth=t_FifoShallow
			TaggedUspmvCStreamType l_data2accRowBank[t_DdrWidth][t_Interleaves];
			#pragma HLS data_pack variable=l_data2accRowBank
			#pragma HLS stream variable=l_data2accRowBank depth=t_FifoShallow
			DataStreamType l_data2mergeC[t_DdrWidth];
			#pragma HLS data_pack variable=l_data2mergeC
			#pragma HLS stream variable=l_data2mergeC depth=t_FifoShallow
      ParamStreamType l_param2pairB;
      #pragma HLS data_pack variable=l_param2pairB
      #pragma HLS stream variable=l_param2pairB
      ParamStreamType l_param2multAB;
      #pragma HLS data_pack variable=l_param2multAB
      #pragma HLS stream variable=l_param2multAB
      ParamStreamType l_param2formC;
      #pragma HLS data_pack variable=l_param2formC
      #pragma HLS stream variable=l_param2formC
      ParamStreamType l_param2xBarRow;
      #pragma HLS data_pack variable=l_param2xBarRow
      #pragma HLS stream variable=l_param2xBarRow
      ParamStreamType l_param2rowInterleave[t_DdrWidth];
      #pragma HLS data_pack variable=l_param2rowInterleave
      #pragma HLS stream variable=l_param2rowInterleave
      ParamStreamType l_param2accRowBank[t_DdrWidth];
      #pragma HLS data_pack variable=l_param2accRowBank
      #pragma HLS stream variable=l_param2accRowBank
      ParamStreamType l_param2mergeC[t_DdrWidth];
      #pragma HLS data_pack variable=l_param2mergeC
      #pragma HLS stream variable=l_param2mergeC
		
			#pragma HLS DATAFLOW
      readCol(p_inParams, p_inBs, l_param2pairB, l_idx2pairB, l_data2pairB, t_StageId);
			pairB(l_param2pairB, l_idx2pairB, l_data2pairB, l_param2multAB, l_dataB2multAB, t_StageId);
			multAB(l_dataB2multAB, l_data2formC, l_param2multAB, l_param2formC, t_StageId); 
			formC(l_data2formC, l_data2xBarRow, l_param2formC, l_param2xBarRow, t_StageId);
			xBarRow(l_data2xBarRow, l_data2rowInterleave, l_param2xBarRow, l_param2rowInterleave, t_StageId); 
			for (unsigned int w=0; w<t_DdrWidth; ++w) {
			#pragma HLS unroll
				rowInterleave(l_data2rowInterleave[w], l_data2accRowBank[w], l_param2rowInterleave[w], l_param2accRowBank[w], t_StageId);
				accRowBank(l_data2accRowBank[w], l_data2mergeC[w], l_param2accRowBank[w], l_param2mergeC[w], t_StageId);
			}
			mergeC(l_data2mergeC,p_outCs, l_param2mergeC, p_outParams, t_StageId);
		}

		//initiate sparse matrices with data from device memory
		void
		loadA(
			DataDdrWideType *p_aAddr
		) {
			//load all As
			unsigned int l_offset=0;
			//load A descripts: NNZs, Ms and Ks
			WideConv<DataDdrWideType, ParamWideType> l_convNnz;
			//WideType<ParamType, t_Stages> l_nnzVals;
			ParamWideType l_nnzVals;
			#pragma HLS ARRAY_PARTITION variable=l_nnzVals dim=1 complete
			for (unsigned int i=0; i<t_AddrBlocks; ++i){
			#pragma HLS PIPELINE
				DataDdrWideType l_data = p_aAddr[l_offset+i];
				ParamWideType l_addr = l_convNnz.convert(l_data); 
				#pragma HLS ARRAY_PARTITION variable=l_addr dim=1 complete
				for (unsigned int j=0; j<t_ParamWidth; ++j) {
					l_nnzVals[i*t_ParamWidth+j] = l_addr[j];
				}
			}
			for (unsigned int i=0; i<t_Stages; ++i) {
			#pragma HLS UNROLL
				m_NnzBlocks[i] = (l_nnzVals[i] / t_DdrWidth);
			}
			l_offset += t_AddrBlocks;

			WideConv<DataDdrWideType, IdxDoubleDdrWideType> l_conv;
			//WideType<t_IdxType, t_Stages+1> l_mSize;
			IdxDoubleDdrWideType l_mSize;
			#pragma HLS ARRAY_PARTITION variable=l_mSize dim=1 complete
			for (unsigned int i=0; i<t_AddrBlocks; ++i){
			#pragma HLS PIPELINE
				DataDdrWideType l_data = p_aAddr[l_offset+i];
				IdxDoubleDdrWideType l_addr = l_conv.convert(l_data); 
				#pragma HLS ARRAY_PARTITION variable=l_addr dim=1 complete
				for (unsigned int j=0; j<t_DoubleDdrWidth; ++j) {
					l_mSize[i*t_DoubleDdrWidth+j] = l_addr[j];
				}
			}
			for (unsigned int i=0; i<t_Stages; ++i) {
			#pragma HLS UNROLL
				m_Mblocks[i] = (l_mSize[i] / t_DdrWidth);
			}
			for (unsigned int i=1; i<t_Stages; ++i) {
			#pragma HLS PIPELINE
				m_Kblocks[i] = m_Mblocks[i-1];
			}
			m_Kblocks[0] = l_mSize[t_Stages]/t_DdrWidth;

			l_offset += t_AddrBlocks;

			//load Prelus for each stage
			for (unsigned int i=0; i<t_AddrBlocks; ++i) {
			#pragma HLS PIPELINE
				DataDdrWideType l_data = p_aAddr[l_offset+i];
				#pragma HLS ARRAY_PARTITION variable=l_data complete
				for (unsigned int j=0; j<t_DdrWidth; ++j) {
					if ((i*t_DdrWidth+j) < t_Stages) {
						m_Prelus[i*t_DdrWidth+j] = l_data[j];
					}
				}				
			}
			l_offset += t_AddrBlocks;
			//load A col row and val
			for (unsigned int l_stage=0; l_stage<t_Stages; ++l_stage) {
				//read Col, Row and Val of each Sparse matrix
				//load Col and row
				unsigned int l_nnzBlocks = m_NnzBlocks[l_stage];
				for (unsigned int i=0; i<l_nnzBlocks; ++i) {
				#pragma HLS PIPELINE
					DataDdrWideType l_val = p_aAddr[l_offset + i];
					WideConv<DataDdrWideType, IdxDoubleDdrWideType> l_conv;
					IdxDoubleDdrWideType l_colRow = l_conv.convert(l_val);
					#pragma HLS ARRAY_PARTITION variable=l_colRow dim=1 complete
					IdxUramType l_uramCol[t_IdxNumUramPerDdr];
					#pragma HLS ARRAY_PARTITION variable=l_uramCol dim=1 complete
					#pragma HLS ARRAY_PARTITION variable = l_uramCol dim=2 complete
					IdxUramType l_uramRow[t_IdxNumUramPerDdr];
					#pragma HLS ARRAY_PARTITION variable=l_uramRow dim=1 complete
					#pragma HLS ARRAY_PARTITION variable = l_uramRow dim=2 complete
					for (unsigned int j=0; j<t_IdxNumUramPerDdr; ++j) {
						for (unsigned int k=0; k<t_IdxUramWidth; ++k) {
							l_uramCol[j][k] = l_colRow[j*t_IdxUramWidth*2+k*2];
							l_uramRow[j][k] = l_colRow[j*t_IdxUramWidth*2+k*2+1];
						}
						m_Acol[l_stage][j][i] = l_uramCol[j];
						m_Arow[l_stage][j][i] = l_uramRow[j];
					}
				}
				l_offset += l_nnzBlocks;

				//load value
				for (unsigned int i=0; i<l_nnzBlocks; ++i) {
				#pragma HLS PIPELINE
					DataDdrWideType l_val = p_aAddr[l_offset + i];
					#pragma HLS ARRAY_PARTITION variable=l_val dim=1 complete
					DataUramType l_uramVal[t_NumUramPerDdr];
					#pragma HLS ARRAY_PARTITION variable=l_uramVal dim=1 complete
					#pragma HLS ARRAY_PARTITION variable = l_uramVal dim=2 complete
					for (unsigned int j=0; j<t_NumUramPerDdr; ++j) {
						for (unsigned int k=0; k<t_UramWidth; ++k) {
							l_uramVal[j][k] = l_val[j*t_UramWidth+k];
						}
						m_Adata[l_stage][j][i] = l_uramVal[j];
					}
				}
				l_offset += l_nnzBlocks;
			}
		}

		void loadB (
			DataDdrWideType *p_bAddr,
			DataDdrWideStreamType &p_outBs,
      ParamStreamType &p_outParams,
			unsigned int p_numRuns
		) {
      //stream out parameters
      //p_outParams.write(p_numRuns);
      for (unsigned int i=0; i<t_Stages; ++i) {
      #pragma HLS PIPELINE
        unsigned int l_nnzBlocks = m_NnzBlocks[i];
        unsigned int l_mBlocks = m_Mblocks[i];
        unsigned int l_kBlocks = m_Kblocks[i];
      	p_outParams.write(p_numRuns);
        p_outParams.write(l_nnzBlocks);
        p_outParams.write(l_mBlocks);
        p_outParams.write(l_kBlocks);
      }
			//loadB
			unsigned int l_offset=0;
			unsigned int l_run=0;
			unsigned int l_kBlocks = m_Kblocks[0];
			while (l_run<p_numRuns) {
				for (unsigned int j=0; j<l_kBlocks; ++j) {
				#pragma HLS PIPELINE
					DataDdrWideType l_data = p_bAddr[l_offset+j];
					p_outBs.write(l_data);
				}
				l_offset += l_kBlocks;
				l_run++;
			}		
		}
		
		void
		storeC(
			DataDdrWideStreamType &p_inCs,
			DataDdrWideType *p_cAddr,
      ParamStreamType &p_inParams
		){
      unsigned int p_numRuns = p_inParams.read();

      unsigned int l_mBlocks;
      p_inParams.read(l_mBlocks);

			DataDdrWideType *l_cAddr = p_cAddr;
			unsigned int l_run=0;
			while (l_run < p_numRuns) {
				for (unsigned int i=0; i<l_mBlocks; ++i){
				#pragma HLS PIPELINE
					DataDdrWideType l_data = p_inCs.read();
					l_cAddr[i] = l_data;
				}
				l_cAddr += l_mBlocks;
				l_run++;
			}
		}
	
		void
		streamUspmv(
			DataDdrWideType *p_bAddr,
			DataDdrWideType *p_cAddr,
			unsigned int p_numRuns
		)
		{
			DataDdrWideStreamType l_bS[t_StagesPlusOne];
			#pragma HLS data_pack variable=l_bS
			#pragma HLS stream variable=l_bS 
      ParamStreamType l_paramS[t_StagesPlusOne];
      #pragma HLS data_pack variable=l_paramS
      #pragma HLS stream variable=l_paramS 
 
			#pragma HLS resource variable=m_Acol core=XPM_MEMORY uram
			#pragma HLS data_pack variable=m_Acol
			#pragma HLS ARRAY_PARTITION variable=m_Acol dim=1 complete
			#pragma HLS ARRAY_PARTITION variable=m_Acol dim=2 complete
			
      #pragma HLS resource variable=m_Arow core=XPM_MEMORY uram
			#pragma HLS data_pack variable=m_Arow
			#pragma HLS ARRAY_PARTITION variable=m_Arow dim=1 complete
			#pragma HLS ARRAY_PARTITION variable=m_Arow dim=2 complete
      
			#pragma HLS resource variable=m_Adata core=XPM_MEMORY uram
			#pragma HLS data_pack variable=m_Adata
			#pragma HLS ARRAY_PARTITION variable=m_Adata dim=1 complete
			#pragma HLS ARRAY_PARTITION variable=m_Adata dim=2 complete

			#pragma HLS ARRAY_PARTITION variable=m_Prelus complete
			
			#pragma HLS DATAFLOW
			loadB (p_bAddr, l_bS[0], l_paramS[0], p_numRuns);
			for (unsigned int i=0; i<t_Stages; ++i) {
			#pragma HLS unroll
			  spmvCompute(l_bS[i], l_bS[i+1], l_paramS[i], l_paramS[i+1], i);
			}
			//spmvCompute(l_bS[0], l_bS[1], l_paramS[0], l_paramS[1], 0);
			storeC(l_bS[t_Stages],p_cAddr, l_paramS[t_Stages]);
		}	
		
		void
		runUspmv(
			DataDdrWideType *p_DdrRd,
			DataDdrWideType *p_DdrWr,
			UspmvArgsType &p_Args
		) {
			#pragma HLS inline off

			DataDdrWideType *l_cAddr = p_DdrWr + p_Args.m_Coffset * DataDdrWideType::per4k();
			DataDdrWideType *l_bAddr = p_DdrRd + p_Args.m_Boffset * DataDdrWideType::per4k();
			DataDdrWideType *l_aAddr = p_DdrRd + p_Args.m_Aoffset * DataDdrWideType::per4k();
		  unsigned int l_NumRuns = p_Args.m_NumRuns;
		
			loadA(l_aAddr);
			streamUspmv(l_bAddr,l_cAddr, l_NumRuns);
		}
};

} // namespace
#endif

