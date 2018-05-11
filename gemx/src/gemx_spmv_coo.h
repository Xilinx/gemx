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
 *  @brief Sparse matrix vector multiply  C += A * B
 *  sparse matrix A is stored in Coordinate Formate (COO): 32-bit row, 32-bit col, nnz val
 *	vector B and C will be read from DDR and stored into URAM
 *
 *  $DateTime: 2017/11/29 09:20:31 $
 */

#ifndef GEMX_SPMV_COO_H
#define GEMX_SPMV_COO_H

#include "assert.h"
#include "hls_stream.h"
#include "gemx_spmv_coo_types.h"
#include "gemx_kargs.h"

namespace gemx {

//      

template <
    typename t_FloatType,
		typename t_IdxType,
    unsigned int t_DdrWidth,       // DDR width in t_FloatType
		unsigned int t_NnzWords,			 // number of t_DdrWidth elements for block-wise A loader
    unsigned int t_kVectorBlocks,  // controls max size of the B vector. max_num_B = t_kVectorBlocks * t_DdrWidth
    unsigned int t_mVectorBlocks,  // controls max size of the C vector. max_num_C = t_mVectorBlocks * t_DdrWidth * t_UramGroups
		unsigned int t_UramGroups = 6 
  >
class SpmvCoo
{
 	private: 
		static const unsigned int t_FloatSize = sizeof(t_FloatType);
		static const unsigned int t_IdxSize = sizeof(t_IdxType);
		static const unsigned int t_DdrNumBytes = t_DdrWidth * t_FloatSize;
		static const unsigned int t_UramWidth = 8 / t_FloatSize;
		static const unsigned int t_NumUramPerDdr = t_DdrWidth / t_UramWidth;
		static const unsigned int t_InterLeaves = t_UramGroups* t_UramWidth;
		static const unsigned int t_UramBWidth = t_UramWidth * t_NumUramPerDdr;
    
  public:
    typedef SpmvArgs SpmvArgsType;
		
		static const unsigned int t_NumIdxPerDdr = t_DdrNumBytes / t_IdxSize;
		static const unsigned int t_NumIdxPairPerDdr = t_NumIdxPerDdr / 2;
		static const unsigned int t_IdxWords = t_NnzWords * (t_DdrWidth / t_NumIdxPerDdr);
		static const unsigned int t_IdxReadPerData = t_DdrWidth / t_NumIdxPairPerDdr; //number of DDR IDx READs per DDR data read
		static const unsigned int t_NnzValIdxWords = t_NnzWords * (1+t_IdxReadPerData);

		typedef WideType<t_IdxType, t_NumIdxPerDdr> IdxWideType;
		typedef WideType<t_FloatType, t_DdrWidth> DdrWideType;
		typedef WideType<t_FloatType, t_UramWidth> UramWideType; 

		typedef SpmCoo<t_FloatType, t_IdxType, t_NumUramPerDdr, t_UramWidth> SpmCooType;
		
		typedef SpmCol<t_FloatType, t_IdxType, t_NumUramPerDdr, t_InterLeaves> SpmColType;
		typedef SpmAB<t_FloatType, t_IdxType, t_NumUramPerDdr, t_InterLeaves> SpmABType;
		typedef WideType<SpmCooType, t_DdrWidth> SpmCooWideType;
		typedef WideType<SpmCooType, t_NumUramPerDdr> SpmCooUramWideType;

		typedef SpmC<t_FloatType, t_IdxType, t_NumUramPerDdr, t_UramWidth> SpmCType;

    typedef hls::stream<DdrWideType> DdrWideStreamType;
		typedef hls::stream<IdxWideType> IdxWideStreamType;
		typedef hls::stream<SpmCooWideType> SpmCooWideStreamType;
		typedef hls::stream<SpmCooUramWideType> SpmCooUramWideStreamType;
		typedef hls::stream<SpmColType> SpmColStreamType;
		typedef hls::stream<SpmCooType> SpmCooStreamType;
		typedef hls::stream<SpmABType> SpmABStreamType;
		typedef hls::stream<SpmCType> SpmCStreamType;
		typedef hls::stream<bool>ControlStreamType;
		typedef hls::stream<t_FloatType> FloatStreamType;
		typedef hls::stream<t_IdxType> IdxStreamType;
		typedef hls::stream<uint8_t> ByteStreamType;
   	typedef hls::stream<UramWideType> WordBStreamType; 
  private:
		//| -------------- DdrWord -----------------|
		//|--URAM_0--| ... |--URAM_t_NumUramPerDdr--|
		//.																					.
		//.																					.t_kVectorBlocks for m_UramB 
		//.																					.
		//|--URAM_0--| ... |--URAM_t_NumUramPerDdr--|

		//|-DdrWord * t_UramGroups -----------------|
		//|--URAM_0--| ... |--URAM_t_NumUramPerDdr--|
		//.																					.
		//.																					. 
		//.																					.t_mVectorBlocks for m_UramC
		//|--URAM_0--| ... |--URAM_t_NumUramPerDdr--|
		UramWideType m_UramB[t_NumUramPerDdr][t_kVectorBlocks];
		UramWideType m_UramC[t_NumUramPerDdr][t_UramGroups][t_mVectorBlocks];

    static const unsigned int t_Debug = 0;

  private:

		void
		loadB(DdrWideType *p_bAddr, unsigned int p_kBlocks) {
			// Load entire B into URAM
			LOOP_BLOAD:for(unsigned int l_kBlock = 0; l_kBlock < p_kBlocks; ++l_kBlock) {
				#pragma HLS PIPELINE
				DdrWideType l_val = p_bAddr[l_kBlock];
				#pragma HLS ARRAY_PARTITION variable=l_val complete
				LOOP_D:for(int b = 0; b < t_NumUramPerDdr; ++b){
					UramWideType l_word; 
					#pragma HLS ARRAY_PARTITION variable=l_word complete
					LOOP_W:for(int w = 0; w < t_UramWidth; ++w) {
						l_word[w] = l_val[w*t_NumUramPerDdr + b];
					}	
					m_UramB[b][l_kBlock] = l_word;	
				} 
			}
		}
		
		void 
		loadC(DdrWideType *p_cAddr, unsigned int p_mBlocks) {
			//load entire C to URAM
			LOOP_CLOAD:for(unsigned int l_mBlock = 0; l_mBlock	< p_mBlocks; ++l_mBlock) {
				LOOP_CLOAD_GROUP:for(unsigned int l_group=0; l_group < t_UramGroups; ++l_group) {
				#pragma HLS PIPELINE				
				DdrWideType l_val = p_cAddr[l_mBlock*t_UramGroups+l_group];
				#pragma HLS ARRAY_PARTITION variable=l_val complete
				LOOP_D:for(int b = 0; b < t_NumUramPerDdr; ++b) {
				#pragma HLS UNROLL
					UramWideType l_word;
					#pragma HLS ARRAY_PARTITION variable=l_word complete
					LOOP_W:for(int w = 0; w < t_UramWidth; ++w) {
						l_word[w] = l_val[w*t_NumUramPerDdr + b];
					}
					m_UramC[b][l_group][l_mBlock] = l_word;
				}
			}
			}
		}

		void
		storeC(DdrWideType *p_cAddr, unsigned int p_mBlocks) {
			//read C from URAM and store it into DDR
			LOOP_CSTORE:for(unsigned int l_mBlock=0; l_mBlock < p_mBlocks; ++l_mBlock) {
				for(int g=0; g<t_UramGroups; ++g) {
					#pragma HLS PIPELINE
					DdrWideType l_val;
					#pragma HLS ARRAY_PARTITION variable=l_val complete
					LOOP_D:for(int b = 0; b < t_NumUramPerDdr; ++b) {
						UramWideType l_word;
						#pragma HLS ARRAY_PARTITION variable=l_word complete
						l_word = m_UramC[b][g][l_mBlock];
						LOOP_W:for(int w = 0; w < t_UramWidth; ++w) {
							l_val[w*t_NumUramPerDdr+ b] = l_word[w];
						}
					}
					p_cAddr[l_mBlock*t_UramGroups+g] = l_val;
				}
			}
		}

		void
		loadA(DdrWideType *p_aAddr, unsigned int p_nnzBlocks, DdrWideStreamType &p_outS) {
			//load indic and data of A and push them into the stream
			for (unsigned int l_nnzBlock = 0; l_nnzBlock < p_nnzBlocks; ++l_nnzBlock) {
				unsigned int l_offset = l_nnzBlock * t_NnzValIdxWords;
				for (int i = 0; i < t_NnzValIdxWords; ++i) {
					#pragma HLS PIPELINE
					DdrWideType l_val = p_aAddr[l_offset + i];
					p_outS.write(l_val);
				} 
			}
			
		}

		void
		mergeIdxData(DdrWideStreamType &p_inS, unsigned int p_nnzBlocks, 
								SpmCooWideStreamType &p_spmCooS) {
			
			assert ((t_DdrWidth % t_NumIdxPairPerDdr) == 0);
			WideConv<DdrWideType, IdxWideType> l_conv;

			unsigned int l_nnzWords = p_nnzBlocks * t_NnzWords;
		
			for (unsigned int l_nnzWord = 0; l_nnzWord < l_nnzWords; ++l_nnzWord) {
			#pragma HLS PIPELINE
				SpmCooWideType l_cooWide;
				#pragma HLS array_partition variable=l_cooWide complete
				//read data
				DdrWideType l_dataWide = p_inS.read();
				//read idx and form SpmCooWideType
				for (unsigned int i = 0; i < t_IdxReadPerData; ++i) {
					DdrWideType l_val = p_inS.read();
					IdxWideType l_idxWide = l_conv.convert(l_val);
					#pragma HLS array_partition variable=l_idxWide complete
					for (unsigned int j = 0; j < t_NumIdxPairPerDdr; ++j) {
						#pragma HLS UNROLL 
						l_cooWide[i*t_NumIdxPairPerDdr+j].getCol() = l_idxWide[j*2];
						l_cooWide[i*t_NumIdxPairPerDdr+j].getRow() = l_idxWide[j*2+1];
					} 
				}
				for (unsigned int i = 0; i < t_DdrWidth; ++i) {
				#pragma HLS UNROLL
					l_cooWide[i].getVal() = l_dataWide[i];
				}
				p_spmCooS.write(l_cooWide);

			}	
		}

		void
		processWideCol(SpmCooWideStreamType &p_inS, unsigned int p_nnzBlocks, 
									 SpmCooUramWideStreamType &p_spmCooS,
									 ControlStreamType &p_cntCooS){

			unsigned int l_nnzWords = p_nnzBlocks * t_NnzWords;

			for (unsigned int l_nnzWord = 0; l_nnzWord < l_nnzWords; ++l_nnzWord) {
			#pragma HLS PIPELINE REWIND
				SpmCooWideType l_val = p_inS.read();
				SpmCooUramWideType l_coo[t_UramWidth];
				#pragma HLS array_partition variable=l_coo complete dim=1 	
				for (int i = 0; i < t_UramWidth; ++i) {
					#pragma HLS UNROLL 
					for (int j = 0; j < t_NumUramPerDdr; ++j) {
						l_coo[i][j] = l_val[i*t_NumUramPerDdr +j];
					}
				}

				for (int i = 0; i < t_UramWidth; ++i) {
						p_spmCooS.write(l_coo[i]);
				} 
			}
			p_cntCooS.write(true);
		}

		////////////////////////////////////////	
		//distribute the data in SpmCooHalfWide type to different col bank channels
		///////////////////////////////////////
		void	
		xBarSplitCol(SpmCooUramWideStreamType &p_inS, ControlStreamType &p_inCntS, 
									SpmCooStreamType p_outS[t_NumUramPerDdr][t_NumUramPerDdr], ControlStreamType &p_outCntS) {
			bool l_exit = false;
			bool l_preDone = false;
			bool l_activity = true;
			while (!l_exit) {
				#pragma HLS PIPELINE
				if (l_preDone && !l_activity && p_inS.empty()) {
					l_exit = true;
				}
				bool l_unused;
				if (p_inCntS.read_nb(l_unused)) {
					l_preDone = true;
				}
				//l_activity = false;
				SpmCooUramWideType l_val;
				if (p_inS.read_nb(l_val)) {
					for (int w = 0; w < t_NumUramPerDdr; ++w) {
					#pragma HLS UNROLL 
						unsigned int l_bank = l_val[w].getColBank();
						p_outS[w][l_bank].write(l_val[w]);
					}
					l_activity = true;
				}
				else {
					l_activity = false;
				}
			}
			p_outCntS.write(true);
		} 

		void
		xBarMergeCol(SpmCooStreamType p_inS[t_NumUramPerDdr][t_NumUramPerDdr], ControlStreamType &p_inCntS, 
									SpmCooStreamType p_outS[t_NumUramPerDdr], ControlStreamType p_outCntS[t_NumUramPerDdr]) {
			bool l_exit = false;
			bool l_preDone = false;
			BoolArr<t_NumUramPerDdr> l_activity(true);
			
			uint8_t l_idIncr=0;

			while (!l_exit) {
				#pragma HLS PIPELINE

				if (l_preDone && !l_activity.Or()) {
					l_exit = true;
				}

				bool l_unused;
				if (p_inCntS.read_nb(l_unused)) {
					l_preDone = true;
				}

				for (int b = 0; b < t_NumUramPerDdr; ++b) {
					#pragma HLS UNROLL
					
					unsigned int l_idx = 0;
					for (int w = 0; w < t_NumUramPerDdr; ++w) {
						#pragma HLS UNROLL
						//unsigned int l_word = (b + w + l_idIncr) % t_NumUramPerDdr;
						//if (!p_inS[l_word][b].empty()) {
						if (!p_inS[w][b].empty()) {
							//l_idx = l_word;
							l_idx = w;
							break;
						}
					}

					SpmCooType l_val;
					if (p_inS[l_idx][b].read_nb(l_val)) {
						p_outS[b].write(l_val);
						l_activity[b] = true;
					}
					else {
						l_activity[b] = false;
					}
				}
				//l_idIncr = (l_idIncr+1) % t_NumUramPerDdr;			
			}

			for (int b = 0; b < t_NumUramPerDdr; ++b) {
				#pragma HLS UNROLL
				p_outCntS[b].write(true);
			}
		}
		
		void
		extractA(SpmCooStreamType &p_inS, ControlStreamType &p_inCntS, 
					IdxStreamType &p_outRowS, IdxStreamType &p_outColOffsetS, ByteStreamType &p_outColIndexS,
					FloatStreamType &p_outValS, ControlStreamType &p_outCntS) {
			
			bool l_exit=false;
			bool l_activity=true;
			bool l_preActivity=true;

			while (!l_exit) {
				#pragma HLS PIPELINE
				if (!l_preActivity && !l_activity) {
					l_exit = true;
				}
				SpmCooType l_val;
				if (p_inS.read_nb(l_val)) {
					p_outRowS.write(l_val.getRow());
					p_outColOffsetS.write((t_IdxType)l_val.getColOffset());
					p_outColIndexS.write((uint8_t)l_val.getColIndex());
					p_outValS.write(l_val.getVal());
					l_activity = true;
				}
				else {
					l_activity = false;
				}
				bool l_unused;
				if (p_inCntS.read_nb(l_unused)){
					l_preActivity = false;
				}
			} //end while
			p_outCntS.write(true);
		}

		void
		readUramB(IdxStreamType &p_inS, ControlStreamType &p_inCntS, 
					WordBStreamType &p_outS, ControlStreamType &p_outCntS, unsigned int t_BankId) {
			
			bool l_exit=false;
			bool l_activity=true;
			bool l_preActivity=true;

			while (!l_exit) {
				#pragma HLS PIPELINE
				if (!l_preActivity && !l_activity) {
					l_exit = true;
				}
				t_IdxType l_val;
				if (p_inS.read_nb(l_val)) {
					UramWideType l_wordB = m_UramB[t_BankId][l_val];
					p_outS.write(l_wordB);
					l_activity = true;
				}
				else {
					l_activity = false;
				}
				bool l_unused;
				if (p_inCntS.read_nb(l_unused)){
					l_preActivity = false;
				}
			} //end while
			p_outCntS.write(true);
		}

		void
		readB(WordBStreamType &p_inWordS, ByteStreamType &p_inIdxS, ControlStreamType &p_inCntS,
					FloatStreamType &p_outS, ControlStreamType &p_outCntS) {
			
			bool l_exit=false;
			bool l_activity=true;
			bool l_preActivity=true;

			while (!l_exit) {
				#pragma HLS PIPELINE
				if (!l_preActivity && !l_activity) {
					l_exit = true;
				}
				uint8_t l_idx;
				UramWideType l_wordB;
				#pragma HLS ARRAY_PARTITION variable=l_wordB complete
				if (p_inIdxS.read_nb(l_idx)) {
					p_inWordS.read(l_wordB);
					p_outS.write(l_wordB[l_idx]);
					l_activity = true;
				}
				else {
					l_activity = false;
				}
				bool l_unused;
				if (p_inCntS.read_nb(l_unused)){
					l_preActivity = false;
				}
			} //end while
			p_outCntS.write(true);
		} 

		void
		formAB(IdxStreamType &p_inRowS, FloatStreamType &p_inValBs,FloatStreamType &p_inValAs,
					ControlStreamType &p_inCntS, SpmABStreamType &p_outS, ControlStreamType &p_outCntS) {
			
			bool l_exit=false;
			bool l_activity=true;
			bool l_preActivity=true;

			while (!l_exit) {
				//#pragma HLS PIPELINE
				if (!l_preActivity && !l_activity) {
					l_exit = true;
				}
				t_IdxType l_row;
				t_FloatType l_valA, l_valB;
				if (p_inRowS.read_nb(l_row)) {
					p_inValAs.read(l_valA);
					p_inValBs.read(l_valB);
					SpmABType l_valOut(l_valA, l_valB, l_row);
					p_outS.write(l_valOut);
					l_activity = true;
				}
				else {
					l_activity = false;
				}
				bool l_unused;
				if (p_inCntS.read_nb(l_unused)){
					l_preActivity = false;
				}
			} //end while
			p_outCntS.write(true);
		}

		void
		xBarSplitRow(SpmABStreamType p_inS[t_NumUramPerDdr], ControlStreamType p_inCntS[t_NumUramPerDdr],
									SpmABStreamType p_outS[t_NumUramPerDdr][t_NumUramPerDdr], ControlStreamType &p_outCntS) {
			bool l_exit = false;
			BoolArr<t_NumUramPerDdr> l_activity(true);
			BoolArr<t_NumUramPerDdr> l_preActivity(true);

			while (!l_exit) {
				#pragma HLS PIPELINE

				if (!l_preActivity.Or() && !l_activity.Or()){
					l_exit = true;
				}
				//l_activity.Reset();
	
				for (int w = 0; w < t_NumUramPerDdr; ++w) {
					#pragma HLS UNROLL
					SpmABType l_val;
					if (p_inS[w].read_nb(l_val)) {
						unsigned int l_rowBank = l_val.getRowBank();
						p_outS[w][l_rowBank].write(l_val);
						l_activity[w] = true;
					}
					else {
						l_activity[w] = false;
					}

					bool l_unused;
					if (p_inCntS[w].read_nb(l_unused)){
						l_preActivity[w] = false;
					}
				}
			}

			p_outCntS.write(true);
		}

		void
		xBarMergeRow(SpmABStreamType p_inS[t_NumUramPerDdr][t_NumUramPerDdr], ControlStreamType &p_inCntS,
									SpmABStreamType p_outS[t_NumUramPerDdr], ControlStreamType p_outCntS[t_NumUramPerDdr]) {
			
			bool l_exit = false;
			bool l_preDone = false;
			BoolArr<t_NumUramPerDdr> l_activity(true);
			
			uint8_t l_idIncr=0;
			while (!l_exit) {
				#pragma HLS PIPELINE

				if (l_preDone && !l_activity.Or()) {
					l_exit = true;
				}
				
				bool l_unused;
				if (p_inCntS.read_nb(l_unused)){
					l_preDone = true;
				}
				//l_activity.Reset();

				for (int b = 0; b < t_NumUramPerDdr; ++b) {
					#pragma HLS UNROLL
					unsigned int l_idx = 0;
					for (int bb = 0; bb < t_NumUramPerDdr; ++bb) {
						//unsigned int l_word = (b+bb+l_idIncr) % t_NumUramPerDdr;	
						//if (!p_inS[l_word][b].empty()){
						if (!p_inS[bb][b].empty()){
							//l_idx = l_word;
							l_idx = bb;
							break;
						}
					}

					SpmABType l_val;
					if (p_inS[l_idx][b].read_nb(l_val)){
						p_outS[b].write(l_val);
						l_activity[b] = true;
					}
					else {
						l_activity[b] = false;
					}
				}
				//l_idIncr = (l_idIncr + 1) % t_NumUramPerDdr;
			}

			for (int b = 0; b < t_NumUramPerDdr; ++b) {
				p_outCntS[b].write(true);
			}
 
		}

		void
		multAB(SpmABStreamType p_inS[t_NumUramPerDdr], ControlStreamType p_inCntS[t_NumUramPerDdr],
						SpmColStreamType p_outS[t_NumUramPerDdr], ControlStreamType p_outCntS[t_NumUramPerDdr]) {

			bool l_exit = false;
			BoolArr<t_NumUramPerDdr> l_preActivity(true);
			BoolArr<t_NumUramPerDdr> l_activity(true);

			while (!l_exit) {
				#pragma HLS PIPELINE
				if ((!(l_preActivity.Or())) && (!(l_activity.Or()))) {
					l_exit = true;
				}
				//l_activity.Reset();

				LOOP_MACROW_CALC:for (int b = 0; b < t_NumUramPerDdr; ++b) {
					#pragma HLS UNROLL
					SpmABType l_val;
					if (p_inS[b].read_nb(l_val)) {
						t_FloatType l_valA = l_val.getA();
						t_FloatType l_valB = l_val.getB();
						SpmColType l_valCol;
						l_valCol.getVal() = l_valA * l_valB;
						l_valCol.getRow() = l_val.getRow();
						p_outS[b].write(l_valCol);
						l_activity[b] = true;
					}
					else {
						l_activity[b] = false;
					}

					bool l_unused;
					if (p_inCntS[b].read_nb(l_unused)) {
						l_preActivity[b] = false;
					}
				}
			}
			
			for (int b = 0; b < t_NumUramPerDdr; ++b) {
			#pragma HLS UNROLL 
				p_outCntS[b].write(true);
			}
		}

		void
		interLeaveUnit(SpmColStreamType &p_inS, ControlStreamType &p_inCntS,
						SpmColStreamType p_outS[t_InterLeaves], ControlStreamType &p_outCntS) {

			bool l_exit = false;
			bool l_preDone = false;
			bool l_activity = true;

			while (!l_exit) {
				#pragma HLS PIPELINE
				if (l_preDone && !l_activity && p_inS.empty()) {
					l_exit = true;
				}
				//l_activity = false;

				SpmColType l_val;
				if (p_inS.read_nb(l_val)) {
					t_IdxType l_row = l_val.getRow();
					unsigned int l_rowGroup = l_val.getRowIndex();
					p_outS[l_rowGroup].write(l_val);
					//p_outS[l_id % t_InterLeaves].write(l_val);
					l_activity = true;
				}
				else {
					l_activity = false;
				}

					bool l_unused;
					if (p_inCntS.read_nb(l_unused)) {
						l_preDone = true;
					}
			}
			p_outCntS.write(true);
		}
	void
	aggCUnit(SpmColStreamType p_inS[t_InterLeaves], ControlStreamType &p_inCntS,
					SpmColStreamType p_outS[t_InterLeaves], ControlStreamType &p_outCntS) {
		bool l_exit = false;
		bool l_preDone = false;
		BoolArr<t_InterLeaves> l_activity(false);

		SpmColType l_aggC[t_InterLeaves];
		#pragma HLS array_partition variable=l_aggC complete dim=1

		for (int i=0; i<t_InterLeaves; ++i) {
		#pragma HLS UNROLL 
			l_aggC[i].getRow()=0;
			l_aggC[i].getVal()=0;
		}

		while (!l_exit) {
		#pragma HLS PIPELINE
			if (l_preDone && !l_activity.Or()) {
				l_exit=true;
			}
			
			for (int i=0; i<t_InterLeaves; ++i) {
			#pragma HLS UNROLL
				SpmColType l_val;
				if (p_inS[i].read_nb(l_val)) {
					if (l_aggC[i].getRow() != l_val.getRow()) {
						p_outS[i].write(l_aggC[i]);
						l_aggC[i].getVal() = l_val.getVal();
						l_aggC[i].getRow() = l_val.getRow();
					}
					else {
						 l_aggC[i].getVal() += l_val.getVal(); 
					}
					l_activity[i] = true;
				}
				else {
					l_activity[i] = false;
				}		
			}
			bool l_unused;
			if (p_inCntS.read_nb(l_unused)) {
				l_preDone=true;
			}
		}

		for (int i=0; i<t_InterLeaves; ++i) {
		#pragma HLS UNROLL
			if (l_aggC[i].getVal()!=0) {
				p_outS[i].write(l_aggC[i]);
			}
		}
		p_outCntS.write(true);
	}

	void
	formCUnit(SpmColStreamType p_inS[t_InterLeaves], ControlStreamType &p_inCntS, 
						SpmCStreamType p_outS[t_UramGroups], ControlStreamType &p_outCntS) {
		bool l_exit = false;
		bool l_preDone = false;
		BoolArr<t_InterLeaves> l_activity(true);
		
		SpmCType l_valC[t_UramGroups];
		#pragma HLS array_partition variable=l_valC complete dim=1
		BoolArr<t_UramWidth> l_valCfilled[t_UramGroups];

		for (int i=0; i<t_UramGroups; ++i) {
		#pragma HLS UNROLL
			l_valC[i].init(0,0);
			l_valCfilled[i].Reset();
		}

		while (!l_exit){
		//#pragma HLS PIPELINE II=t_UramWidth
		#pragma HLS PIPELINE 
			if (l_preDone && !l_activity.Or()) {
				l_exit = true;
			}
			//l_activity.Reset();

			for (uint8_t i=0; i<t_UramGroups; ++i){
			#pragma HLS UNROLL
				for (uint8_t j=0; j<t_UramWidth; ++j){
					SpmColType l_val;
					if (p_inS[i*t_UramWidth+j].read_nb(l_val)){
						unsigned int l_rowOffset = l_val.getRowOffset();
						bool isSameOffset = (l_rowOffset == l_valC[i].getRowOffset());
						if ((l_valCfilled[i].Or() && !isSameOffset) || l_valCfilled[i][j]) {
							p_outS[i].write(l_valC[i]);
							//reset the other index value
							for (int k=0; k<t_UramWidth; ++k) {
							#pragma HLS UNROLL
								if (k != j) {
									l_valC[i][k] = 0;
								}
							}
						}
						
						if (!l_valCfilled[i].Or() || !isSameOffset) {
							l_valC[i].getRowOffset() = l_rowOffset;
						}
						l_valC[i][j] = l_val.getVal();
						l_valCfilled[i][j] = true;
						l_activity[i*t_UramWidth+j] = true;
					}
					else {
						l_activity[i*t_UramWidth+j] = false;
					} 
				}	
			}
			
			bool l_unused;
			if (p_inCntS.read_nb(l_unused)) {
				l_preDone = true;
			}
		}
		for (int i=0; i<t_UramGroups; ++i) {
		#pragma HLS UNROLL
			if (l_valCfilled[i].Or()) {
				p_outS[i].write(l_valC[i]);
			}
		}
		p_outCntS.write(true);
	}

	void
	addCUnit(SpmCStreamType p_inS[t_UramGroups], ControlStreamType &p_inCntS, unsigned int t_BankId) {
	
		bool l_exit = false;
		bool l_preDone = false;
		BoolArr<t_UramGroups> l_activity(true);

		while (!l_exit) {
		//#pragma HLS PIPELINE II=t_InterLeaves
		#pragma HLS PIPELINE
	
			if (l_preDone && !l_activity.Or()) {
				l_exit = true;
			}
			//l_activity.Reset();

			LOOP_AGGC:for (int g=0; g < t_UramGroups; ++g) {
			#pragma HLS UNROLL
				SpmCType l_val;
				if (p_inS[g].read_nb(l_val)) {
					unsigned int l_rowOffset = l_val.getRowOffset();
					UramWideType l_wordC = m_UramC[t_BankId][g][l_rowOffset];
					#pragma HLS ARRAY_PARTITION variable=l_wordC complete dim=0
					for (int i=0; i<t_UramWidth; ++i) {
					#pragma HLS UNROLL
						l_wordC[i] += l_val[i];
					}
					m_UramC[t_BankId][g][l_rowOffset] = l_wordC;
					l_activity[g] = true;
				}
				else {
					l_activity[g] = false;
				}

				bool l_unused;
				if (p_inCntS.read_nb(l_unused)) {
					l_preDone = true;
				}	
			}
			}
		}


	void
	multA(DdrWideType *p_aAddr, unsigned int p_nnzBlocks) {

		static const unsigned int t_DepthDeep = 16;
		static const unsigned int t_DepthShallow = 4;

		DdrWideStreamType l_aS;
		#pragma HLS DATA_PACK variable=l_aS
		#pragma HLS STREAM variable=l_aS depth=t_DepthShallow
		SpmCooWideStreamType l_spmWideCooS;
		#pragma HLS DATA_PACK variable=l_spmWideCooS
		#pragma HLS STREAM variable=l_spmWideCooS depth=t_DepthShallow
		SpmCooUramWideStreamType l_spmUramCooS;
		#pragma HLS DATA_PACK variable=l_spmUramCooS
		#pragma HLS STREAM variable=l_spmUramCooS depth=t_DepthShallow
		ControlStreamType l_cntColS;
		#pragma HLS DATA_PACK variable=l_cntColS
		#pragma HLS STREAM variable=l_cntColS depth=t_DepthShallow
		ControlStreamType l_cnt2SplitColS;
		#pragma HLS DATA_PACK variable=l_cnt2SplitColS
		#pragma HLS STREAM variable=l_cnt2SplitColS depth=t_DepthShallow
		SpmCooStreamType l_spm2MergeColS[t_NumUramPerDdr][t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spm2MergeColS
		#pragma HLS STREAM variable=l_spm2MergeColS depth=t_DepthDeep
		#pragma HLS ARRAY_PARTITION variable=l_spm2MergeColS COMPLETE dim=1
		#pragma HLS ARRAY_PARTITION variable=l_spm2MergeColS COMPLETE dim=2
		ControlStreamType l_cnt2MergeColS;
		#pragma HLS DATA_PACK variable=l_cnt2MergeColS
		#pragma HLS STREAM variable=l_cnt2MergeColS depth=t_DepthShallow
		SpmCooStreamType l_spm2extractAs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spm2extractAs
		#pragma HLS STREAM variable=l_spm2extractAs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_spm2extractAs COMPLETE dim=1
		ControlStreamType l_cnt2extractAs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2extractAs
		#pragma HLS STREAM variable=l_cnt2extractAs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2extractAs COMPLETE dim=1
		IdxStreamType l_spmArowS[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spmArowS
		#pragma HLS STREAM variable=l_spmArowS depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_spmArowS COMPLETE dim=1
		IdxStreamType l_spmAcolOffsetS[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spmAcolOffsetS
		#pragma HLS STREAM variable=l_spmAcolOffsetS depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_spmAcolOffsetS COMPLETE dim=1
		ByteStreamType l_spmAcolIndexS[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spmAcolIndexS
		#pragma HLS STREAM variable=l_spmAcolIndexS depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_spmAcolIndexS COMPLETE dim=1
		FloatStreamType l_spmAvalS[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spmAvalS
		#pragma HLS STREAM variable=l_spmAvalS depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_spmAvalS COMPLETE dim=1
		ControlStreamType l_cnt2readUramBs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2readUramBs
		#pragma HLS STREAM variable=l_cnt2readUramBs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2readUramBs COMPLETE dim=1
		WordBStreamType l_wordBs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_wordBs
		#pragma HLS STREAM variable=l_wordBs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_wordBs COMPLETE dim=1
		ControlStreamType l_cnt2readBs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2readBs
		#pragma HLS STREAM variable=l_cnt2readBs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2readBs COMPLETE dim=1
		FloatStreamType l_valBs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_valBs
		#pragma HLS STREAM variable=l_valBs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_valBs COMPLETE dim=1
		ControlStreamType l_cnt2formABs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2formABs
		#pragma HLS STREAM variable=l_cnt2formABs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2formABs COMPLETE dim=1
		SpmABStreamType l_spm2SplitRowS[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spm2SplitRowS
		#pragma HLS STREAM variable=l_spm2SplitRowS depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_spm2SplitRowS COMPLETE dim=1
		ControlStreamType l_cnt2SplitRowS[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2SplitRowS
		#pragma HLS STREAM variable=l_cnt2SplitRowS depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2SplitRowS COMPLETE dim=1
		SpmABStreamType l_spm2MergeRowS[t_NumUramPerDdr][t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spm2MergeRowS
		#pragma HLS STREAM variable=l_spm2MergeRowS depth=t_DepthDeep
		#pragma HLS ARRAY_PARTITION variable=l_spm2MergeRowS COMPLETE dim=1
		#pragma HLS ARRAY_PARTITION variable=l_spm2MergeRowS COMPLETE dim=2
		ControlStreamType l_cnt2MergeRowS;
		#pragma HLS DATA_PACK variable=l_cnt2MergeRowS
		#pragma HLS STREAM variable=l_cnt2MergeRowS depth=t_DepthShallow
		SpmABStreamType l_spm2multABs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spm2multABs
		#pragma HLS STREAM variable=l_spm2multABs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_spm2multABs COMPLETE dim=1
		ControlStreamType l_cnt2multABs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2multABs
		#pragma HLS STREAM variable=l_cnt2multABs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2multABs COMPLETE dim=1
		SpmColStreamType l_spm2interLeaveS[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_spm2interLeaveS
		#pragma HLS STREAM variable=l_spm2interLeaveS depth=t_DepthDeep
		#pragma HLS ARRAY_PARTITION variable=l_spm2interLeaveS COMPLETE dim=1
		ControlStreamType l_cnt2interLeaveS[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2interLeaveS
		#pragma HLS STREAM variable=l_cnt2interLeaveS depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2interLeaveS COMPLETE dim=1
		SpmColStreamType l_spm2aggCs[t_NumUramPerDdr][t_InterLeaves];
		#pragma HLS DATA_PACK variable=l_spm2aggCs
		#pragma HLS STREAM variable=l_spm2aggCs depth=t_DepthDeep
		#pragma HLS ARRAY_PARTITION variable=l_spm2aggCs COMPLETE dim=1
		#pragma HLS ARRAY_PARTITION variable=l_spm2aggCs COMPLETE dim=2
		ControlStreamType l_cnt2aggCs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2aggCs
		#pragma HLS STREAM variable=l_cnt2aggCs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2aggCs COMPLETE dim=1
		SpmColStreamType l_spm2formCs[t_NumUramPerDdr][t_InterLeaves];
		#pragma HLS DATA_PACK variable=l_spm2formCs
		#pragma HLS STREAM variable=l_spm2formCs depth=t_DepthDeep
		#pragma HLS ARRAY_PARTITION variable=l_spm2formCs COMPLETE dim=1
		#pragma HLS ARRAY_PARTITION variable=l_spm2formCs COMPLETE dim=2
		ControlStreamType l_cnt2formCs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2formCs
		#pragma HLS STREAM variable=l_cnt2formCs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2formCs COMPLETE dim=1
		SpmCStreamType l_spm2addCs[t_NumUramPerDdr][t_UramGroups];
		#pragma HLS DATA_PACK variable=l_spm2addCs
		#pragma HLS STREAM variable=l_spm2addCs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_spm2addCs COMPLETE dim=1
		#pragma HLS ARRAY_PARTITION variable=l_spm2addCs COMPLETE dim=2
		ControlStreamType l_cnt2addCs[t_NumUramPerDdr];
		#pragma HLS DATA_PACK variable=l_cnt2addCs
		#pragma HLS STREAM variable=l_cnt2addCs depth=t_DepthShallow
		#pragma HLS ARRAY_PARTITION variable=l_cnt2addCs COMPLETE dim=1
					
		#pragma HLS DATAFLOW
		loadA(p_aAddr, p_nnzBlocks, l_aS);
		mergeIdxData(l_aS, p_nnzBlocks, l_spmWideCooS);
		processWideCol(l_spmWideCooS, p_nnzBlocks, l_spmUramCooS, l_cnt2SplitColS);
		xBarSplitCol(l_spmUramCooS, l_cnt2SplitColS, l_spm2MergeColS, l_cnt2MergeColS);
		xBarMergeCol(l_spm2MergeColS, l_cnt2MergeColS, l_spm2extractAs, l_cnt2extractAs);
		LOOP_PAIRB:for(int b=0; b < t_NumUramPerDdr; ++b) {
		#pragma HLS UNROLL
			extractA(l_spm2extractAs[b], l_cnt2extractAs[b], l_spmArowS[b], l_spmAcolOffsetS[b], l_spmAcolIndexS[b], l_spmAvalS[b], l_cnt2readUramBs[b]);
			readUramB(l_spmAcolOffsetS[b], l_cnt2readUramBs[b], l_wordBs[b], l_cnt2readBs[b], b);
			readB(l_wordBs[b], l_spmAcolIndexS[b], l_cnt2readBs[b], l_valBs[b], l_cnt2formABs[b]);
			formAB(l_spmArowS[b], l_valBs[b], l_spmAvalS[b], l_cnt2formABs[b], l_spm2SplitRowS[b], l_cnt2SplitRowS[b]);
		}
		xBarSplitRow(l_spm2SplitRowS, l_cnt2SplitRowS, l_spm2MergeRowS, l_cnt2MergeRowS);
		xBarMergeRow(l_spm2MergeRowS, l_cnt2MergeRowS, l_spm2multABs, l_cnt2multABs);
		multAB(l_spm2multABs, l_cnt2multABs, l_spm2interLeaveS, l_cnt2interLeaveS);
		LOOP_W_RU:for(int b=0; b < t_NumUramPerDdr; ++b) {
		#pragma HLS UNROLL
			interLeaveUnit(l_spm2interLeaveS[b], l_cnt2interLeaveS[b], l_spm2aggCs[b], l_cnt2aggCs[b]);
			aggCUnit(l_spm2aggCs[b], l_cnt2aggCs[b], l_spm2formCs[b], l_cnt2formCs[b]);
			formCUnit(l_spm2formCs[b], l_cnt2formCs[b], l_spm2addCs[b], l_cnt2addCs[b]);
			addCUnit(l_spm2addCs[b], l_cnt2addCs[b], b);
		}
	}

	public:
		void runSpmv(
			DdrWideType *p_DdrRd,
			DdrWideType *p_DdrWr,
			SpmvArgsType &p_Args
		){
			#pragma HLS inline off
			
			#pragma HLS DATA_PACK variable=m_UramB
			#pragma HLS RESOURCE variable=m_UramB core=XPM_MEMORY uram
			#pragma HLS ARRAY_PARTITION variable=m_UramB complete dim=1
			#pragma HLS DATA_PACK variable=m_UramC 
			#pragma HLS RESOURCE variable=m_UramC core=XPM_MEMORY uram
			#pragma HLS ARRAY_PARTITION variable=m_UramC complete dim=1 
			#pragma HLS ARRAY_PARTITION variable=m_UramC complete dim=2 
			//load B into URAM
			const unsigned int l_kBlocks = p_Args.m_K / t_DdrWidth;
			assert(l_kBlocks * t_DdrWidth == p_Args.m_K);
			assert(l_kBlocks < t_kVectorBlocks);
			DdrWideType *l_bAddr = p_DdrRd + p_Args.m_Boffset * DdrWideType::per4k();
			loadB(l_bAddr, l_kBlocks);

			//load C into URAM
			assert((t_InterLeaves % t_UramWidth) == 0);
			const unsigned int l_mBlocks = p_Args.m_M / (t_DdrWidth * t_UramGroups);
			assert(l_mBlocks * t_DdrWidth * t_UramGroups == p_Args.m_M);
			assert(l_mBlocks < t_mVectorBlocks);
			DdrWideType *l_cAddr = p_DdrWr + p_Args.m_Coffset * DdrWideType::per4k();
			loadC(l_cAddr, l_mBlocks);

			//multA
			const unsigned int l_nnzBlocks = p_Args.m_Nnz / (t_DdrWidth * t_NnzWords);
			assert(l_nnzBlocks * t_DdrWidth * t_NnzWords == p_Args.m_Nnz);

			DdrWideType *l_aAddr = p_DdrRd + p_Args.m_Aoffset * DdrWideType::per4k();
			multA(l_aAddr, l_nnzBlocks);

			//store C
			storeC(l_cAddr, l_mBlocks);
		}
};

} // namespace
#endif

