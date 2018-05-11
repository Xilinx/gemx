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
 *  @brief FCN Kernet argrument handling
 *
 *  $DateTime: 2018/01/24 09:21:43 $
 */

#ifndef GEMX_FCN_KARGS_H
#define GEMX_FCN_KARGS_H

#include "assert.h"
#include "hls_stream.h"
#include "hls/utils/x_hls_utils.h"
#include "gemx_types.h"
#include "gemx_kargs.h"
#include <ap_fixed.h>
#include <stdio.h>
#include <vector>

namespace gemx {

////////////////////////////  ////////////////////////////
class FcnArgs {
  public:
    unsigned int m_Aoffset, m_Boffset, m_Coffset, m_Xoffset,
                 m_M, m_K, m_N,
                 m_Lda, m_Ldb, m_Ldc, m_Ldx;
		int32_t m_postScale;
		int16_t m_PReluVal;
  public:
    FcnArgs() {}
    FcnArgs(
        unsigned int p_Aoffset, unsigned int p_Boffset, unsigned int p_Coffset, unsigned int p_Xoffset,
        unsigned int p_M, unsigned int p_K, unsigned int p_N,
        unsigned int p_Lda, unsigned int p_Ldb, unsigned int p_Ldc, unsigned int p_Ldx,
				int32_t p_postScale, 
				int16_t p_PReluVal
      ) : m_Aoffset(p_Aoffset), m_Boffset(p_Boffset),  m_Coffset(p_Coffset), m_Xoffset(p_Xoffset),
          m_M(p_M), m_K(p_K), m_N(p_N),
          m_Lda(p_Lda),  m_Ldb(p_Ldb),  m_Ldc(p_Ldc), m_Ldx(p_Ldx),
					m_postScale(p_postScale),
					m_PReluVal(p_PReluVal)
      {}
	void
	init(
        unsigned int p_Aoffset, unsigned int p_Boffset, unsigned int p_Coffset, unsigned int p_Xoffset,
        unsigned int p_M, unsigned int p_K, unsigned int p_N,
        unsigned int p_Lda, unsigned int p_Ldb, unsigned int p_Ldc, unsigned int p_Ldx,
				int32_t p_postScale,
				int16_t p_PReluVal) {
      	m_Aoffset=p_Aoffset;
		 	  m_Boffset=p_Boffset;
    	  m_Coffset=p_Coffset;
				m_Xoffset=p_Xoffset;
      	m_M=p_M;
			  m_K=p_K;
	  		m_N=p_N;
	      m_Lda=p_Lda;
			  m_Ldb=p_Ldb;
			  m_Ldc=p_Ldc;
				m_Ldx=p_Ldx;
				m_postScale = p_postScale;
				m_PReluVal = p_PReluVal;
	}
};

 ////////////////////////////  KARGS module  ////////////////////////////

template <
   typename t_FloatType,        // Interface type for client (instead of ap_uint)
   typename t_FloatEqIntType,   // A type compatible with ap_uint<> of same size as t_FloatType
   unsigned int t_DdrWidthFloats,
   unsigned int t_InstrWidth,   // For narrow DDR, >1 allows to fit the instruction (in DdrWords)
   unsigned int t_DdrWidthBits,   // Must match the 8 * sizeof(t_BaseFloat) * t_DdrWidthFloats * t_InstrWidth
   unsigned int t_ArgPipeline
 >
class FcnKargs
{
  public:
    typedef WideType<t_FloatType, t_DdrWidthFloats> DdrFloatType;    
    class DdrInstrType {
      private:
        DdrFloatType v[t_InstrWidth];
      public:
        DdrFloatType &operator[](int i) {return(v[i]);}
        void
        loadFromDdr(
          DdrFloatType *p_Addr,
          unsigned int p_Offset
        ) {
          for(int i = 0; i < t_InstrWidth; ++i) {
            v[i] = p_Addr[p_Offset+i];
          }
        }
        void
        storeToDdr(
          DdrFloatType *p_Addr,
          unsigned int p_Offset
        ) {
          for(int i = 0; i < t_InstrWidth; ++i) {
            p_Addr[p_Offset+i] = reg(v[i]);
          }
        }
    };
    typedef ap_uint< t_DdrWidthBits >   DdrBitType;
    typedef enum {OpControl, OpGemv, OpGemm, OpTransp, OpSpmv, OpResult, OpFail, OpFcn} OpType;
        
  private:
    DdrBitType m_Flat;
    unsigned int m_BitPos;
    unsigned int m_WordPos;
  private:
    // Helper functions for serializing and deserializing
    template<typename T>
    void loadVal(T &var) {
      const int l_sizeBits = 8 * sizeof(var);
      unsigned int l_endBit = m_BitPos + l_sizeBits - 1;
      if (l_endBit >= t_DdrWidthBits) {
        m_WordPos++;
        m_BitPos = 0;
        l_endBit = m_BitPos + l_sizeBits - 1;
      }
      assert(m_WordPos <= 0);
      assert(l_endBit <= t_DdrWidthBits);
      ap_uint<l_sizeBits> l_bitVal = m_Flat.range(l_endBit, m_BitPos);
      //printf("  DEBUG loadVal bf=%s\n", l_bitVal.to_string(16).c_str());
      m_BitPos = l_endBit + 1;
      var = l_bitVal;
    }
    template<typename T>
    void storeVal(T &var) {
      const int l_sizeBits = 8 * sizeof(var);
      ap_uint<l_sizeBits> l_bitVal = var;
      //printf("  DEBUG storeVal bf=%s\n", l_bitVal.to_string(16).c_str());
      unsigned int l_endBit = m_BitPos + l_sizeBits - 1;
      if (l_endBit >= t_DdrWidthBits) {
        m_WordPos++;
        m_BitPos = 0;
        l_endBit = m_BitPos + l_sizeBits - 1;
      }
      assert(m_WordPos <= 0);
      assert(l_endBit <= t_DdrWidthBits);
      m_Flat.range(l_endBit, m_BitPos) = l_bitVal;
      m_BitPos = l_endBit + 1;
    }
    template<typename T>
    void storeValConst(const T var) {
      T l_val = var;
      storeVal(l_val);
    }
    void initPos() {m_WordPos = 0; m_BitPos = 0;}
    
    t_FloatType bits2float(t_FloatEqIntType p_Bits) {
        assert(sizeof(t_FloatType) == sizeof(t_FloatEqIntType));
        union {
            t_FloatType f;
            t_FloatEqIntType b;
          } l_val;
        l_val.b = p_Bits;
        //std::cout << "  DEBUG bits2float  " << p_Bits << "  ->  " << l_val.f << std::endl;
        return l_val.f;
      }
    t_FloatEqIntType float2bits(t_FloatType p_Val) {
        assert(sizeof(t_FloatType) == sizeof(t_FloatEqIntType));
        union {
            t_FloatType f;
            t_FloatEqIntType b;
          } l_val;
        l_val.f = p_Val;
        //std::cout << "  DEBUG float2bits " << p_Val << "  ->  " << l_val.b << std::endl;
        return l_val.b;
      }

    DdrBitType wideFloatToBits(DdrInstrType p_Val) {
        #pragma HLS ARRAY_PARTITION variable=p_Val dim=1
        #pragma HLS ARRAY_PARTITION variable=p_Val dim=2
        DdrBitType l_Val;
        const unsigned int l_FloatWidthInBits = 8 * sizeof(t_FloatType);
        LOOPinstrToBits:for(int i = 0; i < t_InstrWidth; ++i) {
          #pragma HLS unroll
          unsigned int l_offsetBits = i * t_DdrWidthFloats * l_FloatWidthInBits;
          LOOPwideFloatToBits:for(int w = 0; w < t_DdrWidthFloats; ++w) {
            #pragma HLS unroll
            l_Val.range(l_offsetBits + (w+1) * l_FloatWidthInBits - 1,
                        l_offsetBits + w * l_FloatWidthInBits)
               = float2bits(p_Val[i].getVal(w));
          }
        }
        return(l_Val);
      }
    DdrInstrType wideBitsToFloat(DdrBitType p_Val) {
        DdrInstrType l_Val;
        #pragma HLS ARRAY_PARTITION variable=l_Val dim=1
        #pragma HLS ARRAY_PARTITION variable=l_Val dim=2
        const unsigned int l_FloatWidthInBits = 8 * sizeof(t_FloatType);
        LOOPinstrToBits:for(int i = 0; i < t_InstrWidth; ++i) {
          #pragma HLS unroll
          unsigned int l_offsetBits = i * t_DdrWidthFloats * l_FloatWidthInBits;
          LOOPwideBitsToFloat:for(int w = 0; w < t_DdrWidthFloats; ++w) {
            #pragma HLS unroll
            l_Val[i].getVal(w) =  bits2float(p_Val.range(l_offsetBits + (w+1) * l_FloatWidthInBits - 1,
                                           l_offsetBits + w * l_FloatWidthInBits));
          }
        }
        return(l_Val);
      }
    
  public:
    FcnKargs()
      //  : m_Flat(0),
      //    m_IsLastOp(false)
        {m_Flat = 0;}
    OpType
    loadFromInstr(
          DdrInstrType p_Val
        ) {
        m_Flat = wideFloatToBits(p_Val);
        initPos();
        OpType l_op;
        loadVal(reinterpret_cast<unsigned int&>(l_op));
        return(l_op);
      }
    void
    storeToInstr(
          DdrInstrType &p_Val
        ) {
        p_Val = reg(wideBitsToFloat(m_Flat));
      }

    OpType
    load(
          DdrFloatType *p_Addr,
          unsigned int p_Pc
        ) {
        DdrInstrType l_val;
        l_val.loadFromDdr(p_Addr, p_Pc);
        return(loadFromInstr(l_val));
      }
    void
    store(
          DdrFloatType *p_Addr,
          unsigned int p_Pc
        ) {
        DdrInstrType l_val = wideBitsToFloat(m_Flat);
        l_val.storeToDdr(p_Addr, p_Pc);
      }

    InstrResArgs
    getInstrResArgs() {
      InstrResArgs l_args;
      assert(sizeof(l_args) <=  sizeof(m_Flat) - sizeof(OpType));
      loadVal(l_args.m_StartTime);
      loadVal(l_args.m_EndTime);
      assert(l_args.m_StartTime <= l_args.m_EndTime);
      InstrResArgs l_ret = hlsReg<InstrResArgs, t_ArgPipeline>(l_args);
      return l_ret;
    }
    void
    setInstrResArgs(InstrResArgs p_args) {
      assert(sizeof(p_args) <=  sizeof(m_Flat) - sizeof(OpType));
      initPos();
      storeValConst(int(OpResult));
      storeVal(p_args.m_StartTime);
      storeVal(p_args.m_EndTime);
    }

    ControlArgs
    getControlArgs() {
      ControlArgs l_args;
      assert(sizeof(l_args) <=  sizeof(m_Flat) - sizeof(OpType));
      loadVal(l_args.m_IsLastOp);
      loadVal(l_args.m_Noop);
      ControlArgs l_ret = hlsReg<ControlArgs, t_ArgPipeline>(l_args);
      return l_ret;
    }
    void
    setControlArgs(ControlArgs p_args) {
      assert(sizeof(p_args) <=  sizeof(m_Flat) - sizeof(OpType));
      initPos();
      storeValConst(int(OpControl));
      storeVal(p_args.m_IsLastOp);
      storeVal(p_args.m_Noop);
    }


    FcnArgs
    getFcnArgs() {
      FcnArgs l_args;
      assert(sizeof(l_args) <=  sizeof(m_Flat) - sizeof(OpType));
      loadVal(l_args.m_Aoffset);
      loadVal(l_args.m_Boffset);
      loadVal(l_args.m_Coffset);
			loadVal(l_args.m_Xoffset);
      loadVal(l_args.m_M);
      loadVal(l_args.m_K);
      loadVal(l_args.m_N);
      loadVal(l_args.m_Lda);
      loadVal(l_args.m_Ldb);
      loadVal(l_args.m_Ldc);
			loadVal(l_args.m_Ldx);
			loadVal(l_args.m_postScale);
			loadVal(l_args.m_PReluVal);
      FcnArgs l_ret = hlsReg<FcnArgs, t_ArgPipeline>(l_args);
      return l_ret;
    }
    void
    setFcnArgs(FcnArgs p_args) {
      assert(sizeof(p_args) <=  sizeof(m_Flat) - sizeof(OpType));
      initPos();
      storeValConst(int(OpFcn));
      storeVal(p_args.m_Aoffset);
      storeVal(p_args.m_Boffset);
      storeVal(p_args.m_Coffset);
			storeVal(p_args.m_Xoffset);
      storeVal(p_args.m_M);
      storeVal(p_args.m_K);
      storeVal(p_args.m_N);
      storeVal(p_args.m_Lda);
      storeVal(p_args.m_Ldb);
      storeVal(p_args.m_Ldc);
			storeVal(p_args.m_Ldx);
			storeVal(p_args.m_postScale);
			storeVal(p_args.m_PReluVal);
    }

    static unsigned int
    getInstrWidth() {return(t_InstrWidth);}
};

} // namespace

#endif
