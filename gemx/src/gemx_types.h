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
 *  @brief GEMX common datatypes for HLS kernel code.
 *
 *  @author Jindrich Zejda (jzejda@xilinx.com)
 */

#ifndef GEMX_TYPES_H
#define GEMX_TYPES_H

#include <stdint.h>
#include <ostream>
#include <iomanip>

#include <ap_int.h>

// Helper macros for renaming kernel
#define GEMX_PASTER(x,y) x ## y
#define GEMX_EVALUATOR(x,y)  GEMX_PASTER(x,y)


namespace gemx {

// For C++11
//template<class T>
//auto operator<<(std::ostream& os, const T& t) -> decltype(t.print(os), os) 
//{ 
//  t.print(os); 
//  return os; 
//}

#define GEMX_FLOAT_WIDTH 7
#define GEMX_CMP_WIDTH 11

template <
    typename t_FloatType
  >
class TaggedFloat
{
  private:
    t_FloatType m_Val;
    bool m_Flush;
  public:
    TaggedFloat() {}
    TaggedFloat(t_FloatType p_Val, bool p_Flush)
      : m_Val(p_Val),
        m_Flush(p_Flush)
      {}
    TaggedFloat(t_FloatType p_Val)
      : m_Val(p_Val),
        m_Flush(false)
      {}
    t_FloatType &getVal() {return(m_Val);}
    bool &getFlush() {return(m_Flush);}
    TaggedFloat & operator=(t_FloatType p_Val) {m_Val=p_Val; m_Flush=false; return(*this);}
    t_FloatType & operator()() {return(m_Val);}
    void
    print(std::ostream& os) {
        os << std::setw(GEMX_FLOAT_WIDTH) << m_Val << "f" << m_Flush;
      }
};

template <typename T1>
std::ostream& operator<<(std::ostream& os, TaggedFloat<T1>& p_Val) {
  p_Val.print(os);
  return(os);
}


template <
    typename t_FloatType,
    unsigned int t_Bits
  >
class ControlFloat
{
  public:
    typedef ap_uint<t_Bits+3> ControlFloatBitsType;
  private:
    TaggedFloat<t_FloatType> m_Val;
    bool m_Exit;
    bool m_Pass;
  public:
    ControlFloat() {}
    ControlFloat(t_FloatType p_Val, bool p_Flush, bool p_Exit=false, bool p_Pass=false)
      : m_Val(p_Val, p_Flush),
        m_Exit(p_Exit),
        m_Pass(p_Pass)
      {}
    ControlFloat(ControlFloatBitsType p_ValBits) {
        ap_int<t_Bits> l_bitVal = p_ValBits(t_Bits-1, 0);
        t_FloatType l_val = l_bitVal;
        bool l_flush = p_ValBits[t_Bits-3],
             l_exit = p_ValBits[t_Bits-2],
             l_pass = p_ValBits[t_Bits-1];
        *this = ControlFloat(l_val, l_flush, l_exit, l_pass);
      }
    ControlFloatBitsType
    getBits() {
        ControlFloatBitsType l_val;
        l_val(t_Bits-1, 0) = ap_int<t_Bits>((*this)());
        l_val[t_Bits-3] = getFlush();
        l_val[t_Bits-2] = getExit();
        l_val[t_Bits-1] = getPass();
        return(l_val);
      }
    t_FloatType & operator()() {return(m_Val());}
    ControlFloat(t_FloatType p_Val) {ControlFloat(p_Val, false);}
    bool &getFlush() {return(m_Val.getFlush());}
    bool &getExit() {return(m_Exit);}
    bool &getPass() {return(m_Pass);}
    //TaggedFloat & operator=(t_FloatType p_Val) {m_Val=p_Val; m_Flush=false; return(*this);}
    void
    print(std::ostream& os) {
        m_Val.print(os);
        os << std::setw(GEMX_FLOAT_WIDTH) << "e" << m_Exit << "p" << m_Pass;
      }
};

template <typename T1, unsigned int T2>
std::ostream& operator<<(std::ostream& os, ControlFloat<T1, T2>& p_Val) {
  p_Val.print(os);
  return(os);
}


template <typename T, unsigned int t_Width>
class WideType {
  private:
    T  m_Val[t_Width];
    static const unsigned int t_4k = 4096; 
  public:
    typedef T DataType;
    static const unsigned int t_WidthS = t_Width; 
    static const unsigned int t_per4k = t_4k / sizeof(T) / t_Width; 
  public:
    T &getVal(unsigned int i) {return(m_Val[i]);}
    T &operator[](unsigned int p_Idx) {return(m_Val[p_Idx]);}
    T *getValAddr() {return(&m_Val[0]);}
    WideType() {}
    WideType(T p_initScalar) {
        for(int i = 0; i < t_Width; ++i) {
          getVal(i) = p_initScalar;
        }
      }
    T
    shift(T p_ValIn) {
        #pragma HLS inline self
        #pragma HLS data_pack variable=p_ValIn
        T l_valOut = m_Val[t_Width-1];
        WIDE_TYPE_SHIFT:for(int i = t_Width - 1; i > 0; --i) {
          T l_val = m_Val[i - 1];
          #pragma HLS data_pack variable=l_val
          m_Val[i] = l_val;
        }
        m_Val[0] = p_ValIn;
        return(l_valOut);
      }
    T
    shift() {
        #pragma HLS inline self
        T l_valOut = m_Val[t_Width-1];
        WIDE_TYPE_SHIFT:for(int i = t_Width - 1; i > 0; --i) {
          T l_val = m_Val[i - 1];
          #pragma HLS data_pack variable=l_val
          m_Val[i] = l_val;
        }
        return(l_valOut);
      }
    T
    unshift() {
        #pragma HLS inline self
        T l_valOut = m_Val[0];
        WIDE_TYPE_SHIFT:for(int i = 0; i < t_Width - 1; ++i) {
          T l_val = m_Val[i + 1];
          #pragma HLS data_pack variable=l_val
          m_Val[i] = l_val;
        }
        return(l_valOut);
      }
    static const WideType zero() {
        WideType l_zero;
        #pragma HLS data_pack variable=l_zero
        for(int i = 0; i < t_Width; ++i) {
          l_zero[i] = 0;
        }
        return(l_zero);
      }
    static
    unsigned int
    per4k() {
        return(t_per4k);
      }
    void
    print(std::ostream& os) {
        for(int i = 0; i < t_Width; ++i) {
          os << std::setw(GEMX_FLOAT_WIDTH) << getVal(i) << " ";
        }
      }
    
};

template <typename T1, unsigned int T2>
std::ostream& operator<<(std::ostream& os, WideType<T1, T2>& p_Val) {
  p_Val.print(os);
  return(os);
}


template <typename T, unsigned int t_Width>
class ExitTaggedWideType {
  private:
    WideType<T, t_Width> m_Val;
    bool m_Exit;
  public:
    ExitTaggedWideType(WideType<T, t_Width> p_Val, bool p_Exit)
      : m_Val(p_Val),
        m_Exit(p_Exit)
      {}
    ExitTaggedWideType() {}
    WideType<T, t_Width> &getVal() {return m_Val;}
    T &operator[](unsigned int p_Idx) {return(m_Val[p_Idx]);}

    bool getExit() {return(m_Exit);}
    void
    print(std::ostream& os) {
        m_Val.print(os);
        os << " e" << m_Exit;
      }
};

template <typename T1, unsigned int T2>
std::ostream& operator<<(std::ostream& os, ExitTaggedWideType<T1, T2>& p_Val) {
  p_Val.print(os);
  return(os);
}

template <typename T, unsigned int t_Width>
class TaggedWideType {
  private:
    WideType<T, t_Width> m_Val;
    bool m_Flush;
    bool m_Exit;
  public:
    TaggedWideType(WideType<T, t_Width> p_Val, bool p_Flush, bool p_Exit)
      : m_Val(p_Val),
        m_Flush(p_Flush),
        m_Exit(p_Exit)
      {}
    TaggedWideType() {}
    WideType<T, t_Width> &getVal() {return m_Val;}
    T &operator[](unsigned int p_Idx) {return(m_Val[p_Idx]);}

    bool getFlush() {return(m_Flush);}
    bool getExit() {return(m_Exit);}
    WideType<TaggedFloat<T>, t_Width>
    getVectOfTaggedValues() {
      #pragma HLS inline self
      WideType<TaggedFloat<T>, t_Width> l_vect;
      //#pragma HLS data_pack variable=l_vect
      TWT_FORW:for(unsigned int i = 0; i < t_Width; ++i) {
        l_vect.getVal(i) = TaggedFloat<T>(m_Val.getVal(i), m_Flush);
      }
      return(l_vect);
    }
    //void setVal(T p_Val, unsigned int i) {m_Val[i] = p_Val;}
    //void setFlush(bool p_Flush) {m_Flush = p_Flush;}
    //void setExit() {return(m_Exit);}
    void
    print(std::ostream& os) {
        m_Val.print(os);
        os << " f" << m_Flush << " e" << m_Exit;
      }
};

template <typename T1, unsigned int T2>
std::ostream& operator<<(std::ostream& os, TaggedWideType<T1, T2>& p_Val) {
  p_Val.print(os);
  return(os);
}

// Row-major window
template <typename T, unsigned int t_Rows, unsigned int t_Cols>
class WindowRm {
  private:
    WideType< WideType<T, t_Cols>, t_Rows>  m_Val;
  public:
    T &getval(unsigned int p_Row, unsigned int p_Col) { return m_Val.getVal(p_Row).getVal(p_Col);}
    WideType<T, t_Cols> &operator[](unsigned int p_Idx) {return(m_Val[p_Idx]);}
    void
    clear() {
        //#pragma HLS ARRAY_PARTITION variable=m_Val dim=1 complete
        //#pragma HLS ARRAY_PARTITION variable=m_Val dim=2 complete
        WINDOWRM_ROW:for (unsigned int row = 0; row < t_Rows; ++row) {
          #pragma HLS UNROLL
          WINDOWRM_COL:for (unsigned int col = 0; col < t_Cols; ++col) {
            #pragma HLS UNROLL
            getval(row, col) = -999;
          }
        }
      }
    // DOWN (0th row in, the last row out)
    WideType<T, t_Cols>
    shift(
        WideType<T, t_Cols> p_EdgeIn
      ) {
        #pragma HLS inline self
        return(m_Val.shift(p_EdgeIn));
      }
    // DOWN no input
    WideType<T, t_Cols>
    shift() {
        #pragma HLS inline self
        return(m_Val.shift());
      }
    // UP no input
    WideType<T, t_Cols>
    unshift() {
        #pragma HLS inline self
        return(m_Val.unshift());
      }
    // RIGHT
    WideType<T, t_Rows>
    shift_right(
        WideType<T, t_Rows> p_EdgeIn
      ) {
        #pragma HLS inline self
        //#pragma HLS ARRAY_PARTITION variable=m_Val dim=1 complete
        //#pragma HLS ARRAY_PARTITION variable=m_Val dim=2 complete
        #pragma HLS data_pack variable=p_EdgeIn
        WideType<T, t_Cols> l_edgeOut;
        #pragma HLS data_pack variable=l_edgeOut
        // Shift each row
        WINDOWRM_SHIFT_R1:for (unsigned int row = 0; row < t_Rows; ++row) {
          l_edgeOut[row] = m_Val[row].shift(p_EdgeIn[row]);
        }
        return(l_edgeOut);
      }
    void
    print(std::ostream& os) {
        for(int i = 0; i < t_Rows; ++i) {
          os << m_Val.getVal(i) << "\n";
        }
      }
};

template <typename T1, unsigned int T2, unsigned int T3>
std::ostream& operator<<(std::ostream& os, WindowRm<T1, T2, T3>& p_Val) {
  p_Val.print(os);
  return(os);
}


// Column-major window
template <typename T, unsigned int t_Rows, unsigned int t_Cols>
class WindowCm {
  private:
    WindowRm<T, t_Cols, t_Rows>  m_WinRm;
  public:
    T &getval(unsigned int p_Row, unsigned int p_Col) { return m_WinRm.getval(p_Col, p_Row);}
    WideType<T, t_Rows> &operator[](unsigned int p_Idx) {return(m_WinRm[p_Idx]);}
    void
    clear() {m_WinRm.clear(); }
    WideType<T, t_Rows>
    shift(WideType<T, t_Rows> p_EdgeIn) {return(m_WinRm.shift(p_EdgeIn));}
    WideType<T, t_Rows>
    shift() {return(m_WinRm.shift());}
    void
    print(std::ostream& os) {
        //m_WinRm.print(os);
        // Printing is still row by row, so custom code
        for(int row = 0; row < t_Rows; ++row) {
          for(int col = 0; col < t_Cols; ++col) {
            os << std::setw(GEMX_FLOAT_WIDTH) << getval(row, col);
          }
          os << "\n";
        }
      }
};

template <typename T1, unsigned int T2, unsigned int T3>
std::ostream& operator<<(std::ostream& os, WindowCm<T1, T2, T3>& p_Val) {
  p_Val.print(os);
  return(os);
}



template <typename T, unsigned int t_Width>
class TriangWindow {
  private:
    T m_Val[t_Width][t_Width];
    T &getval(unsigned int p_Row, unsigned int p_Col) { return m_Val[p_Row][p_Col];}
  public:
    void
    clear() {
        #pragma HLS ARRAY_PARTITION variable=m_Val dim=1 complete
        #pragma HLS ARRAY_PARTITION variable=m_Val dim=2 complete
        TRIANGWINDOW_CLEAR_ROW:for (unsigned int row = 0; row < t_Width; ++row) {
          #pragma HLS UNROLL
          TRIANGWINDOW_CLEAR_COL:for (unsigned int col = t_Width - row - 1; col < t_Width; ++col) {
            #pragma HLS UNROLL
            getval(row, col) = -999;
          }
        }
      }
    WideType<T, t_Width>
    shiftDiagInEdgeOut(
        WideType<T, t_Width> p_DiagIn
      ) {
        #pragma HLS ARRAY_PARTITION variable=m_Val dim=1 complete
        #pragma HLS ARRAY_PARTITION variable=m_Val dim=2 complete
        #pragma HLS data_pack variable=p_DiagIn
        WideType<T, t_Width> l_edgeOut;
        #pragma HLS data_pack variable=l_edgeOut
        for (unsigned int i = 0; i < t_Width; ++i) {
          getval(i, t_Width - i - 1) = p_DiagIn.getVal(i);
          l_edgeOut.getVal(i) = getval(i, t_Width-1);
        }
        for (unsigned int row = 0; row < t_Width; ++row) {
          for (unsigned int col = t_Width - 1; col > t_Width - row - 1; --col) {
            getval(row, col) = getval(row, col - 1);
          }
        }
        return(l_edgeOut);
      }
    void
    print(std::ostream& os) {
        for (unsigned int row = 0; row < t_Width; ++row) {
          for (unsigned int col = 0; col < t_Width; ++col) {
            if (col < t_Width - row - 1) {
              os << std::setw(GEMX_FLOAT_WIDTH + 2) << "- ";
            } else {
              os << std::setw(GEMX_FLOAT_WIDTH) << getval(row, col);
            }
          }
          os << "\n";
        }
      }
};

template <typename T1, unsigned int T2>
std::ostream& operator<<(std::ostream& os, TriangWindow<T1, T2>& p_Val) {
  p_Val.print(os);
  return(os);
}


/////////////////////////    SPMV types    /////////////////////////

//   Int ranges :  otherMsb  p_Mod  p_Discard 
//   p_Val         |-------| |----| |-------| 
template <typename T>
T
CalcMod(T p_Val, T p_Mod, T p_Discard = 1) {
  return (p_Val / p_Discard) % p_Mod;
}

template <typename t_FloatType, int t_NumIdxBits, int t_ColAddIdxBits, int t_NumBanks>
class SpmvAd {
  public:
  private:
    t_FloatType  m_ValA;
    #if GEMX_spmvPadA
      t_FloatType  m_Pad;
    #endif
    unsigned short m_Col;
    unsigned short m_Row;
    static const unsigned int t_4k = 4096; 
  public:
    static const unsigned int t_per4k = t_4k / (4 + 2 + 2); 
  public:
    SpmvAd() {}
    SpmvAd(t_FloatType p_A, unsigned int p_Row, unsigned int p_Col)
     : m_ValA(p_A), m_Row(p_Row), m_Col(p_Col)
     {
       assert(t_NumIdxBits == 8 * sizeof(m_Row));
       assert(t_NumIdxBits == 8 * sizeof(m_Col));
       assert(t_ColAddIdxBits < t_NumIdxBits);
     }
    unsigned int getCol() {return m_Col;}
    unsigned int getRow() {return m_Row;}
    unsigned int getColBank() {return CalcMod<unsigned int>(m_Col, t_NumBanks);}
    unsigned int getColOffset() {return m_Col / t_NumBanks;}
    t_FloatType getA() {return m_ValA;}
    void
    print(std::ostream& os) {
        os << std::setw(GEMX_FLOAT_WIDTH) << getRow() << " "
           << std::setw(GEMX_FLOAT_WIDTH) << getCol() << "    "
           << std::setw(GEMX_FLOAT_WIDTH) << getA();
      }
};
template <typename T1, int T2, int T3, int T4>
std::ostream& operator<<(std::ostream& os, SpmvAd<T1, T2, T3, T4>& p_Val) {
  p_Val.print(os);
  return(os);
}

template <typename t_FloatType, int t_NumIdxBits, int t_ColAddIdxBits, int t_NumBanks>
class SpmvA {
  private:
    static const int t_RowIdxBits = t_NumIdxBits - t_ColAddIdxBits;
    static const int t_ColIdxBits = t_NumIdxBits + t_ColAddIdxBits;
    static const int t_RowIdxMask = (1 << t_RowIdxBits) - 1;
    static const int t_ShortIdxMask = (1 << t_NumIdxBits) - 1;
  public:
    typedef ap_uint<t_RowIdxBits> SpmvRowIdxType;
    typedef ap_uint<t_ColIdxBits> SpmvColIdxType;
    typedef SpmvAd<t_FloatType, t_NumIdxBits, t_ColAddIdxBits, t_NumBanks> AdType;
    static const unsigned int t_maxRowIdx =  (1 << t_RowIdxBits);
  private:
    t_FloatType  m_ValA;
    #if GEMX_spmvPadA
      t_FloatType  m_Pad;
    #endif
    SpmvColIdxType m_Col;
    SpmvRowIdxType m_Row;
  public:
    SpmvA() {}
    SpmvA(AdType p_ValNdx)
     : m_ValA(p_ValNdx.getA()) {
        assert(t_RowIdxBits + t_ColIdxBits == 2 * t_NumIdxBits);
        unsigned int l_dRow = p_ValNdx.getRow();
        unsigned int l_dCol = p_ValNdx.getCol();
        m_Row = l_dRow & t_RowIdxMask;
        //assert(m_Row < 15360);
        m_Col = l_dCol | ((l_dRow & ~t_RowIdxMask) << t_ColAddIdxBits); 
      }
    AdType
    getAsAd() {
      unsigned int l_row = getRow();
      unsigned int l_col = getCol();
      AdType l_val(getA(),
                   l_row | ((l_col & ~t_ShortIdxMask) >> t_ColAddIdxBits),
                   l_col & t_ShortIdxMask);
      return(l_val);
    }
    
    SpmvA(t_FloatType p_A, unsigned int p_Row, unsigned int p_Col)
     : m_ValA(p_A), m_Row(p_Row), m_Col(p_Col)
     {
       assert(p_Row < (1 << t_RowIdxBits));
       //assert(p_Row < 15360);
       assert(p_Col < (1 << t_ColIdxBits));
       unsigned int l_col = m_Col;
       assert(l_col == p_Col);
       assert(getCol() == p_Col);
     }
    unsigned int getCol() {return m_Col;} // using proper SpmvRow/ColIdxType did not work
    unsigned int getRow() {return m_Row;}
    unsigned int getColBank() {return CalcMod<unsigned int>(m_Col, t_NumBanks);}
    unsigned int getColOffset() {return m_Col / t_NumBanks;}
    t_FloatType getA() {return m_ValA;}
    void
    print(std::ostream& os) {
        os << std::setw(GEMX_FLOAT_WIDTH) << int(getRow()) << " "
           << std::setw(GEMX_FLOAT_WIDTH) << int(getCol()) << "    "
           << std::setw(GEMX_FLOAT_WIDTH) << getA();
      }
};
template <typename T1, int T2, int T3, int T4>
std::ostream& operator<<(std::ostream& os, SpmvA<T1, T2, T3, T4>& p_Val) {
  p_Val.print(os);
  return(os);
}

template <typename t_FloatType, int t_NumIdxBits, int t_ColAddIdxBits, int t_NumBanks, int t_NumGroups>
class SpmvAB {
  private:
    static const int t_RowIdxBits = t_NumIdxBits - t_ColAddIdxBits;
    static const int t_ColIdxBits = t_NumIdxBits + t_ColAddIdxBits;
    static const int t_RowIdxMask = (1 << t_RowIdxBits) - 1;
  public:
    typedef ap_uint<t_RowIdxBits> SpmvRowIdxType;
  private:
    t_FloatType  m_ValA;
    t_FloatType  m_ValB;
    SpmvRowIdxType m_Row;
  public:
    SpmvAB() {}
    SpmvAB(t_FloatType p_A, t_FloatType p_B, unsigned int p_Row)
     : m_ValA(p_A), m_ValB(p_B), m_Row(p_Row)
      {
        //assert(m_Row < 15360);
      }
    unsigned int getRow() {return m_Row;}
    //   Row bits : msb offset group bank 
    //              |--------| |---| |--| 
    unsigned int getRowBank() {return CalcMod<unsigned int>(m_Row, t_NumBanks);}
    unsigned int getRowGroup() {return CalcMod<unsigned int>(m_Row, t_NumGroups, t_NumBanks);}
    unsigned int getRowOffset() {return getRow() / (t_NumGroups * t_NumBanks);}
    t_FloatType getA() {return m_ValA;}
    t_FloatType getB() {return m_ValB;}
    void setRowOffsetIntoRow(unsigned int p_RowOffset) {m_Row = p_RowOffset;}
    void
    print(std::ostream& os) {
        os << std::setw(GEMX_FLOAT_WIDTH) << getRow() << " "
           << std::setw(GEMX_FLOAT_WIDTH) << getB() << " "
           << std::setw(GEMX_FLOAT_WIDTH) << getA();
      }
};
template <typename T1, int T2, int T3, int T4, int T5>
std::ostream& operator<<(std::ostream& os, SpmvAB<T1, T2, T3, T4, T5>& p_Val) {
  p_Val.print(os);
  return(os);
}

template <typename t_FloatType, int t_NumIdxBits, int t_ColAddIdxBits,  int t_NumBanks, int t_NumGroups>
class SpmvC {
  private:
    static const int t_RowIdxBits = t_NumIdxBits - t_ColAddIdxBits;
    static const int t_ColIdxBits = t_NumIdxBits + t_ColAddIdxBits;
    static const int t_RowIdxMask = (1 << t_RowIdxBits) - 1;
  public:
    typedef ap_uint<t_RowIdxBits> SpmvRowIdxType;
  private:
    t_FloatType  m_ValC;
    SpmvRowIdxType m_Row;
  public:
    SpmvC() {}
    SpmvC(t_FloatType p_C, unsigned int p_Row)
     : m_ValC(p_C), m_Row(p_Row)
      {
        //assert(m_Row < 15360);
      }
    unsigned int getRow() {return m_Row;}
    void setRow(unsigned int p_Row) {m_Row = p_Row;}
    //   Row bits : msb offset bank 
    //              |--------| |--| 
    unsigned int getRowBank() {return CalcMod<unsigned int>(m_Row, t_NumBanks);}
    unsigned int getRowGroup() {return CalcMod<unsigned int>(m_Row, t_NumGroups, t_NumBanks);}
    unsigned int getRowOffset() {return m_Row / (t_NumGroups * t_NumBanks);}
    t_FloatType &getC() {return m_ValC;}
    unsigned int getRowOffsetStoredAsRow() {return m_Row;}
    void
    print(std::ostream& os) {
        os << std::setw(GEMX_FLOAT_WIDTH) << getRow() << " "
           << std::setw(GEMX_FLOAT_WIDTH) << getC();
      }
};
template <typename T1, int T2, int T3, int T4, int T5>
std::ostream& operator<<(std::ostream& os, SpmvC<T1, T2, T3, T4, T5>& p_Val) {
  p_Val.print(os);
  return(os);
}


/////////////////////////    Control and helper types    /////////////////////////


class ControlType {
  public:
    typedef enum {OpStream, OpExit} OpType;
  private:
    OpType m_OpCode;
  public:
    ControlType() {}
    ControlType(OpType p_OpCode)
      : m_OpCode(p_OpCode) {}
    OpType getOpCode() {return(m_OpCode);}
};


template<class T, uint8_t t_NumCycles>
T
hlsReg(T p_In)
{
  #pragma HLS INLINE self off
  #pragma HLS INTERFACE ap_none port=return register
  if (t_NumCycles == 1) {
    return p_In;
  } else {
    return hlsReg<T, uint8_t(t_NumCycles - 1)> (p_In);
  }
}

template<class T>
T
hlsReg(T p_In)
{
  return hlsReg<T, 1> (p_In);
}

template <typename T>
bool cmpVal(float p_TolRel, float p_TolAbs, T vRef, T v,
            std::string p_Prefix, bool &p_exactMatch,
            unsigned int p_Verbose) {
  float l_diffAbs = abs(v - vRef);
  float l_diffRel = l_diffAbs;
  if (vRef != 0) {
    l_diffRel /= abs(vRef);
  }
  p_exactMatch = (vRef == v);
  bool l_status = p_exactMatch || (l_diffRel <= p_TolRel) || (l_diffAbs <= p_TolAbs);
  if ((p_Verbose >= 3) ||
      ((p_Verbose >= 2) && !p_exactMatch) || 
      ((p_Verbose >= 1) && !l_status)) {
    std::cout << p_Prefix
              << "  ValRef " << std::left << std::setw(GEMX_CMP_WIDTH) << vRef
              << " Val "     << std::left << std::setw(GEMX_CMP_WIDTH) << v
              << "  DifRel " << std::left << std::setw(GEMX_CMP_WIDTH) << l_diffRel
              << " DifAbs "  << std::left << std::setw(GEMX_CMP_WIDTH) << l_diffAbs
              << "  Status " << l_status
              << "\n";
  }
  return(l_status);
}

template <unsigned int W>
class BoolArr {
  private:
    bool m_Val[W];
  public:
    BoolArr(bool p_Init) {
        for(unsigned int i = 0; i < W; ++i) {
          #pragma HLS UNROLL
          m_Val[i] = p_Init;
        }
      }
    bool & operator[](unsigned int p_Idx) {return m_Val[p_Idx];}
    bool And() {
        bool l_ret = true;
        for(unsigned int i = 0; i < W; ++i) {
          #pragma HLS UNROLL
          #pragma HLS ARRAY_PARTITION variable=m_Val COMPLETE
          l_ret = l_ret && m_Val[i];
        }
        return(l_ret);
      }
    bool Or() {
        bool l_ret = false;
        for(unsigned int i = 0; i < W; ++i) {
          #pragma HLS UNROLL
          #pragma HLS ARRAY_PARTITION variable=m_Val COMPLETE
          l_ret = l_ret || m_Val[i];
        }
        return(l_ret);
      }
    void Reset() {
        for(unsigned int i = 0; i < W; ++i) {
          #pragma HLS UNROLL
          #pragma HLS ARRAY_PARTITION variable=m_Val COMPLETE
          m_Val[i] = false;
        }
      }
};

template <class S, int W>
bool
streamsAreEmpty(S p_Sin[W]) {
  #pragma HLS inline self
  bool l_allEmpty = true;
  LOOP_S_IDX:for (int w = 0; w < W; ++w) {
    #pragma HLS UNROLL
    l_allEmpty = l_allEmpty && p_Sin[w].empty();
  }
  return(l_allEmpty);
}

// LCM
template<int V1, int V2>
struct findGCD {
  enum { result = findGCD<V2, V1%V2>::result };
};
template<int V1>
struct findGCD<V1,0> {
    enum { result = V1 };
};
template<int V1, int V2>
struct findLCM {
   enum { result = (V1/findGCD<V1,V2>::result) * V2 };
};


class SpmvAdesc {
  private:
    unsigned int m_Nnz;
    unsigned int m_Offset; // in pages
    static const unsigned int t_4k = 4096; 
  public:
    static const unsigned int t_per4k = t_4k / (sizeof(m_Nnz) + sizeof(m_Offset)); 
  public:
    SpmvAdesc() {}
    SpmvAdesc(unsigned int p_Nnz, unsigned int p_Offset)
      : m_Nnz(p_Nnz), m_Offset(p_Offset)
      {}
    unsigned int getNnz() {return m_Nnz;}
    unsigned int getOffset() {return m_Offset;}
    void
    print(std::ostream& os) {
        os << std::setw(GEMX_FLOAT_WIDTH) << getNnz() << " "
           << std::setw(GEMX_FLOAT_WIDTH) << getOffset();
      }
};
inline
std::ostream& operator<<(std::ostream& os, SpmvAdesc & p_Val) {
  p_Val.print(os);
  return(os);
}
    


//  Bit converter
template <typename T>
class BitConv {
  public:
    static const unsigned int t_SizeOf = sizeof(T);
    static const unsigned int t_NumBits = 8 * sizeof(T);
    typedef ap_uint<t_NumBits> BitsType;
  public:
    BitsType
    toBits(T p_Val) {
      return p_Val;
    }
    T
    toType(BitsType p_Val) {
      return p_Val;
    }
 };
 
template<>
inline
BitConv<float>::BitsType
BitConv<float>::toBits(float p_Val) {
  union {
    float f;
    unsigned int i;
  } u;
  u.f = p_Val;
  return(u.i);
}

template<>
inline
float
BitConv<float>::toType(BitConv<float>::BitsType p_Val) {
  union {
    float f;
    unsigned int i;
  } u;
  u.i = p_Val;
  return(u.f);
}

template<>
inline
BitConv<SpmvAdesc>::BitsType
BitConv<SpmvAdesc>::toBits(SpmvAdesc p_Val) {
  unsigned long int i1 = p_Val.getNnz();
  unsigned long int i2 = p_Val.getOffset();
  unsigned long int i = (i1 << 32) | i2;
  return(i);
}

template<>
inline
SpmvAdesc
BitConv<SpmvAdesc>::toType(BitConv<SpmvAdesc>::BitsType p_Val) {
  unsigned long int i2 = p_Val >> 32 ;
  unsigned long int i1 = p_Val & ((1ul << 32) - 1);
  SpmvAdesc l_ret(i1, i2);
  return(l_ret);
}


// Type converter - for vectors of different lengths and types
template <typename TS, typename TD>
class WideConv {
  private:
    static const unsigned int t_ws = TS::t_WidthS;
    static const unsigned int t_wd = TD::t_WidthS;
    typedef BitConv<typename TS::DataType> ConvSType;
    typedef BitConv<typename TD::DataType> ConvDType;
    static const unsigned int t_bs = ConvSType::t_NumBits;
    static const unsigned int t_bd = ConvDType::t_NumBits;
    static const unsigned int l_numBits = t_ws * t_bs;
  private:
    ap_uint<l_numBits> l_bits;
  public:
    inline
    TD
    convert(TS p_Src) {
      TD l_dst;
      ConvSType l_convS;
      assert(t_wd * t_bd == l_numBits);
      for(int ws = 0; ws < t_ws; ++ws) {
        l_bits.range(t_bs * (ws+1) - 1, t_bs * ws) = l_convS.toBits(p_Src[ws]);
      }
      ConvDType l_convD;
      for(int wd = 0; wd < t_wd; ++wd) {
        l_dst[wd] = l_convD.toType(l_bits.range(t_bd * (wd+1) - 1, t_bd * wd));
      }

      return(l_dst);
    }
};

// Memory allocation descriptor for FPGA, file, and program data exchange
// It never allocates nor freed memory
class MemDesc {
  private:
    size_t m_Num4kPages;
    void* m_PageSpace;
    static const unsigned int t_4k = 4096; 
  public:
	MemDesc() {}
    MemDesc(size_t p_Num4kPages, void* p_PageSpace) :
      m_Num4kPages(p_Num4kPages),
      m_PageSpace(p_PageSpace) {}
    size_t
    sizeBytes() {return m_Num4kPages * t_4k;}
    void*
    data() {return m_PageSpace;}
    size_t
    sizePages() {return m_Num4kPages;}
};




} // namespace
#endif
