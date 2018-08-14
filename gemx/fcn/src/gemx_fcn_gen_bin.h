/**********
 * Copyright (c) 2018, Xilinx, Inc.
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
 *  @brief FCN instruction encoder testcase generator
 *
 *  $DateTime: 2018/01/25 10:26:36 $
 */

#ifndef GEMX_FCN_GEN_BIN_H
#define GEMX_FCN_GEN_BIN_H

#include <stdio.h>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdlib.h>
#include <chrono>
#include <bitset>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
//#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/algorithm/string.hpp>

#include "gemx_fcn_kernel.h"

////////////////////////  COMMON  ////////////////////////

// Common types
typedef GEMX_dataType FloatType;
typedef gemx::DdrMatrixShape DdrMatrixShapeType;

// VLIV processing types
typedef gemx::FcnKargs<
    GEMX_dataType, GEMX_dataEqIntType,
    GEMX_ddrWidth, GEMX_argInstrWidth,
    GEMX_argInstrWidth * GEMX_ddrWidth * sizeof(GEMX_dataType) * 8,
    GEMX_argPipeline
  > KargsType;
typedef KargsType::OpType KargsOpType;
typedef KargsType::DdrFloatType DdrFloatType;
typedef gemx::ControlArgs ControlArgsType;
typedef FcnType::FcnArgsType FcnArgsType;
typedef DdrMatrixShapeType::FormatType MatFormatType;
#define GEMX_maxNumInstr 64
#define VERBOSE 1 
typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePointType;
inline void
showTimeData(std::string p_Task, TimePointType &t1, TimePointType &t2, double *p_TimeMsOut = 0)
{
  t2 = std::chrono::high_resolution_clock::now();    
  std::chrono::duration<double> l_durationSec = t2 - t1;
  double l_timeMs = l_durationSec.count() * 1e3;
  if (p_TimeMsOut) {
    *p_TimeMsOut = l_timeMs;
  }
  (VERBOSE > 0) && std::cout << p_Task
            << "  " << std::fixed << std::setprecision(6)
            << l_timeMs << " msec\n";
}

template<
  typename T,
  unsigned int t_PageSize
>
class Page {
  public:
    typedef std::array<T, t_PageSize> PageType;
  private:
    PageType m_Page;
  public:
    FloatType& operator[](unsigned int p_Idx) {return m_Page[p_Idx];}
    Page() {
      //m_Page.fill(0);
      for(int i = 0 ; i < t_PageSize; ++i) {
        m_Page[i] = 0;
      }
    }
};

class PageHandleDescriptor {
  public:
    unsigned int m_StartPage;
    unsigned int m_SizePages;
    //std::string m_HandleName;
  public:
    PageHandleDescriptor() : m_StartPage(0), m_SizePages(0) {}
    PageHandleDescriptor(unsigned int p_StartPage, unsigned int p_SizePages)
      : m_StartPage(p_StartPage), m_SizePages(p_SizePages)
      {}
    PageHandleDescriptor(const PageHandleDescriptor & p_Val)
      : m_StartPage(p_Val.m_StartPage), m_SizePages(p_Val.m_SizePages)
      {}
    bool operator<(const PageHandleDescriptor &p_Other) const {
        return(m_StartPage < p_Other.m_StartPage);
      }
};

typedef std::array<uint8_t, GEMX_instructionSizeBytes> InstrControlType;
typedef std::vector<Page<uint8_t, GEMX_pageSizeBytes > > PageVectorType;
typedef Page<uint8_t, GEMX_pageSizeBytes> PageType;

template <
    typename t_FloatType  // to simplify client-side interfaces
  >
class Program {
  public:
    typedef std::array<InstrControlType, GEMX_maxNumInstr> InstrType;
  private:
    PageVectorType m_PageVector;
    unsigned int m_NumInstr;
    std::map<std::string, PageHandleDescriptor> m_Handles;
  private:
    // Utilities
    std::ifstream::pos_type getFileSize(std::string p_FileName)
    {
      std::ifstream in(p_FileName.c_str(), std::ifstream::ate | std::ifstream::binary);
      return in.tellg(); 
    }
  public:
    Program()
      : m_NumInstr(0)
      {
        // Reserve instruction and result pages
        m_PageVector.resize(GEMX_dataPage);
      }
    Program(size_t p_NumPages)  // Constructor typically used to store output from FPGA
      : m_NumInstr(0)
      {
        // Reserve space for entire output FPGA image
        m_PageVector.resize(p_NumPages);
      }
		void
		init (size_t p_NumPages) {
			m_NumInstr=0;
			m_PageVector.resize(p_NumPages);
		}
    unsigned int
    allocPages(std::string p_Handle, bool &p_NewAlloc, size_t p_NumElements) { // unit: t_FloatType
        
        assert(p_NumElements > 0);
        size_t l_numPages = (p_NumElements * sizeof(t_FloatType)
                             + GEMX_pageSizeBytes - 1)  /  GEMX_pageSizeBytes;
        unsigned int l_startPage = 0;
        PageHandleDescriptor l_desc = m_Handles[p_Handle];
        if (l_desc.m_StartPage == 0) {
          l_startPage = m_PageVector.size();
          m_PageVector.resize(l_startPage + l_numPages);
          m_Handles[p_Handle] = PageHandleDescriptor(l_startPage, l_numPages);
          p_NewAlloc = true;
        } else {
          assert(l_desc.m_SizePages == l_numPages);
          l_startPage = l_desc.m_StartPage;
          p_NewAlloc = false;
        }
        //std::cout << "  DEBUG allocPages Start page for " << p_Handle << " is " << l_startPage << "\n";
        return(l_startPage);
      }
    t_FloatType *
    getPageAddr(unsigned int p_PageIdx) {
        t_FloatType* l_addr = (t_FloatType*)&m_PageVector[p_PageIdx];
        return(l_addr);
      }
    DdrFloatType *
    getBaseInstrAddr() {return (DdrFloatType*)&m_PageVector[GEMX_codePage];}
    DdrFloatType *
    getBaseResAddr() {return (DdrFloatType*)&m_PageVector[GEMX_resPage];}
    DdrFloatType *
    addInstr() {
        assert(m_NumInstr * sizeof(InstrControlType) <= GEMX_pageSizeBytes);
        InstrControlType *l_instrBased = (InstrControlType *)&m_PageVector[GEMX_codePage];
        DdrFloatType* l_instrAdd = (DdrFloatType*)&l_instrBased[m_NumInstr];
        ++m_NumInstr;
        return(l_instrAdd);
      }
    bool
    writeToBinFile(std::string p_FileName)
    {
      bool ok = false;  
      std::ofstream l_of(p_FileName.c_str(), std::ios::binary);
      if (l_of.is_open()) {
        size_t l_sizeBytes =  sizeof(m_PageVector[0]) * m_PageVector.size();
        l_of.write((char*)&m_PageVector[0], l_sizeBytes);
        if (l_of.tellp() == l_sizeBytes) {
          std::cout << "INFO: wrote " << l_sizeBytes << " bytes to " << p_FileName << "\n";
          ok = true;
        } else {
          std::cout << "ERROR: wrote only " << l_of.tellp() << " bytes to " << p_FileName << "\n";
        }
        l_of.close();
      }
      return(ok);
    }
    bool
    readFromBinFile(std::string p_FileName)
    {
      bool ok = false;  
      // Bin file existence
      std::ifstream l_if(p_FileName.c_str(), std::ios::binary);
      if (l_if.is_open()) {
        // Bin file size
        size_t l_FileSize = getFileSize(p_FileName);
        std::cout << "INFO: loading " + p_FileName + " of size " << l_FileSize << "\n";
        assert(l_FileSize > 0);
        size_t l_FileSizeInPages = l_FileSize / sizeof(m_PageVector[0]);
        assert(l_FileSize % sizeof(m_PageVector[0]) == 0);

        // Bin file storage
        m_PageVector.reserve(l_FileSizeInPages);
        m_PageVector.resize(l_FileSizeInPages);

        // Read the bin file
        l_if.read((char*) &m_PageVector[0], l_FileSize);
        if (l_if) {
          std::cout << "INFO: loaded " << l_FileSize << " bytes from " << p_FileName << "\n";
          ok = true;
        } else {
          m_PageVector.clear();
          std::cout << "ERROR: loaded only " << l_if.gcount() << " bytes from " << p_FileName << "\n";
        }
        l_if.close();

      } else {
        std::cout << "ERROR: failed to open file " + p_FileName + "\n";
      }
      return(ok);
    }
    gemx::MemDesc
    getMemDesc(void) {
      assert(m_PageVector.size() > 0);
      gemx::MemDesc l_MemDesc(m_PageVector.size(), m_PageVector.data());
      return(l_MemDesc);
    }

};

typedef Program<GEMX_dataType> ProgramType;

// Matrix descriptor with data itself stored in caller's space
template < typename T>
class Mat
{
  private:
    unsigned int m_Rows, m_Cols, m_Ld; 
    T *m_Addr;
  public:
	Mat()
	{}
    Mat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld, T *p_Addr)
      : m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_Addr(p_Addr)
      {}
    Mat& operator=(const Mat& p_Src) {
        assert(p_Src.rows() == rows());
        assert(p_Src.cols() == cols());
        for (unsigned int row = 0; row < m_Rows; ++row) {
          for (unsigned int col = 0; col < m_Cols; ++col) {
            m_Addr[row][col] = p_Src.getVal(row, col);
          }
        }
        return *this;
      }
    inline T &getVal(unsigned int p_Row, unsigned int p_Col) {return m_Addr[p_Row * ld() + p_Col];}
    inline unsigned int rows() {return m_Rows;}
    inline unsigned int cols() {return m_Cols;}
    inline unsigned int ld() {return m_Ld;}
	void
	init(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld, T *p_Addr) {
		m_Rows = p_Rows;
		m_Cols = p_Cols;
		m_Ld = p_Ld;
		m_Addr = p_Addr;
    }
    void
    fillMod(T p_Max, T p_First = 0) {
        T l_val = p_First;
        for (unsigned int row = 0; row < m_Rows; ++row) {
          for (unsigned int col = 0; col < ld(); ++col) {
            getVal(row, col) = l_val;
            l_val++;
            l_val %= p_Max;
          }
        }
      }
		void
		fillFromFile(std::istream& p_Is) {
			T l_val;
			for (unsigned int row=0; row < m_Rows; ++row) {
				for (unsigned int col=0; col < ld(); ++col) {
					p_Is >> l_val;
					getVal(row,col) = l_val;
				}
			}
		}
    void
    multiply(Mat & p_A, Mat & p_B) {
        T l_val = 0;
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        for (unsigned int row = 0; row < rows(); ++row) {
          for (unsigned int col = 0; col < cols(); ++col) {
            T l_val = 0;
            for (unsigned int k = 0; k < p_A.cols(); ++k) {
              l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
            }
            //std::cout << "    DEBUG multiply setting row=" << row << " col=" << col << std::endl;
            getVal(row, col) = l_val;
          }
        }
      }
		void matMultWithScaleAndPRelu(Mat & p_A, Mat & p_B, Mat<GEMX_XdataType> & p_X,  int32_t p_postScale, int16_t p_PReluVal) {
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
				assert(p_X.rows() == rows());
				assert(p_X.cols() == cols());
        for (unsigned int row = 0; row < rows(); ++row) {
          for (unsigned int col = 0; col < cols(); ++col) {
            int64_t l_val = 0;
            for (unsigned int k = 0; k < p_A.cols(); ++k) {
              l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
//							if ((row==6) && (col == 0)) {
//									if (p_B.getVal(k, col) != 0) {
//										std::cout << " A[6," << std::dec << k << "]= " << p_A.getVal(row, k) << std::hex << " 0x" << p_A.getVal(row, k);
//										std::cout << " B[" << std::dec << k << ",0]= " << p_B.getVal(k,col) <<  std::hex << " 0x" << p_B.getVal(k,col);
//										std::cout << " A*B+C = " << std::dec << l_val << std::hex << " 0x" << l_val << "\n";
//									}
//							}
            }

//						if ((row == 6) && (col == 0)) {
//							std::bitset<64> l_bVal{l_val};
//							std::cout << "C[6,0]= " << l_bVal << "\n";
//						}
						l_val += p_X.getVal(row,col);
						unsigned int l_psShift = p_postScale & 0x00ff;
						int64_t l_psVal = p_postScale >> 8;
						l_val = l_val * l_psVal;
						l_val = (l_val >> l_psShift);
						T l_entry = (T)(l_val);
						if (l_entry < 0) {
							l_entry = l_entry * (p_PReluVal >> 6) >> (p_PReluVal & 0x003f);
						}
						getVal(row, col) = l_entry;
          }
        }
		}
    
		void
    print(std::ostream& os) {
        os << m_Rows << "x" << m_Cols << " Ld=" << m_Ld << "\n";
        unsigned int l_cols = 
          cols(); // normal matrix
          //ld();; // parent matrix (within Ld
        for (unsigned int row = 0; row < rows(); ++row) {
          for (unsigned int col = 0; col < l_cols; ++col) {
            os << std::setw(GEMX_FLOAT_WIDTH) << int(getVal(row, col)) << " ";
          }
          os << "\n";
        }
      }
    bool
    cmp(float p_TolRel, float p_TolAbs, Mat &p_Ref) {
        bool ok = true;
        unsigned int l_verbose = 1;  // 0 none, 1 if not exactly equal, 2 if passed tolerance, 3 show all
        unsigned int l_numExactMatches = 0, l_numMismatches = 0;
        for (unsigned int row = 0; row < rows(); ++row) {
          for (unsigned int col = 0; col < cols(); ++col) {
            std::string l_Prefix = "      row " + std::to_string(row) + " col " + std::to_string(col); 
            T v = getVal(row, col);
            T vRef = p_Ref.getVal(row, col);
            bool l_exactMatch = false;
            bool l_ok = gemx::cmpVal<T>(p_TolRel, p_TolAbs, vRef, v, l_Prefix, l_exactMatch, 1);
            ok = ok && l_ok;
            if (l_exactMatch) {
              l_numExactMatches++;
            }
            if (!l_ok) {
              l_numMismatches++;
            }
          }
        }
        unsigned int l_total = rows() * cols();
        unsigned int l_withinTolerance = l_total - l_numExactMatches - l_numMismatches;
        std::cout << "  Compared " << l_total << " values:"
                  << "  exact match " << l_numExactMatches
                  << "  within tolerance " << l_withinTolerance
                  << "  mismatch " << l_numMismatches << "\n";
        return(ok);
      }

};
template <typename T1>
std::ostream& operator<<(std::ostream& os, Mat<T1>& p_Val) {
  p_Val.print(os);
  return(os);
}


// Float specialization
typedef Mat<float> MatType_ForFloat;

template<>
void
MatType_ForFloat::fillMod(float p_Max, float p_First) {
  const bool l_small = false;
  if (l_small) {
    // for easier debug of matrices
    float l_val = p_First;
    for (unsigned int row = 0; row < m_Rows; ++row) {
      for (unsigned int col = 0; col < ld(); ++col) {
        getVal(row, col) = l_val;
        l_val += 0.3;
        if (l_val > p_Max) {
          l_val -= p_Max;
        }
      }
    }
  } else {
    // for better float robustness of large matrices
    float l_val = 1.0;
    float l_step = 0.3;
    float l_drift = 0.00001;
    float l_sign = 1;
    for (unsigned int row = 0; row < m_Rows; ++row) {
      for (unsigned int col = 0; col < ld(); ++col) {
        getVal(row, col) = l_val;
        l_val += l_sign * l_step;
        l_step += l_drift;
        l_sign = -l_sign;
        if (l_val > p_Max) {
          l_val -= p_Max;
        }
      }
    }
  }
}
template<>
void
MatType_ForFloat::print(std::ostream& os) {
    os << m_Rows << "x" << m_Cols << " Ld=" << m_Ld << "\n";
    unsigned int l_cols = 
      cols(); // normal matrix
      //ld();; // parent matrix (within Ld
    for (unsigned int row = 0; row < rows(); ++row) {
      for (unsigned int col = 0; col < l_cols; ++col) {
        os << std::setw(GEMX_FLOAT_WIDTH) << std::fixed << std::setprecision(3) << getVal(row, col) << " ";
      }
      os << "\n";
    }
  }

////  Typedefs
typedef Mat<GEMX_dataType> MatType;
typedef Mat<GEMX_XdataType> XMatType;


////////////////////////  CONTROL  ////////////////////////
class GenControl
{
  public:
    void
    addInstr(
      ProgramType &p_Program,
      bool p_IsLastOp,
      bool p_Noop
    ) {
    
    // Instruction
    ControlArgsType l_controlArgs(
        p_IsLastOp, p_Noop
      );
    KargsType l_kargs;
    l_kargs.setControlArgs(l_controlArgs);
    l_kargs.store(p_Program.addInstr(), 0);

    std::cout << "Added CONTROL  IsLastOp=" << p_IsLastOp << " Noop=" << p_Noop << "  ";
  }
  
  void
  show(
      ProgramType &p_Program,
      ControlArgsType p_ControlArgs
    ) {
      bool l_isLastOp = p_ControlArgs.m_IsLastOp,
           l_Noop = p_ControlArgs.m_Noop;
      std::cout << "\n###########  Op Control  ###########\n"
        << "  IsLastOp=" << l_isLastOp
        << " Noop=" << l_Noop
        << "\n";
    }
      
};

  
////////////////////////  FCN  ////////////////////////
class GenFcn
{
  public:
    bool
    checkDim(std::string p_VarName, unsigned int p_Val, unsigned int p_Mod, unsigned int p_Min) {
      bool l_ok = true;
      if (p_Val % p_Mod != 0) {
        std::cerr << "ERROR: " << p_VarName << " " << p_Val << " must be multiple of " << p_Mod << "\n";
        l_ok = false;
      }
      if (p_Val < p_Min) {
        std::cerr << "ERROR: " << p_VarName << " " << p_Val << " must be at least " << p_Min << "\n";
        l_ok = false;
      }
      return(l_ok);
    }

    bool
    check(
      unsigned int p_M,
      unsigned int p_K,
      unsigned int p_N,
      unsigned int p_LdA,
      unsigned int p_LdB,
      unsigned int p_LdC,
			unsigned int p_LdX
    ) {
        bool ok = true;
        
        const unsigned int l_Edge = GEMX_ddrWidth;
        const unsigned int l_kMin = 2 * GEMX_ddrWidth; // due to kernel Cout control to save area
        
        ok = checkDim("M", p_M, l_Edge, 1) &&
             checkDim("K", p_K, l_Edge, 2 * l_Edge) &&
             checkDim("N", p_N, l_Edge, 1) &&
             checkDim("LdA", p_LdA, l_Edge, p_K) &&
             checkDim("LdB", p_LdB, l_Edge, p_N) &&
             checkDim("LdC", p_LdC, l_Edge, p_N) &&
						 checkDim("LdX", p_LdX, l_Edge, p_N);
        return(ok);
      }
      void addInstrFromFile(
	      ProgramType &p_Program,
	      std::string p_FileName,
	      bool p_WithGolden
      ) {
	      unsigned int l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX;
	      unsigned int l_pageA, l_pageB, l_pageC, l_pageX;
	      int32_t l_postScale, l_PRelu;
	      std::string l_handleA, l_handleB, l_handleC, l_handleX;
	      bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocX;
	      
	      std::cout <<"INFO: loading input matrices file " << p_FileName << "\n";
	      std::ifstream l_fs(p_FileName.c_str(), std::ios_base::in | std::ios_base::binary);
		      
	      boost::iostreams::filtering_istream l_bs;
	      l_bs.push(l_fs);
	      
	      bool l_good = l_bs.good();

	      bool ok=true;
	      const unsigned int l_Edge = GEMX_ddrWidth;
	      const unsigned int l_kMin = 2 * GEMX_ddrWidth;

	      if (l_good) {
		      while (l_bs.peek()=='#') l_bs.ignore(2048, '\n');
		      l_bs >> l_postScale >> l_PRelu;
		      //read A dimensions
		      l_bs >> l_handleA >> l_M >> l_K >> l_LdA;
		      std::cout << "INFO " << l_handleA << " " << l_M << " " << l_K << " " << l_LdA << "\n";
		      //check A dimention
		      ok = checkDim("M", l_M, l_Edge, 1) &&
					checkDim("K", l_K, l_Edge, 2*l_Edge) &&
					checkDim("LdA", l_LdA, l_Edge, l_K);
		      if (!ok) {
			      return;
		      }
		      //allocate host memory and initialize it with data from the file
		      l_pageA = p_Program.allocPages(l_handleA, l_newAllocA, l_M * l_LdA);
		      MatType l_matA(l_M, l_K, l_LdA, p_Program.getPageAddr(l_pageA));
		      if (l_newAllocA) {
			      l_matA.fillFromFile(l_bs);	
		      }
	      
		      //read matrix B dimensions
		      unsigned int l_bK;
		      l_bs >> l_handleB >> l_bK >> l_N >> l_LdB;
		      std::cout << "INFO " << l_handleB << " " << l_bK << " " << l_N << " " << l_LdB << "\n";
		      assert(l_bK == l_K);
		      ok = checkDim("N", l_N, l_Edge, 1) &&
					checkDim("LdB", l_LdB, l_Edge, l_N);
		      if (!ok) {
			      boost::iostreams::close(l_bs);
			      return;
		      }

		      l_pageB = p_Program.allocPages(l_handleB, l_newAllocB, l_K * l_LdB);
		      MatType l_matB(l_K, l_N, l_LdB, p_Program.getPageAddr(l_pageB));
		      if (l_newAllocB) {
			      l_matB.fillFromFile(l_bs);	
		      }
		      
		      //read matrix X dimensions
		      unsigned int l_xM, l_xN;
		      l_bs >> l_handleX >> l_xM >> l_xN >> l_LdX;
		      std::cout << "INFO " << l_handleX << " " << l_xM << " " << l_xN << " " << l_LdX << "\n";
		      assert(l_xM == l_M);
		      assert(l_xN == l_N);
		      ok = checkDim("LdX", l_LdX, l_Edge, l_N);
		      if (!ok) {
			      boost::iostreams::close(l_bs);
			      return;
		      }

		      l_pageX = p_Program.allocPages(l_handleX, l_newAllocX, l_M*l_N*(sizeof(GEMX_XdataType)/sizeof(GEMX_dataType)));
		      XMatType l_matX(l_M, l_N, l_LdX, (GEMX_XdataType *) p_Program.getPageAddr(l_pageX));
		      if (l_newAllocX) {
			      l_matX.fillFromFile(l_bs);
		      }
		      //read matrix C dimensions
		      unsigned int l_cM, l_cN;
		      l_bs >> l_handleC >> l_cM >> l_cN >> l_LdC;
		      std::cout << "INFO " << l_handleC << " " << l_cM << " " << l_cN << " " << l_LdC << "\n";
		      assert(l_cM == l_M);
		      assert(l_cN == l_N);
		      ok = checkDim("LdC", l_LdC, l_Edge, l_N);
		      if (!ok) {
			      boost::iostreams::close(l_bs);
			      return;
		      }	
		      l_pageC = p_Program.allocPages(l_handleC, l_newAllocC, l_M * l_LdC);
		      MatType l_matC(l_M, l_N, l_LdC, p_Program.getPageAddr(l_pageC));

		      if (p_WithGolden){
				l_matC.matMultWithScaleAndPRelu(l_matA, l_matB, l_matX, l_postScale, l_PRelu);
		      }

		      FcnArgsType l_fcnArgs(
			      l_pageA, l_pageB, l_pageC, l_pageX,
			      l_M, l_K, l_N,
			      l_LdA, l_LdB, l_LdC, l_LdX,
			      l_postScale,
			      l_PRelu
		      );
		      KargsType l_kargs;
		      l_kargs.setFcnArgs(l_fcnArgs);
		      l_kargs.store(p_Program.addInstr(),0);

		      std::cout << "Added FCN" << l_M << "x" << l_K << "x" << l_N << " postScale: " << l_postScale << " PReluVal: " << l_PRelu << "  ";
		      boost::iostreams::close(l_bs);
	      }
	      else {
		      std::cout << "ERROR: bad filename" << "\n";
	      }
      }
      
            void addInstrFromFiles(
	      unsigned int l_instrIndex,
	      ProgramType &p_Program,
	      std::string p_InsName, std::string p_MatAName, std::string p_MatBName, std::string p_MatXName,
	      bool p_WithGolden
      ) {
	      unsigned int l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX;
	      unsigned int l_pageA, l_pageB, l_pageC, l_pageX;
	      int32_t l_postScale, l_PRelu;
	      std::string l_handleA, l_handleB, l_handleC, l_handleX;
	      bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocX;
	      
	      std::cout <<"INFO: loading input matrices files " 
	      << p_InsName <<" "<<p_MatAName<<" "<<p_MatBName<<" "<<p_MatXName<<"\n";
	      std::ifstream l_fs_ins(p_InsName.c_str(),   std::ios_base::in | std::ios_base::binary);
	      std::ifstream l_fs_matA(p_MatAName.c_str(), std::ios_base::in | std::ios_base::binary);
	      std::ifstream l_fs_matB(p_MatBName.c_str(), std::ios_base::in | std::ios_base::binary);
	      std::ifstream l_fs_matX(p_MatXName.c_str(), std::ios_base::in | std::ios_base::binary);

	      boost::iostreams::filtering_istream l_bs_ins;
	      l_bs_ins.push(l_fs_ins);
	      boost::iostreams::filtering_istream l_bs_matA;
	      l_bs_matA.push(l_fs_matA);
	      boost::iostreams::filtering_istream l_bs_matB;
	      l_bs_matB.push(l_fs_matB);
	      boost::iostreams::filtering_istream l_bs_matX;
	      l_bs_matX.push(l_fs_matX);
	      
	      bool l_good = l_bs_ins.good() && l_bs_matA.good() && l_bs_matB.good() && l_bs_matX.good();

	      bool ok=true;
	      const unsigned int l_Edge = GEMX_ddrWidth;
	      const unsigned int l_kMin = 2 * GEMX_ddrWidth;

	      if (l_good) {
		      unsigned int l_index;
		      while (l_bs_ins.peek()=='#') l_bs_ins.ignore(2048, '\n');
		      l_bs_ins >> l_index;
		      while (l_index < l_instrIndex){
			l_bs_ins.ignore(2048, '\n');
			l_bs_ins >> l_index;		
		      }
		      std::cout<<"INFO instr number : "<<l_index<<"\n";
		      l_bs_ins >> l_postScale >> l_PRelu;
		      std::cout << "INFO " << l_postScale << " " << l_PRelu << "\n";
		      //read A dimensions
		      l_bs_ins >> l_handleA >> l_M >> l_K >> l_LdA;
		      std::cout << "INFO " << l_handleA << " " << l_M << " " << l_K << " " << l_LdA << "\n";
		      //check A dimention
		      ok = checkDim("M", l_M, l_Edge, 1) &&
					checkDim("K", l_K, l_Edge, 2*l_Edge) &&
					checkDim("LdA", l_LdA, l_Edge, l_K);
		      if (!ok) {
			      return;
		      }
		      //allocate host memory and initialize it with data from the file
		      l_pageA = p_Program.allocPages(l_handleA, l_newAllocA, l_M * l_LdA);
		      MatType l_matA(l_M, l_K, l_LdA, p_Program.getPageAddr(l_pageA));
		      while (l_bs_matA.peek()=='#') l_bs_matA.ignore(2048, '\n');
		      if (l_newAllocA) {
			      l_matA.fillFromFile(l_bs_matA);	
		      }
	      
		      //read matrix B dimensions
		      unsigned int l_bK;
		      l_bs_ins >> l_handleB >> l_bK >> l_N >> l_LdB;
		      std::cout << "INFO " << l_handleB << " " << l_bK << " " << l_N << " " << l_LdB << "\n";
		      assert(l_bK == l_K);
		      ok = checkDim("N", l_N, l_Edge, 1) &&
					checkDim("LdB", l_LdB, l_Edge, l_N);
		      if (!ok) {
			      boost::iostreams::close(l_bs_ins);
			      return;
		      }

		      l_pageB = p_Program.allocPages(l_handleB, l_newAllocB, l_K * l_LdB);
		      MatType l_matB(l_K, l_N, l_LdB, p_Program.getPageAddr(l_pageB));
		      while (l_bs_matB.peek()=='#') l_bs_matB.ignore(2048, '\n');
		      if (l_newAllocB) {
			      l_matB.fillFromFile(l_bs_matB);	
		      }
		      
		      //read matrix X dimensions
		      unsigned int l_xM, l_xN;
		      l_bs_ins >> l_handleX >> l_xM >> l_xN >> l_LdX;
		      std::cout << "INFO " << l_handleX << " " << l_xM << " " << l_xN << " " << l_LdX << "\n";
		      assert(l_xM == l_M);
		      assert(l_xN == l_N);
		      ok = checkDim("LdX", l_LdX, l_Edge, l_N);
		      if (!ok) {
			      boost::iostreams::close(l_bs_ins);
			      return;
		      }

		      l_pageX = p_Program.allocPages(l_handleX, l_newAllocX, l_M*l_N*(sizeof(GEMX_XdataType)/sizeof(GEMX_dataType)));
		      XMatType l_matX(l_M, l_N, l_LdX, (GEMX_XdataType *) p_Program.getPageAddr(l_pageX));
		      while (l_bs_matX.peek()=='#') l_bs_matX.ignore(2048, '\n');
		      if (l_newAllocX) {
			      l_matX.fillFromFile(l_bs_matX);
		      }
		      //read matrix C dimensions
		      unsigned int l_cM, l_cN;
		      l_bs_ins >> l_handleC >> l_cM >> l_cN >> l_LdC;
		      std::cout << "INFO " << l_handleC << " " << l_cM << " " << l_cN << " " << l_LdC << "\n";
		      assert(l_cM == l_M);
		      assert(l_cN == l_N);
		      ok = checkDim("LdC", l_LdC, l_Edge, l_N);
		      if (!ok) {
			      boost::iostreams::close(l_bs_ins);
			      return;
		      }	
		      l_pageC = p_Program.allocPages(l_handleC, l_newAllocC, l_M * l_LdC);
		      MatType l_matC(l_M, l_N, l_LdC, p_Program.getPageAddr(l_pageC));

		      if (p_WithGolden){
				l_matC.matMultWithScaleAndPRelu(l_matA, l_matB, l_matX, l_postScale, l_PRelu);
		      }

		      FcnArgsType l_fcnArgs(
			      l_pageA, l_pageB, l_pageC, l_pageX,
			      l_M, l_K, l_N,
			      l_LdA, l_LdB, l_LdC, l_LdX,
			      l_postScale,
			      l_PRelu
		      );
		      KargsType l_kargs;
		      l_kargs.setFcnArgs(l_fcnArgs);
		      l_kargs.store(p_Program.addInstr(),0);

		      std::cout << "Added FCN" << l_M << "x" << l_K << "x" << l_N << " postScale: " << l_postScale << " PReluVal: " << l_PRelu << "  ";
		      boost::iostreams::close(l_bs_ins);
		      boost::iostreams::close(l_bs_matA);
		      boost::iostreams::close(l_bs_matB);
		      boost::iostreams::close(l_bs_matX);
		      
	      } else {
		      std::cout << "ERROR: bad filename" << "\n";
	      }
      }
      
    void
    addInstr(
      ProgramType &p_Program,
      unsigned int p_M,
      unsigned int p_K,
      unsigned int p_N,
      unsigned int p_LdA,
      unsigned int p_LdB,
      unsigned int p_LdC,
			unsigned int p_LdX,
			int32_t p_postScale,
			int16_t p_PReluVal,
      std::string p_handleA,
      std::string p_handleB,
      std::string p_handleC,
			std::string p_handleX,
      bool p_WithGolden
    ) {
    
    // Allocate all pages before getting any address
    bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocX;
    unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA, p_M * p_LdA);
    unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, p_K * p_LdB);
    unsigned int l_pageC = p_Program.allocPages(p_handleC, l_newAllocC, p_M * p_LdC);
		unsigned int l_pageX = p_Program.allocPages(p_handleX, l_newAllocX, p_M * p_LdX * (sizeof(GEMX_XdataType)/sizeof(GEMX_dataType)));
    
    // Get addresses where matrices are stored
    MatType l_matA(p_M, p_K, p_LdA, p_Program.getPageAddr(l_pageA));
    MatType l_matB(p_K, p_N, p_LdB, p_Program.getPageAddr(l_pageB));
		XMatType l_matX(p_M, p_N, p_LdX, (GEMX_XdataType *) p_Program.getPageAddr(l_pageX));
    MatType l_matC(p_M, p_N, p_LdC, p_Program.getPageAddr(l_pageC));
    
    // Instruction
    FcnArgsType l_fcnArgs(
        l_pageA, l_pageB, l_pageC, l_pageX,
        p_M, p_K, p_N,
        p_LdA, p_LdB, p_LdC, p_LdX,
				p_postScale,
				p_PReluVal
      );
    KargsType l_kargs;
    l_kargs.setFcnArgs(l_fcnArgs);
    l_kargs.store(p_Program.addInstr(), 0);

    if (l_newAllocA) {
      l_matA.fillMod(67, 1);
    }
    if (l_newAllocB) {
      l_matB.fillMod(129, 65);
    }
		if (l_newAllocX) {
			l_matX.fillMod(1,0);
		}
  
    // Calculate reference C = A * B
    if (p_WithGolden) {
			l_matC.matMultWithScaleAndPRelu(l_matA, l_matB, l_matX, p_postScale, p_PReluVal);
			//l_matC.multiply(l_matA, l_matB);
    }
    std::cout << "Added FCN" << p_M << "x" << p_K << "x" << p_N << " postScale: " << p_postScale << " PReluVal: " << p_PReluVal << "  ";
  }
  void
  show(
      ProgramType &p_Program,
      FcnArgsType p_FcnArgs) {
      unsigned int 	l_M = p_FcnArgs.m_M,
                   	l_K = p_FcnArgs.m_K,
                   	l_N = p_FcnArgs.m_N,
                   	l_ldA = p_FcnArgs.m_Lda,
                   	l_ldB = p_FcnArgs.m_Ldb,
                   	l_ldC = p_FcnArgs.m_Ldc,
										l_ldX = p_FcnArgs.m_Ldx;
			int32_t				l_postScale = p_FcnArgs.m_postScale;
			int16_t			 	l_PReluVal = p_FcnArgs.m_PReluVal;

      MatType l_matA(l_M, l_K, l_ldA, p_Program.getPageAddr(p_FcnArgs.m_Aoffset));
      MatType l_matB(l_K, l_N, l_ldB, p_Program.getPageAddr(p_FcnArgs.m_Boffset));
			XMatType l_matX(l_M, l_N, l_ldX, (GEMX_XdataType *)p_Program.getPageAddr(p_FcnArgs.m_Xoffset));
      MatType l_matC(l_M, l_N, l_ldC, p_Program.getPageAddr(p_FcnArgs.m_Coffset));
      std::cout << "\n###########  Op Fcn  ###########\n"
        << "  C = A * B + X  postScale PReluVal " << "\n"
				 << l_M << "x" << l_N << " = " << l_M << "x" << l_K << " * " << l_K << "x" << l_N << " + " << l_M << " x " << l_N <<"\n"
				<< l_postScale << " " << l_PReluVal << "\n"
        << "  A " << l_matA << "\n"
        << "  B " << l_matB << "\n"
				<< "  X " << l_matX << "\n"
        << "  C " << l_matC << "\n";
    }
  bool
  compare(
      float p_TolRel, float p_TolAbs, 
      ProgramType &p_Program0, ProgramType &p_Program1,
      FcnArgsType p_FcnArgs
    ) {
      unsigned int 	l_M = p_FcnArgs.m_M,
                   	l_K = p_FcnArgs.m_K,
                   	l_N = p_FcnArgs.m_N,
                   	l_ldA = p_FcnArgs.m_Lda,
                   	l_ldB = p_FcnArgs.m_Ldb,
                   	l_ldC = p_FcnArgs.m_Ldc;
			int32_t				l_postScale = p_FcnArgs.m_postScale;
			int16_t			 	l_PReluVal = p_FcnArgs.m_PReluVal;

      MatType l_matC0(l_M, l_N, l_ldC, p_Program0.getPageAddr(p_FcnArgs.m_Coffset)),
              l_matC1(l_M, l_N, l_ldC, p_Program1.getPageAddr(p_FcnArgs.m_Coffset));
      std::cout << "\n###########  Op Fcn  ###########\n"
        << "  C = A * B + X  postScale PReluVal "
				<< l_M << "x" << l_N << " = " << l_M << "x" << l_K << " * " << l_K << "x" << l_N << " + " << l_M << " x " << l_N <<"\n"
				<< l_postScale << l_PReluVal << "\n"
        << "  Comparing ...\n";
      bool ok = l_matC1.cmp(p_TolRel, p_TolAbs, l_matC0);
      std::cout << "Fcn C " << (ok ? "Matches" : "Differs") << "\n";
      return(ok);
    }
      
};
  
#endif
