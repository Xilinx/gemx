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
 *  @brief GEMV testcase generator
 *
 *  $DateTime: 2017/11/22 14:19:13 $
 */

#ifndef GEMX_GEN_BIN_H
#define GEMX_GEN_BIN_H

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

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
//#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/algorithm/string.hpp>

#include "gemx_kernel.h"

////////////////////////  COMMON  ////////////////////////

// Common types
typedef GEMX_dataType FloatType;

// VLIV processing types
typedef gemx::Kargs<
    GEMX_dataType, GEMX_dataEqIntType,
    GEMX_ddrWidth, GEMX_argInstrWidth,
    GEMX_argInstrWidth * GEMX_ddrWidth * sizeof(GEMX_dataType) * 8,
    GEMX_argPipeline
  > KargsType;
typedef KargsType::OpType KargsOpType;
typedef KargsType::DdrFloatType DdrFloatType;
typedef gemx::ControlArgs ControlArgsType;
typedef GemvType::GemvArgsType GemvArgsType;
typedef GemmType::GemmArgsType GemmArgsType;
typedef TranspType::TranspArgsType TranspArgsType;
typedef SpmvType::SpmvArgsType SpmvArgsType;
typedef gemx::DdrMatrixShape DdrMatrixShapeType;
typedef DdrMatrixShapeType::FormatType MatFormatType;
#define GEMX_maxNumInstr 64
#define GEMX_transpEdgeSize (GEMX_ddrWidth * GEMX_transpBlocks)
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


class MtxRow {
  private:
    unsigned int m_Row, m_Col ;
    double m_Val;
  public:
    MtxRow() : m_Row(0), m_Col(0), m_Val(0) {}
    MtxRow(double p_Val, unsigned int p_Row, unsigned int p_Col)
      : m_Row(p_Row), m_Col(p_Col), m_Val(p_Val) {}
    unsigned int getRow() {return m_Row;}
    unsigned int getCol() {return m_Col;}
    double getVal() {return m_Val;}
    void
    scan(std::istream& p_Is) {
        p_Is >>  m_Row >> m_Col >> m_Val;
        if ((m_Row <= 0) || (m_Col <= 0))  {
          std::cerr << "  Error: invalid MTX file line row=" << m_Row
                    << " col=" << m_Col << " val=" << m_Val << "\n";
          assert(0);
        }
        // Indices start from 1 in MTX; 0 locally
        m_Row--;
        m_Col--;
      }
    friend bool
    operator<(MtxRow &a, MtxRow &b) {
        if (a.getRow() < b.getRow()) {
          return(true);
        } else if (a.getRow() == b.getRow()) {
          if (a.getCol() < b.getCol()) {
            return(true);
          }
        }
        return(false);
      }
    void
    print(std::ostream& os) {
        os << std::setw(GEMX_FLOAT_WIDTH) << int(getRow()) << " "
           << std::setw(GEMX_FLOAT_WIDTH) << int(getCol()) << "    "
           << std::setw(GEMX_FLOAT_WIDTH) << getVal();
      }
};
inline
std::ostream& operator<<(std::ostream& os, MtxRow &p_Val) {
  p_Val.print(os);
  return(os);
}


// Sparse matrix descriptor with data itself stored in caller's space
template < typename Tddr,  typename TmatD, typename Tmat>
class SpMat
{
  public:
    typedef TmatD  SpmvAdType;
    typedef Tmat  SpmvAType;
    typedef gemx::SpmvAdesc SpmvAdescType;
    static const unsigned int t_numSpmvPerPage = SpmvAdType::t_per4k;
    static const unsigned int t_numDescPerPage = SpmvAdescType::t_per4k;
    static const unsigned int t_numDdrWordsPerPage = SpmvType::DdrWideType::t_per4k;
    static const unsigned int t_RowsInCblock = SpmvType::t_RowsInCblock;
  private:
    unsigned int m_Rows, m_Cols, m_Nnz, m_Cblocks,
                 m_AstartIdx = GEMX_spmvNumCblocks * t_numDescPerPage / t_numSpmvPerPage;
    union {
      Tddr *Ddr;
      TmatD *Mat;
      SpmvAdescType *Desc;
    } m_Addr;
  public:
    	SpMat()
	{}
    SpMat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Nnz, unsigned int p_Cblocks, Tddr *p_Addr)
      : m_Rows(p_Rows), m_Cols(p_Cols), m_Nnz(p_Nnz), m_Cblocks(p_Cblocks) {
        m_Addr.Ddr = p_Addr;
        0 && std::cout << "DEBUG: sizeof(Tddr)=" << sizeof(Tddr)
                  << "  SpmvType::getDdrWidth()=" << SpmvType::getDdrWidth()
                  << "  sizeof(Tmat)=" << sizeof(TmatD)
                  << "  SpmvType::getSpmvWidth()=" << SpmvType::getSpmvWidth()
                  << std::endl;
                  
        assert(sizeof(Tddr) * SpmvType::getDdrWidth() == sizeof(TmatD) * SpmvType::getSpmvWidth());  // Make sure compiler padding and sizes are correct
        assert (m_AstartIdx * t_numSpmvPerPage ==  GEMX_spmvNumCblocks * t_numDescPerPage); // Desc pages
        assert(t_RowsInCblock <= SpmvAType::t_maxRowIdx); // Any stored row must be indexable
        assert(t_RowsInCblock > 0);  // incorrect groups or ddr with wrt ColAddIdxBits
      }
    SpMat& operator=(const SpMat& p_Src) {
        assert(p_Src.rows() == rows());
        assert(p_Src.cols() == cols());
        for (unsigned int i = 0; i < m_Nnz; ++i) {
          getVal(i) = p_Src.getVal(i);
        }
        return *this;
      }
    inline unsigned int rows() {return m_Rows;}
    inline unsigned int cols() {return m_Cols;}
    inline unsigned int nnz() {return m_Nnz;}

    inline SpmvAdescType &getDesc(unsigned int p_Cblock) {
        assert(p_Cblock < GEMX_spmvNumCblocks);
        return m_Addr.Desc[p_Cblock];
      }
    inline TmatD &getVal(unsigned int p_Idx) {
        //assert(p_Idx < nnz());
        return m_Addr.Mat[m_AstartIdx + p_Idx];
      }
    void 
    init(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Nnz, unsigned int p_Cblocks, Tddr *p_Addr){
		m_Rows = p_Rows;
		m_Cols = p_Cols;
		m_Nnz = p_Nnz;
		m_Cblocks = p_Cblocks;
		m_Addr.Ddr = p_Addr;
    
    }
    
    void
    fillMod(Tddr p_Max) {
        std::vector<MtxRow> l_rows;
        
        Tddr l_d = 17;
        unsigned int row = 0, col = 0;
        unsigned int numCols =  nnz() / rows();
        assert(numCols > 0);
        unsigned int colStep = cols() / numCols - 1;
        assert(colStep > 0);
        for (unsigned int i = 0; i < m_Nnz; ++i) {
          l_d++;
          l_d %= p_Max;
          assert(row < rows());
          assert(col < cols());
          MtxRow l_m(l_d, row, col);
          l_rows.push_back(l_m);
          //std::cout << "  DEBUG fillMod\n    l_m = " << l_m
          //          << "\n    l_D = " << l_mD << "\n";
          if (i % numCols == numCols - 1) {
            row++;
            col = 0;
          }
          col += colStep;
	  if (row >= rows()) {
	    row--;
	  }
        }
        fillFromVector(l_rows);
      }
    void
    fillFromVector(std::vector<MtxRow> p_Rows) {
        assert(p_Rows.size() ==  nnz());
        
        // Partition the matrix
        std::vector<std::vector<MtxRow>> l_part;
        l_part.reserve(GEMX_spmvNumCblocks);
        for (unsigned int i = 0; i < m_Nnz; ++i) {
          MtxRow l_row = p_Rows[i];
          unsigned int l_cBlock = l_row.getRow() / t_RowsInCblock;
          if (l_cBlock >= l_part.size()) {
            l_part.resize(l_cBlock + 1);
          }
          l_part[l_cBlock].push_back(l_row);
        }
        m_Cblocks = l_part.size();
        
        // Pad each partition nnzs to align with ddr width
        const unsigned int l_spmvAlignNnz = GEMX_spmvWidth;
        for (unsigned int l_cBlock = 0; l_cBlock < m_Cblocks; ++l_cBlock) {
          unsigned int l_nnz = l_part[l_cBlock].size();
          unsigned int l_nnzAligned =l_spmvAlignNnz * ((l_nnz + l_spmvAlignNnz - 1) / l_spmvAlignNnz);
          assert(l_nnzAligned >= l_nnz);
          l_part[l_cBlock].resize(l_nnzAligned);
        }
        
        // Break long rows
        const unsigned int l_rowUnits = GEMX_spmvWidth * GEMX_spmvMacGroups;
        for (unsigned int l_cBlock = 0; l_cBlock < m_Cblocks; ++l_cBlock) {
          std::array<std::queue<MtxRow>, l_rowUnits> l_rowQueues;
          // Separate per row unit
          unsigned int l_nnz = l_part[l_cBlock].size();
          for(auto & l_row : l_part[l_cBlock]) {
            l_rowQueues[l_row.getRow() % l_rowUnits].push(l_row);
          }
          l_part[l_cBlock].clear();
          // Aggregate to max row length
          const unsigned int l_rowBreak = 16; // This should roughly match the smallest chain of t_FifoDepthDeep
          unsigned int l_doneNnzs = 0;
          while (l_doneNnzs < l_nnz) {
            for (unsigned int l_rowUnit = 0 ; l_rowUnit < l_rowUnits; l_rowUnit++) {
              for (unsigned int i = 0 ; i < l_rowBreak; i++) {
                if (l_rowQueues[l_rowUnit].empty()) {
                  break;
                } else {
                  l_part[l_cBlock].push_back(l_rowQueues[l_rowUnit].front());
                  l_rowQueues[l_rowUnit].pop();
                  l_doneNnzs++;
                }
              }
            }
          }
        }
        
        // Create block descriptors
        unsigned int l_startIdx = 0;
        for (unsigned int l_cBlock = 0; l_cBlock < m_Cblocks; ++l_cBlock) {
          unsigned int l_nnz = l_part[l_cBlock].size();
          SpmvAdescType l_desc(l_nnz, l_startIdx / t_numSpmvPerPage);
          getDesc(l_cBlock) = l_desc;
          l_startIdx += l_nnz;
          // Align start to 4kB
          while ((l_startIdx % t_numSpmvPerPage) != 0) {
            l_startIdx++;
          }
        }
        std::cout << "INFO: Spmv fillFromVector number of partitions " << m_Cblocks << "\n";
        
        // Fill the A matrix data
        unsigned int is = 0, l_rowCblock = 0;
        for (unsigned int l_cBlock = 0; l_cBlock < l_part.size(); ++l_cBlock) {
          SpmvAdescType l_desc = getDesc(l_cBlock);
          for (unsigned int i = 0; i < l_desc.getNnz(); ++i) {
            MtxRow l_row = l_part[l_cBlock][i];
            Tmat l_m(Tddr(l_row.getVal()), l_row.getRow() % t_RowsInCblock, l_row.getCol());
            TmatD l_mD = l_m.getAsAd();
            getVal(l_desc.getOffset() * t_numSpmvPerPage + i) = l_mD;
            
            0 && std::cout << "  DEBUG fillFromVector"
                      << "  l_cBlock=" << l_cBlock
                      << "  i=" << i
                      << "  l_m = " << l_m
                      << "  l_D = " << l_mD
                      << "\n";
          }
        }
      }

    std::vector<MtxRow>
    getNnzVector() {
        std::vector<MtxRow> l_rows;
        for (unsigned int l_cBlock = 0; l_cBlock < m_Cblocks; ++l_cBlock) {
          SpmvAdescType l_desc = getDesc(l_cBlock);
          for (unsigned int i = 0; i < l_desc.getNnz(); ++i) {
            typename SpMat::SpmvAdType l_Ad = getVal(l_desc.getOffset() * t_numSpmvPerPage + i);
            typename SpMat::SpmvAType l_A(l_Ad);
            unsigned int row = l_A.getRow(),
                         col = l_A.getCol();
            if (l_A.getA() != 0) {
              MtxRow l_mr(l_A.getA(), l_cBlock * t_RowsInCblock + row, col);
              l_rows.push_back(l_mr);
            }
            0 && std::cout << "  DEBUG getNnzVector"
                      << "  l_cBlock=" << l_cBlock
                      << "  i=" << i
                      << "  l_m = " << l_A
                      << "  l_D = " << l_Ad
                      << "\n";
          }
        }
        return(l_rows);
      }
    
    void
    print(std::ostream& os) {
        os << "%%MatrixMarket matrix coordinate real general\n"
           << "% Rows Columns Entries\n";
        os << rows() << "  " << cols() << "  " << nnz() << "\n";
        for (unsigned int l_cBlock = 0; l_cBlock < m_Cblocks; ++l_cBlock) {
          SpmvAdescType l_desc = getDesc(l_cBlock);
          for (unsigned int i = 0; i < l_desc.getNnz(); ++i) {
            typename SpMat::SpmvAdType l_Ad = getVal(l_desc.getOffset() * t_numSpmvPerPage + i);
            typename SpMat::SpmvAType l_A(l_Ad);
            unsigned int row = l_A.getRow(),
                         col = l_A.getCol();
            //os << l_Ad << " Ad\n";
            //os << l_A << " A\n\n";
            MtxRow l_mr(l_A.getA(), l_cBlock * t_RowsInCblock + row + 1, col + 1);
            os << l_mr << "\n";
          }
        }
      }
};
template <typename T1, typename T2, typename T3>
std::ostream& operator<<(std::ostream& os, SpMat<T1, T2, T3>& p_Val) {
  p_Val.print(os);
  return(os);
}




// Matrix descriptor with data itself stored in caller's space
template < typename T, typename TspD, typename Tsp>
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
    void
    multiplyGf(Mat & p_A, Mat & p_B, unsigned int p_EdgeWidth) {
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        std::cout << "  DEBUG multiplyGf rows=" << rows() << "  cols=" << cols() << "\n";
        for (unsigned int rowBlock = 0; rowBlock < rows() / p_EdgeWidth ; ++rowBlock) {
          for (unsigned int colBlock = 0; colBlock < cols() / p_EdgeWidth ; ++colBlock) {
            for (unsigned int row = 0; row < rows(); ++row) {
              for (unsigned int col = 0; col < cols(); ++col) {
                T l_val = 0;
                for (unsigned int k = 0; k < p_A.cols(); ++k) {
                  l_val += p_A.getVal(k + rowBlock * p_EdgeWidth, col + colBlock * p_EdgeWidth) *
                           p_B.getVal(k + rowBlock * p_EdgeWidth, col + colBlock * p_EdgeWidth);
                }
                getVal(row + rowBlock * p_EdgeWidth, col + colBlock * p_EdgeWidth) = l_val;
                std::cout << "DEBUG multiplyGf after k-loop " << *this << "\n"; 
              }
            }
          }
        }
      }
    // Matrix A is in GvA format (also dimensions are wider and shorter)
    // The p_rowEdgeWidth just inficates the compute array intake edge to allow for matrix dimension adjustment
    void
    multiplyGemvGf(Mat & p_A, Mat & p_B, unsigned int p_rowEdgeWidth) {
        assert(p_A.rows() * p_rowEdgeWidth == rows());
        assert(p_A.cols() == p_B.rows() * p_rowEdgeWidth);
        assert(p_B.cols() == cols());
        std::cout << "  DEBUG multiplyGvA format rows=" << rows() << "  cols=" << cols() << "\n";
        // Rows here are mblocks, cols are within the mblock 
        for (unsigned int row = 0; row < p_A.rows() ; ++row) {  // A is already in block format
          for (unsigned int col = 0; col < p_A.cols() ; ++col) {
            unsigned int k = col / p_rowEdgeWidth;
            unsigned int w = col % p_rowEdgeWidth;
            T l_a = p_A.getVal(row, col);
            T l_b = p_B.getVal(k, 0);
            getVal(w + row * p_rowEdgeWidth, 0)  += l_a * l_b;
            //std::cout << "        += a * b  = " << l_a << " * " << l_b << "\n";
          }
          //std::cout << "    DEBUG multiplyGemvGf after k-loop " << *this << "\n"; 
        }
      }
    void
    multiplySpmv(SpMat<T, TspD, Tsp> & p_A, Mat & p_B) {
        T l_val = 0;
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        std::vector<MtxRow> l_rows =  p_A.getNnzVector();
        for (MtxRow &l_row : l_rows) {
          unsigned int row = l_row.getRow(),
                       col = l_row.getCol();
          double l_val = l_row.getVal();
          getVal(row, 0) += l_val * p_B.getVal(col, 0);
          //std::cout << "DEBUG multiplySpmv row=" << row << " col=" << col << "  "
          //          << l_val << " * " << p_B.getVal(col, 0)
          //          << " was added to " << getVal(row, 0) << "\n";
        }
      }
    void
    transpose(Mat & p_A) {
        for (unsigned int row = 0; row < rows(); ++row) {
          for (unsigned int col = 0; col < cols() ; ++col) {
            getVal(row, col) = p_A.getVal(col, row);
          }
        }
        std::swap(m_Rows, m_Cols);
      }
    void
    transposeGva(Mat & p_A, unsigned int p_rowEdgeWidth, unsigned int p_colEdgeWidth) {
        unsigned int l_pos = 0;
        for (unsigned int rowBlock = 0; rowBlock < p_A.rows() / p_rowEdgeWidth ; ++rowBlock) {
          for (unsigned int colBlock = 0; colBlock < p_A.cols() / p_colEdgeWidth ; ++colBlock) {
            for (unsigned int col = 0; col < p_colEdgeWidth; ++col) {
              for (unsigned int row = 0; row < p_rowEdgeWidth; ++row) {
                getVal(l_pos / cols(), l_pos % cols())  =
                  p_A.getVal(row + rowBlock * p_rowEdgeWidth, col + colBlock * p_colEdgeWidth);
                l_pos++;
              }
              //std::cout << "DEBUG transposeGva step " << *this << "\n"; 
            }
          }
        }
        std::swap(m_Rows, m_Cols);
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
template <typename T1, typename T2, typename T3>
std::ostream& operator<<(std::ostream& os, Mat<T1, T2, T3>& p_Val) {
  p_Val.print(os);
  return(os);
}




// Float specialization

typedef gemx::Spmv<
    float, int,
    GEMX_ddrWidth, GEMX_spmvWidth,
    GEMX_spmvkVectorBlocks, GEMX_spmvmVectorBlocks,
    GEMX_spmvMacGroups,
    GEMX_spmvColAddIdxBits,
    GEMX_spmvNumCblocks,
    GEMX_spmvFloatPerDesc
  > SpmvType_ForFloat;

typedef SpmvType_ForFloat::SpmvAType SpmvAType_ForFloat;
typedef SpmvType_ForFloat::SpmvAdType SpmvAdType_ForFloat;
typedef SpMat<float, SpmvAdType_ForFloat, SpmvAType_ForFloat> SpMatType_ForFloat;
typedef Mat<float, SpmvAdType_ForFloat, SpmvAType_ForFloat> MatType_ForFloat;

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


template<>
void
SpMatType_ForFloat::fillMod(float p_Max) {
  std::vector<MtxRow> l_rows;
  
  float l_d = 17;
  unsigned int row = 0, col = 0;
  unsigned int numCols =  nnz() / rows();
  assert(numCols > 0);
  unsigned int colStep = cols() / numCols - 1;
  assert(colStep > 0);
  for (unsigned int i = 0; i < m_Nnz; ++i) {
    l_d += 0.3;
    if (l_d > p_Max) {
      l_d -= p_Max;
    }
    assert(row < rows());
    assert(col < cols());
    MtxRow l_m(l_d, row, col);
    l_rows.push_back(l_m);
    if (i % numCols == numCols - 1) {
      row++;
      col = 0;
    }
    col += colStep;
    if (row >= rows()) {
      row--;
    }
  }
  fillFromVector(l_rows);
}


////  Typedefs
typedef SpmvType::SpmvAdType SpmvAdType;
typedef SpmvType::SpmvAType SpmvAType;
typedef Mat<GEMX_dataType, SpmvAdType, SpmvAType > MatType;
typedef SpMat<GEMX_dataType, SpmvAdType, SpmvAType > SpMatType;



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

  
////////////////////////  GEMV  ////////////////////////
class GenGemv
{
  public:
    bool
    check(
      unsigned int p_M,
      unsigned int p_K,
      unsigned int p_LdA
    ) {
        bool ok = true;
        
        const unsigned int l_mEdge = GEMX_gemvmGroups * GEMX_ddrWidth;
        const unsigned int l_kEdge = GEMX_transpBlocks * GEMX_ddrWidth;
        
        if (p_M % l_mEdge != 0) {
          std::cerr << "ERROR: gemv  M dimension " << p_M << " must be multiple of "
                    << l_mEdge << "\n";
          ok = false;
        }
        if (p_K % (l_kEdge) != 0) {
          std::cerr << "ERROR: gemv  K dimension " << p_K << " must be multiple of "
                    << l_kEdge << "\n";
          ok = false;
        }        
        return(ok);
      }
    //__attribute__ ((noinline))
    void
    addInstr(
      ProgramType &p_Program,
      unsigned int p_M,
      unsigned int p_K,
      unsigned int p_Lda,
      std::string p_handleA,
      std::string p_handleB,
      std::string p_handleC,
      bool p_WithGolden
    ) {
    
    // Allocate all pages before getting any address
    bool l_newAllocA, l_newAllocB, l_newAllocC;
    unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA, p_M * p_Lda);
    unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, p_K * 1);
    unsigned int l_pageC = p_Program.allocPages(p_handleC, l_newAllocC, p_M * 1);
    
    // Get addresses where matrices are stored
    MatType l_matA(p_M, p_K, p_Lda, p_Program.getPageAddr(l_pageA));
    MatType l_matB(p_K, 1, 1,       p_Program.getPageAddr(l_pageB));
    MatType l_matC(p_M, 1, 1,       p_Program.getPageAddr(l_pageC));
    
    // Instruction
    GemvArgsType l_gemvArgs(
        l_pageA, l_pageB, l_pageC,
        p_M, p_K, p_Lda
      );
    KargsType l_kargs;
    l_kargs.setGemvArgs(l_gemvArgs);
    l_kargs.store(p_Program.addInstr(), 0);

    if (l_newAllocA) {
      l_matA.fillMod(std::numeric_limits<GEMX_dataType>::max());
    }
    if (l_newAllocB) {
      l_matB.fillMod(7);
    }
    
    // Calculate reference C = A * B
    if (p_WithGolden) {
      l_matC.multiply(l_matA, l_matB);
    }
    std::cout << "Added GEMV " << p_M << "x" << p_K << "  ";
  }
  
  void
  show(
      ProgramType &p_Program,
      GemvArgsType p_GemvArgs
    ) {
      unsigned int l_M = p_GemvArgs.m_M,
                   l_K = p_GemvArgs.m_K,
				   l_Lda = p_GemvArgs.m_Lda;
      MatType l_matA(l_M, l_K, l_Lda, p_Program.getPageAddr(p_GemvArgs.m_Aoffset));
      MatType l_matB(l_K, 1,   1,   p_Program.getPageAddr(p_GemvArgs.m_Boffset));
      MatType l_matC(l_M, 1,   1,   p_Program.getPageAddr(p_GemvArgs.m_Coffset));
      std::cout << "\n###########  Op Gemv  ###########\n"
        << "  C = A * B  "
        << l_M << "x" << 1 << " = " << l_M << "x" << l_K << " * " << l_K << "x" << 1
        << "  A " << l_matA << "\n"
        << "  B " << l_matB << "\n"
        << "  C " << l_matC << "\n";
    }
  bool
  compare(
      float p_TolRel, float p_TolAbs, 
      ProgramType &p_Program0, ProgramType &p_Program1,
      GemvArgsType p_GemvArgs
    ) {
      unsigned int l_M = p_GemvArgs.m_M,
                   l_K = p_GemvArgs.m_K;
      MatType l_matC0(l_M, 1,   1,   p_Program0.getPageAddr(p_GemvArgs.m_Coffset)),
              l_matC1(l_M, 1,   1,   p_Program1.getPageAddr(p_GemvArgs.m_Coffset));
      std::cout << "\n###########  Op Gemv  ###########\n"
        << "  C = A * B  "
        << l_M << "x" << 1 << " = " << l_M << "x" << l_K << " * " << l_K << "x" << 1
        << "  Comparing ...\n";
      bool ok = l_matC1.cmp(p_TolRel, p_TolAbs, l_matC0);
      std::cout << "Gemv C " << (ok ? "Matches" : "Differs") << "\n";
      return(ok);
    }
      
};

  
////////////////////////  GEMM  ////////////////////////
class GenGemm
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
      unsigned int p_LdC
    ) {
        bool ok = true;
        
        const unsigned int l_Edge = GEMX_ddrWidth;
        const unsigned int l_kMin = 2 * GEMX_ddrWidth; // due to kernel Cout control to save area
        
        ok = checkDim("M", p_M, l_Edge, 1) &&
             checkDim("K", p_K, l_Edge, 2 * l_Edge) &&
             checkDim("N", p_N, l_Edge, 1) &&
             checkDim("LdA", p_LdA, l_Edge, p_K) &&
             checkDim("LdB", p_LdB, l_Edge, p_N) &&
             checkDim("LdC", p_LdC, l_Edge, p_N);
        return(ok);
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
      std::string p_handleA,
      std::string p_handleB,
      std::string p_handleC,
      bool p_WithGolden
    ) {
    
    // Allocate all pages before getting any address
    bool l_newAllocA, l_newAllocB, l_newAllocC;
    unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA, p_M * p_LdA);
    unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, p_K * p_LdB);
    unsigned int l_pageC = p_Program.allocPages(p_handleC, l_newAllocC, p_M * p_LdC);
    
    // Get addresses where matrices are stored
    MatType l_matA(p_M, p_K, p_LdA, p_Program.getPageAddr(l_pageA));
    MatType l_matB(p_K, p_N, p_LdB, p_Program.getPageAddr(l_pageB));
    MatType l_matC(p_M, p_N, p_LdC, p_Program.getPageAddr(l_pageC));
    
    // Instruction
    GemmArgsType l_gemmArgs(
        l_pageA, l_pageB, l_pageC,
        p_M, p_K, p_N,
        p_LdA, p_LdB, p_LdC
      );
    KargsType l_kargs;
    l_kargs.setGemmArgs(l_gemmArgs);
    l_kargs.store(p_Program.addInstr(), 0);

    if (l_newAllocA) {
      l_matA.fillMod(67, 1);
    }
    if (l_newAllocB) {
      l_matB.fillMod(129, 65);
    }
  
    // Calculate reference C = A * B
    if (p_WithGolden) {
      l_matC.multiply(l_matA, l_matB);
    }
    std::cout << "Added GEMM " << p_M << "x" << p_K << "x" << p_N << "  ";
  }
  void
  show(
      ProgramType &p_Program,
      GemmArgsType p_GemmArgs) {
      unsigned int l_M = p_GemmArgs.m_M,
                   l_K = p_GemmArgs.m_K,
                   l_N = p_GemmArgs.m_N,
                   l_ldA = p_GemmArgs.m_Lda,
                   l_ldB = p_GemmArgs.m_Ldb,
                   l_ldC = p_GemmArgs.m_Ldc;
      MatType l_matA(l_M, l_K, l_ldA, p_Program.getPageAddr(p_GemmArgs.m_Aoffset));
      MatType l_matB(l_K, l_N, l_ldB, p_Program.getPageAddr(p_GemmArgs.m_Boffset));
      MatType l_matC(l_M, l_N, l_ldC, p_Program.getPageAddr(p_GemmArgs.m_Coffset));
      std::cout << "\n###########  Op Gemm  ###########\n"
        << "  C = A * B  "
        << l_M << "x" << l_N << " = " << l_M << "x" << l_K << " * " << l_K << "x" << l_N << "\n"
        << "  A " << l_matA << "\n"
        << "  B " << l_matB << "\n"
        << "  C " << l_matC << "\n";
    }
  bool
  compare(
      float p_TolRel, float p_TolAbs, 
      ProgramType &p_Program0, ProgramType &p_Program1,
      GemmArgsType p_GemmArgs
    ) {
      unsigned int l_M = p_GemmArgs.m_M,
                   l_K = p_GemmArgs.m_K,
                   l_N = p_GemmArgs.m_N,
                   l_ldA = p_GemmArgs.m_Lda,
                   l_ldB = p_GemmArgs.m_Ldb,
                   l_ldC = p_GemmArgs.m_Ldc;
      MatType l_matC0(l_M, l_N, l_ldC, p_Program0.getPageAddr(p_GemmArgs.m_Coffset)),
              l_matC1(l_M, l_N, l_ldC, p_Program1.getPageAddr(p_GemmArgs.m_Coffset));
      std::cout << "\n###########  Op Gemm  ###########\n"
        << "  C = A * B  "
        << l_M << "x" << l_N << " = " << l_M << "x" << l_K << " * " << l_K << "x" << l_N << "\n"
        << "  Comparing ...\n";
      bool ok = l_matC1.cmp(p_TolRel, p_TolAbs, l_matC0);
      std::cout << "Gemm C " << (ok ? "Matches" : "Differs") << "\n";
      return(ok);
    }
      
};
  
////////////////////////  TRANSP  ////////////////////////
class GenTransp
{
  public:
    typedef gemx::DdrMatrixShape::FormatType MatFormatType;

  public:
    bool
    check(
      unsigned int p_M,
      unsigned int p_N,
      unsigned int p_LdIn,
      unsigned int p_LdOut,
      MatFormatType p_FormatA,
      MatFormatType p_FormatB
    ) {
        bool ok = true;
        
        if (p_FormatB == MatFormatType::GvA) {
          if (p_LdOut != 0) {
            std::cerr << "ERROR: transp  LdOut " << p_LdOut << " is auto computed for GVA matrix, use 0 " << "\n";
            ok = false;
          }
        } else {
          if (p_LdOut < p_M) {
            std::cerr << "ERROR: transp  LdOut " << p_LdOut << " is smaller than p_M " << p_M << "\n";
            ok = false;
          }
        }
        
        if (p_LdIn < p_N) {
          std::cerr << "ERROR: transp  LdIn " << p_LdIn << " is smaller than p_N " << p_N << "\n";
          ok = false;
        }
        if (p_M % GEMX_transpEdgeSize != 0) {
          std::cerr << "ERROR: transp  p_M " << p_M << " is not divisible by GEMX_transpEdgeSize " << GEMX_transpEdgeSize << "\n";
          ok = false;
        }
        if (p_N % GEMX_transpEdgeSize != 0) {
          std::cerr << "ERROR: transp  p_N " << p_N << " is not divisible by GEMX_transpEdgeSize " << GEMX_transpEdgeSize << "\n";
          ok = false;
        }
        if (p_LdIn % GEMX_ddrWidth != 0) {
          std::cerr << "ERROR: transp  p_LdIn " << p_LdIn << " is not divisible by GEMX_ddrWidth " << GEMX_ddrWidth << "\n";
          ok = false;
        }
        if (p_LdOut % GEMX_ddrWidth != 0) {
          std::cerr << "ERROR: transp  p_LdOut " << p_LdOut << " is not divisible by GEMX_ddrWidth " << GEMX_ddrWidth << "\n";
          ok = false;
        }
        if (p_FormatA == MatFormatType::Unknown) {
          std::cerr << "ERROR: transp  formatA " << p_FormatA << " is not valid\n";
          ok = false;
        }
        if (p_FormatB == MatFormatType::Unknown) {
          std::cerr << "ERROR: transp  formatB " << p_FormatB << " is not valid\n";
          ok = false;
        }
        
        return(ok);
      }
    void
    addInstr(
      ProgramType &p_Program,
      unsigned int p_M,
      unsigned int p_N,
      unsigned int p_LdIn,
      unsigned int p_LdOut,
      MatFormatType p_FormatA,
      MatFormatType p_FormatB,
      std::string p_handleA,
      std::string p_handleB,
      bool p_WithGolden
    ) {
    
    // Dimensions
    unsigned int l_outRows = p_N;
    unsigned int l_outCols = p_M;
    unsigned int l_outLd = p_LdOut;
    if (p_FormatB == MatFormatType::GvA) {
      unsigned int l_Width = GEMX_ddrWidth;
      l_outRows = p_M / l_Width;
      l_outCols = p_N * l_Width;
      l_outLd = l_outCols;
    }
    
    // Allocate all pages before getting any address
    bool l_newAllocA, l_newAllocB;
    unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA, p_M * p_LdIn);
    unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, l_outRows * l_outLd);
    
    // Get addresses where matrices are stored
    MatType l_matA(p_M, p_N, p_LdIn, p_Program.getPageAddr(l_pageA));
    MatType l_matB(l_outRows, l_outCols, l_outLd, p_Program.getPageAddr(l_pageB));
    
    // Instruction
    DdrMatrixShapeType l_srcShape(l_pageA,  p_M, p_N, p_LdIn,  0, p_FormatA),
                       l_dstShape(l_pageB, l_outRows, l_outCols, l_outLd, 0, p_FormatB);
    TranspArgsType l_transpArgs(
        l_srcShape, l_dstShape
      );
    KargsType l_kargs;
    l_kargs.setTranspArgs(l_transpArgs);
    l_kargs.store(p_Program.addInstr(), 0);

    if (l_newAllocA) {
      l_matA.fillMod(std::numeric_limits<GEMX_dataType>::max());
    }
    if (l_newAllocB) {
      l_matB.fillMod(7);
    }
    
    // Calculate reference C = A * B
    if (p_WithGolden) {
      if (p_FormatB == MatFormatType::Cm) {
        l_matB.transpose(l_matA);
      } else if (p_FormatB == MatFormatType::GvA) {
        l_matB.transposeGva(l_matA, GEMX_ddrWidth * GEMX_gemvmGroups, GEMX_ddrWidth);
      } else {
        assert(false);
      }
    }
    std::cout << "Added TRANSP " << p_M << "x" << p_N << "  ";
  }
  void
  show(
      ProgramType &p_Program,
      TranspArgsType p_TranspArgs
    ) {
      DdrMatrixShapeType l_src = p_TranspArgs.m_Src,
                          l_dst = p_TranspArgs.m_Dst;
      unsigned int l_pageA = l_src.m_Offset,
                   l_pageB = l_dst.m_Offset;
      MatType l_matA(l_src.m_Rows, l_src.m_Cols, l_src.m_Ld, p_Program.getPageAddr(l_pageA));
      MatType l_matB(l_dst.m_Rows, l_dst.m_Cols, l_dst.m_Ld, p_Program.getPageAddr(l_pageB));
      std::cout << "\n###########  Op Transp  ###########\n"
        << "  " << l_src << "  ->  " << l_dst << "\n"
        << "  A  Page=" << l_pageA << "  " << l_matA << "\n"
        << "  B  Page=" << l_pageB << "  " << l_matB << "\n";
    }
  bool
  compare(
      float p_TolRel, float p_TolAbs, 
      ProgramType &p_Program0, ProgramType &p_Program1,
      TranspArgsType p_TranspArgs
    ) {
      DdrMatrixShapeType l_src = p_TranspArgs.m_Src,
                         l_dst = p_TranspArgs.m_Dst;
      unsigned int l_pageA = l_src.m_Offset,
                   l_pageB = l_dst.m_Offset;
      MatType l_matB0(l_dst.m_Rows, l_dst.m_Cols, l_dst.m_Ld, p_Program0.getPageAddr(l_pageB)),
              l_matB1(l_dst.m_Rows, l_dst.m_Cols, l_dst.m_Ld, p_Program1.getPageAddr(l_pageB));
      std::cout << "\n###########  Op Transp  ###########\n"
        << "  " << l_src << "  ->  " << l_dst << "\n"
        << "  Comparing ...\n";
      bool ok = l_matB1.cmp(p_TolRel, p_TolAbs, l_matB0);
      std::cout << "Transp B " << (ok ? "Matches" : "Differs") << "\n";
      return(ok);
    }
};


////////////////////////  SPMV  ////////////////////////

class MtxFile
{
  public:
  
  private:
    std::string m_FileName;
    bool m_Good;
    unsigned int m_M, m_K, m_Nnz;
    std::vector<MtxRow> m_Rows;
  private:
    void align( unsigned int &dst, unsigned int width) {dst = width * ((dst + width - 1) / width);}
  public:
    bool good() {return(m_Good);}
    unsigned int rows() {return(m_M);}
    unsigned int cols() {return(m_K);}
    unsigned int nnz() {return(m_Nnz);}
    std::string fileName() {return(m_FileName);}
    MtxFile(std::string p_FileName)
      : m_Good(false),
        m_M(0), m_K(0), m_Nnz(0), 
        m_FileName(p_FileName)
      {
        if (m_FileName != "none") {
          std::cout << "INFO: loading Mtx file  " << m_FileName << "\n";
          std::ifstream l_fs(m_FileName.c_str(), std::ios_base::in | std::ios_base::binary);
          
          boost::iostreams::filtering_istream l_bs;
          std::string l_ext = boost::to_lower_copy<std::string>(m_FileName.substr(m_FileName.find_last_of(".") + 1));
          if (l_ext == "gz") {
            l_bs.push(boost::iostreams::gzip_decompressor());
          // Bzip2 did not work properly in boost 1.64
          //} else if (l_ext == "bz2") {
          //  l_bs.push(boost::iostreams::bzip2_decompressor());
          } else if (l_ext == "mtx") {
            // noop
          } else{
            std::cerr << "ERROR: MtxFile failed due to unknown extension \"" << l_ext << "\", file  "
                      << m_FileName << "\n";
            assert(0);
          }
          l_bs.push(l_fs);
          
          m_Good = l_bs.good();
          if (m_Good) {
            while (l_bs.peek() == '%') l_bs.ignore(2048, '\n');
            l_bs >>  m_M >> m_K >> m_Nnz;
            for (unsigned int i = 0; i < nnz(); ++i) {
              MtxRow l_row;
              l_row.scan(l_bs);
              m_Rows.push_back(l_row);
            }
            boost::iostreams::close(l_bs);
            // Sort to make canonical
            sort(m_Rows.begin(), m_Rows.end());
            // Pad with 0s
            while (m_Nnz % GEMX_spmvWidth != 0) {
              std::cout << "INFO: Added padding row to the mtx data\n";
              MtxRow l_row;
              m_Rows.push_back(l_row);
              m_Nnz++;
            }
            // Adjust dimensions - needs to be aligned to both GEMX_spmvWidth and GEMX_ddrWidth
            align (m_M, GEMX_spmvWidth * GEMX_spmvMacGroups);  // Align for loadC
            //align (m_M, GEMX_ddrWidth);
            assert(m_M % GEMX_spmvWidth == 0);
            align (m_K, GEMX_ddrWidth);  
            std::cout << "INFO: loaded mtx file"
                     << "  M " << rows()
                     << "  K " << cols()
                     << "  Nnz " << nnz()
                     << "\n";
          }
        }
      }
    std::vector<MtxRow> &getRows() {return(m_Rows);}
};

class GenSpmv
{
  public:
    bool
    check(
      unsigned int &p_M,  // The check() modifies the dimensions when loading from a file
      unsigned int &p_K,
      unsigned int &p_Nnz,
      MtxFile &p_MtxFile
    ) {
        bool ok = true;
        
        // m_C
        const unsigned int l_mEdge = GEMX_spmvWidth * GEMX_spmvMacGroups;
        const unsigned int l_mMax = l_mEdge * GEMX_spmvmVectorBlocks * GEMX_ddrWidth * GEMX_spmvNumCblocks;
        // m_B
        const unsigned int l_kEdge = GEMX_ddrWidth;
        const unsigned int l_kMax = l_kEdge * GEMX_spmvkVectorBlocks * GEMX_spmvWidth;
        
        if (!p_MtxFile.good() && (p_MtxFile.fileName() != "none")) {
          std::cerr << "ERROR: spmv  mtxFile " << p_MtxFile.fileName()
                    << " must exist or use none for auto-generated data"
                    << "\n";
          ok = false;
        }
        
        // Use the file only
        if (p_MtxFile.good()) {
          if ((p_M != 0) || (p_K != 0) || (p_Nnz != 0)) {
            std::cerr << "ERROR: spmv  M, K, Nnz must be 0 when using mtx file: "
                      << "  M " << p_M
                      << "  K " << p_K
                      << "  Nnz " << p_Nnz
                      << "\n";
          }
          p_M = p_MtxFile.rows();
          p_K = p_MtxFile.cols();
          p_Nnz = p_MtxFile.nnz();
        }
        
        if (p_Nnz == 0) {
          std::cerr << "ERROR: spmv  Nnz must be non-0, it is " << p_Nnz << "\n";
          ok = false;
        }
        if (p_Nnz % GEMX_spmvWidth != 0) {
          std::cerr << "ERROR: spmv  Nnz " << p_Nnz << " must be multiple of GEMX_spmvWidth "
                    << GEMX_spmvWidth << "\n";
          ok = false;
        }
        //if (p_Nnz < p_M) {
        //  std::cerr << "ERROR: spmv  Nnz " << p_Nnz << " must be greater than number of rows M "
        //            << p_M << "\n";
        //  ok = false;
        //}
        if (p_M > l_mMax) {
          std::cerr << "ERROR: spmv  M dimension " << p_M << " is larger than max supported " << l_mMax
                    << "   Recompile the kernel with larger GEMX_spmvmVectorBlocks\n";
          ok = false;
        }
        if (p_K > l_kMax) {
          std::cerr << "ERROR: spmv  K dimension " << p_K << " is larger than max supported " << l_kMax
                    << "  Recompile the kernel with larger GEMX_spmvkVectorBlocks\n";
          ok = false;
        }        
        if (p_M % l_mEdge != 0) {
          std::cerr << "ERROR: spmv  M dimension " << p_M << " must be multiple of "
                    << l_mEdge << "\n";
          ok = false;
        }
        if (p_K % (l_kEdge) != 0) {
          std::cerr << "ERROR: spmv  K dimension " << p_K << " must be multiple of "
                    << l_kEdge << "\n";
          ok = false;
        }        
        return(ok);
      }

    void
    addInstr(
      ProgramType &p_Program,
      unsigned int p_M,
      unsigned int p_K,
      unsigned int p_Nnz,
      MtxFile p_MtxFile,
      std::string p_handleA,
      std::string p_handleB,
      std::string p_handleC,
      bool p_WithGolden
    ) {
    
    // Allocate all pages before getting any address
    bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocD;
    // A, D; Descriptors simply prefix the A body
    const unsigned int l_numDescPages = (GEMX_spmvNumCblocks + SpMatType::t_numDescPerPage - 1) /
                                        SpMatType::t_numDescPerPage;
    unsigned int l_numDescDdrWords = l_numDescPages * SpMatType::t_numDdrWordsPerPage;
    const unsigned int l_numPaddingPages = GEMX_spmvNumCblocks;
    const unsigned int l_numPaddingDdrWords = l_numPaddingPages * SpMatType::t_numDdrWordsPerPage;
    unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA,
                                                l_numDescDdrWords * GEMX_ddrWidth +
                                                p_Nnz * GEMX_ddrWidth / GEMX_spmvWidth +
                                                l_numPaddingDdrWords * GEMX_ddrWidth);
    // B, C
    unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, p_K * 1);
    unsigned int l_pageC = p_Program.allocPages(p_handleC, l_newAllocC, p_M * 1);
    
    // Get addresses where matrices are stored
    SpMatType l_matA(p_M, p_K, p_Nnz, 0, p_Program.getPageAddr(l_pageA));
    MatType l_matB(p_K, 1, 1,       p_Program.getPageAddr(l_pageB));
    MatType l_matC(p_M, 1, 1,       p_Program.getPageAddr(l_pageC));
    
    // Large matrix support
    unsigned int l_Cblocks = (p_M + SpmvType::getRowsInCblock() - 1) / SpmvType::getRowsInCblock();
    
    // Instruction
    SpmvArgsType l_spmvArgs(
        l_pageA, l_pageB, l_pageC,
        p_M, p_K, p_Nnz, l_Cblocks, l_numDescPages
      );
    KargsType l_kargs;
    l_kargs.setSpmvArgs(l_spmvArgs);
    l_kargs.store(p_Program.addInstr(), 0);
    
    if (l_newAllocA) {
      if (p_MtxFile.good()) {
        l_matA.fillFromVector(p_MtxFile.getRows());
      } else {
        l_matA.fillMod(std::numeric_limits<GEMX_dataType>::max());
      }
    }
    if (l_newAllocB) {
      l_matB.fillMod(9);
    }
    
    // Calculate reference C = A * B
    if (p_WithGolden) {
      l_matC.multiplySpmv(l_matA, l_matB);
    }
    std::cout << "Added SPMV " << p_M << "x" << p_K << " Nnz=" << p_Nnz << "  ";
    //std::cout << "DEBUG A:\n" << l_matA << "\n";
  }
  
  void
  show(
      ProgramType &p_Program,
      SpmvArgsType p_SpmvArgs
    ) {
      unsigned int l_M = p_SpmvArgs.m_M,
                   l_K = p_SpmvArgs.m_K,
                   l_Nnz = p_SpmvArgs.m_Nnz,
                   l_Cblocks = p_SpmvArgs.m_Cblocks;
      SpMatType l_matA(l_M, l_K, l_Nnz, l_Cblocks, p_Program.getPageAddr(p_SpmvArgs.m_Aoffset));
      MatType l_matB(l_K, 1,   1,   p_Program.getPageAddr(p_SpmvArgs.m_Boffset));
      MatType l_matC(l_M, 1,   1,   p_Program.getPageAddr(p_SpmvArgs.m_Coffset));
      std::cout << "\n###########  Op Spmv  ###########\n"
        << "  C = A * B  "
        << l_M << "x" << 1 << " = " << l_M << "x" << l_K << " * " << l_K << "x" << 1
        << "  Nnz=" << l_Nnz << "\n"
        << "  A\n" << l_matA << "\n"
        << "  B " << l_matB << "\n"
        << "  C " << l_matC << "\n";
    }
  bool
  compare(
      float p_TolRel, float p_TolAbs, 
      ProgramType &p_Program0, ProgramType &p_Program1,
      SpmvArgsType p_SpmvArgs
    ) {
      unsigned int l_M = p_SpmvArgs.m_M,
                   l_K = p_SpmvArgs.m_K,
                   l_Nnz = p_SpmvArgs.m_Nnz;
      MatType l_matC0(l_M, 1,   1,   p_Program0.getPageAddr(p_SpmvArgs.m_Coffset)),
              l_matC1(l_M, 1,   1,   p_Program1.getPageAddr(p_SpmvArgs.m_Coffset));
      std::cout << "\n###########  Op Spmv  ###########\n"
        << "  C = A * B  "
        << l_M << "x" << 1 << " = " << l_M << "x" << l_K << " * " << l_K << "x" << 1
        << "  Nnz=" << l_Nnz << "\n"
        << "  Comparing ...\n";
      bool ok = l_matC1.cmp(p_TolRel, p_TolAbs, l_matC0);
      std::cout << "Spmv C " << (ok ? "Matches" : "Differs") << "\n";
      return(ok);
    }
      
};


#endif
