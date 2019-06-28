/*
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

*/
#ifndef GEMX_MATRIX_H
#define GEMX_MATRIX_H

#include <iostream>
#include <fstream> 
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
//#include <boost/iostreams/filter/bzip2.hpp>

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
    struct compareCol {
        bool operator() (MtxRow &a, MtxRow &b) {
          if (a.getCol() < b.getCol()) {
            return(true);
          } else if (a.getCol() == b.getCol()) {
            if (a.getRow() < b.getRow()) {
              return(true);
            }
          }
          return(false);
        }
    };

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

/*

classes for Sparse Matrix descriptor with data itself stored in caller's space
 UspMat - for dense sparse matrix multiply
 SpMat - for spmv with no URAM usage
 SpMatUram - for spmv with URAM usage

*/

#if (GEMX_runSpmv==1) && (GEMX_useURAM==0)
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
    static const unsigned int t_ColsInBblock = GEMX_spmvWidth * GEMX_spmvkVectorBlocks * GEMX_ddrWidth;
  private:
    unsigned int m_Rows, m_Cols, m_Nnz, m_Bblocks, m_Cblocks,
                 m_AstartIdx = GEMX_spmvNumCblocks * t_numDescPerPage / t_numSpmvPerPage;
    union {
      Tddr *Ddr;
      TmatD *Mat;
      SpmvAdescType *Desc;
    } m_Addr;
  public:
    SpMat() {}
    SpMat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Nnz, unsigned int p_Bblocks, unsigned int p_Cblocks, Tddr *p_Addr)
      : m_Rows(p_Rows), m_Cols(p_Cols), m_Nnz(p_Nnz), m_Bblocks(p_Bblocks), m_Cblocks(p_Cblocks) {
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
    inline unsigned int bBlocks() {return m_Bblocks;}
    inline unsigned int cBlocks() {return m_Cblocks;}

    inline SpmvAdescType &getDesc(unsigned int p_Cblock) {
        assert(p_Cblock < GEMX_spmvNumCblocks);
        return m_Addr.Desc[p_Cblock];
      }
    inline TmatD &getVal(unsigned int p_Idx) {
        //assert(p_Idx < nnz());
        return m_Addr.Mat[m_AstartIdx + p_Idx];
      }
    void 
    init(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Nnz, unsigned int p_Bblocks, unsigned int p_Cblocks, Tddr *p_Addr){
        m_Rows = p_Rows;
        m_Cols = p_Cols;
        m_Nnz = p_Nnz;
        m_Bblocks = p_Bblocks;
        m_Cblocks = p_Cblocks;
        m_Addr.Ddr = p_Addr;    
    }
    
    void
    fillMod(Tddr p_Value, Tddr p_Max=std::numeric_limits<GEMX_dataType>::max()) {
        std::vector<MtxRow> l_rows;
        unsigned int row = 0, col = 0;
        unsigned int numCols = ( nnz() / rows() > 0 ) ? ( nnz() / rows() ) : 1;  
        unsigned int colStep = cols() / numCols - 1;
        assert(colStep > 0); //check sparsity
        unsigned int addStep = 1;
        for (unsigned int i = 0; i < m_Nnz; ++i) {
          p_Value++;
          p_Value %= p_Max;
          assert(row < rows());
          assert(col < cols());
          MtxRow l_m(p_Value, row, col);
          l_rows.push_back(l_m);
          row++;
          col++;
          if ( cols()>rows() ){
            if (col >= cols() || row >= rows()){
             row = 0;
             col = addStep;
             addStep++;
            }
          } else {
            if ( col >= cols() || row >= rows() ){
             col = 0;
             row = addStep;
             addStep++;
            }
          }
        }  
        fillFromVector(l_rows);
      }
    void
    fillFromVector(std::vector<MtxRow> p_Rows) {
        assert(p_Rows.size() ==  nnz());
        m_Cblocks = (m_Rows + t_RowsInCblock -1 ) / t_RowsInCblock;
        m_Bblocks = (m_Cols + t_ColsInBblock -1 ) / t_ColsInBblock;
        unsigned int l_totalBlocks = m_Bblocks * m_Cblocks;
        std::vector<std::vector<MtxRow>> l_part;
        l_part.resize(l_totalBlocks);
        for (auto & l_row: p_Rows) {
          unsigned int l_Bblock = l_row.getCol() / t_ColsInBblock;
          unsigned int l_Cblock = l_row.getRow() / t_RowsInCblock;
          l_part[l_Bblock*m_Cblocks + l_Cblock].push_back(l_row);
        }
        unsigned int l_startIdx = 0;
        const unsigned int l_spmvAlignNnz = GEMX_spmvWidth;
        const unsigned int l_rowUnits = GEMX_spmvWidth * GEMX_spmvMacGroups;
        const unsigned int l_rowBreak = 16; // this should roughly match the smallest chain of t_FifoDepthDeep
        for (unsigned int l_block = 0; l_block<l_totalBlocks; ++l_block) {
          unsigned int l_nnz = l_part[l_block].size();
          if (l_nnz != 0) {
            //sort l_part[l_block] along row
            //sort(l_part[l_block].begin(), l_part[l_block].end());
            unsigned int l_nnzAligned = l_spmvAlignNnz * ((l_nnz + l_spmvAlignNnz - 1) / l_spmvAlignNnz);
            assert(l_nnzAligned >= l_nnz);
            l_part[l_block].resize(l_nnzAligned);
          }
          l_nnz = l_part[l_block].size();
          //seperate per row unit
          std::array<std::queue<MtxRow>, l_rowUnits> l_rowQueues;
          for (auto & l_row: l_part[l_block]) {
            l_rowQueues[l_row.getRow() % l_rowUnits].push(l_row);
          }
          l_part[l_block].clear();
          //aggregate to max row length
          unsigned int l_doneNnzs=0;
          while (l_doneNnzs < l_nnz) {
            for (unsigned int l_rowUnit=0; l_rowUnit < l_rowUnits; ++l_rowUnit) {
                for (unsigned int i=0; i < l_rowBreak; ++i) {
                  if (l_rowQueues[l_rowUnit].empty()) {
                    break;
                  } else {
                    l_part[l_block].push_back(l_rowQueues[l_rowUnit].front());
                    l_rowQueues[l_rowUnit].pop();
                    l_doneNnzs++;
                  }
                }
            }
          }

          //create block description
          unsigned int i=0;
          for (unsigned int i = 0; i < l_nnz; ++i) {
                MtxRow l_row = l_part[l_block][i];
                Tmat l_m(Tddr(l_row.getVal()), l_row.getRow() % t_RowsInCblock, l_row.getCol() % t_ColsInBblock);

                TmatD l_mD = l_m.getAsAd();
                getVal(l_startIdx + i) = l_mD;

                0 && std::cout << "  DEBUG fillFromVector"
                            << "  l_block=" << l_block
                            << "  i=" << i
                            << "  l_m = " << l_m
                            << "  l_D = " << l_mD
                            << "\n";
          }
          SpmvAdescType l_desc(l_nnz, l_startIdx / t_numSpmvPerPage);
          getDesc(l_block) = l_desc;

          l_startIdx += l_nnz;
          // Align start to 4kB
          while ((l_startIdx % t_numSpmvPerPage) != 0) {
            l_startIdx++;
          }
        }
    }
    
    std::vector<MtxRow>
    getNnzVector() {
        std::vector<MtxRow> l_rows;
          for (unsigned int l_bBlock = 0; l_bBlock < m_Bblocks; ++l_bBlock) {
            for (unsigned int l_cBlock = 0; l_cBlock < m_Cblocks; ++l_cBlock) {
                SpmvAdescType l_desc = getDesc(l_bBlock * m_Cblocks + l_cBlock);
                for (unsigned int i = 0; i < l_desc.getNnz(); ++i) {
                  typename SpMat::SpmvAdType l_Ad = getVal(l_desc.getOffset() * t_numSpmvPerPage + i);
                  typename SpMat::SpmvAType l_A(l_Ad);
                  unsigned int row = l_A.getRow(),
                               col = l_A.getCol();
                  if (l_A.getA() != 0) {
                    MtxRow l_mr(l_A.getA(), l_cBlock * t_RowsInCblock + row, l_bBlock * t_ColsInBblock + col);
                    l_rows.push_back(l_mr);
                  }
                  0 && std::cout << "  DEBUG getNnzVector"
                                << "  l_bBlock=" << l_bBlock
                                << "  l_cBlock=" << l_cBlock
                                << "  i=" << i
                                << "  l_m = " << l_A
                                << "  l_D = " << l_Ad
                                << "\n";
                }
            }
          }
        return(l_rows);
      }
    
    void
    print(std::ostream& os) {
        os << "%%MatrixMarket matrix coordinate real general\n"
           << "% Rows Columns Entries\n";
        os << rows() << "  " << cols() << "  " << nnz() << "\n";
          for (unsigned int l_bBlock = 0; l_bBlock < m_Bblocks; ++l_bBlock) {
            for (unsigned int l_cBlock = 0; l_cBlock < m_Cblocks; ++l_cBlock) {
                SpmvAdescType l_desc = getDesc(l_cBlock);
                for (unsigned int i = 0; i < l_desc.getNnz(); ++i) {
                  typename SpMat::SpmvAdType l_Ad = getVal(l_desc.getOffset() * t_numSpmvPerPage + i);
                  typename SpMat::SpmvAType l_A(l_Ad);
                  unsigned int row = l_A.getRow(),
                               col = l_A.getCol();
                  //os << l_Ad << " Ad\n";
                  //os << l_A << " A\n\n";
                  MtxRow l_mr(l_A.getA(), l_cBlock * t_RowsInCblock + row + 1, l_bBlock * t_ColsInBblock + col + 1);
                  os << l_mr << "\n";
                }
            }
          }
      }
};
template <typename T1, typename T2, typename T3>
std::ostream& operator<<(std::ostream& os, SpMat<T1, T2, T3>& p_Val) {
  p_Val.print(os);
  return(os);
}
#endif


#if (GEMX_runSpmv==1) | (GEMX_runUspmv==1)
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
    MtxFile() {}
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
          std::string l_ext = m_FileName.substr(m_FileName.find_last_of(".") + 1);
          std::transform(l_ext.begin(), l_ext.end(), l_ext.begin(), ::tolower);

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
            //sort(m_Rows.begin(), m_Rows.end());
            // Pad with 0s
            #if GEMX_runSpmv==1
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
            #elif GEMX_runUspmv==1
            while (m_Nnz % GEMX_ddrWidth != 0) {
              std::cout << "INFO: Added padding row to the mtx data\n";
              MtxRow l_row;
              m_Rows.push_back(l_row);
              m_Nnz++;
            }
            // Adjust dimensions - needs to be aligned to both GEMX_spmvWidth and GEMX_ddrWidth
            align (m_M, GEMX_ddrWidth * GEMX_uspmvInterleaves);  // Align for loadC
            //align (m_M, GEMX_ddrWidth);
            assert(m_M % GEMX_ddrWidth == 0);
            align (m_K, GEMX_ddrWidth);  
            std::cout << "INFO: loaded mtx file"
                      << "  M " << rows()
                      << "  K " << cols()
                      << "  Nnz " << nnz()
                      << "\n";
            #endif
          }
        }
      }
    std::vector<MtxRow> &getRows() {return(m_Rows);}
};

#endif

# if GEMX_runUspmv==1

#define PageSize 4096
template <typename t_FloatType, typename t_IdxType, unsigned int t_Stages, unsigned int t_DdrWidth>
class UspMat
{
    public:
        static const unsigned int t_DoubleDdrWidth = t_DdrWidth*2;
        static const unsigned int t_StageBlocks = (t_Stages + t_DoubleDdrWidth -1) / t_DoubleDdrWidth;
        static const unsigned int t_FloatsPerPage = PageSize / sizeof(t_FloatType);
        //static const unsigned int t_DescSize = (((t_StageBlocks * t_DdrWidth * 3) + t_FloatsPerPage -1)/ t_FloatsPerPage) * t_FloatsPerPage;
        static const unsigned int t_DescSize = ((t_StageBlocks * t_DoubleDdrWidth * 3)+1)/2;

    private:
        unsigned int m_NumRuns;
        unsigned int m_NnzBase=0;
        unsigned int m_Mbase = t_StageBlocks * t_DoubleDdrWidth;
        unsigned int m_PreluBase = t_StageBlocks * t_DdrWidth * 2;
        unsigned int m_AidBase = t_DescSize * 2;
        unsigned int m_AdatBase = t_DescSize;
        union {
            unsigned int *Nnzs;
            t_IdxType *Idx;
            t_FloatType *Dat;
        } m_Ddr;

    public:
        UspMat (unsigned int p_numRuns, t_FloatType *p_ddr) {
            m_NumRuns = p_numRuns;
            m_Ddr.Dat = p_ddr;
        }

        inline unsigned int &getNnzs(unsigned int p_id) {
            return m_Ddr.Nnzs[m_NnzBase + p_id];
        }

        inline t_IdxType &getRows(unsigned int p_id) {
            return m_Ddr.Idx[m_Mbase + p_id];
        }

        inline t_IdxType &getCols(unsigned int p_id) {
            if (p_id == 0) {
        return m_Ddr.Idx[m_Mbase + t_Stages];
            }
            else {
        return m_Ddr.Idx[m_Mbase + p_id-1];
            }
        }

        inline t_IdxType &getIdx(unsigned int p_id) {
            return m_Ddr.Idx[m_AidBase + p_id];
        }
        
        inline t_FloatType &getVal(unsigned int p_id) {
            return m_Ddr.Dat[m_AdatBase + p_id];
        }

        inline t_FloatType &getPrelu(unsigned int p_id) {
            return m_Ddr.Dat[m_PreluBase + p_id];
        }
        
    void
    fillFromVector(std::array<MtxFile, t_Stages> &p_mtxFiles, t_FloatType *p_pRelu) {

        unsigned int l_datOffset = 0;
        unsigned int l_idxOffset = 0;
        for (unsigned int i=0; i<t_Stages; ++i) {
            MtxFile l_mtxFile = p_mtxFiles[i];
            unsigned l_nnzs = l_mtxFile.nnz();
            unsigned l_rows = l_mtxFile.rows();
            unsigned l_cols = l_mtxFile.cols();
            std::vector<MtxRow> l_mtxData = l_mtxFile.getRows();
            sort(l_mtxData.begin(), l_mtxData.end(), MtxRow::compareCol()); 
            for (unsigned int j=0; j<l_nnzs; ++j) {
              MtxRow l_row = l_mtxData[j];
              getIdx(l_idxOffset+j*2)=l_row.getCol();
              getIdx(l_idxOffset+j*2+1)=l_row.getRow();
            }    
            for (unsigned int j=0; j<l_nnzs; ++j) {
              MtxRow l_row = l_mtxData[j];
              getVal(l_datOffset + l_nnzs + j) = l_row.getVal();
            }
            getNnzs(i)=l_nnzs;
            getRows(i)=l_rows;
            getPrelu(i) = p_pRelu[i];
            if (i==0) {
              getCols(i)=l_cols;
            }
            l_datOffset += l_nnzs*2;    
            l_idxOffset += l_nnzs*4;
        }
    } 
    
    void
    fillMod(t_FloatType p_value, unsigned int *p_m, unsigned int *p_k, unsigned int *p_nnz, t_FloatType *p_pRelu){ 
        unsigned int l_datOffset = 0;
        unsigned int l_idxOffset = 0;
        t_FloatType p_Max=std::numeric_limits<GEMX_dataType>::max();
        for (unsigned int i=0; i<t_Stages; ++i) {
            unsigned l_nnzs = p_nnz[i];
            unsigned l_rows = p_m[i];
            unsigned l_cols = p_k[i];
            
            unsigned int numCols = ( l_nnzs / l_rows > 0 ) ? ( l_nnzs / l_rows ) : 1;  
            unsigned int colStep = l_cols / numCols - 1;
            assert(colStep > 0); //check sparsity
            unsigned int addStep = 1;
            unsigned int row = 0, col = 0;
            for (unsigned int j=0; j<l_nnzs; ++j) {
              p_value += 0.3;
              if (p_value > p_Max) {
                p_value -= p_Max;
              }
              getIdx(l_idxOffset+j*2) = col;
              getIdx(l_idxOffset+j*2+1) = row;
              row++;
              col++;
              if ( l_cols>l_rows ){
                if (col >= l_cols || row >= l_rows){
                  row = 0;
                  col = addStep;
                  addStep++;
                }
              } else {
                if ( col >= l_cols || row >= l_rows ){
                  col = 0;
                  row = addStep;
                  addStep++;
                }
              }
              getVal(l_datOffset + l_nnzs + j) = p_value;
            }
            getNnzs(i)=l_nnzs;
            getRows(i)=l_rows;
            getPrelu(i) = p_pRelu[i];
            if (i==0) {
              getCols(i)=l_cols;
            }
            l_datOffset += l_nnzs*2;    
            l_idxOffset += l_nnzs*4;
        }
      
    }
    
    void
    print(std::ostream& os) {
        os << "layers: " << t_Stages << "\n";
        unsigned int l_valBase = 0;
        for (unsigned int i=0; i<t_Stages; ++i) {
            unsigned int l_nnzs = getNnzs(i);
            os << "layer " << i << " matrix\n";
            os << "%%MatrixMarket matrix coordinate real general\n"
               << "% Rows Columns Entries\n";
            os << getRows(i) << " " << getCols(i) << " " << l_nnzs << "\n";
            for (unsigned int j=0; j<l_nnzs; ++j) {
        MtxRow l_mr(getVal(l_valBase+l_nnzs+j), getIdx(l_valBase*2 + j*2+1), getIdx(l_valBase*2 + j*2));
        os << l_mr << "\n";
            }
            l_valBase += l_nnzs*2;        
        }
    }
};
template <typename T1, typename T2, unsigned int T3, unsigned int T4>
std::ostream& operator<<(std::ostream& os, UspMat<T1, T2, T3, T4>& p_val) {
    p_val.print(os);
    return(os);
}

#endif

#if (GEMX_runSpmv==1) && (GEMX_useURAM==1)
// Sparse matrix descriptor with data itself stored in caller's space for using URAM
template < typename Tdata,  typename Tidx>
class SpMatUram
{
  private:
    unsigned int m_Rows, m_Cols, m_Nnz;
             Tdata *m_DataAddr;
             Tidx  *m_IdxAddr;
  public:
        static const unsigned int t_NumData = (sizeof(GEMX_idxType)*2/sizeof(GEMX_dataType))*GEMX_ddrWidth+GEMX_ddrWidth;
        static const unsigned int t_NumIdx = (sizeof(GEMX_idxType)*2/sizeof(GEMX_dataType)+1)*sizeof(GEMX_dataType)*GEMX_ddrWidth / sizeof(GEMX_idxType);
        static const unsigned int t_NumUramPerDdr = GEMX_ddrWidth / (8/sizeof(Tdata)); 
  public:
    SpMatUram(){}
    SpMatUram(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Nnz, Tdata *p_DataAddr)
      : m_Rows(p_Rows), m_Cols(p_Cols), m_Nnz(p_Nnz), m_DataAddr(p_DataAddr), m_IdxAddr((Tidx*)(p_DataAddr+GEMX_ddrWidth)) {
                  
      }
    SpMatUram& operator=(const SpMatUram& p_Src) {
        assert(p_Src.rows() == rows());
        assert(p_Src.cols() == cols());
        for (unsigned int i = 0; i < m_Nnz; ++i) {
          getVal(i) = p_Src.getVal(i);
          getCol(i) = p_Src.getCol(i);
          getRow(i) = p_Src.getRow(i);
        }
        return *this;
      }
    inline unsigned int rows() {return m_Rows;}
    inline unsigned int cols() {return m_Cols;}
    inline unsigned int nnz() {return m_Nnz;}
    inline Tdata &getVal(unsigned int p_id) {return m_DataAddr[(p_id/GEMX_ddrWidth)*t_NumData+(p_id%GEMX_ddrWidth)];}
    inline Tidx &getCol(unsigned int p_id) {return m_IdxAddr[(p_id/GEMX_ddrWidth)*t_NumIdx + (p_id % GEMX_ddrWidth)*2];}
    inline Tidx &getRow(unsigned int p_id) {return m_IdxAddr[(p_id/GEMX_ddrWidth)*t_NumIdx + (p_id % GEMX_ddrWidth)*2+1];}

    void 
    init(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Nnz, Tdata *p_DataAddr){
        m_Rows = p_Rows;
        m_Cols = p_Cols;
        m_Nnz = p_Nnz;
        m_DataAddr = p_DataAddr;
        m_IdxAddr = (Tidx*) (p_DataAddr+GEMX_ddrWidth);
    }
    
    void
    fillMod(Tdata p_Value, Tdata p_Max=std::numeric_limits<GEMX_dataType>::max()) {
        std::vector<MtxRow> l_rows;
        unsigned int row = 0, col = 0;
        unsigned int numCols = ( nnz() / rows() > 0 ) ? ( nnz() / rows() ) : 1;  
        unsigned int colStep = cols() / numCols - 1;
        assert(colStep > 0); //check sparsity
        unsigned int addStep = 1;
        for (unsigned int i = 0; i < m_Nnz; ++i) {
          p_Value++;
          p_Value %= p_Max;
          assert(row < rows());
          assert(col < cols());
          MtxRow l_m(p_Value, row, col);
          l_rows.push_back(l_m);
          row++;
          col++;
          if ( cols()>rows() ){
            if (col >= cols() || row >= rows()){
             row = 0;
             col = addStep;
             addStep++;
            }
          } else {
            if ( col >= cols() || row >= rows() ){
             col = 0;
             row = addStep;
             addStep++;
            }
          }
        }  
        fillFromVector(l_rows);
      }
    void
    fillFromVector(std::vector<MtxRow> p_Rows) {
      assert(p_Rows.size() ==  nnz());
      for (unsigned int i = 0; i < m_Nnz; ++i) {
        MtxRow l_row = p_Rows[i];
        getVal(i) = l_row.getVal();
        getRow(i) = l_row.getRow();        
        getCol(i) = l_row.getCol();
      }
    }
    
    void
    fillFromVectorWithReorder(std::vector<MtxRow> p_Rows) {
            assert(p_Rows.size() ==  nnz());
        unsigned int i=0;
        unsigned int l_blocks = m_Nnz / (t_NumUramPerDdr * t_NumUramPerDdr);
        for (unsigned int c = 0; c < t_NumUramPerDdr; ++c) {
            for (unsigned int b = 0; b < l_blocks; ++b) {
                for (unsigned int r = 0; r < t_NumUramPerDdr; ++r) {
                     MtxRow l_row = p_Rows[i];
                     i++;
                     getVal(b*t_NumUramPerDdr*t_NumUramPerDdr+r*t_NumUramPerDdr+c) = l_row.getVal();
                     getRow(b*t_NumUramPerDdr*t_NumUramPerDdr+r*t_NumUramPerDdr+c) = l_row.getRow();
                     getCol(b*t_NumUramPerDdr*t_NumUramPerDdr+r*t_NumUramPerDdr+c) = l_row.getCol();
                }
            }        
            }
    }

    std::vector<MtxRow>
    getNnzVector() {
        std::vector<MtxRow> l_rows;
        for (unsigned int i = 0; i < m_Nnz; ++i) {
              MtxRow l_mr(getVal(i), getRow(i), getCol(i));
              l_rows.push_back(l_mr);
        }
        return(l_rows);
      }
    
    void
    print(std::ostream& os) {
        os << "%%MatrixMarket matrix coordinate real general\n"
           << "% Rows Columns Entries\n";
        os << rows() << "  " << cols() << "  " << nnz() << "\n";
        for (unsigned int i = 0; i < m_Nnz; ++i) {
            MtxRow l_mr(getVal(i), getRow(i), getCol(i));
            os << l_mr << "\n";
      }
};
};
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, SpMatUram<T1, T2>& p_Val) {
  p_Val.print(os);
  return(os);
}

class MtxFileUram
{
  public:
  
  private:
    std::string m_FileName;
    bool m_Good;
    bool m_isDiag;
    unsigned int m_M, m_K, m_Nnz;
    std::vector<MtxRow> m_Rows;
  private:
    void align( unsigned int &dst, unsigned int width) {dst = width * ((dst + width - 1) / width);}
  public:
    bool good() {return(m_Good);}
    bool isDiag() {return(m_isDiag);}
    unsigned int rows() {return(m_M);}
    unsigned int cols() {return(m_K);}
    unsigned int nnz() {return(m_Nnz);}
    std::string fileName() {return(m_FileName);}
    MtxFileUram(std::string p_FileName)
      : m_Good(false),
        m_isDiag(true),
        m_M(0), m_K(0), m_Nnz(0), 
        m_FileName(p_FileName)
      {
        if (m_FileName != "none") {
          std::cout << "INFO: loading Mtx file  " << m_FileName << "\n";
          std::ifstream l_fs(m_FileName.c_str(), std::ios_base::in | std::ios_base::binary);
          
          boost::iostreams::filtering_istream l_bs;
          std::string l_ext = m_FileName.substr(m_FileName.find_last_of(".") + 1);
          std::transform(l_ext.begin(), l_ext.end(), l_ext.begin(), ::tolower);
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
          unsigned int l_curRow=0;
          unsigned int l_curCol=0;
          if (m_Good) {
            while (l_bs.peek() == '%') l_bs.ignore(2048, '\n');
            l_bs >>  m_M >> m_K >> m_Nnz;
            for (unsigned int i = 0; i < nnz(); ++i) {
                MtxRow l_row;
                l_row.scan(l_bs);
                if (i==0) {
                    l_curRow = l_row.getRow();
                    l_curCol = l_row.getCol();
                } else {
                    if ((l_row.getRow() != (l_curRow+1)) || (l_row.getCol() != (l_curCol+1))){
                        m_isDiag = false;
                    }
                l_curRow = l_row.getRow();
                l_curCol = l_row.getCol();
                }
                m_Rows.push_back(l_row);
            }
            boost::iostreams::close(l_bs);
            // Sort to make canonical
            //sort(m_Rows.begin(), m_Rows.end());
            // Pad with 0s
            while (m_Nnz % (GEMX_ddrWidth * GEMX_nnzBlocks) != 0) {
                std::cout << "INFO: Added padding row to the mtx data\n";
                MtxRow l_row;
                m_Rows.push_back(l_row);
                m_Nnz++;
            }
            // Adjust dimensions - needs to be aligned to both GEMX_ddrWidth
            align (m_M, GEMX_ddrWidth * GEMX_spmvUramGroups);  // Align for loadC
            assert(m_M % GEMX_ddrWidth == 0);
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

#endif

// Regula Matrix descriptor with data itself stored in caller's space
template < typename T>
class DenseMat
{
  private:
    unsigned int m_Rows, m_Cols, m_Ld; 
    T *m_Addr;
  public:
        DenseMat()
        {}
    DenseMat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld, T *p_Addr)
      : m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_Addr(p_Addr)
      {}
    DenseMat& operator=(const DenseMat& p_Src) {
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
      
   void fillModFromFile(std::string l_fs){
     std::ifstream inputFile(l_fs);
     T l_val;
     if (inputFile.good()) {
        for (unsigned int row=0; row < m_Rows; ++row) {
          for (unsigned int col=0; col < ld(); ++col) {
             inputFile >> l_val;
             getVal(row,col) = l_val;

        }
      }
     }else{
       std::cout<<"no file no loading!\n";
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
    #if GEMX_runTransp==1
    //Golden comparsion functions for transp engine
    void
    transpose(DenseMat & p_A) {
        for (unsigned int row = 0; row < rows(); ++row) {
          for (unsigned int col = 0; col < cols() ; ++col) {
            getVal(row, col) = p_A.getVal(col, row);
          }
        }
        std::swap(m_Rows, m_Cols);
    }
    void
    transposeGva(DenseMat & p_A, unsigned int p_rowEdgeWidth, unsigned int p_colEdgeWidth) {
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
    #endif
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
    cmp(float p_TolRel, float p_TolAbs, DenseMat &p_Ref) {
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
std::ostream& operator<<(std::ostream& os, DenseMat<T1>& p_Val) {
  p_Val.print(os);
  return(os);
}

/*Type define and float specialization*/

#if GEMX_runSpmv==1

  #if GEMX_useURAM==0
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

  typedef SpmvType::SpmvAdType SpmvAdType;
  typedef SpmvType::SpmvAType SpmvAType;
  typedef SpMat<GEMX_dataType, SpmvAdType, SpmvAType > SpMatType;
  #else
  typedef SpMatUram<float, GEMX_idxType> SpMatType_ForFloat;
  typedef SpMatUram<GEMX_dataType, GEMX_idxType > SpMatType;
  #endif
  
  template<>
  void
  SpMatType_ForFloat::fillMod(float p_Value, float p_Max) {
    std::vector<MtxRow> l_rows;
    unsigned int row = 0, col = 0;
    unsigned int numCols = ( nnz() / rows() > 0 ) ? ( nnz() / rows() ) : 1;  
    unsigned int colStep = cols() / numCols - 1;
    assert(colStep > 0); //check sparsity
    unsigned int addStep = 1;
    for (unsigned int i = 0; i < m_Nnz; ++i) {
        p_Value += 0.3;
        if (p_Value > p_Max) {
          p_Value -= p_Max;
        }
        assert(row < rows());
        assert(col < cols());
        MtxRow l_m(p_Value, row, col);
        l_rows.push_back(l_m);
        row++;
        col++;
        if ( cols()>rows() ){
          if (col >= cols() || row >= rows()){
            row = 0;
            col = addStep;
            addStep++;
          }
        } else {
          if ( col >= cols() || row >= rows() ){
            col = 0;
            row = addStep;
            addStep++;
          }
        }
      }  
    fillFromVector(l_rows);
  }

#endif


typedef DenseMat<float> MatType_ForFloat;

typedef DenseMat<GEMX_dataType> MatType;

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

#endif
