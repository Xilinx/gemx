/**********
* Copyright (c) 2017-2019, Xilinx, Inc.
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
#ifndef _SPMV_BRAM_HOST_H_
#define _SPMV_BRAM_HOST_H_

#include "xhost.h"
#include "gemm_host.h"
#include "gemx_util.h"

#include <queue>

using namespace std;
namespace gemx {
  
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
}; 

template <typename t_FloatType>
class SpmvAd {
  public:
  private:
    t_FloatType  m_ValA;
    unsigned short m_Col;
    unsigned short m_Row;
    static const unsigned int t_4k = 4096; 
  public:
    static const unsigned int t_per4k = t_4k / (4 + 2 + 2); 
  public:
    SpmvAd() {}
    SpmvAd(t_FloatType p_A, unsigned int p_Col, unsigned int p_Row)
     : m_ValA(p_A), m_Col(p_Col), m_Row(p_Row)
     {}
    unsigned int getCol() {return m_Col;}
    unsigned int getRow() {return m_Row;}
    t_FloatType getA() {return m_ValA;}
};

// Sparse matrix descriptor with data itself stored in caller's space
template < typename Tddr, typename TmatD>
class SpMat
{
  public:
    typedef SpmvAdesc SpmvAdescType;
    static const unsigned int t_numSpmvPerPage = TmatD::t_per4k;
    static const unsigned int t_numDescPerPage = SpmvAdescType::t_per4k;
  private:
    unsigned int m_Rows, m_Cols, m_Nnz, m_Bblocks, m_Cblocks,
                 m_AstartIdx = 1024 * t_numDescPerPage / t_numSpmvPerPage;
    union {
      Tddr *Ddr;
      TmatD *Mat;
      SpmvAdescType *Desc;
    } m_Addr;
  public:
    SpMat(){}
    SpMat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Nnz, unsigned int p_Bblocks, unsigned int p_Cblocks, Tddr *p_Addr)
      : m_Rows(p_Rows), m_Cols(p_Cols), m_Nnz(p_Nnz), m_Bblocks(p_Bblocks), m_Cblocks(p_Cblocks) {
        m_Addr.Ddr = p_Addr;
      }
      
    inline SpmvAdescType &getDesc(unsigned int p_Cblock) {
        return m_Addr.Desc[p_Cblock];
      }
    inline TmatD &getVal(unsigned int p_Idx) {
        return m_Addr.Mat[m_AstartIdx + p_Idx];
      }

    void
    fillFromVector(vector<MtxRow> p_Rows, unsigned int t_RowsInCblock, unsigned int t_ColsInBblock, unsigned int spmv_width) {
          unsigned int l_totalBlocks = m_Bblocks * m_Cblocks;
          vector<vector<MtxRow>> l_part;
          l_part.resize(l_totalBlocks);
          for (auto & l_row: p_Rows) {
              unsigned int l_Bblock = l_row.getCol() / t_ColsInBblock;
              unsigned int l_Cblock = l_row.getRow() / t_RowsInCblock;
              l_part[l_Bblock*m_Cblocks + l_Cblock].push_back(l_row);
          }
          unsigned int l_startIdx = 0;
          unsigned int l_spmvAlignNnz = spmv_width;
          const unsigned int l_rowUnits = 8 * 12; //GEMX_spmvMacGroups
          const unsigned int l_rowBreak = 16; // this should roughly match the smallest chain of t_FifoDepthDeep
          for (unsigned int l_block = 0; l_block<l_totalBlocks; ++l_block) {
              unsigned int l_nnz = l_part[l_block].size();
              if (l_nnz != 0) {
                unsigned int l_nnzAligned = l_spmvAlignNnz * ((l_nnz + l_spmvAlignNnz - 1) / l_spmvAlignNnz);
                assert(l_nnzAligned >= l_nnz);
                l_part[l_block].resize(l_nnzAligned);
              }
              l_nnz = l_part[l_block].size();
              //seperate per row unit
              array<queue<MtxRow>, l_rowUnits> l_rowQueues;
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
              const int t_ShortIdxMask = (1 << 16) - 1;
              const int t_ColAddIdxBits = 2; //GEMX_spmvColAddIdxBits=2
              for (unsigned int i = 0; i < l_nnz; ++i) {
                    MtxRow l_mtx_row = l_part[l_block][i];
                    unsigned int l_row = l_mtx_row.getRow() % t_RowsInCblock;
                    unsigned int l_col = l_mtx_row.getCol() % t_ColsInBblock;
                    TmatD l_mD(Tddr(l_mtx_row.getVal()), l_col & t_ShortIdxMask, l_row | ((l_col & ~t_ShortIdxMask) >> t_ColAddIdxBits));
                    getVal(l_startIdx + i) = l_mD;    
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
};
  
class SpmvArgsBram: public kArgs {
public:
    virtual ~SpmvArgsBram() {
    }
    SpmvArgsBram() = delete;
    SpmvArgsBram ( unsigned int p_Aoffset, unsigned int p_Boffset, unsigned int p_Coffset, unsigned int M, unsigned int K, unsigned int Nnz, unsigned int p_Bblocks, unsigned int p_Cblocks, unsigned int p_DescPages) :
        m_spmv_args( { int(OpSpmv), p_Aoffset, p_Boffset, p_Coffset, M, K, Nnz, p_Bblocks, p_Cblocks, p_DescPages, 0, 0, 0, 0, 0, 0} ){
    }

    size_t sizeInBytes() {
        return sizeof(m_spmv_args);
    }
    char *asByteArray() {
        return reinterpret_cast<char*>(&m_spmv_args);
    }
protected:
    struct {
        int m_optype;
        unsigned int m_Aoffset, m_Boffset, m_Coffset, m_M, m_K, m_Nnz, m_Bblocks, m_Cblocks, m_DescPages;
        unsigned int dummy[6];
    } m_spmv_args;
};

template<typename HType>
class SPMVBRAMHost : public GEMMHost<HType> {
public:
    SPMVBRAMHost() = delete;
    virtual ~SPMVBRAMHost(){
    }

    SPMVBRAMHost(const SPMVBRAMHost<HType> &) = delete;

    SPMVBRAMHost(const string & xclbin, const string & kernelName, const unsigned ddrBank, const string & device) : GEMMHost<HType> ( xclbin, kernelName, ddrBank, device)
    {
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift) {
        cerr << "GEMM operation not supported" << endl;
        return false;
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType & C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, unsigned int lda, unsigned int ldb, unsigned int ldc, unsigned int ldx, int postScale, int postShift) {
        cerr << "GEMM operation not supported" << endl;
        return false;
    } 
    
    virtual void* SendSpToFpgaFloat(int * row, int * col, float * data, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks){
       vector<MtxRow> l_rows;     
       for(int i = 0; i<nnz; ++i){
         MtxRow l_m(data[i], row[i], col[i]);
         l_rows.push_back(l_m);
       }
       typedef SpmvAd<float> SpmvAdType; 
       unsigned int l_Cblocks = (m + capacity_Cblocks - 1) / capacity_Cblocks;
       unsigned int l_Bblocks = (k + capacity_Bblocks - 1) / capacity_Bblocks;
       unsigned int l_numDescPages = (num_cblocks + SpmvAdesc::t_per4k - 1) / SpmvAdesc::t_per4k; 
       unsigned int l_numDescDdrWords = l_numDescPages * 4096 / sizeof(float) / ddr_width;
       unsigned int l_numPaddingDdrWords = num_cblocks * 4096 / sizeof(float) / ddr_width;
       float *A = new float[l_numDescDdrWords * ddr_width + nnz * ddr_width / spmv_width + l_numPaddingDdrWords * ddr_width];
       SpMat<float,SpmvAdType> MatA(m,k,nnz,l_Bblocks,l_Cblocks,A);
       MatA.fillFromVector(l_rows, capacity_Cblocks, capacity_Bblocks, spmv_width);
       this->SendToFPGA(A, A, (unsigned long long)((l_numDescDdrWords * ddr_width + nnz * ddr_width / spmv_width + l_numPaddingDdrWords * ddr_width)*sizeof(float))); 
       return A;
    }
    
    virtual void* SendSpToFpgaInt(int * row, int * col, float * data, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks){
       vector<MtxRow> l_rows;     
       for(int i = 0; i<nnz; ++i){
         MtxRow l_m(data[i], row[i], col[i]);
         l_rows.push_back(l_m);
       }
       typedef SpmvAd<int> SpmvAdType; 
       unsigned int l_Cblocks = (m + capacity_Cblocks - 1) / capacity_Cblocks;
       unsigned int l_Bblocks = (k + capacity_Bblocks - 1) / capacity_Bblocks;
       unsigned int l_numDescPages = (num_cblocks + SpmvAdesc::t_per4k - 1) / SpmvAdesc::t_per4k; 
       unsigned int l_numDescDdrWords = l_numDescPages * 4096 / sizeof(int) / ddr_width;
       unsigned int l_numPaddingDdrWords = num_cblocks * 4096 / sizeof(int) / ddr_width;
       int *A = new int[l_numDescDdrWords * ddr_width + nnz * ddr_width / spmv_width + l_numPaddingDdrWords * ddr_width];
       SpMat<int,SpmvAdType> MatA(m,k,nnz,l_Bblocks,l_Cblocks,A);
       MatA.fillFromVector(l_rows, capacity_Cblocks, capacity_Bblocks, spmv_width);
       
       this->SendToFPGA(A, A, (unsigned long long)((l_numDescDdrWords * ddr_width + nnz * ddr_width / spmv_width + l_numPaddingDdrWords * ddr_width)*sizeof(int))); 
       return A;
    }
    
    virtual bool AddSPMVOp(const HType & A, const HType & B, const HType & C, unsigned int m, unsigned int k, unsigned int nnz, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks){     
      if (this->_hostMat.find(A) == this->_hostMat.end()
                || this->_hostMat.find(B) == this->_hostMat.end()
                || this->_hostMat.find(C) == this->_hostMat.end()) {
            cerr << "Matrix not found!" << endl;
            return false;
        }
       
       unsigned long long A_off = 0, B_off = 0, C_off = 0;
       xclGetMemObjDeviceAddress(this->_devHandle[A].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &A_off);
       xclGetMemObjDeviceAddress(this->_devHandle[B].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &B_off);
       xclGetMemObjDeviceAddress(this->_devHandle[C].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &C_off);
       //cout << "A_dev_addr: " << A_off << " B_dev_addr: " << B_off << " C_dev_addr: " << C_off << endl;
       
       assert(A_off > this->_ddrDeviceBaseAddr);
       assert(B_off > this->_ddrDeviceBaseAddr);
       assert(C_off > this->_ddrDeviceBaseAddr);
       
       A_off -= this->_ddrDeviceBaseAddr;
       B_off -= this->_ddrDeviceBaseAddr;
       C_off -= this->_ddrDeviceBaseAddr;

       assert(A_off % this->PAGE_SIZE == 0);  
       assert(B_off % this->PAGE_SIZE == 0);
       assert(C_off % this->PAGE_SIZE == 0);

       A_off /= this->PAGE_SIZE;
       B_off /= this->PAGE_SIZE;
       C_off /= this->PAGE_SIZE;
       unsigned int l_numDescPages = (num_cblocks + SpmvAdesc::t_per4k - 1) / SpmvAdesc::t_per4k; 
       unsigned int l_Cblocks = (m + capacity_Cblocks - 1) / capacity_Cblocks;
       unsigned int l_Bblocks = (k + capacity_Bblocks - 1) / capacity_Bblocks;


       SpmvArgsBram args(A_off, B_off, C_off, m, k, nnz, l_Bblocks, l_Cblocks, l_numDescPages);
       this->AddInstr (&args);  
       return true;
    }
       
};

}


#endif
