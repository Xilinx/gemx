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
#ifndef _GEMX_UTIL_H
#define _GEMX_UTIL_H

#include <chrono>
#include <assert.h>
#include <iomanip>
#include <sstream>
#include <thread>
#include <queue>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

using namespace std;

namespace gemx{

    class XTimer
    {
        public:
            XTimer() : beg_(clock_::now()) {}
            void reset() { beg_ = clock_::now(); }
            double elapsed() const {
                return chrono::duration_cast<second_>
                    (clock_::now() - beg_).count(); }

        private:
            typedef chrono::high_resolution_clock clock_;
            typedef chrono::duration<double, ratio<1> > second_;
            chrono::time_point<clock_> beg_;
    };


    // Matrix descriptor with data itself stored in caller's space
    template<typename T>
        class Mat 
        {
            private:
                unsigned int m_Rows, m_Cols, m_Ld, m_buf_sz;
                bool m_ownmem;
                T *m_Addr;

            public:
                const static size_t GEMX_CMP_WIDTH = 11;
                Mat() = delete;
                ~Mat() {
                    if (m_ownmem && m_Addr) {
                        free(m_Addr);
                    }
                }

                Mat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld) :
                    m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_ownmem(true) {
                        m_buf_sz = sizeof(T) * p_Rows * p_Ld;
                        posix_memalign((void**) &m_Addr, 4096, m_buf_sz);
                    }

                Mat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld, T *p_Addr) :
                    m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_Addr(p_Addr), m_ownmem(false) {
                        m_buf_sz = sizeof(T) * p_Rows * p_Ld;
                    }

                Mat& operator=(const Mat& p_Src) {
                    assert(p_Src.rows() == rows());
                    assert(p_Src.cols() == cols());
                    for (unsigned int row = 0; row < m_Rows; ++row) {
                        for (unsigned int col = 0; col < m_Ld; ++col) {
                            m_Addr[row][col] = p_Src.getVal(row, col);
                        }
                    }
                    return *this;
                }

                unsigned int buf_sz(){
                    return m_buf_sz;
                }

                T*& data() {
                    return m_Addr;
                }

                inline T &getVal(unsigned int p_Row, unsigned int p_Col) {
                    return m_Addr[p_Row * ld() + p_Col];
                }
                inline unsigned int rows() {
                    return m_Rows;
                }
                inline unsigned int cols() {
                    return m_Cols;
                }
                inline unsigned int ld() {
                    return m_Ld;
                }

                void init(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld, T *p_Addr) {
                    m_Rows = p_Rows;
                    m_Cols = p_Cols;
                    m_Ld = p_Ld;
                    m_Addr = p_Addr;
                }

                void fillModRange(T p_Min, T p_Max) {
                    T l_val = p_Min;
                    for (unsigned int row = 0; row < m_Rows; ++row) {
                        for (unsigned int col = 0; col < ld(); ++col) {
                            getVal(row, col) = l_val++;
                            if ( l_val > p_Max ) l_val = p_Min; 
                        }
                    }
                }
                void fill(T p_val) {
                    for (unsigned int row = 0; row < m_Rows; ++row) {
                        for (unsigned int col = 0; col < ld(); ++col) {
                            getVal(row, col) = p_val;
                        }
                    }
                }

                void fillMod(T p_Max, T p_First = 0) {
                    T l_val = p_First;
                    for (unsigned int row = 0; row < m_Rows; ++row) {
                        for (unsigned int col = 0; col < ld(); ++col) {
                            getVal(row, col) = l_val;
                            l_val++;
                            l_val %= p_Max;
                        }
                    }
                }

                void multiply(Mat & p_A, Mat & p_B) {
                    assert(p_A.rows() == rows());
                    assert(p_A.cols() == p_B.rows());
                    assert(p_B.cols() == cols());
                    for (unsigned int row = 0; row < rows(); ++row) {
                        for (unsigned int col = 0; col < cols(); ++col) {
                            int64_t l_val = 0;
                            for (unsigned int k = 0; k < p_A.cols(); ++k) {
                                l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
                            }
                            getVal(row, col) = (T)l_val;
                        }
                    }
                }

                void multiplyAddScale(Mat & p_A, Mat & p_B,  Mat<int> & p_X, int postScaleVal, int postScaleShift) {
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
                            }
                            l_val += p_X.getVal(row, col);
                            l_val = (l_val >> postScaleShift ) * postScaleVal;
                            getVal(row, col) = (T)(l_val);
                        }
                    }
                }

                void matMultWithScaleAndPRelu(Mat & p_A, Mat & p_B, Mat<int> & p_X,  int32_t p_postScale, int16_t p_PReluVal) {
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
                            }
                            l_val += p_X.getVal(row,col);
                            unsigned int l_psShift = p_postScale & 0x00ff;
                            unsigned int l_psVal = p_postScale >> 8;
                            l_val = (l_val >> l_psShift) * l_psVal;
                            T l_entry = (T)(l_val);
                            if (l_entry < 0) {
                                l_entry = (l_entry  >> (p_PReluVal & 0x003f))* (T)(p_PReluVal >> 6);
                            }
                            getVal(row, col) = l_entry;
                        }
                    }
                }

                bool cmp(float p_TolRel, float p_TolAbs, Mat &p_Ref) {
                    bool ok = true;
                    unsigned int l_verbose = 1; // 0 none, 1 if not exactly equal, 2 if passed tolerance, 3 show all
                    unsigned int l_numExactMatches = 0, l_numMismatches = 0;
                    for (unsigned int row = 0; row < rows(); ++row) {
                        for (unsigned int col = 0; col < cols(); ++col) {
                            string l_Prefix = "      row " + to_string(row) + " col " + to_string(col);
                            T v = getVal(row, col);
                            T vRef = p_Ref.getVal(row, col);
                            bool l_exactMatch = false;
                            bool l_ok = cmpVal(p_TolRel, p_TolAbs, vRef, v, l_Prefix, l_exactMatch, 1);
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
                    cout << "  Compared " << l_total << " values:" << "  exact match "
                        << l_numExactMatches << "  within tolerance "
                        << l_withinTolerance << "  mismatch " << l_numMismatches
                        << "\n";
                    return (ok);
                }

                bool cmpVal(float p_TolRel, float p_TolAbs, T vRef, T v, string p_Prefix, bool &p_exactMatch, unsigned int p_Verbose) {
                    float l_diffAbs = max( v - vRef, vRef-v);
                    float l_diffRel = l_diffAbs;
                    if (vRef > 0) {
                        l_diffRel /= vRef;
                    }
                    if (vRef < 0) {
                        l_diffRel /= (-vRef);
                    }

                    p_exactMatch = (vRef == v);
                    bool l_status = p_exactMatch || (l_diffRel <= p_TolRel) || (l_diffAbs <= p_TolAbs);
                    if ((p_Verbose >= 3) || ((p_Verbose >= 2) && !p_exactMatch) || ((p_Verbose >= 1) && !l_status)) {
                        cout << p_Prefix << "  ValRef " << left
                            << setw(GEMX_CMP_WIDTH) << vRef << " Val " << left
                            << setw(GEMX_CMP_WIDTH) << v << "  DifRel "
                            << left << setw(GEMX_CMP_WIDTH) << l_diffRel
                            << " DifAbs " << left << setw(GEMX_CMP_WIDTH)
                            << l_diffAbs << "  Status " << l_status << "\n";
                    }
                    return (l_status);
                }

        };


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
                scan(istream& p_Is) {
                    p_Is >>  m_Row >> m_Col >> m_Val;
                    if ((m_Row <= 0) || (m_Col <= 0))  {
                        cerr << "  Error: invalid MTX file line row=" << m_Row
                            << " col=" << m_Col << " val=" << m_Val << "\n";
                        assert(0);
                    }
                    // Indices start from 1 in MTX; 0 locally
                    m_Row--;
                    m_Col--;
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




    };

    class MtxFile
    {
        public:

        private:
            string m_FileName;
            bool m_Good;
            unsigned int m_M, m_K, m_Nnz;
            vector<MtxRow> m_Rows;
        private:
            void align( unsigned int &dst, unsigned int width) {dst = width * ((dst + width - 1) / width);}
        public:
            MtxFile() {}
            bool good() {return(m_Good);}
            unsigned int rows() {return(m_M);}
            unsigned int cols() {return(m_K);}
            unsigned int nnz() {return(m_Nnz);}
            string fileName() {return(m_FileName);}
            MtxFile(string p_FileName)
                : m_FileName(p_FileName) ,
                m_Good(false),
                m_M(0), m_K(0), m_Nnz(0) 
        {
            if (m_FileName != "none") {
                cout << "INFO: loading Mtx file  " << m_FileName << "\n";
                ifstream l_fs(m_FileName.c_str(), ios_base::in | ios_base::binary);

                boost::iostreams::filtering_istream l_bs;
                string l_ext = m_FileName.substr(m_FileName.find_last_of(".") + 1);
                transform(l_ext.begin(), l_ext.end(), l_ext.begin(), ::tolower);

                if (l_ext == "gz") {
                    l_bs.push(boost::iostreams::gzip_decompressor());
                } else if (l_ext == "mtx") {
                    // noop
                } else{
                    cerr << "ERROR: MtxFile failed due to unknown extension \"" << l_ext << "\", file  "
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
#if GEMX_runSpmv==1
                    while (m_Nnz % GEMX_spmvWidth != 0) {
                        cout << "INFO: Added padding row to the mtx data\n";
                        MtxRow l_row;
                        m_Rows.push_back(l_row);
                        m_Nnz++;
                    }
                    // Adjust dimensions - needs to be aligned to both GEMX_spmvWidth and GEMX_ddrWidth
                    align (m_M, GEMX_spmvWidth * GEMX_spmvMacGroups);  // Align for loadC
                    assert(m_M % GEMX_spmvWidth == 0);
                    align (m_K, GEMX_ddrWidth);  
                    cout << "INFO: loaded mtx file"
                        << "  M " << rows()
                        << "  K " << cols()
                        << "  Nnz " << nnz()
                        << "\n";
#elif GEMX_runUspmv==1
                    while (m_Nnz % GEMX_ddrWidth != 0) {
                        cout << "INFO: Added padding row to the mtx data\n";
                        MtxRow l_row;
                        m_Rows.push_back(l_row);
                        m_Nnz++;
                    }
                    // Adjust dimensions - needs to be aligned to both GEMX_spmvWidth and GEMX_ddrWidth
                    align (m_M, GEMX_ddrWidth * GEMX_uspmvInterleaves);  // Align for loadC
                    //align (m_M, GEMX_ddrWidth);
                    assert(m_M % GEMX_ddrWidth == 0);
                    align (m_K, GEMX_ddrWidth);  
                    cout << "INFO: loaded mtx file"
                        << "  M " << rows()
                        << "  K " << cols()
                        << "  Nnz " << nnz()
                        << "\n";
#endif
                }
            }
        }
            vector<MtxRow> &getRows() {return(m_Rows);}
            void setIndex(unsigned int p_m, unsigned int p_k, unsigned int p_nnz){
                m_M=p_m;
                m_K=p_k;
                m_Nnz=p_nnz;    
            }
    };  


    template <typename t_FloatType, typename t_IdxType>
        class UspMat
        {
            private:
                unsigned int m_NnzBase=0;
                unsigned int m_Mbase;
                unsigned int m_PreluBase;
                unsigned int m_AidBase;
                unsigned int m_Abase;
                unsigned int m_AdatBase;
                unsigned int m_Stages;
                union {
                    unsigned int *Nnzs;
                    t_IdxType *Idx;
                    t_FloatType *Dat;
                } m_Ddr;

            public:
                UspMat (t_FloatType *p_ddr, unsigned int t_DdrWidth, unsigned int t_Stages) {
                    unsigned int t_DoubleDdrWidth = t_DdrWidth*2;
                    unsigned int t_StageBlocks = (t_Stages + t_DoubleDdrWidth -1) / t_DoubleDdrWidth;
                    unsigned int t_DescSize = ((t_StageBlocks * t_DoubleDdrWidth * 3)+1)/2;
                    m_Mbase = t_StageBlocks * t_DoubleDdrWidth;
                    m_PreluBase = t_StageBlocks * t_DdrWidth * 2;
                    m_AidBase = t_DescSize * 2;
                    m_AdatBase = t_DescSize;
                    m_Stages = t_Stages;
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
                        return m_Ddr.Idx[m_Mbase + m_Stages];
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

                void fillFromVector(uint16_t* row, uint16_t* col, float* data, int* row_size, int* col_size, int* nnz_size, float* p_pRelu) {
                    unsigned int l_datOffset = 0;
                    unsigned int l_idxOffset = 0;
                    unsigned int l_array_offset = 0;
                    for (unsigned int i=0; i<m_Stages; ++i) {
                        unsigned l_nnzs = nnz_size[i];
                        unsigned l_rows = row_size[i];
                        unsigned l_cols = col_size[i];
                        for (unsigned int j=0; j<l_nnzs; ++j) {
                            getIdx(l_idxOffset+j*2)=col[l_array_offset+j];
                            getIdx(l_idxOffset+j*2+1)=row[l_array_offset+j];
                        }
                        for (unsigned int j=0; j<l_nnzs; ++j) {
                            getVal(l_datOffset + l_nnzs + j) =  data[l_array_offset+j];
                        }
                        getNnzs(i)=nnz_size[i];
                        getRows(i)=row_size[i];
                        getPrelu(i) = p_pRelu[i];
                        if (i==0) {
                            getCols(i)=col_size[i];
                        }
                        l_datOffset += l_nnzs*2;
                        l_idxOffset += l_nnzs*4;
                        l_array_offset = l_array_offset + l_nnzs;
                    }
                }
                void fillFromMtxFile(vector<MtxFile> &p_mtxFiles, t_FloatType *p_pRelu) {

                    unsigned int l_datOffset = 0;
                    unsigned int l_idxOffset = 0;
                    for (unsigned int i=0; i<m_Stages; ++i) {
                        MtxFile l_mtxFile = p_mtxFiles[i];
                        unsigned l_nnzs = l_mtxFile.nnz();
                        unsigned l_rows = l_mtxFile.rows();
                        unsigned l_cols = l_mtxFile.cols();
                        vector<MtxRow> l_mtxData = l_mtxFile.getRows();
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

                inline unsigned int rows() {return m_Rows;}
                inline unsigned int cols() {return m_Cols;}
                inline unsigned int nnz() {return m_Nnz;}
                inline unsigned int bBlocks() {return m_Bblocks;}
                inline unsigned int cBlocks() {return m_Cblocks;}


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

                vector<MtxRow> fillMod(float p_Value, unsigned int t_RowsInCblock, unsigned int t_ColsInBblock, unsigned int spmv_width) {
                    vector<MtxRow> l_rows;
                    unsigned int row = 0, col = 0;
                    unsigned int numCols = ( nnz() / rows() > 0 ) ? ( nnz() / rows() ) : 1;  
                    unsigned int colStep = cols() / numCols - 1;
                    assert(colStep > 0); //check sparsity
                    unsigned int addStep = 1;
                    for (unsigned int i = 0; i < m_Nnz; ++i) {
                        p_Value += 0.3;
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
                    fillFromVector(l_rows, t_RowsInCblock, t_ColsInBblock, spmv_width);
                    return l_rows;
                }

        };



}

#endif
