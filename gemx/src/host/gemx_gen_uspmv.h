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
 *  @brief GEMX USPMV instruction generator
 *
 *  $DateTime: 2018/05/08 03:56:09 $
 */

#ifndef GEMX_GEN_USPMV_H
#define GEMX_GEN_USPMV_H

#include "gemx_gen_bin.h"
#include "gemx_matrix.h"


template<typename t_FloatType, typename t_IdxType, unsigned int t_Stages, unsigned int t_DdrWidth>
    void
    multiplySingle(UspMat<t_FloatType, t_IdxType, t_Stages, t_DdrWidth> &p_a, std::vector<t_FloatType> &p_b, unsigned int p_mtxId, std::vector<t_FloatType> &p_c){
        unsigned int l_mtxBaseIdx = 0;
        for (unsigned int i=0; i<p_mtxId; ++i) {
            l_mtxBaseIdx += p_a.getNnzs(i)*2;
        }
        unsigned int l_nnz = p_a.getNnzs(p_mtxId);
        for (unsigned int i=0; i<l_nnz; ++i) {
            unsigned int l_col = p_a.getIdx(l_mtxBaseIdx*2 + i*2);
            unsigned int l_row = p_a.getIdx(l_mtxBaseIdx*2+ i*2+1);
            t_FloatType l_val = p_a.getVal(l_mtxBaseIdx+l_nnz+i);
            t_FloatType l_b = (p_b[l_col]<0)? (p_a.getPrelu(p_mtxId-1) * p_b[l_col]): p_b[l_col];
            p_c[l_row] += l_val * l_b;
        }
}
    
template<typename t_FloatType, typename t_IdxType, unsigned int t_Stages, unsigned int t_DdrWidth>
    void
    spmm_ref(UspMat<t_FloatType, t_IdxType, t_Stages, t_DdrWidth> &p_a, DenseMat<t_FloatType> &p_b, unsigned int p_numRuns, DenseMat<t_FloatType> &p_c) {
        //define vector arrays for storing intermediate SPMV results from each stage spmv engine
        std::array<std::vector<t_FloatType>, GEMX_uspmvStages> l_immVectors;
        for (unsigned int i=0; i<p_numRuns; ++i) {
            for (unsigned int i=0; i<GEMX_uspmvStages; ++i) {
              unsigned int l_rows = p_a.getRows(i);
               l_immVectors[i].resize(l_rows);
               for (unsigned int j=0; j<l_rows; ++j) {
                 l_immVectors[i][j] = 0;
               }
            }    
            //compute multiplication with input p_b to get first intermediate vector
            unsigned int l_nnz = p_a.getNnzs(0);
            for (unsigned int j=0; j<l_nnz; ++j) {
              unsigned int l_col = p_a.getIdx(j*2);
              unsigned int l_row = p_a.getIdx(j*2+1);
              t_FloatType l_val = p_a.getVal(l_nnz +j);
              l_immVectors[0][l_row] += l_val * p_b.getVal(i, l_col);    
            }
            for (unsigned int j=1; j<GEMX_uspmvStages; ++j) {
              multiplySingle<t_FloatType, t_IdxType, t_Stages, t_DdrWidth>(p_a, l_immVectors[j-1], j, l_immVectors[j]);
            }
            //assign l_immVectors[t_Stages-1] to p_c
            for (unsigned int l_cCol=0; l_cCol<p_a.getRows(GEMX_uspmvStages-1); ++l_cCol) {
              p_c.getVal(i,l_cCol) = (l_immVectors[GEMX_uspmvStages-1][l_cCol]<0)? l_immVectors[GEMX_uspmvStages-1][l_cCol] * p_a.getPrelu(GEMX_uspmvStages-1): l_immVectors[GEMX_uspmvStages-1][l_cCol];
            }
        }
}

template <typename t_FloatType, typename t_IdxType, unsigned int t_Stages, unsigned int t_DdrWidth>
class GenUspmv
{
    private:
        static const unsigned int t_FloatSize = sizeof(t_FloatType);
        static const unsigned int t_UramWidth = 8 / t_FloatSize;
        static const unsigned int t_NumUramPerDdr = t_DdrWidth / t_UramWidth; //number of URAM slices used to store one data DDR
    public:
    bool
    check(
      unsigned int *p_m,
      unsigned int *p_k,
      unsigned int *p_nnz,
      std::array<MtxFile, t_Stages> &p_mtxFile
    ) {
        bool ok = true;       
        assert(t_NumUramPerDdr * t_UramWidth == GEMX_ddrWidth); 
        assert(sizeof(t_IdxType)*2 == sizeof(t_FloatType));
        assert(t_Stages < (t_DdrWidth*2));
        const unsigned int l_kEdge = GEMX_ddrWidth;
        const unsigned int l_mEdge = GEMX_ddrWidth * GEMX_uspmvInterleaves;
        const unsigned int l_maxNnz = GEMX_ddrWidth * GEMX_uspmvNnzVectorBlocks;
        const unsigned int l_maxM = GEMX_ddrWidth * GEMX_uspmvMvectorBlocks; 
       
        for (unsigned int i=0; i<t_Stages; ++i) { 
          if (!p_mtxFile[i].good() && (p_mtxFile[i].fileName() != "none")) {
              std::cerr << "ERROR: spmv  mtxFile " << p_mtxFile[i].fileName()
                        << " must exist or use none for auto-generated data"
                        << "\n";
              ok = false;
          }         
          // Use the file only
          if (p_mtxFile[i].good()) {
              if ((p_m[i] != 0) || (p_k[i] != 0) || (p_nnz[i] != 0)) {
                  std::cerr << "ERROR: spmv  M, K, Nnz must be 0 when using mtx file: "
                            << "  M " << p_m[i]
                            << "  K " << p_k[i]
                            << "  Nnz " << p_nnz[i]
                            << "\n";
              }
              p_m[i] = p_mtxFile[i].rows();
              p_k[i] = p_mtxFile[i].cols();
              p_nnz[i] = p_mtxFile[i].nnz();              
          } 
          if (p_m[i] % l_mEdge != 0) {
              std::cout << "INFO: spmv  M dimension " << p_m[i] << " must be multiple of "
                        << l_mEdge << "\n";
              while(p_m[i] % l_mEdge != 0){
                std::cout << "INFO: add one to given m\n";
                p_m[i]++;
              }    
          }
          if (p_k[i] % l_kEdge != 0) {
              std::cout << "INFO: spmv  K dimension " << p_k[i] << " must be multiple of "
                        << l_kEdge << "\n";
              while(p_k[i] % l_kEdge != 0){
                std::cout << "INFO: add one to given K\n";
                p_k[i]++;
              }
          }          
          if (p_nnz[i] == 0) {
              std::cerr << "ERROR: spmv  Nnz must be non-0, it is " << p_nnz[i] << "\n";
              ok = false;
          }
          if (p_nnz[i] > l_maxNnz) {
              std::cerr << "ERROR: spmv Nnz " << p_nnz[i] << "must be less than or equal to " << l_maxNnz << "\n";
              ok = false;
          }
          if (p_m[i] > l_maxM) {
              std::cerr << "ERROR: spmv M dimension " << p_m[i] << "must be less than or equal to " << l_maxM << "\n";
              ok = false;
          }
        }
        return(ok);
      }

    void
    addInstr(
      Program<t_FloatType> &p_program,
      unsigned int *p_m,
      unsigned int *p_k,
      unsigned int *p_nnz,
      t_FloatType *p_pRelu,
      unsigned int p_numRuns,
      std::array<MtxFile, t_Stages> &p_mtxFile,
      std::string p_handleA,
      std::string p_handleB,
      std::string p_handleC,
      bool p_withGolden
    ) {
    
        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB, l_newAllocC;
        
        //allocate memory for all sparse matrices
        unsigned int l_aSize = UspMat<t_FloatType, t_IdxType, t_Stages, t_DdrWidth>::t_DescSize; //used to store description of each sparse matrix, nnz, m and k sizes
        for (unsigned int i=0; i<t_Stages; ++i) {
            l_aSize += p_nnz[i] * 2;
        }
        unsigned int l_pageA = p_program.allocPages(p_handleA, l_newAllocA, l_aSize);
        // B, C
        unsigned int l_pageB = p_program.allocPages(p_handleB, l_newAllocB, p_k[0] * p_numRuns);
        unsigned int l_pageC = p_program.allocPages(p_handleC, l_newAllocC, p_m[t_Stages-1] * p_numRuns);
        
        // Get addresses where matrices are stored
        UspMat<t_FloatType, t_IdxType, t_Stages, t_DdrWidth> l_matA(p_numRuns, p_program.getPageAddr(l_pageA));
        DenseMat<t_FloatType> l_matB(p_numRuns, p_k[0], p_k[0], p_program.getPageAddr(l_pageB));
        DenseMat<t_FloatType> l_matC(p_numRuns, p_m[t_Stages-1], p_m[t_Stages-1], p_program.getPageAddr(l_pageC));
      
        // Instruction
        gemx::UspmvArgs l_uspmvArgs(
            l_pageA, l_pageB, l_pageC, p_numRuns 
          );
        KargsType l_kargs;
        l_kargs.setUspmvArgs(l_uspmvArgs);
        l_kargs.store(p_program.addInstr(), 0);
        
        if (l_newAllocA) {
          if (p_mtxFile[0].good()){
            l_matA.fillFromVector(p_mtxFile, p_pRelu);
          } else {
            l_matA.fillMod(1, p_m, p_k, p_nnz,p_pRelu);
          }
        }
        if (l_newAllocB) {
            l_matB.fillMod(9);
        }
        
        // Calculate reference C = A * B
        if (p_withGolden) {
            spmm_ref<t_FloatType, t_IdxType, t_Stages, t_DdrWidth>(l_matA, l_matB, p_numRuns, l_matC);
        }
        std::cout << "Added USPMV\n";
        for (unsigned int i=0; i<t_Stages; ++i) {
            std::cout << p_m[i] << "x" << p_k[i] << " Nnz=" << p_nnz[i] << "\n ";
        }
        std::cout << "run " << p_numRuns << "times\n";
      }
    void
    show(
        Program<t_FloatType> &p_program,
        gemx::UspmvArgs p_uspmvArgs
    ) {
        unsigned int l_numRuns = p_uspmvArgs.m_NumRuns;
        UspMat<t_FloatType, t_IdxType, t_Stages, t_DdrWidth> l_matA(l_numRuns, p_program.getPageAddr(p_uspmvArgs.m_Aoffset));
        unsigned int l_cols_0 = l_matA.getCols(0);
        unsigned int l_rows_last = l_matA.getRows(t_Stages-1);
        MatType l_matB(l_numRuns, l_cols_0, l_cols_0, p_program.getPageAddr(p_uspmvArgs.m_Boffset));
        MatType l_matC(l_numRuns, l_rows_last, l_rows_last, p_program.getPageAddr(p_uspmvArgs.m_Coffset));
        std::cout << "\nn###########  Op Uspmv  ###########\n"
                  << " C = A * B "
                  << " layers = " << t_Stages << "\n"
                  << "  A\n" << l_matA << "\n"
                  << "  B " << l_matB << "\n"
                  << "  C " << l_matC << "\n";
    }
    
    bool
    compare (
        float p_TolRel, 
        float p_TolAbs,
        Program<t_FloatType> &p_program0, 
        Program<t_FloatType> &p_program1,
        gemx::UspmvArgs p_uspmvArgs
    ) {
        unsigned int l_numRuns = p_uspmvArgs.m_NumRuns;
        UspMat<t_FloatType, t_IdxType, t_Stages, t_DdrWidth> l_matA(l_numRuns, p_program0.getPageAddr(p_uspmvArgs.m_Aoffset));
        unsigned int l_cols_0 = l_matA.getCols(0);
        unsigned int l_rows_last = l_matA.getRows(t_Stages-1);
        DenseMat<t_FloatType> l_matC0(l_numRuns, l_rows_last, l_rows_last, p_program0.getPageAddr(p_uspmvArgs.m_Coffset));
        DenseMat<t_FloatType> l_matC1(l_numRuns, l_rows_last, l_rows_last, p_program1.getPageAddr(p_uspmvArgs.m_Coffset));    

        std::cout << "\n###########  Op Uspmv  ###########\n"
                  << "  C = A * B  "
                  << " layers = " << t_Stages << "\n"
                  << "  Comparing ...\n";
        bool ok = l_matC1.cmp(p_TolRel, p_TolAbs, l_matC0);
        std::cout << "Spmv C " << (ok ? "Matches" : "Differs") << "\n";
        return(ok);
      }
};

#endif
