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
#ifndef GEMX_GEN_SPMV_H
#define GEMX_GEN_SPMV_H 

#include "gemx_gen_bin.h"
#include "gemx_matrix.h"


typedef SpmvType::SpmvArgsType SpmvArgsType;


#if GEMX_useURAM==0

template <typename t_FloatType, typename t2, typename t3>
void
spmv_ref(SpMat<t_FloatType, t2, t3> &p_a, DenseMat<t_FloatType> &p_b, DenseMat<t_FloatType> &p_c, bool p_usePrelu) {
  t_FloatType l_val = 0;
  std::vector<MtxRow> l_rows =  p_a.getNnzVector();
  for (MtxRow &l_row : l_rows) {
        unsigned int row = l_row.getRow(),
                     col = l_row.getCol();
        l_val = l_row.getVal();
        p_c.getVal(row, 0) += l_val * p_b.getVal(col, 0);
      }
  if (p_usePrelu) {
        for (unsigned int i=0; i<p_c.rows(); ++i) {
          t_FloatType l_valRow = p_c.getVal(i,0);
          p_c.getVal(i,0) = (l_valRow<0)? 0: l_valRow;
        }
  }
  return;
}


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
        // m_B
        const unsigned int l_kEdge = GEMX_ddrWidth;
        const unsigned int l_bBlockSize = GEMX_spmvWidth * GEMX_spmvkVectorBlocks * GEMX_ddrWidth;
        const unsigned int l_cBlockSize = GEMX_spmvWidth * GEMX_spmvMacGroups * GEMX_spmvmVectorBlocks * GEMX_ddrWidth;        
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
          
        } else {
          if (p_Nnz % GEMX_spmvWidth != 0) {
            std::cout << "INFO: spmv  Nnz " << p_Nnz << " must be multiple of GEMX_spmvWidth "
                      << GEMX_spmvWidth << "\n";
            while(p_Nnz % GEMX_spmvWidth != 0){
                std::cout << "INFO: add one to given number of non-zero element\n";
                p_Nnz++;
            }
          }
          if (p_M % l_mEdge != 0) {
            std::cout << "INFO: spmv  M dimension " << p_M << " must be multiple of "
                    << l_mEdge << "\n";
            while(p_M % l_mEdge != 0){
                std::cout << "INFO: add one to given m\n";
                p_M++;
            }    
          }
          if (p_K % l_kEdge != 0) {
            std::cout << "INFO: spmv  K dimension " << p_K << " must be multiple of "
                    << l_kEdge << "\n";
            while(p_K % l_kEdge != 0){
                std::cout << "INFO: add one to given K\n";
                p_K++;
            }
          }   
        }
        
        if (p_Nnz == 0) {
          std::cerr << "ERROR: spmv  Nnz must be non-0, it is " << p_Nnz << "\n";
          ok = false;
        }
        unsigned int l_bBlocks = (p_K + l_bBlockSize - 1) / l_bBlockSize;
        unsigned int l_cBlocks = (p_M + l_cBlockSize - 1) / l_cBlockSize;
        unsigned int l_totalBlocks = l_bBlocks * l_cBlocks;
        unsigned int l_maxBlocks = GEMX_spmvNumCblocks;
        if (l_totalBlocks > l_maxBlocks) {
          std::cerr << "ERROR: spmv  total blocks " << l_totalBlocks << " is larger than max supported " << l_maxBlocks << "   Recompile the kernel with larger GEMX_spmvNumCblocks\n";
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
      bool p_usePrelu,
      bool p_WithGolden
    ) {
        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocD;
        //Large matrix support
        unsigned int l_Cblocks = (p_M + SpmvType::getRowsInCblock() - 1) / SpmvType::getRowsInCblock();
        unsigned int l_Bblocks = (p_K + SpMatType::t_ColsInBblock-1) / SpMatType::t_ColsInBblock;
        // A, D; Descriptors simply prefix the A body
        const unsigned int l_numDescPages = (GEMX_spmvNumCblocks + SpMatType::t_numDescPerPage - 1) / SpMatType::t_numDescPerPage;
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
        SpMatType l_matA(p_M, p_K, p_Nnz, l_Bblocks, l_Cblocks, p_Program.getPageAddr(l_pageA));
        MatType l_matB(p_K, 1, 1,       p_Program.getPageAddr(l_pageB));
        MatType l_matC(p_M, 1, 1,       p_Program.getPageAddr(l_pageC));
      
        // Instruction
        SpmvArgsType l_spmvArgs(
            l_pageA, l_pageB, l_pageC,
            p_M, p_K, p_Nnz, l_Bblocks, l_Cblocks, l_numDescPages, p_usePrelu
          );
        KargsType l_kargs;
        l_kargs.setSpmvArgs(l_spmvArgs);
        l_kargs.store(p_Program.addInstr(), 0);
        
        if (l_newAllocA) {
          if (p_MtxFile.good()) {
            l_matA.fillFromVector(p_MtxFile.getRows());
          } else {
            l_matA.fillMod(17);
          }
        }
        if (l_newAllocB) {
          l_matB.fillMod(9);
        }

        // Calculate reference C = A * B
        if (p_WithGolden) {
          spmv_ref<GEMX_dataType, SpmvAdType, SpmvAType>(l_matA, l_matB, l_matC, p_usePrelu);
        }
        std::cout << "Added SPMV " << p_M << "x" << p_K << " Nnz=" << p_Nnz << "  ";
        //std::cout << "DEBUG A:\n" << l_matA << "\n";
  }
  
    void
    addInstrReadVector(
      ProgramType &p_Program,
      unsigned int p_M,
      unsigned int p_K,
      unsigned int p_Nnz,
      MtxFile p_MtxFile,
      std::string l_vectorFileName,
      std::string p_handleA,
      std::string p_handleB,
      std::string p_handleC,
      bool p_usePrelu,
      bool p_WithGolden
    ) {
        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocD;
        //Large matrix support
        unsigned int l_Cblocks = (p_M + SpmvType::getRowsInCblock() - 1) / SpmvType::getRowsInCblock();
        unsigned int l_Bblocks = (p_K + SpMatType::t_ColsInBblock-1) / SpMatType::t_ColsInBblock;
        // A, D; Descriptors simply prefix the A body
        const unsigned int l_numDescPages = (GEMX_spmvNumCblocks + SpMatType::t_numDescPerPage - 1) / SpMatType::t_numDescPerPage;
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
        SpMatType l_matA(p_M, p_K, p_Nnz, l_Bblocks, l_Cblocks, p_Program.getPageAddr(l_pageA));
        MatType l_matB(p_K, 1, 1,       p_Program.getPageAddr(l_pageB));
        MatType l_matC(p_M, 1, 1,       p_Program.getPageAddr(l_pageC));
      
        // Instruction
        SpmvArgsType l_spmvArgs(
            l_pageA, l_pageB, l_pageC,
            p_M, p_K, p_Nnz, l_Bblocks, l_Cblocks, l_numDescPages, p_usePrelu
          );
        KargsType l_kargs;
        l_kargs.setSpmvArgs(l_spmvArgs);
        l_kargs.store(p_Program.addInstr(), 0);
        
        if (l_newAllocA) {
          if (p_MtxFile.good()) {
            l_matA.fillFromVector(p_MtxFile.getRows());
          } else {
            l_matA.fillMod(17);
          }
        }
        if (l_newAllocB) {
          l_matB.fillModFromFile(l_vectorFileName);
        }

        // Calculate reference C = A * B
        if (p_WithGolden) {
          spmv_ref<GEMX_dataType, SpmvAdType, SpmvAType>(l_matA, l_matB, l_matC, p_usePrelu);
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
                     l_Bblocks = p_SpmvArgs.m_Bblocks,
                     l_Cblocks = p_SpmvArgs.m_Cblocks;
        SpMatType l_matA(l_M, l_K, l_Nnz, l_Bblocks, l_Cblocks, p_Program.getPageAddr(p_SpmvArgs.m_Aoffset));
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

#else

template <typename t_FloatType, typename t_IdxType>
void
spmv_ref(SpMatUram<t_FloatType, t_IdxType> &p_a, DenseMat<t_FloatType> &p_b, DenseMat<t_FloatType> &p_c) {
    t_FloatType l_val = 0;
    std::vector<MtxRow> l_rows =  p_a.getNnzVector();
    for (MtxRow &l_row : l_rows) {
          unsigned int row = l_row.getRow(),
                       col = l_row.getCol();
          l_val = l_row.getVal();
          p_c.getVal(row, 0) += l_val * p_b.getVal(col, 0);
        }

    return;
}

class GenSpmvUram
{
  public:
    bool
    check(
      unsigned int &p_M,  // The check() modifies the dimensions when loading from a file
      unsigned int &p_K,
      unsigned int &p_Nnz,
      MtxFileUram &p_MtxFile
    ) {
        bool ok = true;        
        // m_C
        const unsigned int l_mEdge = GEMX_ddrWidth * GEMX_spmvUramGroups;
        const unsigned int l_mMax = GEMX_ddrWidth * GEMX_spmvMmaxBlocks * GEMX_spmvUramGroups;
        // m_B
        const unsigned int l_kEdge = GEMX_ddrWidth;
        const unsigned int l_kMax = l_kEdge * GEMX_spmvKmaxBlocks;
        
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
        }else{
          if (p_Nnz % (GEMX_ddrWidth * GEMX_nnzBlocks) != 0) {
            std::cout << "INFO: spmv  Nnz " << p_Nnz << " must be multiple of GEMX_ddrWidth * GEMX_nnzBlocks "
                    << GEMX_ddrWidth << "*" << GEMX_nnzBlocks << "\n";
            while(p_Nnz % (GEMX_ddrWidth * GEMX_nnzBlocks) != 0){
                std::cout << "INFO: add one to given number of non-zero element\n";
                p_Nnz++;
            }
          }
          if (p_M % l_mEdge != 0) {
            std::cout << "INFO: spmv  M dimension " << p_M << " must be multiple of "
                    << l_mEdge << "\n";
            while(p_M % l_mEdge != 0){
                std::cout << "INFO: add one to given m\n";
                p_M++;
            }    
          }
          if (p_K % l_kEdge != 0) {
            std::cout << "INFO: spmv  K dimension " << p_K << " must be multiple of "
                    << l_kEdge << "\n";
            while(p_K % l_kEdge != 0){
                std::cout << "INFO: add one to given K\n";
                p_K++;
            }
          }   
        }
        
        if (p_Nnz == 0) {
          std::cerr << "ERROR: spmv  Nnz must be non-0, it is " << p_Nnz << "\n";
          ok = false;
        }
        
        if (p_M > l_mMax) {
          std::cerr << "ERROR: spmv  M dimension " << p_M << " is larger than max supported " << l_mMax
                    << "   Recompile the kernel with larger GEMX_spmvMmaxBlocks\n";
          ok = false;
        }
        if (p_K > l_kMax) {
          std::cerr << "ERROR: spmv  K dimension " << p_K << " is larger than max supported " << l_kMax
                    << "  Recompile the kernel with larger GEMX_spmvKmaxBlocks\n";
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
      MtxFileUram p_MtxFile,
      std::string p_handleA,
      std::string p_handleB,
      std::string p_handleC,
      bool p_WithGolden
    ) {
        
        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB, l_newAllocC;
        
        unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA, p_Nnz+p_Nnz*2*sizeof(GEMX_idxType)/sizeof(GEMX_dataType));
        // B, C
        unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, p_K * 1);
        unsigned int l_pageC = p_Program.allocPages(p_handleC, l_newAllocC, p_M * 1);
        
        // Get addresses where matrices are stored
        SpMatType l_matA(p_M, p_K, p_Nnz, p_Program.getPageAddr(l_pageA));
        MatType l_matB(p_K, 1, 1,       p_Program.getPageAddr(l_pageB));
        MatType l_matC(p_M, 1, 1,       p_Program.getPageAddr(l_pageC));
        
        // Instruction
        SpmvArgsType l_spmvArgs(
            l_pageA, l_pageB, l_pageC,
            p_M, p_K, p_Nnz, 0, 0, 0, 0
          );
        KargsType l_kargs;
        l_kargs.setSpmvArgs(l_spmvArgs);
        l_kargs.store(p_Program.addInstr(), 0);
        
        if (l_newAllocA) {
          if (p_MtxFile.good()) {
            if (p_MtxFile.isDiag()) {
              l_matA.fillFromVector(p_MtxFile.getRows());
            } else {
              l_matA.fillFromVectorWithReorder(p_MtxFile.getRows());
            }
          } else {
            l_matA.fillMod(17);
          }
        }
        if (l_newAllocB) {
          l_matB.fillMod(9);
        }
        
        // Calculate reference C = A * B
        if (p_WithGolden) {
          spmv_ref<GEMX_dataType, GEMX_idxType>(l_matA, l_matB, l_matC);
          
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
                     l_Nnz = p_SpmvArgs.m_Nnz;
        SpMatType l_matA(l_M, l_K, l_Nnz, p_Program.getPageAddr(p_SpmvArgs.m_Aoffset));
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
#endif 
