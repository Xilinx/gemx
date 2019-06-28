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
#ifndef _USPMV_HOST_H_
#define _USPMV_HOST_H_

#include "xhost.h"
#include "gemm_host.h"
#include "gemx_util.h"

using namespace std;
namespace gemx {

class USpmvArgs: public kArgs {
public:
    virtual ~USpmvArgs() {
    }
    USpmvArgs() = delete;
    USpmvArgs ( unsigned int p_Aoffset, unsigned int p_Boffset, unsigned int p_Coffset, unsigned int p_numRuns) :
        m_uspmv_args( { int(OpUspmv), p_Aoffset, p_Boffset, p_Coffset, p_numRuns, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ){
    }

    size_t sizeInBytes() {
        return sizeof(m_uspmv_args);
    }
    char *asByteArray() {
        return reinterpret_cast<char*>(&m_uspmv_args);
    }
protected:
    struct {
        int m_optype;
        unsigned int m_Aoffset, m_Boffset, m_Coffset, m_numRuns;
        unsigned int dummy[11];
    } m_uspmv_args;
};


template<typename HType>
class USPMVHost : public GEMMHost<HType> {
public:
    USPMVHost() = delete;
    virtual ~USPMVHost(){
    }

    USPMVHost(const USPMVHost<HType> &) = delete;

    USPMVHost(const string & xclbin, const string & kernelName) : GEMMHost<HType> ( xclbin, kernelName)
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
       
    virtual void* SendUSpMat(uint16_t* row, uint16_t* col, float* data, int* row_size, int* col_size, int* nnz_size, float* p_pRelu, unsigned int t_DdrWidth, unsigned int t_Stages){
      unsigned int t_DoubleDdrWidth = t_DdrWidth*2;
      unsigned int t_StageBlocks = (t_Stages + t_DoubleDdrWidth -1) / t_DoubleDdrWidth;
      unsigned int l_aSize = ((t_StageBlocks * t_DoubleDdrWidth * 3)+1)/2;
      for (unsigned int i=0; i<t_Stages; ++i) {
        l_aSize += nnz_size[i] * 2;
      }
      float *A = new float[l_aSize];
      UspMat<float,uint16_t> MatA(A, t_DdrWidth, t_Stages);
      MatA.fillFromVector(row, col, data, row_size, col_size, nnz_size, p_pRelu);
      this->SendToFPGA((float*)A, A,(unsigned long long)l_aSize * sizeof(float));
      return A;
    }
    
    virtual bool AddUSPMVOp(const HType & A, const HType & B, const HType & C, unsigned int numRuns){     
      if (this->_hostMat.find(A) == this->_hostMat.end()
                || this->_hostMat.find(B) == this->_hostMat.end()
                || this->_hostMat.find(C) == this->_hostMat.end()) {
            cerr << "Matrix not found!" << endl;
            return false;
      }
       
     unsigned long long A_off = 0, B_off = 0, C_off = 0;
     xclGetMemObjDeviceAddress(this->_devHandle[A](), (this->_fpga_stream->getDevice())(), sizeof(unsigned long long), &A_off);
     xclGetMemObjDeviceAddress(this->_devHandle[B](), (this->_fpga_stream->getDevice())(), sizeof(unsigned long long), &B_off);
     xclGetMemObjDeviceAddress(this->_devHandle[C](), (this->_fpga_stream->getDevice())(), sizeof(unsigned long long), &C_off);
       
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
              
     USpmvArgs args(A_off, B_off, C_off, numRuns);
     this->AddInstr (&args);  
     return true;
   }
    
    virtual bool AddUSPMVDevOp(const HType & A, const HType & B, const HType & C, unsigned int numRuns){     
       if (this->_hostMatPageOffset.find(A) == this->_hostMatPageOffset.end()
          || this->_hostMatPageOffset.find(B) == this->_hostMatPageOffset.end()
          || this->_hostMatPageOffset.find(C) == this->_hostMatPageOffset.end()) {
          cerr << "Matrix not found!" << endl;
          return false;
       }
       unsigned long long A_off = 0, B_off = 0, C_off = 0;
       A_off = this->GetMatOffset(A);
       B_off = this->GetMatOffset(B);
       C_off = this->GetMatOffset(C);
       USpmvArgs args(A_off, B_off, C_off, numRuns);
       this->AddInstr (&args);  
       return true;
    }
};


template<typename HType>
class USPMVDevHost : public GEMMHost<HType> {
public:
    USPMVDevHost() = delete;
    virtual ~USPMVDevHost(){
    }

    USPMVDevHost(const USPMVDevHost<HType> &) = delete;

    USPMVDevHost(const string & xclbin, const string & kernelName) : GEMMHost<HType> ( xclbin, kernelName)
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
    
    virtual void* AddUSpDevBuf(uint16_t* row, uint16_t* col, float* data, char* A_str, int* row_size, int* col_size, int* nnz_size, float* p_pRelu, unsigned int t_DdrWidth, unsigned int t_Stages){
      unsigned int t_DoubleDdrWidth = t_DdrWidth*2;
      unsigned int t_StageBlocks = (t_Stages + t_DoubleDdrWidth -1) / t_DoubleDdrWidth;
      unsigned int l_aSize = ((t_StageBlocks * t_DoubleDdrWidth * 3)+1)/2;
      for (unsigned int i=0; i<t_Stages; ++i) {
        l_aSize += nnz_size[i] * 2;
      }
      float *A = (float*) this->AddDevBuf(A_str, l_aSize* sizeof(float));
      UspMat<float,uint16_t> MatA(A, t_DdrWidth, t_Stages);
      MatA.fillFromVector(row, col, data, row_size, col_size, nnz_size, p_pRelu);
      
      return A;
    }  
    
    virtual bool AddUSPMVDevOp(const HType & A, const HType & B, const HType & C, unsigned int numRuns){     
       if (this->_hostMatPageOffset.find(A) == this->_hostMatPageOffset.end()
          || this->_hostMatPageOffset.find(B) == this->_hostMatPageOffset.end()
          || this->_hostMatPageOffset.find(C) == this->_hostMatPageOffset.end()) {
          cerr << "Matrix not found!" << endl;
          return false;
       }
       unsigned long long A_off = 0, B_off = 0, C_off = 0;
       A_off = this->GetMatOffset(A);
       B_off = this->GetMatOffset(B);
       C_off = this->GetMatOffset(C);
       USpmvArgs args(A_off, B_off, C_off, numRuns);
       this->AddInstr (&args);  
       return true;
    }
};

}

#endif
