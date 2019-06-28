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
#ifndef _SPMV_HOST_H_
#define _SPMV_HOST_H_

#include "xhost.h"
#include "gemm_host.h"
#include "gemx_util.h"


using namespace std;
namespace gemx {
  
class SpmvArgs: public kArgs {
public:
    virtual ~SpmvArgs() {
    }
    SpmvArgs() = delete;
    SpmvArgs ( unsigned int p_Aoffset, unsigned int p_Boffset, unsigned int p_Coffset, unsigned int M, unsigned int K, unsigned int Nnz, unsigned int p_Bblocks, unsigned int p_Cblocks, unsigned int p_DescPages, bool p_pRelu) :
        m_spmv_args( { int(OpSpmv), p_Aoffset, p_Boffset, p_Coffset, M, K, Nnz, p_Bblocks, p_Cblocks, p_DescPages, p_pRelu, 0, 0, 0, 0, 0} ){
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
        bool m_Prelu;
        unsigned int dummy[5];
    } m_spmv_args;
};

template<typename HType>
class SPMVHost : public GEMMHost<HType> {
public:
    SPMVHost() = delete;
    virtual ~SPMVHost(){
    }

    SPMVHost(const SPMVHost<HType> &) = delete;

    SPMVHost(const string & xclbin, const string & kernelName) : GEMMHost<HType> ( xclbin, kernelName)
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
        for(unsigned int i = 0; i<nnz; ++i){
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
        for(unsigned int i = 0; i<nnz; ++i){
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
      
    virtual bool AddSPMVOp(const HType & A, const HType & B, const HType & C, unsigned int m, unsigned int k, unsigned int nnz, bool l_pRelu, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks){     
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

        SpmvArgs args(A_off, B_off, C_off, m, k, nnz, l_Bblocks, l_Cblocks, l_numDescPages, l_pRelu);
        this->AddInstr (&args);  
        return true;
    }
       
};

template<typename HType>
class SPMVDevHost : public GEMMHost<HType> {
public:
    SPMVDevHost() = delete;
    virtual ~SPMVDevHost(){
    }

    SPMVDevHost(const SPMVDevHost<HType> &) = delete;

    SPMVDevHost(const string & xclbin, const string & kernelName) : GEMMHost<HType> ( xclbin, kernelName)
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
    
    virtual void* AddSpDevBuf(int * row, int * col, float * data, char* A_str, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks){
        vector<MtxRow> l_rows;     
        for(unsigned int i = 0; i<nnz; ++i){
          MtxRow l_m(data[i], row[i], col[i]);
          l_rows.push_back(l_m);
        }
        typedef SpmvAd<float> SpmvAdType; 
        unsigned int l_Cblocks = (m + capacity_Cblocks - 1) / capacity_Cblocks;
        unsigned int l_Bblocks = (k + capacity_Bblocks - 1) / capacity_Bblocks;        
        unsigned int l_numDescPages = (num_cblocks + SpmvAdesc::t_per4k - 1) / SpmvAdesc::t_per4k; 
        unsigned int l_numDescDdrWords = l_numDescPages * 4096 / sizeof(float) / ddr_width;
        unsigned int l_numPaddingDdrWords = num_cblocks * 4096 / sizeof(float) / ddr_width;       
        unsigned int l_aSize = (l_numDescDdrWords * ddr_width + nnz * ddr_width / spmv_width + l_numPaddingDdrWords * ddr_width) * sizeof(float);
        float *A = (float*) this->AddDevBuf(A_str, l_aSize);
        SpMat<float,SpmvAdType> MatA(m,k,nnz,l_Bblocks,l_Cblocks,A);
        MatA.fillFromVector(l_rows, capacity_Cblocks, capacity_Bblocks, spmv_width);
        return A;
    }
    
    virtual bool AddSPMVDevOp(const HType & A, const HType & B, const HType & C, unsigned int m, unsigned int k, unsigned int nnz, bool l_pRelu, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks){     
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
        unsigned int l_numDescPages = (num_cblocks + SpmvAdesc::t_per4k - 1) / SpmvAdesc::t_per4k; 
        unsigned int l_Cblocks = (m + capacity_Cblocks - 1) / capacity_Cblocks;
        unsigned int l_Bblocks = (k + capacity_Bblocks - 1) / capacity_Bblocks;
        SpmvArgs args(A_off, B_off, C_off, m, k, nnz, l_Bblocks, l_Cblocks, l_numDescPages, l_pRelu);
        this->AddInstr (&args);  
        return true;
    }
       
};



}


#endif
