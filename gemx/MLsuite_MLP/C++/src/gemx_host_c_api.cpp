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
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "gemm_host.h"
#include "fcn_host.h"
#include "spmv_host.h"
#include "uspmv_host.h"
#include "xhost.h"
#include "gemx_util.h"
#include "gemx_host_c_api.h"

//#define GEMX_PERF_DBG
using namespace gemx;
using namespace std;


class GEMXHostProfiler {
    public:
        unordered_map < string, double> func_time;
        unordered_map < string, unsigned long long> func_calls;
        static GEMXHostProfiler& Instance() {
            static GEMXHostProfiler theInstance;
            return theInstance;
        }
    protected:
        GEMXHostProfiler() {

        }
};
template<typename HType>
class GEMXHostHandle {
    public:
        vector<shared_ptr<GEMMHost<HType>>> gh_ptr;
        static GEMXHostHandle& Instance() {
            static GEMXHostHandle theInstance;
            return theInstance;
        }
    protected:
        GEMXHostHandle() {

        }
};


    template<typename T>
static void print(char *name,T * A, int m, int n)
{
    ofstream myfile;
    string fName = name;
    fName += ".c";
    myfile.open (fName.c_str());

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            myfile << A[i*m + j] << " ";
        }
        myfile << "\n";
    }
    myfile.close();
}

void MakeFCNHost(char *xclbin, unsigned int nPE)
{
    for (unsigned i = 0; i < nPE; i++)
    {
        string kName = GEMMHost<short*>::getKernelName(i);
        GEMXHostHandle<void*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<void*> > (new gemx::FCNHost<void*>(xclbin, kName) ));
    }
}

void MakeGEMMHost(char *xclbin, unsigned int nPE)
{
    for (unsigned i = 0; i < nPE; i++)
    {
        string kName = GEMMHost<short*>::getKernelName(i);
        GEMXHostHandle<void*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<void*> > ( new gemx::GEMMHost<void*>(xclbin, kName) ));
    }
}

void MakeUSPMVHost(char *xclbin, unsigned int nPE) { 
    for (unsigned i = 0; i < nPE; i++)
    {
        string kName = GEMMHost<void*>::getKernelName(i);
        GEMXHostHandle<void*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<void*> > ( new gemx::USPMVHost<void*>(xclbin, kName) ));
    }
}

void MakeSPMVHost(char *xclbin, unsigned int nPE) {  
    for (unsigned i = 0; i < nPE; i++)
    {
        string kName = GEMMHost<void*>::getKernelName(i);
        GEMXHostHandle<void*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<void*> > ( new gemx::SPMVHost<void*>(xclbin, kName) ));
    }
}

void SendToFPGAShrt(short *A, unsigned long long num_elem, unsigned PE, bool sync_send)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->SendToFPGA(A, A, sizeof(short) *num_elem, sync_send);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendToFPGAShrt"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendToFPGAShrt"]++;
#endif
}

void SendToFPGAInt(int *A, unsigned long long num_elem, unsigned PE,bool sync_send)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->SendToFPGA(A, A, sizeof(int) *num_elem, sync_send);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendToFPGAInt"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendToFPGAInt"]++;
#endif

}

void SendToFPGAFloat(float *A, unsigned long long num_elem, unsigned PE,bool sync_send)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->SendToFPGA(A, A, sizeof(float) *num_elem, sync_send);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendToFPGAFloat"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendToFPGAFloat"]++;
#endif

}

void* SendUSpMat(uint16_t* row, uint16_t* col, float* data, int* row_size, int* col_size, int* nnz_size, float* p_pRelu, unsigned int t_DdrWidth, unsigned int t_Stages, unsigned PE){
    gemx::XTimer t;
    gemx::USPMVHost<void*>* spmv_ptr = static_cast< gemx::USPMVHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    void* ret = spmv_ptr->SendUSpMat(row,col,data,row_size,col_size,nnz_size,p_pRelu, t_DdrWidth, t_Stages);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendUSpMat"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendUSpMat"]++;
#endif
    return ret;
}

void* SendSpToFpgaFloat(int *row, int *col, float *data, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE){
    gemx::SPMVHost<void*>* spmv_ptr = static_cast< gemx::SPMVHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    void* ret = spmv_ptr->SendSpToFpgaFloat(row,col,data,m,k,nnz,ddr_width,spmv_width,num_cblocks,capacity_Cblocks,capacity_Bblocks);
    return ret;
}

void* SendSpToFpgaInt(int *row, int *col, float *data, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE){
    gemx::SPMVHost<void*>* spmv_ptr = static_cast< gemx::SPMVHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    void* ret = spmv_ptr->SendSpToFpgaInt(row,col,data,m,k,nnz,ddr_width,spmv_width,num_cblocks,capacity_Cblocks,capacity_Bblocks);
    return ret;
}

void* GetFromFPGA(short *A, unsigned PE, bool sync_get)
{
    gemx::XTimer t;
    void * ptr = GEMXHostHandle<void*>::Instance().gh_ptr[PE]->GetMat(A, true, sync_get);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["GetFromFPGA"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["GetFromFPGA"]++;
#endif
    return ptr;
}

void* GetFromFPGAInt(int *A, unsigned PE, bool sync_get)
{
    gemx::XTimer t;
    void * ptr = GEMXHostHandle<void*>::Instance().gh_ptr[PE]->GetMat(A, true, sync_get);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["GetFromFPGA"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["GetFromFPGA"]++;
#endif
    return ptr;
}

void* GetFromFPGAFloat(float *A, unsigned PE, bool sync_get)
{
    gemx::XTimer t;
    void * ptr = GEMXHostHandle<void*>::Instance().gh_ptr[PE]->GetMat(A, true, sync_get);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["GetFromFPGA"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["GetFromFPGA"]++;
#endif
    return ptr;
}

bool AddFCNOp(void * A, void * B, void *C, void * bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha,unsigned PE)
{
    gemx::XTimer t;
    //cout << C << " = " << A << " * " << B << " + " << bias << endl;
    gemx::FCNHost<void*>* fcn_ptr = static_cast< gemx::FCNHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    bool ret = fcn_ptr->AddFCNOp(A, B, C, bias, m,k,n, postScale, postShift, PReLUScale, PReLUAlpha);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["AddFCNOp"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["AddFCNOp"]++;
#endif
    return ret;
}

bool AddGEMMOp(void * A, void * B, void *C, void * bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, unsigned PE)
{
    //cout << C << " = " << A << " * " << B << " + " << bias << endl;
    return GEMXHostHandle<void*>::Instance().gh_ptr[PE]->AddGEMMOp(A, B, C, bias, m,k,n, postScale, postShift);
}

bool AddSPMVOp(void *A, void * B, void *C, unsigned int m, unsigned int k, unsigned int nnz, bool l_pRelu, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE)
{
    gemx::XTimer t;
    gemx::SPMVHost<void*>* spmv_ptr = static_cast< gemx::SPMVHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    bool ret = spmv_ptr->AddSPMVOp(A, B, C, m, k, nnz, l_pRelu, num_cblocks, capacity_Cblocks, capacity_Bblocks);
    return ret;
}

bool AddUSPMVOp(void *A, void * B, void *C, unsigned int numRuns, unsigned PE)
{
    gemx::USPMVHost<void*>* spmv_ptr = static_cast< gemx::USPMVHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    bool ret = spmv_ptr->AddUSPMVOp(A, B, C, numRuns);
    return ret;
}

void Execute (bool sync_exec, unsigned PE)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->Execute(sync_exec);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["Execute"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["Execute"]++;
#endif

}

void Wait (unsigned PE)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->Wait();
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["Wait"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["Wait"]++;
#endif
}

void ClearInstrBuf(unsigned PE)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->ClearInstrBuf();
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["ClearInstrBuf"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["ClearInstrBuf"]++;
#endif
}

void PrintStats()
{
    for ( auto p : GEMXHostProfiler::Instance().func_time)
    {
        cout << p.first << ": " << (p.second * 1000.0) / GEMXHostProfiler::Instance().func_calls[p.first]  << " ms " << GEMXHostProfiler::Instance().func_calls[p.first] << " calls" << endl;
    }
}

void int16_gemm(short * A, short * B, short *X, short * C, unsigned int M, unsigned int K, unsigned int N ) {
    using namespace std;
    using namespace gemx;
    cout << "A_ptr: " << A << " B_ptr: " << B << " C_ptr: " << C << " X_ptr: " << X << endl;
    shared_ptr<GEMMHost<void*>> host_ptr = GEMXHostHandle<void*>::Instance().gh_ptr[0];
    host_ptr->SendToFPGA((short*)A, A, sizeof(short)*M*K);
    host_ptr->SendToFPGA((short*)B, B, sizeof(short)*K*N);
    host_ptr->SendToFPGA((short*)C, C, sizeof(short)*M*N);
    host_ptr->SendToFPGA((short*)X, X, sizeof(short)*M*N);
    host_ptr->AddGEMMOp(A, B, C, (short*)X, M,K,N, 1,0);
    host_ptr->Execute();
    host_ptr->GetMat(C, true, true);
}

//C callable functions for using librar-based memory allocation
void MakeStrGEMMHost(const char *xclbin, unsigned int nPE)
{
    for (unsigned i = 0; i < nPE; i++)
    {
        string kName = GEMMHost<short*>::getKernelName(i);
        string str_xclbin(xclbin);
        GEMXHostHandle<char*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<char*> > ( new gemx::GEMMHost<char*>(str_xclbin, kName) ));
    }
}

void MakeStrFCNHost(const char *xclbin, unsigned int nPE)
{
    for (unsigned i = 0; i < nPE; i++)
    {
        string kName = GEMMHost<short*>::getKernelName(i);
        string str_xclbin(xclbin);
        GEMXHostHandle<char*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<char*> > ( new gemx::FCNHost<char*>(str_xclbin, kName) ));
    }
}

void MakeStrSPMVHost(const char *xclbin, unsigned int nPE)
{
    for (unsigned i = 0; i < nPE; i++)
    {
        string kName = GEMMHost<short*>::getKernelName(i);
        string str_xclbin(xclbin);
        GEMXHostHandle<char*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<char*> > ( new gemx::SPMVDevHost<char*>(str_xclbin, kName) ));
    }
}

void MakeStrUSPMVHost(const char *xclbin, unsigned int nPE)
{
    for (unsigned i = 0; i < nPE; i++)
    {
        string kName = GEMMHost<short*>::getKernelName(i);
        string str_xclbin(xclbin);
        GEMXHostHandle<char*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<char*> > ( new gemx::USPMVDevHost<char*>(str_xclbin, kName) ));
    }
}

bool AllocProgBuf(unsigned int buf_sz, unsigned PE)
{
    gemx::XTimer t;
    bool l_res = GEMXHostHandle<char*>::Instance().gh_ptr[PE]->AllocProgBuf(buf_sz);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["AllocProgBuf"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["AllocProgBuf"]++;
#endif
    return l_res;
}

void* AddDevBuf(char* A, unsigned int buf_sz, unsigned PE)
{
    gemx::XTimer t;
    void* l_ptr = GEMXHostHandle<char*>::Instance().gh_ptr[PE]->AddDevBuf(A, buf_sz);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["AddDevBuf"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["AddDevBuf"]++;
#endif
    return l_ptr;
}

void* AddSpDevBuf(int * row, int * col, float * data, char* A, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE)
{
    gemx::XTimer t;
    gemx::SPMVDevHost<char*>* spmv_ptr = static_cast< gemx::SPMVDevHost<char*> *> (GEMXHostHandle<char*>::Instance().gh_ptr[PE].get());
    void* ret = spmv_ptr->AddSpDevBuf(row,col,data,A, m,k,nnz,ddr_width,spmv_width,num_cblocks,capacity_Cblocks,capacity_Bblocks);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["AddSpDevBuf"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["AddSpDevBuf"]++;
#endif
    return ret;
}

void* AddUSpDevBuf(uint16_t* row, uint16_t* col, float* data, char* A, int* row_size, int* col_size, int* nnz_size, float* p_pRelu, unsigned int t_DdrWidth, unsigned int t_Stages, unsigned PE)
{    
    gemx::XTimer t;    
    gemx::USPMVDevHost<char*>* spmv_ptr = static_cast< gemx::USPMVDevHost<char*> *> (GEMXHostHandle<char*>::Instance().gh_ptr[PE].get());
    void* ret = spmv_ptr->AddUSpDevBuf(row,col,data,A, row_size,col_size,nnz_size,p_pRelu, t_DdrWidth, t_Stages);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["AddUSpDevBuf"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["AddUSpDevBuf"]++;
#endif
    return ret;
}

void SendDevBuf(char* A, unsigned PE, bool sync_send)
{
    gemx::XTimer t;
    GEMXHostHandle<char*>::Instance().gh_ptr[PE]->SendDevBuf(A, sync_send);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendToFPGAShrt"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendToFPGAShrt"]++;
#endif
}

void* GetDevBuf(char* A, unsigned PE, bool sync_get)
{
    gemx::XTimer t;
    void* l_ptr = GEMXHostHandle<char*>::Instance().gh_ptr[PE]->GetDevBuf(A, true, sync_get);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["GetFromFPGA"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["GetFromFPGA"]++;
#endif
    return l_ptr;
}

bool AddGEMMDevOp(char* A, char* B, char*C, char* bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, unsigned PE)
{
    return GEMXHostHandle<char*>::Instance().gh_ptr[PE]->AddGEMMDevOp(A, B, C, bias, m,k,n, postScale, postShift);
}

bool AddFCNDevOp(char* A, char* B, char*C, char* bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha, unsigned PE)
{
    gemx::FCNHost<char*>* fcn_ptr = static_cast< gemx::FCNHost<char*> *> (GEMXHostHandle<char*>::Instance().gh_ptr[PE].get());
    bool ret = fcn_ptr->AddFCNDevOp(A, B, C, bias, m,k,n, postScale, postShift, PReLUScale, PReLUAlpha);
    return ret;
}

bool AddSPMVDevOp(char* A, char* B, char*C, unsigned int m, unsigned int k, unsigned int nnz, bool l_pRelu, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE)
{
    gemx::SPMVDevHost<char*>* spmv_ptr = static_cast< gemx::SPMVDevHost<char*> *> (GEMXHostHandle<char*>::Instance().gh_ptr[PE].get());
    bool ret = spmv_ptr->AddSPMVDevOp(A, B, C, m,k,nnz, l_pRelu, num_cblocks, capacity_Cblocks, capacity_Bblocks);
    return ret;
}

bool AddUSPMVDevOp(char* A, char* B, char*C, unsigned int numRuns, unsigned PE)
{
    gemx::USPMVDevHost<char*>* uspmv_ptr = static_cast< gemx::USPMVDevHost<char*> *> (GEMXHostHandle<char*>::Instance().gh_ptr[PE].get());
    bool ret = uspmv_ptr->AddUSPMVDevOp(A, B, C, numRuns);
    return ret;
}

void ExecuteDev (bool sync_exec, unsigned PE)
{
    gemx::XTimer t;
    GEMXHostHandle<char*>::Instance().gh_ptr[PE]->ExecuteDev(sync_exec);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["ExecuteDev"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["ExecuteDev"]++;
#endif
}
