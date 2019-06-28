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
// namespace

extern "C" {

void MakeFCNHost(char *xclbin, unsigned int nPE);
void MakeGEMMHost(char *xclbin, unsigned int nPE);
void MakeUSPMVHost(char *xclbin, unsigned int nPE);
void MakeSPMVHost(char *xclbin, unsigned int nPE);

void SendToFPGAShrt(short *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void SendToFPGAInt(int *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void SendToFPGAFloat(float *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void* SendUSpMat(uint16_t* row, uint16_t* col, float* data, int* row_size, int* col_size, int* nnz_size, float* p_pRelu, unsigned int t_DdrWidth, unsigned int t_Stages, unsigned PE);
void* SendSpToFpgaFloat(int *row, int *col, float *data, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE);
void* SendSpToFpgaInt(int *row, int *col, float *data, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE);

void* GetFromFPGA( short *A, unsigned PE, bool sync_get);
void* GetFromFPGAInt( int *A, unsigned PE, bool sync_get);
void* GetFromFPGAFloat( float *A, unsigned PE, bool sync_get);
void Wait (unsigned PE);
void ClearInstrBuf (unsigned PE);
void PrintStats();
bool AddFCNOp( void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha, unsigned PE);
bool AddGEMMOp( void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, unsigned PE);
bool AddUSPMVOp(void *A, void * B, void *C, unsigned int numRuns, unsigned PE);
bool AddSPMVOp(void *A, void * B, void *C, unsigned int m, unsigned int k, unsigned int nnz, bool l_pRelu, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE);

void Execute (bool sync_exec, unsigned PE);

void int16_gemm(short * A, short * B, short * X, short *C, unsigned int M, unsigned int K, unsigned int N );

//C callable functions for using librar-based memory allocation
void MakeStrGEMMHost(const char *xclbin, unsigned int nPE);
void MakeStrFCNHost(const char *xclbin, unsigned int nPE);
void MakeStrSPMVHost(const char *xclbin, unsigned int nPE);
void MakeStrUSPMVHost(const char *xclbin, unsigned int nPE);

bool AllocProgBuf(unsigned int buf_sz, unsigned PE);
void* AddDevBuf(char* A, unsigned int buf_sz, unsigned PE);
void* AddSpDevBuf(int * row, int * col, float * data, char* A, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE);
void* AddUSpDevBuf(uint16_t* row, uint16_t* col, float* data, char* A, int* row_size, int* col_size, int* nnz_size, float* p_pRelu, unsigned int t_DdrWidth, unsigned int t_Stages, unsigned PE);
void SendDevBuf(char* A, unsigned PE, bool sync_send);
void* GetDevBuf(char* A, unsigned PE, bool sync_get);
bool AddGEMMDevOp(char* A, char* B, char*C, char* bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, unsigned PE);
bool AddFCNDevOp(char* A, char* B, char*C, char* bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha, unsigned PE);
bool AddSPMVDevOp(char* A, char* B, char*C, unsigned int m, unsigned int k, unsigned int nnz, bool l_pRelu, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE);
bool AddUSPMVDevOp(char* A, char* B, char*C, unsigned int numRuns, unsigned PE);


void ExecuteDev (bool sync_exec, unsigned PE);

}

