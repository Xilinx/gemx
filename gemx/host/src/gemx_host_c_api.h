// namespace

extern "C" {

void MakeFCNHost(char *xclbin, char* device, unsigned int nPE);
void MakeGEMMHost(char *xclbin, char* device, unsigned int nPE);
void MakeSPMVHost(char *xclbin, char* device, unsigned int nPE);
void MakeSPMVBRAMHost(char *xclbin, char* device, unsigned int nPE);

void SendToFPGAShrt(short *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void SendToFPGAInt(int *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void SendToFPGAFloat(float *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void* SendSpToFpgaFloat(int *row, int *col, float *data, unsigned int nnz, unsigned int ddr_width, unsigned PE);
void* SendSpToFpgaFloatBRAM(int *row, int *col, float *data, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE);
void* SendSpToFpgaInt(int *row, int *col, float *data, unsigned int nnz, unsigned int ddr_width, unsigned PE);
void* SendSpToFpgaIntBRAM(int *row, int *col, float *data, unsigned int m, unsigned int k, unsigned int nnz, unsigned int ddr_width, unsigned int spmv_width, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE);
//void SendToFPGAShrt_dbg( char * name, short *A, int m, int n, bool sync_send);
//void SendToFPGAInt_dbg( char * name, int *A, int m, int n, bool sync_send);

void* GetFromFPGA( short *A, unsigned PE, bool sync_get);
void* GetFromFPGAInt( int *A, unsigned PE, bool sync_get);
void* GetFromFPGAFloat( float *A, unsigned PE, bool sync_get);
void Wait (unsigned PE);
void ClearInstrBuf (unsigned PE);
void PrintStats();
bool AddFCNOp( void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha, unsigned PE);
bool AddGEMMOp( void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, unsigned PE);
bool AddSPMVOp(void *A, void * B, void *C, unsigned int m, unsigned int k, unsigned int nnz, unsigned PE);
bool AddSPMVBRAMOp(void *A, void * B, void *C, unsigned int m, unsigned int k, unsigned int nnz, unsigned int num_cblocks, unsigned int capacity_Cblocks, unsigned int capacity_Bblocks, unsigned PE);

int GetFreq ();
void Execute (bool sync_exec, unsigned PE);

void int16_gemm(short * A, short * B, short * X, short *C, unsigned int M, unsigned int K, unsigned int N );

}

