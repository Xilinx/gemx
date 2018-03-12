#include "gemx_host.h"
#include <iostream>
#include <fstream>
using namespace std;

template<typename T>
void print(char *name,T * A, int m, int n){
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

gemx::FCNHost<short*> * MakeFCNHost(char *xclbin, char * kernName) {
    return new gemx::FCNHost<short*>(xclbin, kernName);
}

void DestroyFCNHost(gemx::FCNHost<short*> * ptr) {
    delete ptr;
}

void SendToFPGAShrt(gemx::FCNHost<short*> * gh, short *A, unsigned long long num_elem, bool sync_send){
    SendToFPGA(gh, A, sizeof(short) * num_elem, sync_send);
}

void SendToFPGAInt(gemx::FCNHost<short*> * gh, int *A, unsigned long long num_elem, bool sync_send){
    SendToFPGA(gh, A, sizeof(int) *num_elem, sync_send);
}

void SendToFPGAShrt_dbg(gemx::FCNHost<short*> * gh, char * name, short *A, int m, int n, bool sync_send){
    print<short>(name, A, m,n);
    SendToFPGA(gh, name, sizeof(short) * m * n, sync_send);
}

void SendToFPGAInt_dbg(gemx::FCNHost<short*> * gh, char * name, int *A, int m, int n, bool sync_send){
    print<int>(name, A, m,n);
    SendToFPGA(gh, name, sizeof(int) * m * n, sync_send);
}
void SendToFPGA(gemx::FCNHost<short*> * gh, void * A, unsigned long long buf_sz, bool sync_send) {
    gh->SendToFPGA((short*)A, A, buf_sz, sync_send);
}
void* GetFromFPGA(gemx::FCNHost<short*> * gh, short *A, bool sync_get)
{
    return gh->GetMat((short*)A, true, sync_get);
}

bool AddFCNOp(gemx::FCNHost<short*> * gh, void * A, void * B, void *C, void * bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha )
{
    //cout << C << " = " << A << " * " << B << " + " << bias << endl;
    return gh->AddFCNOp((short*)A, (short*)B,(short*)C, (short*)bias, m,k,n, postScale, postShift, PReLUScale, PReLUAlpha);
}

void Execute (gemx::FCNHost<short*> * gh)
{
    gh->Execute();
}

void Wait (gemx::FCNHost<short*> * gh)
{
    gh->Wait();
}




gemx::GEMMHost<short*> * MakeGEMMHost(char *xclbin, char * kernName) {
    return new gemx::GEMMHost<short*>(xclbin, kernName);
}

void DestroyGEMMHost(gemx::GEMMHost<short*> * ptr) {
    delete ptr;
}

void SendToFPGAShrt_GEMM(gemx::GEMMHost<short*> * gh, short *A, unsigned long long num_elem, bool sync_send){
    SendToFPGA_GEMM(gh, A, sizeof(short) * num_elem, sync_send);
}

void SendToFPGAInt_GEMM(gemx::GEMMHost<short*> * gh, int *A, unsigned long long num_elem, bool sync_send){
    SendToFPGA_GEMM(gh, A, sizeof(int) *num_elem, sync_send);
}

void SendToFPGAShrt_dbg_GEMM(gemx::GEMMHost<short*> * gh, char * name, short *A, int m, int n, bool sync_send){
    print<short>(name, A, m,n);
    SendToFPGA_GEMM(gh, name, sizeof(short) * m * n, sync_send);
}

void SendToFPGAInt_dbg_GEMM(gemx::GEMMHost<short*> * gh, char * name, int *A, int m, int n, bool sync_send){
    print<int>(name, A, m,n);
    SendToFPGA_GEMM(gh, name, sizeof(int) * m * n, sync_send);
}
void SendToFPGA_GEMM(gemx::GEMMHost<short*> * gh, void * A, unsigned long long buf_sz, bool sync_send) {
    gh->SendToFPGA((short*)A, A, buf_sz, sync_send);
}
void* GetFromFPGA_GEMM(gemx::GEMMHost<short*> * gh, short *A, bool sync_get)
{
    return gh->GetMat((short*)A, true, sync_get);
}
bool AddGEMMOp(gemx::GEMMHost<short*> * gh, void * A, void * B, void *C, void * bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift)
{
    //cout << C << " = " << A << " * " << B << " + " << bias << endl;
    return gh->AddGEMMOp((short*)A, (short*)B,(short*)C, (short*)bias, m,k,n, postScale, postShift);
}

void Execute_GEMM (gemx::GEMMHost<short*> * gh)
{
    gh->Execute();
}

void Wait_GEMM (gemx::GEMMHost<short*> * gh)
{
    gh->Wait();
}


#if 0
void int16GEMM(gemx::FCNHost<short*> * gh, void * A, void * B,
        void * C, unsigned int M, unsigned int K, unsigned int N,
        unsigned int lda, unsigned int ldb, unsigned int ldc, bool sendA,
        bool sendB, bool sendC) {
    using namespace std;
    using namespace gemx;
    short * shrt_A = (short*) A;
    short * shrt_B = (short*) B;
    short * shrt_C = (short*) C;

    cout << "A_ptr: " << A << " B_ptr: " << B << " C_ptr: " << C << endl;
    /*
     gh->AddMat(A, A, M, K, lda);
     gh->AddMat(B, B, K, N, ldb);
     gh->AddMat(shrt_C, shrt_C, M, N, ldc );
     gh->SendToFPGA(A);
     gh->SendToFPGA(B);
     gh->SendToFPGA(shrt_C);
     */

    if (sendA) gh->SendToFPGA(shrt_A, shrt_A, M, K, lda);
    if (sendB) gh->SendToFPGA(shrt_B, shrt_B, K, N, ldb);
    if (sendC) gh->SendToFPGA(shrt_C, shrt_C, M, N, ldc);

    gh->AddFCNOp(shrt_A, shrt_B, shrt_C, 0, 1, 0, 1, 0);
    //gh->AddGEMMOp(shrt_A, shrt_B, shrt_C);

    gh->Execute();

    std::shared_ptr<gemx::Mat<short> > C_mat = gh->GetMat(shrt_C, true);
    /*
     std::shared_ptr< gemx::Mat<short> > A_mat = gh->GetMat ( A , true );
     std::shared_ptr< gemx::Mat<short> > B_mat = gh->GetMat ( B , true );
     //Mat<short> l_matC_cpu ( M, N, N);

     //l_matC_cpu.multiply(*A_mat, *B_mat);
     //l_matC_cpu.cmp(0, 0, *C_mat);

     //std::cout << A_mat->rows() << " " << A_mat->cols() << " " << A_mat->ld() << std::endl;
     //std::cout << B_mat->rows() << " " << B_mat->cols() << " " << B_mat->ld() << std::endl;
     //std::cout << C_mat->rows() << " " << C_mat->cols() << " " << C_mat->ld() << std::endl;
     *
     */
}
#endif
