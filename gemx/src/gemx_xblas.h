#ifndef GEMX_XBLAS_H__
#define GEMX_XBLAS_H__

#include <vector>

#ifdef  __cplusplus
extern "C" {
#endif

enum CBLAS_ORDER { CblasRowMajor, CblascolMajor };
enum CBLAS_TRANSPOSE { CblasNoTrans, CblasTrans };

void xblas_sgemm(const enum CBLAS_ORDER __Order, const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N, const int __K, const GEMX_dataType __alpha, std::vector<GEMX_dataType>& __A, const int __lda, std::vector<GEMX_dataType>& __B, const int __ldb,GEMX_dataType __beta, std::vector<GEMX_dataType>& __C, const int __ldc);

void clearMemory();

#ifdef __cplusplus
}
#endif


#endif //define GEMX_XBLAS_H__