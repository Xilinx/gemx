#!/bin/bash

echo "Please make sure SDx 2017.4 environment is set up properly before running this script"
echo "Please select the application (gemm_perf, spmv_perf, gemm_test_python, gemx_func_test):"
read app_name

if [ $app_name = "gemm_perf" ]; then
  make clean
  make gemm_perf GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4
elif [ $app_name = "spmv_perf" ]; then
  make clean
  make spmv_perf GEMX_dataType=float GEMX_datEqIntType=int32_t GEMX_ddrWidth=16 GEMX_runSpmv=1 GEMX_runGemm=0 
elif [ $app_name = "gemm_test_python" ]; then
  make clean
  make gemm_test_python GEMX_keepMacBits=1 GEMX_gemmNBlocks=1 GEMX_splitMesh=1
elif [ $app_name =  "gemx_func_test" ]; then
  make clean
  make gemx_func_test GEMX_keepMacBits=1 GEMX_splitMesh=1 GEMX_runGemm=1 GEMX_runGemv=1 GEMX_runSpmv=1 GEMX_runTransp=1 GEN_BIN_PROGRAM="gemm 256 256 256  256 256 256 256 1 0 A1 B1 C1 X1 gemv 256 256 288 A2 B2 C2 spmv 96 128 256 none A3 B3 C3 transp 32 32 64 96 rm cm A4 B4"
else
  echo "invalid application name."
fi
exit
  
  
