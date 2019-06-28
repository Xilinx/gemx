 # Copyright 2019 Xilinx, Inc.
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
########################################
# Brief: This example demonstrates Python interface to GEMM matrix math engine run on Xilinx FPGA
# Usage: 
#  export PYTHONPATH=./python  #point the PYTHONPATH to the location of gemx.py file
#  python tests/test_gemm.py --xclbin ./xclbins/u200_201830_1/gemm_short/gemx.xclbin --cfg ./xclbins/u200_201830_1/gemm_short/config_info.dat --gemxlib ./C++/lib/libgemxhost.so
#
# Code structure:
#   The main function in test_gemm.py takes the steps below to offload GEMM operations:
#     1. Read command line options information to args and config_info.dat information to xclbin_opts 
#     2. Create GEMM handle using the above information
#     3. Run test function hard coded in test_gemm.py
#
# Users could add more testcases with different parameters in main function to run them. The Common test functions in test.py could be used as examples to create customised test functions. 
# In test.py, test_basic_randint randomly initializes input matrices with given matrix sizes.
# def test_basic_randint (self,PE, m, k, n, post_scale):
#
#   int16_max = np.iinfo(np.int16).max
#   int16_min = np.iinfo(np.int16).min
#   int32_max = np.iinfo(np.int32).max
#   int32_min = np.iinfo(np.int32).min
#   mat_A = np.random.randint(low=int16_min, high=int16_max, size=(m, k), dtype=np.int16)
#   mat_B = np.random.randint(low=int16_min, high=int16_max, size=(k, n), dtype=np.int16)  
#   bias = np.random.randint(low=int32_min, high=int32_max, size=(m, n), dtype=np.int32)      
#   self.test_basic(PE,mat_A, mat_B, bias, post_scale)
# 
# test_basic function takes input matrices, sends matrices and operation instructions to the engine/kernel running on and FPGA card, launches the kernel and then reads the results back. It also calls multiply_and_cmp function to calculate golden results locally and compares golden results to the results from the FPGA.
# 
#   gemx.sendMat(mat_A,PE)
#   gemx.sendMat(mat_B,PE)
#   gemx.sendMat(C_fpga,PE)    
#   gemx.sendMat(bias, PE)
#   gemx.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], PE) # default test_basic will call addGEMMOp
#   gemx.execute(PE)
#   gemx.getMat(C_fpga,PE)
#   self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)

import numpy as np
import gemx
import time
from test import GemmTest

    
#testcases example 
def test_perf_gemm(m, k, n, xclbin_opts, post_scale=[1,0], A_range=32764, B_range=32764, bias_range=32764):
    ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
    m = test.get_padded_size(m, int(xclbin_opts["GEMX_gemmMBlocks"]) * ddrWidth)
    k = test.get_padded_size(k, int(xclbin_opts["GEMX_gemmKBlocks"]) * ddrWidth)
    n = test.get_padded_size(n, int(xclbin_opts["GEMX_gemmNBlocks"]) * ddrWidth)
    mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
    mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)  
    bias = []
    if bias_range != 0:
        bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
    else:
        bias = np.zeros ((m, n), dtype=np.int32, order='C');   
    C_fpga = np.zeros( (m, n), dtype=np.int16)
    start_time = time.time()  
    gemx.sendMat(mat_A)
    gemx.sendMat(mat_B)
    gemx.sendMat(C_fpga)    
    gemx.sendMat(bias)
    gemx.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1])
    gemx.execute()
    gemx.clearInstrBuf()
    gemx.getMat(C_fpga)  
    end_time = time.time()
    total_operations = 2 * m * n * k + m * n * 3
    test.test_perf(end_time-start_time,total_operations,m,k,n,ddrWidth)
    test.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
      
if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test = GemmTest()
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createGEMMHandle(args, xclbin_opts)
  
  test.test_basic_size(512,512,512,xclbin_opts)   
  
  size=256
  while size < 8192:
      test_perf_gemm(size,size,size, xclbin_opts) # run performance measurement
      size =  size * 2
  
