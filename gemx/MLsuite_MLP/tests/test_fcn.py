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
import numpy as np
import gemx
import time
import test
from test import FcnTest

#testcases example 
def test_perf_fcn(m, k, n, xclbin_opts, post_scale=[1,0], A_range=32764, B_range=32764, bias_range=32764):
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
    gemx.addFCNOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1],1,0)
    gemx.execute()
    gemx.clearInstrBuf()
    gemx.getMat(C_fpga)  
    end_time = time.time()
    total_operations = 2 * m * n * k + m * n * 3
    test.test_perf(end_time-start_time,total_operations,m,k,n,ddrWidth)
    test.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
      
def test_multi_fcn(ins_count, m_size, k_size, n_size, post_scale=[1,0], A_range=32764, B_range=32764):
    mat_A=[]
    mat_C=[]
    mat_bias=[]
    ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
    for i in range(ins_count):
      m_size[i] = test.get_padded_size( m_size[i], int(xclbin_opts["GEMX_gemmMBlocks"]) * ddrWidth)
      k_size[i] = test.get_padded_size(k_size[i], int(xclbin_opts["GEMX_gemmKBlocks"]) * ddrWidth)
      n_size[i] = test.get_padded_size(n_size[i], int(xclbin_opts["GEMX_gemmNBlocks"]) * ddrWidth)
      mat_A.append(np.random.randint(low=-A_range, high=A_range, size=(m_size[i], k_size[i]), dtype=np.int16))
      mat_bias.append(np.zeros ((m_size[i], n_size[i]), dtype=np.int32))
      mat_C.append(np.zeros((m_size[i], n_size[i]), dtype=np.int16, order='C'))
    mat_B0 = np.random.randint(low=-B_range, high=B_range, size=(k_size[0], n_size[0]), dtype=np.int16) 
    for i in range(ins_count):
      gemx.sendMat(mat_A[i])
      gemx.sendMat(mat_C[i])
      gemx.sendMat(mat_bias[i])
    gemx.sendMat(mat_B0)
    gemx.addFCNOp (mat_A[0], mat_B0, mat_C[0], mat_bias[0], post_scale[0], post_scale[1],1,0)    
    gemx.addFCNOp (mat_A[1], mat_C[0], mat_C[1], mat_bias[1], post_scale[0], post_scale[1],1,0) 
    gemx.addFCNOp (mat_A[2], mat_C[1], mat_C[2], mat_bias[2], post_scale[0], post_scale[1],1,0) 
    gemx.addFCNOp (mat_A[3], mat_C[2], mat_C[3], mat_bias[3], post_scale[0], post_scale[1],1,0)
    gemx.execute()
    gemx.clearInstrBuf()
    gemx.getMat(mat_C[0])  
    gemx.getMat(mat_C[1]) 
    gemx.getMat(mat_C[2]) 
    gemx.getMat(mat_C[3]) 
    test.multiply_and_cmp(mat_C[3], mat_A[3], mat_C[2], mat_bias[3], m_size[3], n_size[3], post_scale)

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test=FcnTest()
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createFCNHandle( args, xclbin_opts)
  for j in range (1,3):
      for k in range(1,8):
          for i in range (int(xclbin_opts["GEMX_numKernels"])):
              for m,n in ( [0,0], [1,0]):
                  test.test_basic_randint( i, xclbin_opts, [j,k], [m,n], 2048)    
   
  test.test_basic_size(512,512,512, xclbin_opts)   
  
  size=256
  while size < 8192:
      test_perf_fcn(size,size,size,xclbin_opts)  # run performance measurement
      size =  size * 2
