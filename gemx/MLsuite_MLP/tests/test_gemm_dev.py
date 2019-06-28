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
from test import GemmTest

# this test allows you to run sw emu or hw emu with python API

def test_gemm_dev(m,k,n,A_handle,B_handle,C_handle,X_handle):
  # create an empty array with host memory address return by addDevBuf
  # then fill them with random values without changing the host memory address (For result C, just fill with zeros)
  A = gemx.addDevBuf(A_handle,m,k,np.int16) 
  A[:] = test.gen_rand_matrix ( np.int16, m, k)
  B = gemx.addDevBuf(B_handle,k,n,np.int16)
  B[:] = test.gen_rand_matrix ( np.int16, k, n)
  X = gemx.addDevBuf(X_handle,m,n,np.int32)
  X[:] = test.gen_rand_matrix ( np.int32, m, n)
  C = gemx.addDevBuf(C_handle,m,n,np.int16)
  C.fill(0)
  #now, send to FPGA
  gemx.sendDevBuf(A_handle)
  gemx.sendDevBuf(B_handle)
  gemx.sendDevBuf(X_handle)
  gemx.sendDevBuf(C_handle)
  #send instrution
  gemx.addGEMMDevOp(A_handle,B_handle,C_handle,X_handle,m,k,n)
  #execute
  gemx.executeDev()
  #get result C
  gemx.getDevBuf(C_handle)  
  #compare with golden reference
  test.multiply_and_cmp(C, A, B, X, 128, 128, [1,0])
  
if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createStrGEMMHandle(args, xclbin_opts)
  test = GemmTest()
  gemx.allocProgBuf(2097152)
  test_gemm_dev(128, 128, 128, b'A1',b'B1',b'C1',b'X1')
