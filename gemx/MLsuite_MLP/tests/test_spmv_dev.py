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
import math
import scipy.io as sio
import scipy.sparse as sp
from test import SpmvTest

# this test allows you to run sw emu or hw emu with python API

xclbin_opts = []

def get_padded_size (size, min_size):
  size_padded = int( math.ceil( np.float32(size) / min_size ) * min_size ) 
  return size_padded

def common_spmv_dev(row,col,data,m,k,nnz,A_handle,B_handle,C_handle):
  ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
  min_k = ddrWidth
  spmvWidth = int(xclbin_opts["GEMX_spmvWidth"])
  min_m = spmvWidth * int(xclbin_opts["GEMX_spmvMacGroups"])
  m = get_padded_size (m, min_m)
  k = get_padded_size (k, min_k)
  print ("size:",m,k,"nnz:",nnz)  
  A = gemx.addSpDevBuf(row,col,data,A_handle,m,k,nnz,xclbin_opts)  
  B = gemx.addDevBuf(B_handle,k,1,np.float32)
  B_tmp = np.zeros ((k, 1), dtype=np.float32)
  test.fillMod(B_tmp,k)
  B[:]  = B_tmp  
  C = gemx.addDevBuf(C_handle,m,1,np.float32)
  C.fill(0)
  gemx.sendDevBuf(A_handle)
  gemx.sendDevBuf(B_handle)
  gemx.sendDevBuf(C_handle)
  gemx.addSPMVDevOp(A_handle,B_handle,C_handle,m,k,nnz,xclbin_opts)
  gemx.executeDev()
  gemx.getDevBuf(C_handle)  
  test.multiply_and_cmp_spmv(row,col,data,m,k,nnz,B,C)  

def test_spmv_mtxfile_dev(mtxpath,A_handle,B_handle,C_handle):
  matA = sio.mmread(mtxpath)
  if sp.issparse(matA):
     row = (matA.row).astype(np.int32)
     col = (matA.col).astype(np.int32)
     data = (matA.data).astype(np.float32)
     m,k = matA.shape
     nnz = matA.nnz         
     common_spmv_dev(row,col,data,m,k,nnz,A_handle,B_handle,C_handle)
  else:
     print ("only sparse matrix is supported")

def test_spmv_random_dev(m,k,nnz,A_handle,B_handle,C_handle):
  row  = np.random.randint(low=0, high=m, size=(nnz, 1), dtype=np.int32)
  col  = np.random.randint(low=0, high=k, size=(nnz, 1), dtype=np.int32)
  data = np.random.randint(low=-32764, high=32764, size=(nnz, 1), dtype=np.int32)
  data = data.astype(np.float32)
  common_spmv_dev(row,col,data,m,k,nnz,A_handle,B_handle,C_handle)

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createStrSPMVHandle(args, xclbin_opts)
  test = SpmvTest()
  gemx.allocProgBuf(20971520)
  test_spmv_random_dev(96, 128, 256,b'A1',b'B1',b'C1')
  test_spmv_random_dev(96, 128, 256,b'A2',b'B2',b'C2') 

