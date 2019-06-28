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
import test
import random
import scipy.io as sio
import scipy.sparse as sp
from test import SpmvTest

xclbin_opts = [] # config data read from config_info.dat

def get_padded_size (size, min_size):
  size_padded = int( math.ceil( np.float32(size) / min_size ) * min_size ) 
  return size_padded

def common_spmv(row,col,data,m,k,nnz,vector_range):
  if xclbin_opts["GEMX_dataType"] == "float":
    data_type = np.float32
  elif xclbin_opts["GEMX_dataType"] == "int32_t":
    data_type = np.int32
  else:
     raise TypeError("type", xclbin_opts["GEMX_dataType"], "not supported") 
  ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
  min_k = ddrWidth
  spmvWidth = int(xclbin_opts["GEMX_spmvWidth"])
  min_m = spmvWidth * int(xclbin_opts["GEMX_spmvMacGroups"])
  m = get_padded_size (m, min_m)
  k = get_padded_size (k, min_k)
  print ("size:",m,k,"nnz:",nnz)  
  if data_type == np.int32:
     B = np.random.randint(low=-vector_range, high=vector_range, size=(k, 1), dtype=np.int32)
  else:
     B = np.zeros ((k, 1), dtype=np.float32)
     test.fillMod(B,k,vector_range)
  C = np.zeros ((m, 1), dtype=data_type)
  A = gemx.sendSpMat(row,col,data,m,k,nnz,xclbin_opts)  
  gemx.sendMat(B)
  gemx.sendMat(C)
  gemx.addSPMVOp(A,B,C,nnz,xclbin_opts)
  gemx.execute()
  gemx.clearInstrBuf()
  gemx.getMat(C)
  test.multiply_and_cmp_spmv(row,col,data,m,k,nnz,B,C)

def test_spmv_mtxfile(mtxpath,vector_range):
  matA = sio.mmread(mtxpath)
  if sp.issparse(matA):
     row = (matA.row).astype(np.int32)
     col = (matA.col).astype(np.int32)
     data = (matA.data).astype(np.float32)
     m,k = matA.shape
     nnz = matA.nnz         
     common_spmv(row,col,data,m,k,nnz,vector_range)
  else:
     print ("only sparse matrix is supported")

def test_spmv_random(m,k,nnz,vector_range=32764):
  row  = np.random.randint(low=0, high=m, size=(nnz, 1), dtype=np.int32)
  col  = np.random.randint(low=0, high=k, size=(nnz, 1), dtype=np.int32)
  data = np.zeros ((nnz, 1), dtype=np.float32)
  nnz_min = random.randint(-vector_range, vector_range)
  for i in range(nnz):
     nnz_min += 0.3
     data[i,0] = nnz_min
  common_spmv(row,col,data,m,k,nnz,vector_range) 

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test = SpmvTest()
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createSPMVHandle(args, xclbin_opts)
  #mtx file must be in Matrix Market format
  #test_spmv_mtxfile("./data/spmv/c-67.mtx",32764) 
  test_spmv_random(96,128,256,32764)
  test_spmv_random(65472,65472,500000,32764) 
  test_spmv_random(12800,12800,1400000,32764) 
  gemx.printStats()
  
