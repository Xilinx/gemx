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
import test
import scipy.io as sio
from test import UspmvTest

xclbin_opts = [] # config data read from config_info.dat

def common_uspmv(rows,cols,datas,m_sizes,k_sizes,nnz_sizes, num_runs,vector_range):
  ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
  min_k = ddrWidth
  min_m = ddrWidth * int(xclbin_opts["GEMX_uspmvInterleaves"]) 
  for i in range(len(m_sizes)):
     m_sizes[i] = test.get_padded_size (m_sizes[i], min_m)
     k_sizes[i] = test.get_padded_size (k_sizes[i], min_m)
  print ("size:",m_sizes,k_sizes,"nnz:",nnz_sizes) 
  B = np.zeros((num_runs, k_sizes[i]), dtype=np.float32)
  test.fillMod(9, num_runs, k_sizes[i],B)
  B = B.astype(np.float32)
  C_list=[B]
  for i in range(len(m_sizes)):
    C = np.zeros ((num_runs, m_sizes[i]), dtype=np.float32)
    C_list.append(C)
    A = gemx.sendUSpMat(np.array(rows[i]).astype(np.uint16),
                        np.array(cols[i]).astype(np.uint16),
                        np.array(datas[i]),
                        np.array(m_sizes[i],dtype=np.int32),
                        np.array(k_sizes[i],dtype=np.int32),
                        np.array(nnz_sizes[i],dtype=np.int32),
                        np.array(1,dtype=np.float32),
                        xclbin_opts)  
    gemx.sendMat(C_list[i])
    gemx.sendMat(C_list[i+1])
    gemx.addUSPMVOp(A,C_list[i],C_list[i+1],num_runs)
  gemx.execute()
  gemx.clearInstrBuf()
  gemx.getMat(C_list[-1])
  test.multiply_and_cmp_uspmv(rows,cols,datas,m_sizes,k_sizes,B,C_list[-1])

def test_uspmv_mtxfile(mtxpaths,num_runs,vector_range):
  rows = []
  cols = []
  datas = []
  m_sizes = []
  k_sizes = []
  nnz_sizes = []
  for mtx in mtxpaths:
    matA = sio.mmread(mtx)
    row=(matA.row).astype(np.int32)
    col=(matA.col).astype(np.int32)
    data=(matA.data).astype(np.float32)
    m,k = matA.shape
    nnz = matA.nnz        
    test.uspmv_check_maximum(xclbin_opts, m, nnz)
    while nnz %  int(xclbin_opts["GEMX_ddrWidth"]) !=0:
      nnz=nnz+1
      row = (np.append(row,0)).astype(np.uint16)
      col = (np.append(col,0)).astype(np.uint16)
      data = (np.append(data,0)).astype(np.float32)
    ind = np.lexsort((row,col))
    row=row[ind]
    col=col[ind]
    data=data[ind]
    rows.append(row)
    cols.append(col)
    datas.append(data)
    m_sizes.append(m)
    k_sizes.append(k)
    nnz_sizes.append(nnz)
  common_uspmv(rows,cols,datas,m_sizes,k_sizes,nnz_sizes, num_runs, vector_range) 

def test_uspmv_random(m_sizes, k_sizes, nnz_sizes, num_runs, vector_range=32764):
  rows = []
  cols = []
  datas = []
  for i in range(len(m_sizes)):
     test.uspmv_check_maximum(xclbin_opts, m_sizes[i], nnz_sizes[i])
     while nnz_sizes[i] %  int(xclbin_opts["GEMX_ddrWidth"]) !=0:
         nnz_sizes[i] = nnz_sizes[i] + 1
     row = np.random.randint(low=0, high=m_sizes[i], size=(nnz_sizes[i], 1), dtype=np.uint16).flatten()
     col = np.random.randint(low=0, high=k_sizes[i], size=(nnz_sizes[i], 1), dtype=np.uint16).flatten()
     data = np.random.randint(low=0, high=32764, size=(nnz_sizes[i], 1), dtype=np.int32).flatten()
     ind = np.lexsort((row,col))
     row=row[ind]
     col=col[ind]
     data=data[ind]
     data = data.astype(np.float32)
     rows.append(row)
     cols.append(col)
     datas.append(data)
  common_uspmv(rows,cols,datas,m_sizes,k_sizes,nnz_sizes, num_runs, vector_range) 

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test = UspmvTest()
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createUSPMVHandle(args, xclbin_opts)

  #test_uspmv_mtxfile(['PATH_TO_MTXFILE/xx.mtx'],300,32764)
  #array of m sizes, array of k sizes,array of nnz sizes, number of Vectors
  test_uspmv_random([100,25,5],[128,100,25],[12800,2500,125], 300, 32764)
  gemx.printStats()
