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
import scipy.io as sio
from test import UspmvTest

xclbin_opts = [] # config data read from config_info.dat

def common_uspmv_dev(rows,cols,datas,m_sizes,k_sizes,nnz_sizes, num_runs,A_handles,B_handles,C_handles):
  ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
  stages = int(xclbin_opts["GEMX_uspmvStages"])
  min_k = ddrWidth
  min_m = ddrWidth * int(xclbin_opts["GEMX_uspmvInterleaves"]) 
  for i in range(len(m_sizes)):
     m_sizes[i] = test.get_padded_size (m_sizes[i], min_m)
     k_sizes[i] = test.get_padded_size (k_sizes[i], min_m)
  print ("size:",m_sizes,k_sizes,"nnz:",nnz_sizes)    
  B = gemx.addDevBuf(B_handles[0],num_runs, k_sizes[0],np.float32)
  B_tmp = np.zeros((num_runs, k_sizes[0]), dtype=np.float32)
  test.fillMod(9, num_runs, k_sizes[0],B_tmp)
  B[:]=B_tmp
  C_list=[]
  for i in range(len(m_sizes)):
    C = gemx.addDevBuf(C_handles[i],num_runs, m_sizes[-1],np.float32)
    C.fill(0)
    C_list.append(C)
    A = gemx.addUSpDevBuf(np.array(rows[i]).astype(np.uint16),
                        np.array(cols[i]).astype(np.uint16),
                        np.array(datas[i]),
                        A_handles[i],
                        np.array(m_sizes[i],dtype=np.int32),
                        np.array(k_sizes[i],dtype=np.int32),
                        np.array(nnz_sizes[i],dtype=np.int32),
                        np.array(1,dtype=np.float32),xclbin_opts)
    gemx.sendDevBuf(A_handles[i])
    gemx.sendDevBuf(B_handles[i])
    gemx.sendDevBuf(C_handles[i])
    gemx.addUSPMVDevOp(A_handles[i],B_handles[i],C_handles[i],num_runs)

  gemx.executeDev()
  gemx.getDevBuf(C_handles[-1])
  test.multiply_and_cmp_uspmv(rows,cols,datas,m_sizes,k_sizes,B,C_list[-1])

def test_uspmv_mtxfile_dev(mtxpaths,num_runs,A_handles,B_handles,C_handles):
  stages = int(xclbin_opts["GEMX_uspmvStages"])
  if len(mtxpaths) != stages:
     print ('please give mtx files for each stage')
  else:
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
       print (row)
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
     common_uspmv_dev(rows,cols,datas,m_sizes,k_sizes,nnz_sizes, num_runs,A_handles,B_handles,C_handles) 

def test_uspmv_random_dev(m_sizes, k_sizes, nnz_sizes, num_runs,A_handles,B_handles,C_handles):
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
  common_uspmv_dev(rows,cols,datas,m_sizes,k_sizes,nnz_sizes, num_runs,A_handles,B_handles,C_handles) 

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test = UspmvTest()
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createStrUSPMVHandle(args, xclbin_opts)
  gemx.allocProgBuf(20971520)
  #test_uspmv_mtxfile_dev(['../data/spmv/keras_weight_2.mtx'],1,b'A',b'B',b'C')
  #array of m sizes, array of k sizes,array of nnz sizes, number of Vectors
  test_uspmv_random_dev([100,25,5],[128,100,25],[12800,2500,125], 20, [b'A0',b'A1',b'A2'],[b'B',b'C0',b'C1'],[b'C0',b'C1',b'C2'])
  gemx.printStats()
 
