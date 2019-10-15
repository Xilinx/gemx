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
import gemx
import numpy as np
import time
import math
import scipy.io as sio

#usage case:
#1) For testing an example with 4 layers, each weight size is 500*347, 500*500, 2000*500, 8*2000 and input matrix size is 347*1
#    python tests/benchmark_spmv.py --xclbin ./xclbins/u200_201830_1/spmv_float/gemx.xclbin --cfg ./xclbins/u200_201830_1/spmv_float/config_info.dat --gemxlib ./C++/lib/libgemxhost.so --vectors 1 --matrix 500 347 173500 500 500 250000 2000 500 1000000 8 2000 16000
#   python tests/benchmark_spmv.py --xclbin ./xclbins/u200_201830_1/fcn_short/gemx.xclbin --cfg ./xclbins/u200_201830_1/fcn_short/config_info.dat --gemxlib ./C++/lib/libgemxhost.so --engine fcn --vectors 1 --matrix 500 347 1 500 500 1 2000 500 1 8 2000 1
#2) For testing an example with 3 layers, each weight size is 100*128, 25*100, 5*25 and input matrix size is 128*300, no mtx files
#   python tests/benchmark_spmv.py --xclbin ./xclbins/u200_201830_1/spmv_float/gemx.xclbin --cfg ./xclbins/u200_201830_1/spmv_float/config_info.dat --gemxlib ./C++/lib/libgemxhost.so --matrix 100 128 12800 25 100 2500 5 25 125 --vectors 300
#   python tests/benchmark_spmv.py --xclbin ./xclbins/u200_201830_1/fcn_short/gemx.xclbin --cfg ./xclbins/u200_201830_1/fcn_short/config_info.dat --gemxlib ./C++/lib/libgemxhost.so  --engine fcn --vectors 1 --matrix 100 128 300 25 100 300 5 25 300

parser = gemx.default_args()
parser.add_argument('--engine', required = False, choices=['spmv','fcn'], default = 'spmv')
parser.add_argument('--numiter', required = False, type = int, default = 10)
parser.add_argument('-m','--matrix', help='matrix sizes: m k nnz ... or m k n...', nargs="+", type=int, required=False, default = [100,128,12800,25,100,2500,5,25,125])
parser.add_argument('--vectors', required = False, type = int, help='number of vectors', default = 300)
parser.add_argument('--mtx',required = False, help='path to mtx file', nargs="+", default = 'none')

args = parser.parse_args()
xclbin_opt = gemx.parse_cfg ( args.cfg ) 
if args.engine =='spmv':
  gemx.createSPMVHandle(args, xclbin_opt)
else:
  gemx.createFCNHandle(args, xclbin_opt)

A_buf = []
B_buf = []
C_buf = []
bias_buf = []
nnz_size = []

if args.engine =='spmv':
  min_row = int(xclbin_opt["GEMX_spmvMacGroups"]) * int(xclbin_opt["GEMX_spmvWidth"])  
  min_col = int(xclbin_opt["GEMX_ddrWidth"]) 
  if args.mtx =='none':
    #read matrix size from args.matrix
    num_matrix = int(len(args.matrix)/3)
    for i in range(num_matrix):
        m = args.matrix[i*3]
        k = args.matrix[i*3+1]
        nnz = args.matrix[i*3+2]
        #padding part, for size k, we use min_row instead of min_col for multiple matrices 
        m = int( math.ceil( np.float32(m) / min_row ) * min_row ) 
        k = int( math.ceil( np.float32(k) / min_row ) * min_row )
        nnz_size.append(nnz)
        print (m,k,nnz)
        row_array  = np.random.randint(low=0, high=m, size=(nnz, 1), dtype=np.int32)
        col_array  = np.random.randint(low=0, high=k, size=(nnz, 1), dtype=np.int32)
        data_array = np.zeros ((nnz, 1), dtype=np.float32)
        data_array.fill(1)
        A_buf.append(gemx.sendSpMat(row_array,col_array,data_array,m,k,nnz,xclbin_opt))
        B_buf.append(np.zeros ((k, 1), dtype=np.float32))
        C_buf.append(np.zeros ((m, 1), dtype=np.float32))
  else:
    #read matrix from mtx file 
    num_matrix = len(args.mtx)
    for i in range(num_matrix):
        matA = sio.mmread(args.mtx[i])
        row_array = (matA.row).astype(np.int32)
        col_array = (matA.col).astype(np.int32)
        data_array  = (matA.data).astype(np.float32)
        m,k = matA.shape
        nnz = matA.nnz
        nnz_size.append(nnz)
        #padding part, for size k, we use min_row instead of min_col for multiple matrices
        m = int( math.ceil( np.float32(m) / min_row ) * min_row ) 
        k = int( math.ceil( np.float32(k) / min_row ) * min_row )
        print (m,k,nnz)
        A_buf.append(gemx.sendSpMat(row_array,col_array,data_array,m,k,nnz,xclbin_opt))
        B_buf.append(np.zeros ((k, 1), dtype=np.float32))
        C_buf.append(np.zeros ((m, 1), dtype=np.float32))
  
else: #fcn
  min_m = int(xclbin_opt["GEMX_ddrWidth"])  * max (int(xclbin_opt["GEMX_gemmKBlocks"]), int(xclbin_opt["GEMX_gemmMBlocks"]) )
  min_k = int(xclbin_opt["GEMX_ddrWidth"])  * int(xclbin_opt["GEMX_gemmKBlocks"])
  min_n = int(xclbin_opt["GEMX_ddrWidth"])  * max ( int(xclbin_opt["GEMX_gemmKBlocks"]), int(xclbin_opt["GEMX_gemmNBlocks"]) ) 
  if args.mtx =='none':
    num_matrix = len(args.matrix)/3
    for i in range(num_matrix):
        m = args.matrix[i*3]
        k = args.matrix[i*3+1]
        n = args.matrix[i*3+2]
        m = int( math.ceil( np.float32(m) / min_m ) * min_m ) 
        k = int( math.ceil( np.float32(k) / min_k ) * min_k )
        n = int( math.ceil( np.float32(n) / min_n ) * min_n )
        print (m,k,n)
        A = np.zeros ((m, k), dtype=np.int16)
        A.fill(1)
        A_buf.append(A)
        gemx.sendMat(A_buf[i])
        B_buf.append(np.zeros ((k, n), dtype=np.int16))
        C_buf.append(np.zeros ((m, n), dtype=np.int16))
        bias = np.zeros ((m, n), dtype=np.int32)
        bias.fill(1)
        bias_buf.append(bias)
  else:
    #For fcn if read from mtx files, matrix sizes still need to be provided
    num_matrix = len(args.mtx)
    if num_matrix != len(args.matrix)/3:
        raise Exception("please enter sizes for each layer")
    for i in range(num_matrix):
        matA = sio.mmread(args.mtx[i])
        m = args.matrix[i*3]
        k = args.matrix[i*3+1]
        n = args.matrix[i*3+2]
        m = int( math.ceil( np.float32(m) / min_m ) * min_m ) 
        k = int( math.ceil( np.float32(k) / min_k ) * min_k )
        n = int( math.ceil( np.float32(n) / min_n ) * min_n )
        print (m,k,n)
        A = matA.toarray()
        A = A.astype(np.int16)
        A.resize((m,k))
        A_buf.append(A)
        gemx.sendMat(A_buf[i])
        B_buf.append(np.zeros ((k, n), dtype=np.int16))
        C_buf.append(np.zeros ((m, n), dtype=np.int16))
        bias = np.zeros ((m, n), dtype=np.int32)
        bias.fill(1)
        bias_buf.append(bias)
               
B_buf[0].fill(1) #fill vetor B
C_buf.insert(0,B_buf[0])
gemx.sendMat(C_buf[0])
  
for i in range(num_matrix):
  if args.engine =='spmv':
        gemx.sendMat(C_buf[i+1])
        gemx.addSPMVOp(A_buf[i], C_buf[i], C_buf[i+1], nnz_size[i], xclbin_opt,True)
  else:
        gemx.sendMat(C_buf[i+1])     
        gemx.sendMat(bias_buf[i])
        gemx.addFCNOp(A_buf[i], C_buf[i], C_buf[i+1], bias_buf[i], 1, 0, 0, 0)

time.sleep(2)
total_time = 0
for k in range(args.numiter): #interations
    start_time = time.time()
    for j in range(args.vectors):
        gemx.sendMat(C_buf[0])
        for i in range(num_matrix):
            gemx.sendMat(C_buf[i+1])
        gemx.execute()
        gemx.getMat(C_buf[-1])
    total_time += time.time() - start_time

print ("Average FPGA exec time(python): ", (total_time/ args.numiter )*1000, " ms")

#gemx.printStats()
