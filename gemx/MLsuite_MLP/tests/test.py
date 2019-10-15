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
import sys
import random
import subprocess
import math
import scipy.sparse as sp

class Test:
  """
  class provide helper functions to create test cases
  
  """
  def cmp(self,A, B):
    """
    compare if two arrays are totally the same
    
    Parameters
    ----------
    A:         ndarray
               dense matrix in the host memory
    B:         ndarray
               dense matrix in the host memory
    """
    if np.array_equal(A, B):
        print ("Success!\n")
    else:
        print ("not equal :(")
        sys.exit(1)
          
  def cmpWithinTolerance(self,A, B):
    """
    compare if two arrays are equal within tolerance
        
        
    Parameters
    ----------
    A:         ndarray
               dense matrix in the host memory
    B:         ndarray
               dense matrix in the host memory
    """
    if np.allclose(A, B,1e-3,1e-5):
        print ("Success!\n")
    else:
        print (A.shape, B.shape)
        np.savetxt("C.np", A, fmt="%f")
        np.savetxt("C_cpu.np", B, fmt="%f")
        diff = np.isclose(A, B,1e-3,1e-5)
        countDiff = diff.shape[0] - np.count_nonzero(diff)
        print ("not equal, number of mismatches = ", countDiff)
        mismatch = ((diff==0).nonzero())
        print ("mismatches are in ",mismatch[0])
        for i in mismatch[0]:
          print (A[i]," is different from ",B[i])
        sys.exit(1)  
          
  def multiply_and_cmp(self,C, A, B, X, m, n, post_scale, pRelu_val = [1,0]):
    """
    calculate matrix multiply golden result matrix C ( C = relu ((A * B + bias) * postScale >> postShift) ) on CPU and compare result to FPGA result.\n
    will save the results to text files if not equal
    
    Parameters
    ----------
    C:          ndarray
                output dense matrix in the host memory
    A:          ndarray
                dense matrix in the host memory
    B:          ndarray
                dense matrix in the host memory
    X:          ndarray
                dense matrix in the host memory
    post_scale: (int,int)
                (postScale, postShift)
    pRelu_val:  (int,int)
                (PReLUScale, PReLUAlpha) 
    """
    if C.dtype==np.float32:
        C_cpu = np.matmul(A,B,dtype=np.float32) 
        C_cpu = C_cpu + X.astype(np.float32)
        if pRelu_val == [0,0]:
            C_cpu = C_cpu.clip(min=0)
        C_cpu = C_cpu.astype(np.float32)     
    else:
        m64 = np.int64(np.round(np.matmul(np.float64(A), np.float64(B))))  # intermediate accumulation to 64 bits
        #print ("float64 compute elapsed:", time.time() - start_compute)
        #m64 = np.matmul(np.int64(A), np.int64(B)) # intermediate accumulation to 64 bits
        bias64 = np.int64(X)  # bias to 64 bits
        output64 = m64 + bias64
        o64d = output64 * post_scale[0]
        o64m = o64d // (2 ** post_scale[1])
        o64m = np.int16(o64m)
        if pRelu_val != [1,0]:
            for entry in np.nditer(o64m, op_flags=['readwrite']):
                if entry < 0:
                    entry[...] = entry * pRelu_val[0] // (2 ** pRelu_val[1])
        C_cpu = np.int16(o64m)  # scale down for 16 bits    
    if C.dtype==np.float32:
        if np.allclose(C, C_cpu,1e-1,1e-1):
            print ("Success!\n")
        else:
            print ("Not equal!")
            print (C.shape, C_cpu.shape)
            diff = np.isclose(C.flatten(), C_cpu.flatten(),1e-1,1e-1)
            countDiff = diff.shape[0] - np.count_nonzero(diff)
            print ("not equal, number of mismatches = ", countDiff)
            mismatch = ((diff==0).nonzero())
            print ("mismatches are in ",mismatch[0])
            for i in mismatch[0]:
                print (C.flatten()[i]," is different from ",C_cpu.flatten()[i])
            np.savetxt("cpu_out.np", C_cpu, fmt="%f")
            np.savetxt("fpga_out.np", C, fmt="%f")
            np.savetxt("bias.np", X, fmt="%f")
            np.savetxt("A.np", A, fmt="%f")
            np.savetxt("B.np", B, fmt="%f")
            sys.exit(1)              
    else:
        if np.array_equal(C, C_cpu):
            print ("Success!\n")
        else:
            print ("Not equal!")
            print (C.shape, C_cpu.shape)
            np.savetxt("cpu_out.np", C_cpu, fmt="%d")
            np.savetxt("fpga_out.np", C, fmt="%d")
            np.savetxt("bias.np", X, fmt="%d")
            np.savetxt("A.np", A, fmt="%d")
            np.savetxt("B.np", B, fmt="%d")
            sys.exit(1)    
 
  def get_padded_size (self, size, min_size):
    size_padded = int( math.ceil( np.float32(size) / min_size ) * min_size ) 
    return size_padded
  
  def gen_rand_dim (self, min_mult, max):
    rand_dim = random.randint(1, int(max/min_mult))
    return rand_dim * min_mult
  
  def gen_rand_matrix (self, dtype, row, col):
    if dtype == np.float32:
        return np.random.uniform(low=-128, high=128, size=(row, col)).astype(np.float32)
    else:
        max_val = np.iinfo(dtype).max
        min_val = np.iinfo(dtype).min
        return np.random.randint(low=min_val, high=max_val, size=(row, col), dtype=dtype)
      
  def test_basic_randint (self,PE, xclbin_opts, post_scale, max_dim):
    """
    generate random input matrices with random sizes and run the test 
    
    Parameters
    ----------  
    PE:         int
                number of kernels - 1
    post_scale: (int,int)
                (postScale, postShift)
    max_dim:    int
                control the max size generated by the test
    """
    ddrwidth = int(xclbin_opts["GEMX_ddrWidth"])
    rand_m = self.gen_rand_dim ( ddrwidth * int(xclbin_opts["GEMX_gemmMBlocks"]), max_dim )
    rand_k = self.gen_rand_dim ( ddrwidth * int(xclbin_opts["GEMX_gemmKBlocks"]), max_dim )
    rand_n = self.gen_rand_dim ( ddrwidth * int(xclbin_opts["GEMX_gemmNBlocks"]), max_dim )
    if xclbin_opts["GEMX_dataType"]=="short":
        mat_A = self.gen_rand_matrix ( np.int16, rand_m, rand_k)
        mat_B = self.gen_rand_matrix ( np.int16, rand_k, rand_n)
        bias = self.gen_rand_matrix ( np.int32, rand_m, rand_n)   
    else: #float
        mat_A = self.gen_rand_matrix ( np.float32, rand_m, rand_k)
        mat_B = self.gen_rand_matrix ( np.float32, rand_k, rand_n)
        bias = self.gen_rand_matrix ( np.float32, rand_m, rand_n)  
    self.test_basic(PE,xclbin_opts, mat_A, mat_B, bias, post_scale)
      
  def test_basic_size(self, m, k, n, xclbin_opts, PE = 0, post_scale=[1,0]):
    """
    generate random input matrices with given sizes and run the test 
    
    Parameters
    ----------  
    m:           int
                 number of rows of first dense matrix
    k:           int
                 number of cols of first dense matrix
    n:           int
                 number of cols of second dense matrix                
    xclbin_opts: dictionary 
                 information read from config_info.dat used to build the xclbin
    PE:          int
                 number of kernels - 1
    post_scale:  (int,int)
                 (postScale, postShift)
    """
    ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
    padded_m = self.get_padded_size(m, int(xclbin_opts["GEMX_gemmMBlocks"]) * ddrWidth)
    padded_k = self.get_padded_size(k, int(xclbin_opts["GEMX_gemmKBlocks"]) * ddrWidth)
    padded_n = self.get_padded_size(n, int(xclbin_opts["GEMX_gemmNBlocks"]) * ddrWidth)
    
    if xclbin_opts["GEMX_dataType"]=="short":
        mat_A = self.gen_rand_matrix ( np.int16, padded_m, padded_k)
        mat_B = self.gen_rand_matrix ( np.int16, padded_k, padded_n)
        bias = self.gen_rand_matrix ( np.int32, padded_m, padded_n)
    else: #float
        mat_A = self.gen_rand_matrix ( np.float32, padded_m, padded_k)
        mat_B = self.gen_rand_matrix ( np.float32, padded_k, padded_n)
        bias = self.gen_rand_matrix ( np.float32, padded_m, padded_n)
    self.test_basic(PE, xclbin_opts, mat_A, mat_B, bias, post_scale)
      
  def test_basic(self,PE, xclbin_opts, mat_A, mat_B, bias, post_scale = [1,0]):
    m = mat_A.shape[0]
    k = mat_A.shape[1]
    n = mat_B.shape[1]
    print ("test_basic(PE=%d): %d %d %d %d %d" % (PE,m, k, n, post_scale[0], post_scale[1] )) 
    print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
    print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
    print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
    if xclbin_opts["GEMX_dataType"]=="short":
        C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
    else : #float
        C_fpga = np.zeros((m, n), dtype=np.float32, order='C')    
    gemx.sendMat(mat_A,PE)
    gemx.sendMat(mat_B,PE)
    gemx.sendMat(C_fpga,PE)    
    gemx.sendMat(bias, PE)
    gemx.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], PE) # default test_basic will call addGEMMOp
    gemx.execute(PE)
    gemx.clearInstrBuf(PE)
    gemx.getMat(C_fpga,PE)
    self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
   
  def test_perf(self, total_api_time, total_operations, m, k, n, ddrWidth):
    API_Time = total_api_time * 1e3
    perfApiInTops = total_operations/ (API_Time * 1e-3) / 1e12;
    print ("DATA_CSV:DdrWidth,M,K,N,Ops,TimeApiMs,PerfApiTops")
    print ("DATA_CSV:%d,%d,%d,%d,%d,%f,%f" % (ddrWidth,m,k,n,total_operations,API_Time,perfApiInTops))
  
  def check_input(self, m_size, k_size, n_size, xclbin_opts):
    m_block = int(xclbin_opts["GEMX_gemmMBlocks"])
    k_block = int(xclbin_opts["GEMX_gemmKBlocks"])
    n_block = int(xclbin_opts["GEMX_gemmNBlocks"])
    ddr_width = int(xclbin_opts["GEMX_ddrWidth"])
    if m_size%(m_block*ddr_width) !=0:
        print ("m must be multiple of", m_block, "and", ddr_width)
        sys.exit(2)
    elif k_size%(k_block*ddr_width) !=0:
        print ("k must be multiple of", k_block, "and", ddr_width)
        sys.exit(2)
    elif n_size%(n_block*ddr_width) !=0:
        print ("n must be multiple of", n_block, "and", ddr_width)  
        sys.exit(2)
         
  def test_textfiles(self, path_to_a, path_to_b, path_to_bias, post_scale):        
    mat_A = np.loadtxt(path_to_a, dtype=np.int16)
    mat_B = np.loadtxt(path_to_b, dtype=np.int16)
    bias = np.loadtxt(path_to_bias, dtype=np.int32)
    m = mat_A.shape[0]
    n = mat_B.shape[1]
    C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
    gemx.sendMat(mat_A)
    gemx.sendMat(mat_B)
    gemx.sendMat(C_fpga)    
    gemx.sendMat(bias)
    gemx.addGEMMOp (mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1])
    gemx.execute()
    gemx.clearInstrBuf()
    gemx.getMat(C_fpga)  
    self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
      
  def get_freq(self, command): 
    """
    return frequency of the xclbin
  
    Parameters
    ----------    
    command:   string
               path to the xbutil in the machine
    """
    #using command like $XILINX_XRT/bin/xbutil query -d 0
    nextLine_isFreq = False
    freq = 250 # when failed to get board frequency will use 250MHz for reporting
    try:
      proc = subprocess.check_output(command.split())
      for line in proc.splitlines():
        if nextLine_isFreq:
          freq = int(line.split()[1])
          break
        elif b"OCL Frequency" in line:
          nextLine_isFreq = True
    except: 
      print("when failed to get board frequency will use 250MHz for reporting")
    return freq
      
class FcnTest(Test):       
  """
  class provide helper functions to create FCN test cases
  
  """    
  def test_basic_randint (self,PE, xclbin_opts, post_scale, RELU_scale, max_dim):
    """
    generate random input matrices with random sizes and run the test 
    
    Parameters
    ----------  
    PE:         int
                number of kernels - 1
    post_scale: (int,int)
                (postScale, postShift)
    RELU_scale: (int,int)
                (PReLUScale, PReLUAlpha)
    max_dim:    int
                control the max size generated by the test
    """
    ddrwidth = int(xclbin_opts["GEMX_ddrWidth"])
    rand_m = self.gen_rand_dim ( ddrwidth * int(xclbin_opts["GEMX_gemmMBlocks"]), max_dim )
    rand_k = self.gen_rand_dim ( ddrwidth * int(xclbin_opts["GEMX_gemmKBlocks"]), max_dim )
    rand_n = self.gen_rand_dim ( ddrwidth * int(xclbin_opts["GEMX_gemmNBlocks"]), max_dim )
    if xclbin_opts["GEMX_dataType"]=="short":
        mat_A = self.gen_rand_matrix ( np.int16, rand_m, rand_k)
        mat_B = self.gen_rand_matrix ( np.int16, rand_k, rand_n)
        bias = self.gen_rand_matrix ( np.int32, rand_m, rand_n)   
    else: #float
        mat_A = self.gen_rand_matrix ( np.float32, rand_m, rand_k)
        mat_B = self.gen_rand_matrix ( np.float32, rand_k, rand_n)
        bias = self.gen_rand_matrix ( np.float32, rand_m, rand_n)  
    self.test_basic(PE, xclbin_opts, mat_A, mat_B, bias, post_scale, RELU_scale)    
      
  def test_basic(self,PE, xclbin_opts, mat_A, mat_B, bias, post_scale=[1, 0], RELU_scale = [1,0]):
    m = mat_A.shape[0]
    k = mat_A.shape[1]
    n = mat_B.shape[1]
    print ("test Fcn")
    print ("test_basic: %d %d %d %d %d" % (m, k, n, post_scale[0], post_scale[1])) 
    print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
    print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
    print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
    if xclbin_opts["GEMX_dataType"]=="short":
        C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
    else : #float
        C_fpga = np.zeros((m, n), dtype=np.float32, order='C')
    gemx.sendMat(mat_A, PE)
    gemx.sendMat(mat_B, PE)
    gemx.sendMat(C_fpga, PE)    
    gemx.sendMat(bias, PE)
    gemx.addFCNOp (mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], RELU_scale[0], RELU_scale[1], PE)
    gemx.execute(PE)
    gemx.clearInstrBuf(PE)
    gemx.getMat(C_fpga, PE)  
    self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale, RELU_scale)
    
  def test_textfiles(self, path_to_a, path_to_b, path_to_bias,post_scale):        
    mat_A = np.loadtxt(path_to_a, dtype=np.int16)
    mat_B = np.loadtxt(path_to_b, dtype=np.int16)
    bias = np.loadtxt(path_to_bias, dtype=np.int32)
    m = mat_A.shape[0]
    n = mat_B.shape[1]
    C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
    gemx.sendMat(mat_A)
    gemx.sendMat(mat_B)
    gemx.sendMat(C_fpga)    
    gemx.sendMat(bias)
    gemx.addFCNOp (mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], 1, 0)
    gemx.execute()
    gemx.clearInstrBuf()
    gemx.getMat(C_fpga)  
    self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
        
class SpmvTest(Test):
  """
  class provide helper functions to create spmv test cases
  
  """ 
  def multiply_and_cmp_spmv(self,row,col,data,m,k,nnz,B,C):
    """
    calculate sparse matrix multiply golden result matrix C on CPU and compare result to FPGA result.
    
    Parameters
    ----------
    row:         ndarray
                 sparse matrix's row indices
    col:         ndarray
                 sparse matrix's col indices
    data:        ndarray 
                 sparse matrix's non-zero elements 
    m:           int
                 number of rows for this sparse matrix
    k:           int
                 number of cols for this sparse matrix
    nnz:         int
                 number of non-zero elements of this sparse matrix
    C:           ndarray
                 output dense matrix in the host memory
    B:           ndarray
                 dense matrix in the host memory
    """
    if B.dtype == np.int32:
      C_cpu = np.zeros ((m, 1), dtype=np.int32)
      data_cpu = np.zeros ((m, 1), dtype=np.int32)
      data_cpu = data.astype(np.int32)
    elif B.dtype == np.float32:
      C_cpu = np.zeros ((m, 1), dtype=np.float32)
      data_cpu = np.zeros ((m, 1), dtype=np.float32)
      data_cpu = data.astype(np.float32)
    else:
      raise TypeError("type", B.dtype, "not supported") 
    for i in range(nnz):
      C_cpu[row[i]] += B[col[i]] * data_cpu[i]
    self.cmpWithinTolerance(C, C_cpu)
      
  
  def fillMod(self,B,size,Max=32764):
    """
    fill array with values (not random)
    
    Parameters
    ----------
    B:           ndarray
                 dense matrix in the host memory
    size:        int
                 size of the array
    Max:         int
                 control the max value filled to the array 
    """
    l_val = 1.0
    l_step = 0.3
    l_drift = 0.00001
    l_sign = 1
    for i in range(size):
      B[i,0] = l_val
      l_val += l_sign * l_step
      l_step += l_drift
      l_sign = -l_sign;
      if l_val > Max:
        l_val -= Max
  
class UspmvTest(Test):
  """
  class provide helper functions to create uspmv test cases
  
  """   
  def multiply_and_cmp_uspmv(self,rows,cols,datas,m_sizes,k_sizes,B,C):
    stages=len(m_sizes)
    B = np.transpose(B)
    B = sp.coo_matrix(B)
    C_cpu_list=[B]
    for i in range(stages):
      row = rows[i]
      col = cols[i]
      data = datas[i]
      mtx = sp.coo_matrix((data,(row,col)),shape=(m_sizes[i],k_sizes[i]))
      C_cpu =  np.multiply(mtx,C_cpu_list[i])
      C_cpu_list.append(C_cpu)
    C_cpu_final = np.transpose(C_cpu_list[-1].toarray())
    self.cmpWithinTolerance(C, C_cpu_final)

  def uspmv_check_maximum(self, xclbin_opts, m_size, nnz_size):
    nnz_blocks = int(xclbin_opts["GEMX_uspmvNnzVectorBlocks"])
    m_blocks = int(xclbin_opts["GEMX_uspmvMvectorBlocks"])
    ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
    if nnz_size > ddrWidth*nnz_blocks:
      raise Exception("nnz size ", nnz_size, "exceeds the maximum")
    if m_size > ddrWidth*m_blocks:
      raise Exception("m size ", m_size, "exceeds the maximum")
    if nnz_size == 0:
      raise Exception("nnz size could not be zero")
 
  def fillMod(self, Max,row_size,col_size,B):
    l_val = 1.0
    l_step = 0.3
    l_drift = 0.00001
    l_sign = 1
    for i in range(row_size):
      for j in range(col_size):
        B[i,j] = l_val
        l_val += l_sign * l_step
        l_step += l_drift
        l_sign = -l_sign;
        if l_val > Max:
          l_val -= Max
        
class GemmTest(Test):               
  pass
