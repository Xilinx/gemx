import numpy as np
import gemx
import sys
import random
import argparse
import time
import test
from test import FcnTest

def test_multiInstrv1(int_range, m, k, n, add_bias=False):
    print ("test_multiInstrv1: %d %d %d %d" % (int_range, m, k, n)) 
    A = np.random.randint(low=-int_range, high=int_range, size=(m, k), dtype=np.int16)
    B = np.random.randint(low=-int_range, high=int_range, size=(k, n), dtype=np.int16)
    C = np.zeros ((m, n), dtype=np.int16);
    D = np.random.randint(low=-int_range, high=int_range, size=(m, k), dtype=np.int16)
    E = np.zeros ((m, n), dtype=np.int16);
    b0 = np.zeros ((m, n), dtype=np.int32);
        
    b1 = np.zeros ((m, n), dtype=np.int32);
    
    if add_bias == True:
        b0 = np.random.randint(low=-int_range, high=int_range, size=(m, n), dtype=np.int32)
        b1 = np.random.randint(low=-int_range, high=int_range, size=(m, n), dtype=np.int32)
        
    gemx.sendMat(A)
    gemx.sendMat(B)
    gemx.sendMat(b0)
    gemx.sendMat(C)
    gemx.sendMat(D)    
    gemx.sendMat(E)
    gemx.sendMat(b1)         
    gemx.addFCNOp(A, B, C, b0, 1, 13, 307, 10)
    gemx.addFCNOp(D, C, E, b1, 1, 18, 307, 10)
    gemx.execute()
    gemx.getMat(C)
    gemx.getMat(E)
    print("test C")
    test.multiply_and_cmp(C, A, B, b0, m, n, [1, 0])
    print("test E")
    test.multiply_and_cmp(E, D, C, b1, m, n, [1, 0])
      
def test_perf_fcn(A_range, B_range, bias_range, m, k, n, post_scale):
    mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
    mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)  
    bias = []
    if bias_range != 0:
        bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
    else:
        bias = np.zeros ((m, n), dtype=np.int32, order='C');   
    C_fpga = np.zeros( (m, n), dtype=np.int16)
    timePointKernel = []
    timePointKernel.append(time.time()) # current time    
    gemx.sendMat(mat_A)
    gemx.sendMat(mat_B)
    gemx.sendMat(C_fpga)    
    gemx.sendMat(bias)
    gemx.addFCNOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1],1,0)
    timePointKernel.append(time.time()) # send to FPGA
    gemx.execute()
    timePointKernel.append(time.time()) # call kernel
    gemx.getMat(C_fpga)  
    timePointKernel.append(time.time()) # copy from FPGA
    total_operations = 2 * m * n * k + m * n * 3
    total_parallel_operations = 2 * m * n * k
    freq = gemx.getFreq()
    test.test_perf(timePointKernel,total_operations,total_parallel_operations,freq,m,k,n)
    if m > 4096 and n > 4096 and k > 4096:
      print("Skip golden comparision because large matrix size")
    else:
      test.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
      
def test_perf_multi_fcn(ins_count, m_size, k_size, n_size, A_range, B_range, post_scale):
    total_operations = 0
    total_parallel_operations = 0
    mat_A=[]
    mat_C=[]
    mat_bias=[]
    for i in range(ins_count):
      total_operations += 2 * m_size[i] * n_size[i] * k_size[i] + m_size[i] * n_size[i] * 3
      total_parallel_operations += 2 * m_size[i] * n_size[i] * k_size[i]
      mat_A.append(np.random.randint(low=-A_range, high=A_range, size=(m_size[i], k_size[i]), dtype=np.int16))
      mat_bias.append(np.zeros ((m_size[i], n_size[i]), dtype=np.int32))
      mat_C.append(np.zeros((m_size[i], n_size[i]), dtype=np.int16, order='C'))
    mat_B0 = np.random.randint(low=-B_range, high=B_range, size=(k_size[0], n_size[0]), dtype=np.int16) 
    timePointKernel = []
    timePointKernel.append(time.time()) # current time 
    for i in range(ins_count):
      gemx.sendMat(mat_A[i])
      gemx.sendMat(mat_C[i])
      gemx.sendMat(mat_bias[i])
    gemx.sendMat(mat_B0)
    gemx.addFCNOp (mat_A[0], mat_B0, mat_C[0], mat_bias[0], post_scale[0], post_scale[1],1,0)    
    gemx.addFCNOp (mat_A[1], mat_C[0], mat_C[1], mat_bias[1], post_scale[0], post_scale[1],1,0) 
    gemx.addFCNOp (mat_A[2], mat_C[1], mat_C[2], mat_bias[2], post_scale[0], post_scale[1],1,0) 
    gemx.addFCNOp (mat_A[3], mat_C[2], mat_C[3], mat_bias[3], post_scale[0], post_scale[1],1,0)
    timePointKernel.append(time.time()) # send to FPGA
    gemx.execute()
    timePointKernel.append(time.time()) # call kernel
    gemx.getMat(mat_C[0])  
    gemx.getMat(mat_C[1]) 
    gemx.getMat(mat_C[2]) 
    gemx.getMat(mat_C[3]) 
    timePointKernel.append(time.time()) # copy from FPGA
    freq = gemx.getFreq()
    test.test_perf(timePointKernel,total_operations,total_parallel_operations,freq,0,0,0)
    if np.max(m_size) > 4096 and np.max(k_size) > 4096 and np.max(n_size) > 4096:
      print("Skip golden comparision because large matrix size")
    else:
      test.multiply_and_cmp(mat_C[3], mat_A[3], mat_C[2], mat_bias[3], m_size[3], n_size[3], post_scale)

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test=FcnTest()
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createFCNHandle( args, xclbin_opts)

  for j in range (1,30):
      for k in range(1,18):
          for n in range(1000):
              for i in range (int(xclbin_opts["GEMX_numKernels"])):
                  test.test_rand_basic( i, xclbin_opts, [j,k], 2048)    
      
  # test.test_rand_basic (32764, 0, 5, [1,0]) # larger matrix size will lead to hw timeout error in regression test
  test_multiInstrv1(32764, 512, 512, 128, True) 
  
  m_size=np.array([512,512,2048,128])
  k_size=np.array([384,512,512,2048])
  n_size=np.array([32,32,32,32])   
  test_perf_multi_fcn(4, m_size, k_size, n_size, 32764, 32764, [1,0]) # run performance measurement
  gemx.printStats()  