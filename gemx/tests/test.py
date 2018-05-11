import numpy as np
import gemx
import sys
import random
import argparse
import time
import math

# test.py includes all the common test function shared by gemm, fcn and spmv engine
class Test:
  def cmp(self,A, B):
      if np.array_equal(A, B):
          print ("Success!\n")
      else:
          print ("not equal :(")
          sys.exit()
          
  def cmpWithinTolerance(self,A, B):
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
          sys.exit()  
          
  def multiply_and_cmp(self,C, A, B, X, m, n, post_scale):
      # Calculate golden C
      #start_compute = time.time()
      m64 = np.int64(np.round(np.matmul(np.float64(A), np.float64(B))))  # intermediate accumulation to 64 bits
      #print ("float64 compute elapsed:", time.time() - start_compute)
      #m64 = np.matmul(np.int64(A), np.int64(B)) # intermediate accumulation to 64 bits
      bias64 = np.int64(X)  # bias to 64 bits
      output64 = m64 + bias64
      o64d = output64 * post_scale[0]
      o64m = o64d // (2 ** post_scale[1])
      C_cpu = np.int16(o64m)  # scale down for 16 bits
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
          sys.exit();    

  def test_basic_randint (self,PE, m, k, n, post_scale):
      int16_max = np.iinfo(np.int16).max
      int16_min = np.iinfo(np.int16).min
      int32_max = np.iinfo(np.int32).max
      int32_min = np.iinfo(np.int32).min      
      mat_A = np.random.randint(low=int16_min, high=int16_max, size=(m, k), dtype=np.int16)
      mat_B = np.random.randint(low=int16_min, high=int16_max, size=(k, n), dtype=np.int16)  
      bias = np.random.randint(low=int32_min, high=int32_max, size=(m, n), dtype=np.int32)
      
      self.test_basic(PE,mat_A, mat_B, bias, post_scale)
      
  def test_basic_randint_range (self,PE, A_range, B_range, bias_range, m, k, n, post_scale):
      mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
      mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)  
      bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
      self.test_basic(PE,mat_A, mat_B, bias, post_scale)

  def test_basic_randint_shift (self,PE,A_range, A_shift, B_range, B_shift, bias_range, bias_shift, m, k, n, post_scale):
      mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
      mat_A = mat_A + A_shift
      mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)
      mat_B = mat_B + B_shift   
      bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
      self.test_basic(PE,mat_A, mat_B, bias, post_scale)  
      
  def test_rand_basic (self,PE, xclbin_opts, post_scale, max_dim):  
      min_m = 32 * int(xclbin_opts["GEMX_gemmMBlocks"])
      min_k = 32 * int(xclbin_opts["GEMX_gemmKBlocks"])
      min_n = 32 * int(xclbin_opts["GEMX_gemmNBlocks"])      
      rand_m = random.randint(1, int(max_dim/min_m)) 
      rand_k = random.randint(1, int(max_dim/min_k)) 
      rand_n = random.randint(1, int(max_dim/min_n))       
      rand_m = rand_m * min_m 
      rand_k = rand_k * min_k
      rand_n = rand_n * min_n
      self.test_basic_randint(PE, rand_m, rand_k, rand_n, post_scale)
          
  def test_basic(self,PE, mat_A, mat_B, bias, post_scale = [1,1]):
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      print ("test_basic(PE=%d): %d %d %d %d %d" % (PE,m, k, n, post_scale[0], post_scale[1] )) 
      print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
      print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
      print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
      C_fpga = np.zeros( (m, n), dtype=np.int16)
      gemx.sendMat(mat_A,PE)
      gemx.sendMat(mat_B,PE)
      gemx.sendMat(C_fpga,PE)    
      gemx.sendMat(bias, PE)
      gemx.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], PE) # default test_basic will call addGEMMOp
      gemx.execute(PE)
      gemx.getMat(C_fpga,PE)
      self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
   
  def test_perf(self,timePointKernel, total_operations, total_parallel_operations, freq, m, k, n):
      Execute_Time = (timePointKernel[2] - timePointKernel[1])*1e3
      API_Time = (timePointKernel[3] - timePointKernel[0])*1e3
      timeMsAt100pctEff = total_parallel_operations / 2 / 32 / 32 / ( freq * 1e6 ) * 1e3
      effKernelPct = 100 * timeMsAt100pctEff / Execute_Time
      effApiPct = 100 * timeMsAt100pctEff / API_Time
      perfKernelInTops = total_operations / (Execute_Time * 1e-3) / 1e12
      perfApiInTops = total_operations/ (API_Time * 1e-3) / 1e12;
      print ("DATA_CSV:DdrWidth,Freq,M,K,N,Ops,TimeKernelMs,TimeApiMs,EffKernelPct,EffApiPct,PerfKernelTops,PerfApiTops")
      print ("DATA_CSV:32,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f" % (freq,m,k,n,total_operations,Execute_Time,API_Time,effKernelPct,effApiPct,perfKernelInTops,perfApiInTops))
  
  def check_input(self, m_size, k_size, n_size, xclbin_opts):
      m_block = int(xclbin_opts["GEMX_gemmMBlocks"])
      k_block = int(xclbin_opts["GEMX_gemmKBlocks"])
      n_block = int(xclbin_opts["GEMX_gemmNBlocks"])
      ddr_width = int(xclbin_opts["GEMX_ddrWidth"])
      if m_size%(m_block*ddr_width) !=0:
         print ("m must be multiple of", m_block, "and", ddr_width)
         sys.exit()
      elif k_size%(k_block*ddr_width) !=0:
         print ("k must be multiple of", k_block, "and", ddr_width)
         sys.exit()
      elif n_size%(n_block*ddr_width) !=0:
         print ("n must be multiple of", n_block, "and", ddr_width)  
         sys.exit()
        
class FcnTest(Test):       
  def test_basic(self,PE, mat_A, mat_B, bias, post_scale=[1, 1]):
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      print ("test_basic: %d %d %d %d %d" % (m, k, n, post_scale[0], post_scale[1])) 
      print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
      print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
      print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
      C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
      gemx.sendMat(mat_A, PE)
      gemx.sendMat(mat_B, PE)
      gemx.sendMat(C_fpga, PE)    
      gemx.sendMat(bias, PE)
      gemx.addFCNOp (mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], 1, 0, PE)
      gemx.execute(PE)
      gemx.getMat(C_fpga, PE)  
      self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
        
class SpmvTest(Test):
  def multiply_and_cmp_spmv(self,row,col,data,m,k,nnz,B,C):
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
      
  def fillMod(self,B,size,Max):
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

        
class GemmTest(Test):               
  pass