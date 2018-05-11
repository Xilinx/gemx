import numpy as np
import gemx
import sys
import random
import argparse
import time
from test import GemmTest

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
    gemx.addGEMMOp(A, B, C, b0, 1, 0)
    gemx.addGEMMOp(D, C, E, b1, 1, 0)
    gemx.execute()
    gemx.getMat(C)
    gemx.getMat(E)
    print("test C")
    test.multiply_and_cmp(C, A, B, b0, m, n, [1, 0])
    print("test E")
    test.multiply_and_cmp(E, D, C, b1, m, n, [1, 0])

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test=GemmTest()
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createGEMMHandle(args, xclbin_opts)
  
  for PE in range(int(xclbin_opts["GEMX_numKernels"])):
      test.test_basic_randint( PE, 512, 512, 128, [16,17])
      test.test_basic_randint( PE, 256, 512, 128, [2,18])
      test.test_basic_randint( PE, 2048, 512, 128, [4,18])
      test.test_basic_randint( PE, 2048, 512, 128, [128,17])

  # test.test_rand_basic (32764, 0, 5, [1,0]) # larger matrix size will lead to hw timeout error in regression test
  test_multiInstrv1(32764, 512, 512, 128, True) 

 
