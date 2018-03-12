import numpy as np
import gemm
import sys
import random
import argparse

def cmp( A, B):
    if np.array_equal(A, B):
        print ("Success!\n")
    else:
        print ("not equal :(")
        sys.exit()  
        
def multiply_and_cmp(C, A, B, X, m, n, post_scale):
    #Calculate golden C
    m64 = np.matmul(np.int64(A), np.int64(B)) #intermediate accumulation to 64 bits
    bias64 = np.int64(X) #bias to 64 bits
    output64 = m64 + bias64
    o64d = output64*post_scale[0]
    o64m = o64d/(2**post_scale[1])
    C_cpu = np.int16(o64m) #scale down for 16 bits
    C_fpga = C.flatten()
    C_cpu = C_cpu.flatten()  
    C_fpga = np.reshape(C_fpga, (m, n))
    C_cpu = np.reshape(C_cpu, (m, n))
    if np.array_equal(C_fpga, C_cpu):
	print ("Success!\n")
    else:
	print ("Not equal!")
	print (C_fpga.shape, C_cpu.shape)
	np.savetxt("cpu_out.np", C_cpu, fmt="%d")
	np.savetxt("fpga_out.np", C_fpga, fmt="%d")
	np.savetxt("bias.np", X, fmt="%d")
	np.savetxt("A.np", A, fmt="%d")
	np.savetxt("B.np", B, fmt="%d")
	sys.exit();    

def test_basic_randint ( A_range, B_range, bias_range, m, k, n, post_scale):
    mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
    mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)  
    bias = []
    if bias_range != 0:
        bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
    else:
        bias = np.zeros ( (m, n), dtype=np.int32);
    test_basic(mat_A, mat_B, bias, post_scale)

def test_basic_randint_shift ( A_range, A_shift, B_range, B_shift, bias_range, bias_shift, m, k, n, post_scale):
    mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
    mat_A = mat_A + A_shift
    mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)
    mat_B = mat_B + B_shift   
    bias = []
    if bias_range != 0:
        bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
    else:
        bias = np.zeros ( (m, n), dtype=np.int32);    bias = bias + bias_shift
    test_basic(mat_A, mat_B, bias, post_scale)    
    
#def test_basic_gauss ( a_mu, a_sigma, b_mu, b_sigma,  m, k, n, add_bias = False):
#    mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
#    mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)
#    bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
    
#    test_basic(mat_A, mat_B, bias, add_bias)    
    
def test_basic(mat_A, mat_B, bias, post_scale = [1,1]):
    m = mat_A.shape[0]
    k = mat_A.shape[1]
    n = mat_B.shape[1]
    print ("test_basic: %d %d %d %d %d" % (m, k, n, post_scale[0], post_scale[1] )) 
    print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
    print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
    print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
    C_fpga = np.zeros( (m, n), dtype=np.int16)
    gemm.sendMat(mat_A)
    gemm.sendMat(mat_B)
    gemm.sendMat(C_fpga)    
    gemm.sendMat(bias)
    gemm.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1])
    gemm.execute()
    gemm.getMat(C_fpga)  
    if m > 4096 and n > 4096 and k > 4096:
      print("Skip golden comparision because large matrix size")
    else:
      multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)

def test_sendA_first(int_range, m, k, n):
    print ("test_sendA_first: %d %d %d %d" % (int_range, m, k, n )) 
    mat_A = np.random.randint(low=-int_range, high=int_range, size=(m, k), dtype=np.int16)
    gemm.sendMat(mat_A)
    mat_B = np.random.randint(low=-int_range, high=int_range, size=(k, n), dtype=np.int16)
    C_fpga = gemm.matmul(mat_A, mat_B, False)
    C_cpu = np.matmul(mat_A, mat_B)  
    C_cpu = C_cpu.flatten()
    C_fpga = C_fpga.flatten()  
    C_cpu = np.reshape(C_cpu, (m, n))
    C_fpga = np.reshape(C_fpga, (m, n))  
    if np.array_equal(C_fpga, C_cpu):
      print ("Success!\n")
    else:
      print ("not equal :(")
      sys.exit();

def test_multiInstrv1(int_range, sz, add_bias = False):
    print ("test_multiInstrv1: %d %d" % (int_range, sz )) 
    A = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int16)
    B = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int16)
    C = np.zeros ( (sz, sz), dtype=np.int16);
    D = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int16)
    E = np.zeros ( (sz, sz), dtype=np.int16);
    b0 = np.zeros ( (sz, sz), dtype=np.int32);
    b1 = np.zeros ( (sz, sz), dtype=np.int32);
    if add_bias == True:
        b0 = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int32)
        b1 = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int32)       
    gemm.sendMat(A)
    gemm.sendMat(B)
    gemm.sendMat(b0)
    gemm.sendMat(C)
    gemm.sendMat(D)    
    gemm.sendMat(E)
    gemm.sendMat(b1)         
    gemm.addGEMMOp(A,B,C, b0, 1,0)
    gemm.addGEMMOp(C,D,E, b1, 1,0)
    gemm.execute()
    gemm.getMat(C)
    gemm.getMat(E)
    if sz > 4096:
      print("Skip golden comparision because large matrix size")
    else:
      print("test C")
      multiply_and_cmp(C, A, B, b0, sz, sz, [1,0])
      print("test E")
      multiply_and_cmp(E, C, D, b1, sz, sz, [1,0])

def test_multiInstrv2(int_range, sz):
    print ("test_multiInstrv2: %d %d" % (int_range, sz )) 
    A = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int16)
    B = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int16)   
    C = np.zeros ( (sz, sz), dtype=np.int16);
    D = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int16)
    E = np.zeros ( (sz, sz), dtype=np.int16);
    F = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int16)    
    G = np.zeros ( (sz, sz), dtype=np.int16);
    H = np.random.randint(low=-int_range, high=int_range, size=(sz, sz), dtype=np.int16)
    I = np.zeros ( (sz, sz), dtype=np.int16);   
    gemm.sendMat(A)
    gemm.sendMat(B)
    gemm.sendMat(C)
    gemm.sendMat(D)    
    gemm.sendMat(E)
    gemm.sendMat(F)
    gemm.sendMat(G)
    gemm.sendMat(H)
    gemm.sendMat(I)
    b0 = np.zeros ( (sz, sz), dtype=np.int32);
    b1 = np.zeros ( (sz, sz), dtype=np.int32);
    b2 = np.zeros ( (sz, sz), dtype=np.int32);
    b3 = np.zeros ( (sz, sz), dtype=np.int32);
    gemm.sendMat(b0)
    gemm.sendMat(b1)
    gemm.sendMat(b2)
    gemm.sendMat(b3)    
    gemm.addGEMMOp(A,B,C, b0, 1,0)
    gemm.addGEMMOp(C,D,E, b1, 1,0)
    gemm.addGEMMOp(E,F,G, b2, 1,0)
    gemm.addGEMMOp(G,H,I, b3, 1,0)    
    gemm.execute()
    gemm.getMat(I)
    if sz > 4096:
      print("Skip golden comparision because large matrix size")
    else:
      print("test I")
      C_cpu = np.matmul(A,B)
      E_cpu = np.matmul(C_cpu,D)
      G_cpu = np.matmul(E_cpu,F)
      multiply_and_cmp(I, G_cpu, H, b3, sz, sz, [1,0])

def test_rand_basic ( int_range, bias_range, num_iter, post_scale):  
    min_sz_exp = 8 
    for i in range(num_iter):
        print ("test_rand_basic iter: %d" % i)
        rand_m = random.randint(0,5) 
        rand_k = random.randint(0,5) 
        rand_n = random.randint(0,5)       
        rand_m = 2 ** (rand_m + min_sz_exp) 
        rand_k = 2 ** (rand_k + min_sz_exp)
        rand_n = 2 ** (rand_n + min_sz_exp)
        test_basic_randint(int_range, int_range, bias_range, rand_m, rand_k, rand_n, post_scale)
              
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='pyXDNN')
  parser.add_argument('--xclbin', required = True, help='.xclbin file')
  parser.add_argument('--gemxlib', required = True, help='FPGA gemx host shared library')
  args = parser.parse_args()

  gemm.createHandle(args.xclbin, "gemxKernel_0", args.gemxlib)
  
  size = 256
  while size < 8192:
    test_basic_randint( 32764, 32764, 0, size, size, size, [1,1])
    test_basic_randint( 32764, 32764, 0, size, size, size, [4,18])
    size = size * 2
    
  for i in range(5):
    test_basic_randint( 32764, 32764, 0, 512, 512, 32, [16,17])
    test_basic_randint( 32764, 32764, 0, 256, 512, 32, [2,18])
    test_basic_randint( 32764, 32764, 0, 2048, 512, 32, [4,18])
    test_basic_randint( 32764, 32764, 0, 2048, 512, 32, [128,17])
    #test_basic_randint( 32764, 256, 512, 32)  
    #test_basic_randint( 100, 256, 512, 32)
    #test_basic_randint(32764, 256, 512, 256)
    #test_basic_randint(10, 256, 512, 256, True) fail
    #test_basic_randint(32764, 256, 512, 1024)
    #test_basic_randint(32764, 256, 512, 2048)
    #test_basic_randint(100, 16384, 16834, 8192)
    
  test_rand_basic (32764, 0, 5, [1,0])
  test_sendA_first(32764, 256, 512, 1024)
  test_multiInstrv1(32764, 256)  
  
  for m_sz in range(5):
    sz = 2 ** (m_sz+8)
    print ("Size: %d" % sz)
    test_multiInstrv2( 32764, sz)

  gemm.closeHandle()
  
 
