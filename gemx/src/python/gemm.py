from ctypes import *
import timeit
import numpy as np
import sys

class GEMMManager:
  def __init__(self, libFile):
    self._handle = None
    
    self._lib = cdll.LoadLibrary(libFile)
    self._lib.DestroyGEMMHost.argtypes = [c_void_p]
    self._lib.MakeGEMMHost.argtypes = [c_char_p, c_char_p]
    self._lib.MakeGEMMHost.restype = c_void_p
                
    self._lib.SendToFPGAShrt_GEMM.argtypes = [c_void_p, np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), c_ulonglong, c_bool]
    self._lib.SendToFPGAInt_GEMM.argtypes = [c_void_p, np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), c_ulonglong, c_bool]
    
    self._lib.AddGEMMOp.argtypes = [c_void_p,  
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"),  
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), 
                                   c_uint, c_uint, c_uint, c_int, c_int]
    self._lib.AddGEMMOp.restype = c_bool
    
    self._lib.Execute_GEMM.argtypes = [c_void_p]
    self._lib.GetFromFPGA_GEMM.argtypes = [c_void_p,  np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), c_bool]
    self._lib.GetFromFPGA_GEMM.restype = c_void_p
    self._lib.Wait_GEMM.argtypes = [c_void_p]

  def createHandle (self, xclbin, kernel, numHandles):
     b_xclbin = xclbin.encode('utf-8')
     b_kernel = kernel.encode('utf-8')
     self._handle = self._lib.MakeGEMMHost(b_xclbin, b_kernel)
     
  def closeHandle(self): 
    if not self._handle:
      return
    #for h in self._handles:
    self._lib.DestroyGEMMHost(self._handle)

  def addGEMMOp(self, A, B, C, bias, postScale, postShift):
    ret = self._lib.AddGEMMOp(self._handle, A,B, C, bias, c_uint(A.shape[0]), c_uint( A.shape[1] ), c_uint( B.shape[1]), c_int(postScale), c_int(postShift))
    if ret == False:
        sys.exit()
    
  def execute(self):
    self._lib.Execute_GEMM(self._handle)

  def wait(self):
    self._lib.Wait_GEMM(self._handle)    
          
  def sendMat ( self, A,sync_send = False):
    if A.dtype == np.int32:
        self._lib.SendToFPGAInt_GEMM( self._handle, A, c_ulonglong(A.shape[0] *A.shape[1]),  sync_send )
    elif A.dtype == np.int16:
        self._lib.SendToFPGAShrt_GEMM( self._handle, A, c_ulonglong(A.shape[0] *A.shape[1]),  sync_send )        
    else:
        print ("sendMat: ", A.dtype, " type not supported")
        sys.exit()
    #ctypes.data_as(POINTER(c_float))
    
  def getMat(self, A, sync_get = True):
    self._lib.GetFromFPGA_GEMM( self._handle, A, sync_get )
    
  def matmul_addbias (self, A, B, bias, sendA = True, sendB = True, sendBias = True):
    if len (A.shape) != 2:
        print ("A matrix not 2-D")
        sys.exit()
        
    if len (B.shape) != 2:
        print ("B matrix not 2-D")
        sys.exit()
        
    if A.shape[1] != B.shape[0]:
        print ("Can't perform GEMM as A dim[1] %d and B dim[0] %d are different" % (A.shape[1], B.shape[0]) )
        sys.exit()
    
    if sendA == True:
        sendMat(A)
    
    if sendB == True:
        sendMat(B)
    
    if sendBias == True:
        sendMat(bias)
    
    C_fpga = np.zeros( (A.shape[0], B.shape[1]), dtype=np.int16)
    sendMat(C_fpga)    

    sendMat(bias)
    
    addGEMMOp ( A, B, C_fpga, bias, 1, 0)
  
    execute()
    getMat(C_fpga)
    return C_fpga

_gemxManager = None

def sendMat ( A,sync_send=False):
    _gemxManager.sendMat(A,sync_send)

def getMat (A, sync_get = True):
    return _gemxManager.getMat(A, sync_get)
    
def addGEMMOp( A,B,C, bias, postScale, postShift):
    _gemxManager.addGEMMOp(A, B, C, bias, postScale, postShift)
    
def execute():
    _gemxManager.execute()

def wait():
    _gemxManager.wait()    
    
def matmul ( A, B, SendA = True, SendB = True):
    bias = np.zeros ( (A.shape[0], B.shape[1]), dtype=np.int32)   
    return _gemxManager.matmul_addbias(A, B, bias, SendA, SendB,True)

def matmul_addbias ( A, B, bias, SendA = True, SendB = True, SendBias = True):
    return _gemxManager.matmul_addbias(A, B, bias, SendA, SendB, SendBias)

def createManager ( libFile ):
  return GEMMManager(libFile)  
    
def createHandle(xclbin, kernel, libFile, numHandles=1):
  global _gemxManager
  if not _gemxManager:
    _gemxManager = GEMMManager(libFile)  
  return _gemxManager.createHandle(xclbin, kernel, numHandles)

def closeHandle():
  return _gemxManager.closeHandle()
 
