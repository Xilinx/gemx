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

from ctypes import *
import numpy as np
import sys
import argparse
class GEMXManager:
  """
  This class will load the C++ shared library and then specify the required argument and return types for each function in the shared library to use in python side. \n
  .. note:: All the PE in the functions has default value = 0, so no need to put them when xclbin is built with one kernel in it.
  """
  def __init__(self, libFile): 
    self._lib = cdll.LoadLibrary(libFile)
    self._lib.MakeFCNHost.argtypes = [c_char_p, c_uint]
    self._lib.MakeGEMMHost.argtypes = [c_char_p, c_uint]
    self._lib.MakeUSPMVHost.argtypes = [c_char_p, c_uint] 
    self._lib.MakeSPMVHost.argtypes = [c_char_p, c_uint]
    self._lib.SendToFPGAShrt.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), c_ulonglong, c_uint, c_bool]
    self._lib.SendToFPGAInt.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), c_ulonglong, c_uint, c_bool]
    self._lib.SendToFPGAFloat.argtypes = [np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong, c_uint, c_bool]
    self._lib.AddFCNOp.argtypes = [np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"),  
                                  np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), 
                                  np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), 
                                  np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), 
                                  c_uint, c_uint, c_uint, c_int, c_int, c_short, c_short, c_uint]
    self._lib.AddGEMMOp.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"),  
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), 
                                   c_uint, c_uint, c_uint, c_int, c_int, c_uint] 
    self._lib.AddUSPMVOp.argtypes = [c_void_p, 
                                   np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"),  
                                   c_uint, c_uint] 
    self._lib.AddSPMVOp.argtypes = [c_void_p, 
                                   np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"),  
                                   c_uint, c_uint, c_uint, c_bool, c_uint, c_uint, c_uint, c_uint]
    self._lib.SendUSpMat.argtypes= [np.ctypeslib.ndpointer(c_uint16, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_uint16, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
                                       c_uint,c_uint,c_uint]
    self._lib.SendSpToFpgaFloat.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),c_uint,c_uint,c_uint,
                                       c_uint,c_uint,c_uint,c_uint,c_uint,c_uint]
    self._lib.SendSpToFpgaInt.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),c_uint,c_uint,c_uint,
                                       c_uint,c_uint,c_uint,c_uint,c_uint,c_uint]
    self._lib.SendSpToFpgaFloat.restype = c_void_p
    self._lib.SendSpToFpgaInt.restype = c_void_p
    self._lib.SendUSpMat.restype = c_void_p  
    self._lib.AddFCNOp.restype = c_bool
    self._lib.AddGEMMOp.restype = c_bool
    self._lib.AddUSPMVOp.restype = c_bool
    self._lib.AddSPMVOp.restype = c_bool
    self._lib.Execute.argtypes = [c_bool, c_uint]
    self._lib.GetFromFPGA.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), c_uint, c_bool]
    self._lib.GetFromFPGA.restype = c_void_p
    self._lib.GetFromFPGAInt.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), c_uint, c_bool]
    self._lib.GetFromFPGAInt.restype = c_void_p
    self._lib.GetFromFPGAFloat.argtypes = [np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, c_bool]
    self._lib.GetFromFPGAFloat.restype = c_void_p
    self._lib.Wait.argtypes = [c_uint]
    self._lib.ClearInstrBuf.argtypes = [c_uint]
    self._lib.PrintStats.argtypes = []
    # new flow wrapper
    self._lib.MakeStrGEMMHost.argtypes = [c_char_p, c_uint]
    self._lib.MakeStrFCNHost.argtypes = [c_char_p,  c_uint]
    self._lib.MakeStrSPMVHost.argtypes = [c_char_p, c_uint]
    self._lib.MakeStrUSPMVHost.argtypes = [c_char_p, c_uint]
    self._lib.AddDevBuf.argtypes = [c_char_p,c_uint,c_uint]
    self._lib.AddDevBuf.restype = c_void_p
    self._lib.AddSpDevBuf.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
                                       c_char_p,
                                       c_uint,c_uint,c_uint,
                                       c_uint,c_uint,c_uint,c_uint,c_uint,c_uint]
    self._lib.AddSpDevBuf.restype = c_void_p
    self._lib.AddUSpDevBuf.argtypes= [np.ctypeslib.ndpointer(c_uint16, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_uint16, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
                                       c_char_p,
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
                                       c_uint,c_uint,c_uint]
    self._lib.AddUSpDevBuf.restype = c_void_p
    self._lib.AllocProgBuf.argtypes=[c_uint,c_uint]
    self._lib.SendDevBuf.argtypes = [c_char_p,c_uint,c_bool]
    self._lib.AddGEMMDevOp.argtypes=[c_char_p,c_char_p,c_char_p,c_char_p,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint]
    self._lib.AddGEMMDevOp.restype=c_bool
    self._lib.AddFCNDevOp.argtypes=[c_char_p,c_char_p,c_char_p,c_char_p,c_uint,c_uint,c_uint,c_uint,c_uint, c_short, c_short, c_uint]
    self._lib.AddFCNDevOp.restype=c_bool
    self._lib.AddSPMVDevOp.argtypes=[c_char_p,c_char_p,c_char_p,c_uint, c_uint, c_uint, c_bool, c_uint, c_uint, c_uint, c_uint]
    self._lib.AddSPMVDevOp.restype=c_bool
    self._lib.AddUSPMVDevOp.argtypes=[c_char_p,c_char_p,c_char_p,c_uint, c_uint]
    self._lib.AddUSPMVDevOp.restype=c_bool
    self._lib.GetDevBuf.argtypes=[c_char_p,c_uint,c_bool]
    self._lib.GetDevBuf.restype = c_void_p
    self._lib.ExecuteDev.argtypes=[c_bool,c_uint]
        
  def createFCNHandle (self, xclbin, numHandles):
    """
    create FCN Handle
    
    Parameters
    ----------
    xclbin
                file path to FPGA bitstream
    numHandles
                number of kernels in the xclbin
      
    """
    b_xclbin = xclbin.encode('utf-8')
    self._lib.MakeFCNHost(b_xclbin, int(numHandles))
     
  def createGEMMHandle (self, xclbin, numHandles):
    """
    create GEMM Handle
    
    Parameters
    ----------
    xclbin
                file path to FPGA bitstream
    numHandles
                number of kernels in the xclbin
    """
    b_xclbin = xclbin.encode('utf-8')
    self._lib.MakeGEMMHost(b_xclbin, int(numHandles))
    
     
  def createUSPMVHandle (self, xclbin, numHandles):
    """
    create USPMV Handle
    
    Parameters
    ----------
    xclbin
                file path to FPGA bitstream
    numHandles
                number of kernels in the xclbin
    """
    b_xclbin = xclbin.encode('utf-8')
    self._lib.MakeUSPMVHost(b_xclbin, int(numHandles))
      
  def createSPMVHandle (self, xclbin, numHandles):
    """
    create SPMV Handle
    
    Parameters
    ----------
    xclbin
                file path to FPGA bitstream
    numHandles
                number of kernels in the xclbin
    """
    b_xclbin = xclbin.encode('utf-8')
    self._lib.MakeSPMVHost(b_xclbin, int(numHandles))

  def sendMat ( self, A, PE, sync_send = False):
    """
    send dense matrix to kernel
    if sync_send is true, will only create the buffer for that matrix, and will need to send it when executing the kernel
    
    Parameters
    ----------  
    A:         ndarray
               dense matrix in the host memory
    PE:        int
               index of kernel
    sync_send: boolean
               controls when to send the data to kernel. \n
               If false, send immediately, else need to send together when executing the kernel. Default value is false. 
    """
    if A.flags['C_CONTIGUOUS'] == False:
        A = np.ascontiguousarray(A)
        print ("Warning: not C_CONTIGUOUS, performance will be affected")      
    if A.dtype == np.int32:
        self._lib.SendToFPGAInt( A, c_ulonglong(A.size), c_uint(PE), sync_send )
    elif A.dtype == np.int16:
        self._lib.SendToFPGAShrt( A, c_ulonglong(A.size), c_uint(PE), sync_send ) 
    elif A.dtype == np.float32:
        self._lib.SendToFPGAFloat( A, c_ulonglong(A.size), c_uint(PE), sync_send ) 
    else:
        raise TypeError("type", A.dtype, "not supported")
      
  def sendSpMat(self, row, col, data, m, k, nnz, xclbin_opts, PE):
    """
    send sparse matrix to kernel (for spmv engine). 
    
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
    xclbin_opts: dictionary 
                 information read from config_info.dat used to build the xclbin
    PE:          int
                 index of kernel
                 
    Return
    ------
    c_void_p
                 pointer to the start of the host memory for the sparse matrix   
    """
    ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])  
    spmv_width = int(xclbin_opts["GEMX_spmvWidth"])
    num_cblocks = int(xclbin_opts["GEMX_spmvNumCblocks"])
    spmvMacGroups = int(xclbin_opts["GEMX_spmvMacGroups"])
    t_mVectorBlocks =((1 << (16 - int(xclbin_opts["GEMX_spmvColAddIdxBits"]))) // spmv_width // spmvMacGroups // ddrWidth)
    capacity_Cblocks =  int(spmv_width * spmvMacGroups * t_mVectorBlocks * ddrWidth)
    capacity_Bblocks = int(spmv_width * int(xclbin_opts["GEMX_spmvkVectorBlocks"]) * ddrWidth)    
    if xclbin_opts["GEMX_dataType"] == "float":
      return self._lib.SendSpToFpgaFloat(row,col,data, m, k, nnz, ddrWidth, spmv_width, num_cblocks, capacity_Cblocks, capacity_Bblocks, c_uint(PE))
    elif xclbin_opts["GEMX_dataType"] == "int32_t":
      return self._lib.SendSpToFpgaInt(row,col,data, m, k, nnz, ddrWidth, spmv_width, num_cblocks, capacity_Cblocks, capacity_Bblocks, c_uint(PE)) 
    else:
      raise TypeError("type", xclbin_opts["GEMX_dataType"], "not supported")  
  
  def sendUSpMat(self, rows, cols, datas, ms, ks, nnzs, pRelus, xclbin_opts, PE):
    """
    send sparse matrices to kernel (for uspmv engine). \n
    Uspmv engine supports multi-stages usage, so if the prebuilt xclbin is for multi-stages, use this function to send multiple sparse matrices together. \n
    For each matrix, its row index array, col index array and value array need to be sorted to avoid overhead on kernel side.
    
    Parameters
    ----------
    rows:         ndarray
                  all the sparse matrices row indices \n
                  when xclbin is multi-stages, rows should be a concatenated ndarray of all the row indices
    cols:         ndarray
                  all the sparse matrices col indices \n
                  when xclbin is multi-stages, cols should be a concatenated ndarray of all the col indices
    datas:        ndarray 
                  all the sparse matrices non-zero elements \n
                  when xclbin is multi-stages, it should be a concatenated ndarray of all the non-zero elements 
    ms:           ndarray
                  numbers of rows for all the sparse matrices
    ks:           ndarray
                  numbers of cols for all the sparse matrices
    nnzs:         ndarray
                  numbers of non-zero elements for all the sparse matrices
    pRelus:       ndarray
                  numbers to multiply with the output values when output values < 0 
    xclbin_opts:  dictionary 
                  information read from config_info.dat used to build the xclbin
    PE:           int
                  index of kernel
                  
    Return
    ------
    c_void_p
                  pointer to the start of the host memory for the sparse matrices
    """
    return self._lib.SendUSpMat(rows,cols,datas, ms, ks, nnzs, pRelus,int(xclbin_opts["GEMX_ddrWidth"]), int(xclbin_opts["GEMX_uspmvStages"]), c_uint(PE))
  
  def addFCNOp(self, A, B, C, bias, postScale, postShift, PReLUScale, PReLUAlpha, PE):
    """
    create FCN instruction for C = relu ((A * B + bias) * postScale >> postShift) 
    
    Parameters
    ----------
    A:         ndarray
               dense matrix in the host memory
    B:         ndarray
               dense matrix in the host memory
    C:         ndarray
               dense matrix in the host memory
    bias:      ndarray
               dense matrix in the host memory
    postScale: int
               multiply the output values with specific scalar
    postShift: int
               shift the output values with specific scalar
    PReLUScale:int
               multiply the output values with specific scalar when output values < 0 
    PReLUAlpha:int
               shift the output values with specific scalar when output values < 0              
    PE:        int
               index of kernel
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Cannot perform FCN with matrices", A.shape, B.shape )
    if C.shape != bias.shape:
        raise ValueError("Bias matrix shape", bias.shape, "doesn't match output shape", C.shape)
    return self._lib.AddFCNOp( A, B, C, bias, c_uint(A.shape[0]), c_uint( A.shape[1] ), c_uint( B.shape[1]), c_int(postScale), c_int(postShift), c_short(PReLUScale), c_short(PReLUAlpha), c_uint(PE))
  
  def addGEMMOp(self, A, B, C, bias, postScale, postShift, PE):
    """
    create GEMM instruction for C = (A * B + bias) * postScale >> postShift
    
    Parameters
    ----------
    A:         ndarray
               dense matrix in the host memory
    B:         ndarray
               dense matrix in the host memory
    C:         ndarray
               dense matrix in the host memory
    bias:      ndarray
               dense matrix in the host memory
    postScale: int
               multiply the output values with specific scalar
    postShift: int
               shift the output values with specific scalar          
    PE:        int
               index of kernel
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Cannot perform GEMM with matrices", A.shape, B.shape )
    if C.shape != bias.shape:
        raise ValueError("Bias matrix shape", bias.shape, "doesn't match output shape")      
    return self._lib.AddGEMMOp(A,B, C, bias, c_uint(A.shape[0]), c_uint( A.shape[1] ), c_uint( B.shape[1]), c_int(postScale), c_int(postShift), c_uint(PE))
  
  def addSPMVOp(self, A, B, C, nnz, xclbin_opts, relu, PE):    
    """
    create SPMV instruction for C = relu (A (sparse matrix) * B (dense vector) )
    
    Parameters
    ----------
    A:      c_void_p
            pointer to the sparse matrix in the host memory
    B:      ndarray
            dense vector in the host memory
    C:      ndarray
            dense vector in the host memory
    nnz:    int
            number of non-zero elements of this sparse matrix
    relu:   boolean
            when relu is true, for output values < 0, output values = 0    
    PE:     int
            index of kernel
    """
    ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
    spmv_width = int(xclbin_opts["GEMX_spmvWidth"])
    num_cblocks = int(xclbin_opts["GEMX_spmvNumCblocks"])
    spmvMacGroups = int(xclbin_opts["GEMX_spmvMacGroups"])
    t_mVectorBlocks =((1 << (16 - int(xclbin_opts["GEMX_spmvColAddIdxBits"]))) // spmv_width // spmvMacGroups // ddrWidth)
    capacity_Cblocks =  int(spmv_width * spmvMacGroups * t_mVectorBlocks * ddrWidth)
    capacity_Bblocks = int(spmv_width * int(xclbin_opts["GEMX_spmvkVectorBlocks"]) * ddrWidth)
    return self._lib.AddSPMVOp(A,B,C,c_uint(C.shape[0]),c_uint(B.shape[0]),c_uint(nnz),c_bool(relu), c_uint(num_cblocks),c_uint(capacity_Cblocks), c_uint(capacity_Bblocks),c_uint(PE)) 
  
  def addUSPMVOp(self, A, B, C, numRuns,PE):
    """
    create USPMV instruction for C = A (sparse matrix) * B (dense matrix)
    
    Parameters
    ----------
    A:      ndarray of c_void_p
            pointers to all the sparse matrices in the host memory
    B:      ndarray
            dense matrice in the host memory
    C:      ndarray
            dense matrice in the host memory
    numRuns:int
            col size of the first dense matrix B
    PE:     int
            index of kernel
    """
    return self._lib.AddUSPMVOp(A,B,C,numRuns,c_uint(PE))
  
  def execute(self, PE, sync_exec = True):
    """
    send instructions created before to kernel, then start.
    
    Parameters
    ----------  
    PE:        int
               index of kernel
    sync_exec: boolean
               Default is True. \n
               If send some matrices with sync_send = True before, then here need to set sync_exec = False, otherwise those matrices won't be sent to the kernel.\n
               It is suggested to use the default value for sync_send and sync_exec.
    """
    self._lib.Execute(sync_exec, PE)
    
  def wait(self, PE):
    """
    Wait until all events have completed. 
    If using default value for sync_send, sync_exec and sync_get before, there is no need to call this function.
    
    Parameters
    ----------  
    PE:        int
               index of kernel
    """
    self._lib.Wait(PE)
    
  def clearInstrBuf(self, PE):
    """
    Clear the instruction buffer in kernel. \n
    The maximum instructions could be saved in the kernel is 16. Only call this function when previous instructions sent to kernel is > 16.
    
    Parameters
    ----------  
    PE:        int
               index of kernel
    
    """
    self._lib.ClearInstrBuf(PE)      
           
      
  def getMat(self, A, PE, sync_get = True):
    """
    Get the dense matrix from kernel to host memory
    
    Parameters
    ----------  
    A:         ndarray
               dense matrix in the host memory
    PE:        int
               index of kernel 
    sync_get:  boolean
               Default is True. \n
               If true, it indicates that getMat will wait for the end of the transfer. \n
               If false, the wait function call is needed to have received all the data.   
    """
    if A.dtype == np.int16:
        self._lib.GetFromFPGA( A, PE, sync_get )
    elif A.dtype == np.int32:
        self._lib.GetFromFPGAInt( A, PE, sync_get )
    elif A.dtype == np.float32:
        self._lib.GetFromFPGAFloat( A, PE, sync_get )
    else:
        raise TypeError("type", A.dtype, "not supported") 
    
  def printStats(self):
    """
    print time used by functions in C++ side
    """
    self._lib.PrintStats()
    
    
  def createStrGEMMHandle (self, xclbin, numHandles):
    b_xclbin = xclbin.encode('utf-8')
    self._lib.MakeStrGEMMHost(b_xclbin, int(numHandles))
    
  def createStrFCNHandle (self, xclbin, numHandles):
    b_xclbin = xclbin.encode('utf-8')
    self._lib.MakeStrFCNHost(b_xclbin, int(numHandles))
    
  def createStrSPMVHandle (self, xclbin, numHandles):
    b_xclbin = xclbin.encode('utf-8')
    self._lib.MakeStrSPMVHost(b_xclbin, int(numHandles))
    
  def createStrUSPMVHandle (self, xclbin, numHandles):
    b_xclbin = xclbin.encode('utf-8')
    self._lib.MakeStrUSPMVHost(b_xclbin, int(numHandles))
  
  def addDevBuf(self,A,size_row,size_col,datatype,PE):
    buf_size=size_row*size_col*np.dtype(datatype).itemsize
    address = self._lib.AddDevBuf(A,buf_size,PE)
    if datatype==np.int16:
      buff ={'shape':(size_row,size_col),'data':(address,False),'typestr':'<i2'}
    elif datatype==np.int32:
      buff ={'shape':(size_row,size_col),'data':(address,False),'typestr':'<i4'}
    elif datatype==np.float32:
      buff ={'shape':(size_row,size_col),'data':(address,False),'typestr':'<f4'}
    else:
      raise TypeError("type", datatype, "not supported") 
    holder= type('array_wrapper', (), {})()
    holder.__array_interface__=buff
    myArray=np.array(holder,copy=False)
    print(myArray.__array_interface__)
    return myArray   
    
  def addSpDevBuf(self,row,col,data, A, m, k, nnz, xclbin_opts, PE):
    ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
    spmv_width = int(xclbin_opts["GEMX_spmvWidth"])
    spmvMacGroups = int(xclbin_opts["GEMX_spmvMacGroups"])
    num_cblocks = int(xclbin_opts["GEMX_spmvNumCblocks"])
    t_mVectorBlocks =((1 << (16 - int(xclbin_opts["GEMX_spmvColAddIdxBits"]))) // spmv_width // spmvMacGroups // ddrWidth)
    capacity_Cblocks =  int(spmv_width * spmvMacGroups * t_mVectorBlocks * ddrWidth)
    capacity_Bblocks = int(spmv_width * int(xclbin_opts["GEMX_spmvkVectorBlocks"]) * ddrWidth)
    return self._lib.AddSpDevBuf(row,col,data, A, m, k, nnz, ddrWidth, spmv_width, num_cblocks, capacity_Cblocks, capacity_Bblocks, PE)
    
  def addUSpDevBuf(self, rows, cols, datas, A, ms, ks, nnzs, pRelus, xclbin_opts, PE):
    return self._lib.AddUSpDevBuf(rows.astype(c_uint16),cols.astype(c_uint16),datas, A, ms, ks, nnzs, pRelus, int(xclbin_opts["GEMX_ddrWidth"]), int(xclbin_opts["GEMX_uspmvStages"]), c_uint(PE))  
    
  def allocProgBuf(self,buf_sz,PE):
    return self._lib.AllocProgBuf(c_uint(buf_sz),c_uint(PE))
    
  def sendDevBuf(self,A,PE,sync_send):
    self._lib.SendDevBuf(c_char_p(A),c_uint(PE),c_bool(sync_send))
    
  def addGEMMDevOp(self,A,B,C,X,m,k,n,postScale, postShift,PE):
    self._lib.AddGEMMDevOp(A,B,C,X,m,k,n,postScale, postShift,PE)
    
  def addFCNDevOp(self,A,B,C,X,m,k,n,postScale, postShift,PReLUScale, PReLUAlpha,PE):
    self._lib.AddFCNDevOp(A,B,C,X,m,k,n,postScale, postShift,PReLUScale, PReLUAlpha,PE)
  
  def addSPMVDevOp(self, A, B, C, m,k,nnz, xclbin_opts, relu, PE):
    ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
    spmv_width = int(xclbin_opts["GEMX_spmvWidth"])
    num_cblocks = int(xclbin_opts["GEMX_spmvNumCblocks"])
    spmvMacGroups = int(xclbin_opts["GEMX_spmvMacGroups"])
    t_mVectorBlocks =((1 << (16 - int(xclbin_opts["GEMX_spmvColAddIdxBits"]))) // spmv_width // spmvMacGroups // ddrWidth)
    capacity_Cblocks =  int(spmv_width * spmvMacGroups * t_mVectorBlocks * ddrWidth)
    capacity_Bblocks = int(spmv_width * int(xclbin_opts["GEMX_spmvkVectorBlocks"]) * ddrWidth)
    return self._lib.AddSPMVDevOp(A,B,C,m,k,c_uint(nnz),c_bool(relu), c_uint(num_cblocks),c_uint(capacity_Cblocks), c_uint(capacity_Bblocks),c_uint(PE)) 
    
  def addUSPMVDevOp(self, A, B, C, numRuns,PE):
    return self._lib.AddUSPMVDevOp(A,B,C,numRuns,c_uint(PE))
    
  def getDevBuf(self,A,PE,sync_get):
    return self._lib.GetDevBuf(A,PE,sync_get)
  
  def executeDev(self,sync_exec,PE):
    self._lib.ExecuteDev(sync_exec,PE)
    
_gemxManager = None

def addDevBuf(A,size_row,size_col,datatype,PE=0):  
    return _gemxManager.addDevBuf(A,size_row,size_col,datatype,PE)
    
def addSpDevBuf(row,col,data, A, m, k, nnz, xclbin_opts, PE=0):
    return _gemxManager.addSpDevBuf(row,col,data, A, m, k, nnz, xclbin_opts, PE)
    
def addUSpDevBuf(rows,cols,datas, A, ms, ks, nnzs,pRelus, xclbin_opts,PE=0): 
    return _gemxManager.addUSpDevBuf(rows,cols,datas, A, ms, ks, nnzs, pRelus,xclbin_opts,PE)
    
def allocProgBuf(buf_sz,PE=0):
    return _gemxManager.allocProgBuf(buf_sz,PE)
    
def sendDevBuf(A,PE=0,sync_send=False):
    return _gemxManager.sendDevBuf(A,PE,sync_send)
    
def addGEMMDevOp(A,B,C,X,m,k,n,postScale=1, postShift=0,PE=0):
    return _gemxManager.addGEMMDevOp(A,B,C,X,m,k,n,postScale, postShift,PE)

def addFCNDevOp(A,B,C,X,m,k,n,postScale=1, postShift=0,PReLUScale=1, PReLUAlpha=0,PE=0):
    return _gemxManager.addFCNDevOp(A,B,C,X,m,k,n,postScale, postShift,PReLUScale, PReLUAlpha,PE)
    
def addSPMVDevOp( A,B,C,m,k,nnz,xclbin_opts,relu=False, PE=0):
    _gemxManager.addSPMVDevOp(A,B,C,m,k,nnz,xclbin_opts,relu,PE)
    
def addUSPMVDevOp(A,B,C,numRuns, PE=0):
    _gemxManager.addUSPMVDevOp(A,B,C,numRuns,PE)
    
def getDevBuf(A,PE=0,sync_get=True):
    return _gemxManager.getDevBuf(A,PE,sync_get)    

def executeDev(sync_exec=True,PE=0):
    return _gemxManager.executeDev(sync_exec,PE)

def sendMat ( A,PE=0,sync_send=False):
    _gemxManager.sendMat(A,PE,sync_send)
    
def sendSpMat (row,col,data, m, k, nnz, xclbin_opts, PE=0):
    return _gemxManager.sendSpMat(row,col,data, m, k, nnz, xclbin_opts, PE)

def sendUSpMat(rows,cols,datas, ms, ks, nnzs,pRelus, xclbin_opts,PE=0): 
    return _gemxManager.sendUSpMat(rows,cols,datas, ms, ks, nnzs, pRelus,xclbin_opts,PE)

def getMat (A, PE=0, sync_get = True):
    return _gemxManager.getMat(A, PE,sync_get)
    
def addFCNOp( A,B,C, bias, postScale, postShift, PReLUScale, PReLUAlpha,PE=0):
    _gemxManager.addFCNOp(A, B, C, bias, postScale, postShift, PReLUScale, PReLUAlpha, PE)
    
def addGEMMOp( A,B,C, bias, postScale, postShift,PE=0):
    _gemxManager.addGEMMOp(A, B, C, bias, postScale, postShift, PE)

def addSPMVOp( A,B,C,nnz,xclbin_opts,relu=False, PE=0):
    _gemxManager.addSPMVOp(A,B,C,nnz,xclbin_opts,relu,PE)
    
def addUSPMVOp(A,B,C,numRuns, PE=0):
    _gemxManager.addUSPMVOp(A,B,C,numRuns,PE)

def execute(PE=0, sync_exec = True):
    _gemxManager.execute( PE, sync_exec)
    
def wait(PE=0):
    _gemxManager.wait(PE)    

def clearInstrBuf(PE=0):
    _gemxManager.clearInstrBuf(PE)    
      
def createManager ( libFile ):
  global _gemxManager
  if not _gemxManager:
    _gemxManager = GEMXManager(libFile)    
  return True  
    
def createFCNHandle(args, xclbin_opts):
  if int(xclbin_opts['GEMX_runFcn'])!= 1:
     raise Exception('The xclbin does not include fcn engine.')
  createManager (args.gemxlib)
  return _gemxManager.createFCNHandle(args.xclbin, xclbin_opts["GEMX_numKernels"])

def createStrFCNHandle(args, xclbin_opts):
  if int(xclbin_opts['GEMX_runFcn'])!= 1:
     raise Exception('The xclbin does not include fcn engine.')
  createManager (args.gemxlib)
  return _gemxManager.createStrFCNHandle(args.xclbin, xclbin_opts["GEMX_numKernels"])

def createGEMMHandle(args, xclbin_opts):
  if int(xclbin_opts['GEMX_runGemm'])!= 1:
     raise Exception('The xclbin does not include gemm engine.')
  createManager (args.gemxlib)
  return _gemxManager.createGEMMHandle(args.xclbin, xclbin_opts["GEMX_numKernels"])

def createStrGEMMHandle(args, xclbin_opts):
  if int(xclbin_opts['GEMX_runGemm'])!= 1:
     raise Exception('The xclbin does not include gemm engine.')
  createManager (args.gemxlib)
  return _gemxManager.createStrGEMMHandle(args.xclbin, xclbin_opts["GEMX_numKernels"])
  
def createUSPMVHandle(args, xclbin_opts):
  if int(xclbin_opts['GEMX_runUspmv'])!= 1:
     raise Exception('The xclbin does not include uspmv engine.')
  createManager (args.gemxlib)
  return _gemxManager.createUSPMVHandle(args.xclbin, xclbin_opts["GEMX_numKernels"])

def createStrUSPMVHandle(args, xclbin_opts):
  if int(xclbin_opts['GEMX_runUspmv'])!= 1:
     raise Exception('The xclbin does not include uspmv engine.')
  createManager (args.gemxlib)
  return _gemxManager.createStrUSPMVHandle(args.xclbin, xclbin_opts["GEMX_numKernels"])

def createSPMVHandle(args, xclbin_opts):
  if int(xclbin_opts['GEMX_runSpmv'])!= 1:
     raise Exception('The xclbin does not include spmv engine.')
  createManager (args.gemxlib)
  return _gemxManager.createSPMVHandle(args.xclbin, xclbin_opts["GEMX_numKernels"])

def createStrSPMVHandle(args, xclbin_opts):
  if int(xclbin_opts['GEMX_runSpmv'])!= 1:
     raise Exception('The xclbin does not include spmv engine.')
  createManager (args.gemxlib)
  return _gemxManager.createStrSPMVHandle(args.xclbin, xclbin_opts["GEMX_numKernels"])

def printStats():
  return _gemxManager.printStats()
  
def create_fpga_buf ( shape, np_type , PE=0):
    a = np.zeros ( shape, dtype=np_type, order='C')
    _gemxManager.sendMat(a, PE)
    return a

def parse_cfg(filename):
    myvars = {}
    with open(filename) as myfile:
        for line in myfile:
            for word in line.split():
               name, var = word.split("=")
               myvars[name.strip()] = var.rstrip()  
    return myvars

def default_args():
  parser = argparse.ArgumentParser(description='GEMX')
  parser.add_argument('--xclbin', required = True, help='file path to FPGA bitstream')
  parser.add_argument('--gemxlib', required = True, help='file path to GEMX host code shared library')
  parser.add_argument('--cfg', required=True, help='file describing .xclbin properties')
  return parser
    
def processCommandLine():
  parser = default_args()
  args = parser.parse_args()
  xclbin_opts = parse_cfg ( args.cfg ) 
  return args, xclbin_opts
