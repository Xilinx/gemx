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
import math

class GemxRT():
    """
    base class for using the GEMX library in the machine learning examples
    
    Parameters
    ----------
    xclbin_opts: dictionary 
                 information read from config_info.dat used to build the xclbin
    wgt:         list
                 weight read from the model
    bias:        list
                 bias read from the model
    wgt_scale:   list
                 Quantization parameters multiple with weight matrices
    bias_scale:  list
                 Quantization parameters multiple with bias matrices
    post_scale:  list
                 Quantization parameters multiple with output matrices    
    """
    def __init__(self, xclbin_opts, wgt, bias, wgt_scale, bias_scale, post_scale):
      
      #Ensuring min_m and min_n never fall below min_k is needed when chaining multiple GEMM operations
      #If min_m/min_n is less than min_k, using the output of a GEMM call where either dimension 
      #is less than min_k would lead to bad results if it's directly used as input for another GEMM operation  
      ddrwidth = int(xclbin_opts["GEMX_ddrWidth"])
      self.min_m = ddrwidth * max (int(xclbin_opts["GEMX_gemmKBlocks"]), int(xclbin_opts["GEMX_gemmMBlocks"]) )
      self.min_k = ddrwidth * int(xclbin_opts["GEMX_gemmKBlocks"])
      self.min_n = ddrwidth * int(xclbin_opts["GEMX_gemmNBlocks"])
      if type (wgt) != list:
          wgt = [wgt]
      
      if type(bias) != list:
          bias = [bias]
      
      assert len(wgt) == len(wgt_scale)
      assert len(bias) == len(bias_scale)

      self._wshape = []
      for w in wgt:
          self._wshape.append(w.shape)
      if xclbin_opts["GEMX_dataType"] == "float":
          self._qw = wgt
          self._qb = bias
      else:
          self._qw = [np.int16(np.around(a*b)) for a,b in zip(wgt, wgt_scale)]
          self._qb = [np.int32(np.around(a*b)) for a,b in zip(bias, bias_scale)]
      for i,b in enumerate(self._qw):
          b = np.transpose(b)
          self._qw[i] = self.format_for_fpga( b, self.min_m, self.min_k)
          gemx.sendMat(self._qw[i])
          
      #in_row, in_col = self.get_padded_shape(in_dim, self.min_m, self.min_k)
      self.fpga_buf = []
      self.out_dim = None
      self.post_scale = post_scale
      self.batch_sz = 0
        
    def get_padded_shape ( self, shape, min_row, min_col):
      """
      return padded sizes for row and col
     
      Parameters
      ---------- 
      shape:   shape
               a tuple of (actual row, actual col)
      min_row: int
               minimal row size supported
      min_col: int
               minimal col size supported     
               
      Return
      ------
      tuple
               padded row size, padded col size
      """
      row_padded = int( math.ceil( np.float32(shape[0]) / min_row ) * min_row ) 
      col_padded = int( math.ceil( np.float32(shape[1]) / min_col ) * min_col )
      return row_padded,col_padded

    def format_for_fpga ( self, nparr, min_row, min_col):
      """
      pad the numpy array with given minimal row size and col size
      
      Parameters
      ---------- 
      nparr:   ndarray
               array need to be padded
      min_row: int
               minimal row size supported
      min_col: int
               minimal col size supported     
               
      Return
      ------
      ndarray
               padded numpy array
      """      
      row_padded, col_padded = self.get_padded_shape ( nparr.shape, min_row, min_col)
      padded_arr = np.zeros ( (row_padded, col_padded), dtype=nparr.dtype, order='C')
      padded_arr[0:nparr.shape[0], 0:nparr.shape[1]] = nparr
      return padded_arr            
    
    def format_bias (self, b, dim, min_row, min_col):
      if b.ndim == 1:
          b = np.broadcast_to(b, (dim[1],dim[0]) )
      
      b = np.transpose(b)
      b = self.format_for_fpga( b, min_row, min_col)
      gemx.sendMat(b)    
      return b
    
    def init_fpgabuf (self, in_shape ):  
      if self.batch_sz != in_shape[0]:
          self.batch_sz = in_shape[0]
          fpga_buf = []
          buf_dim = [in_shape]
      
          for i in self._wshape:
              buf_dim.append( (i[1], in_shape[1]) )
              
          self.out_dim = buf_dim[-1]
              
          for d in buf_dim:
              d_padded = self.get_padded_shape(d, self.min_m, self.min_n)
              fpga_buf.append ( gemx.create_fpga_buf( d_padded, self._qw[0].dtype ) )
          
          self.fpga_buf = fpga_buf
          
          formatted_bias = []
          for dim,b  in zip (buf_dim[1:], self._qb):
              b = self.format_bias (b, dim, self.min_m, self.min_n)
              formatted_bias.append(b)   
          
          self._qb = formatted_bias           
    
    def loadInstr(self):
      gemx.clearInstrBuf()
      for i,(w_i,b_i) in enumerate( zip( self._qw, self._qb) ):
          gemx.addGEMMOp( w_i , self.fpga_buf[i], self.fpga_buf[i+1], b_i, self.post_scale[i][0], self.post_scale[i][1])
            
    def predict ( self, inp, in_scale, xclbin_opts):
      """
      prepare input matrix for the engine, send all the matrices and instructions to kernel and get the result prediction matrix
      
      Parameters
      ---------- 
      inp:      array
                input matrix
      in_scale: float   
                input scale
      Return
      ------
      array
               result prediction matrix
      
      """
      inp=np.transpose(inp)
      self.init_fpgabuf(inp.shape)
      self.loadInstr()
      if xclbin_opts["GEMX_dataType"] == "float":
        padded_arr = self.format_for_fpga(inp, self.min_k, self.min_n)
        np.copyto(self.fpga_buf[0],  padded_arr, casting='same_kind', where=True)
      else:
        padded_arr = self.format_for_fpga(inp * in_scale, self.min_k, self.min_n)
        np.copyto(self.fpga_buf[0],  np.int16(np.around(padded_arr)), casting='same_kind', where=True)
      gemx.sendMat(self.fpga_buf[0])
      gemx.execute()
      gemx.getMat (self.fpga_buf[-1])
      return np.transpose(self.fpga_buf[-1][:self.out_dim[0],:self.out_dim[1]])                
    
