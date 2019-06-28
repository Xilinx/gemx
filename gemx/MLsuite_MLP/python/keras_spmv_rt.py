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
from gemx_rt import GemxRT                            

class KerasSpmvRT(GemxRT):
    """
    runtime class to use SPMV in the keras example\n
    It will send all the weights matrices loaded from the model to kernel for future usage
    
    Parameters
    ----------
    keras_model: model 
                 pre-trained model
    xclbin_opts: dictionary 
                 information read from config_info.dat used to build the xclbin
    batch_sz:    shape
                 output prediction matrix size
    """
    def __init__(self, keras_model, batch_sz, xclbin_opts):                                                                                     
      self.min_m = int(xclbin_opts["GEMX_spmvMacGroups"]) * int(xclbin_opts["GEMX_spmvWidth"])                                                                                             
      self.min_k = int(xclbin_opts["GEMX_ddrWidth"])                                                                                            
      self._qw = keras_model.get_weights()[0::2]                                                                                               
      #self._qb = keras_model.get_weights()[1::2]                                                                                                                                                                                                                      
      self.A_list = []                                                                                                                       
      self.sizes = []                                                                                                                          
      self.xclbin_opts = xclbin_opts
      for i,wi in enumerate(self._qw):                                                                                                         
          wi = np.transpose(wi)                                                                                                                
          size_r,size_c,size_nnz,row,col,nnz = self.format_for_sparse_fpga(wi,self._qw[i].shape)                                               
          self.sizes.append((size_r,size_c,size_nnz))
          self.A_list.append(gemx.sendSpMat(row,col,nnz,size_r,size_c,size_nnz,xclbin_opts)) 
      self.out_dim = ( batch_sz, keras_model.layers[-1].output_shape[1] )                                                                      
      self.kmodel = keras_model 

    def format_for_sparse_fpga ( self, weight_mat, shape):
      """
      return padded sizes, row index, col index and non-zero elements array of given sparse matrix to fit the format of spmv engine
      
      Parameters
      ---------- 
      weight_mat:   ndarray
                    transpose of the weight sparse matrix
      shape:        shape
                    shape of the sparse matrix               
      Return
      ------
      int
                    padded row size
      int
                    padded col size
      int
                    padded number of non-zero elements
      array         
                    row indices
      array         
                    col indices
      array         
                    non-zero elements
      """   
      size_nnz = np.count_nonzero(weight_mat)
      row_size = max(shape[0],shape[1])
      col_size = max(shape[0],shape[1])
      m_index = np.nonzero(weight_mat)
      m_row = (m_index[0]).astype(np.int32)
      m_col = (m_index[1]).astype(np.int32)
      m_value = (weight_mat[m_row,m_col]).astype(np.float32)
      row_size_padded,col_size_padded = self.get_padded_shape([row_size,col_size], self.min_m, self.min_m)
      return row_size_padded,col_size_padded,size_nnz,m_row,m_col,m_value

    def predict ( self, inp):
      """
      prepare input matrix for the engine, send all the matrices and instructions to kernel and get the result prediction matrix
      
      Parameters
      ---------- 
      inp:      array
                input matrix
                
      Return
      ------
      array
               result prediction matrix
      
      """
      C_list = [[]] * 2
      inp = self.format_for_fpga(inp, 1,1)
      C_list[0] = np.transpose(inp)
      B = (C_list[0][:,0]).astype(np.float32)
      C_vector = [B]
      gemx.sendMat(C_vector[0])
      for i,l in enumerate(self.kmodel.layers):
          C_vector.append(np.zeros ((self.sizes[i][0], 1), dtype=np.float32))
          gemx.sendMat(C_vector[i+1])
          activation = True if l.get_config()['activation'] == 'relu' else False
          gemx.addSPMVOp(self.A_list[i], C_vector[i], C_vector[i+1], self.sizes[i][2], self.xclbin_opts, activation)
      gemx.execute()
      gemx.getMat(C_vector[-1])
      C_list[1] = C_vector[-1]
      for j in range(1, C_list[0].shape[1]):
          C_vector[0][:] = (C_list[0][:,j]).astype(np.float32)
          gemx.sendMat(C_vector[0])
          C_vector[-1].fill(0)
          for i in range(len(self.kmodel.layers)):
              gemx.sendMat(C_vector[i+1])
          gemx.execute()
          gemx.getMat(C_vector[-1])
          C_list[1] = np.append(C_list[1],C_vector[-1],axis=1)
      gemx.clearInstrBuf()
      result = np.transpose(C_list[1])
      return result[:self.out_dim[0],:self.out_dim[1]]
