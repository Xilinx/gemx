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

class GemxUspmvRT(GemxRT):
    """
    base class for using the GEMX library with USPMV engine in the machine learning examples
    
    Parameters
    ----------
    xclbin_opts: dictionary 
                 information read from config_info.dat used to build the xclbin
    wgt:         list
                 weight read from the model
    relu_list:   list
                 list to show that for each layer, if relu or not
    """
    def __init__(self, wgt, relu_list, xclbin_opts):                                                                                                                                                                                                                        
      self.min_m = int(xclbin_opts["GEMX_uspmvInterleaves"]) * int(xclbin_opts["GEMX_ddrWidth"])                                                 
      self.min_k = int(xclbin_opts["GEMX_ddrWidth"])                                                                                            
      self._qw = wgt                                                                                            
      self.A_list = []                                                                                                                       
      self.sizes = []                                                                                                                          
      self.xclbin_opts = xclbin_opts
      self.out_dim = (None, self._qw[-1].shape[1])
      stage_size=int(xclbin_opts["GEMX_uspmvStages"])
      if stage_size != 1:
            m_sizes = np.zeros(shape=(stage_size,),dtype=np.int32)
            k_sizes = np.zeros(shape=(stage_size,),dtype=np.int32)
            nnz_sizes = np.zeros(shape=(stage_size,),dtype=np.int32)
            activation_list=np.zeros(shape=(stage_size,),dtype=np.float32)
            all_rows = []
            all_cols = []
            all_datas = []
      for i,wi in enumerate(self._qw):
            wi = np.transpose(wi)                                                                                                                
            size_r,size_c,size_nnz,rows,cols,datas = self.format_for_sparse_fpga(wi,wi.shape)
            activation = relu_list[i]
            self.sizes.append((size_r,size_c,size_nnz))
            if stage_size==1:
              self.A_list.append(gemx.sendUSpMat(rows,cols,datas,np.array(size_r,dtype=np.int32),np.array(size_c,dtype=np.int32),np.array(size_nnz,dtype=np.int32),np.array(activation,dtype=np.float32),xclbin_opts))
            else:
              all_rows.append(rows)
              all_cols.append(cols)
              all_datas.append(datas)
              m_sizes[i]=size_r
              k_sizes[i]=size_c
              nnz_sizes[i]=size_nnz
              activation_list[i]=activation
      if stage_size != 1: 
            self.A_list.append(gemx.sendUSpMat(np.concatenate(all_rows),np.concatenate(all_cols),np.concatenate(all_datas),m_sizes,k_sizes,nnz_sizes,activation_list,xclbin_opts))  

    def format_for_sparse_fpga ( self, weight_mat, shape):            
      """
      return padded sizes, row index, col index and non-zero elements array of given sparse matrix to fit the format of uspmv engine
      
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
      row_size = shape[0]
      col_size = shape[1]
      m_index = np.nonzero(weight_mat)
      m_row = (m_index[0]).astype(np.uint16)
      m_col = (m_index[1]).astype(np.uint16)
      m_value = (weight_mat[m_row,m_col]).astype(np.float32)
      while size_nnz % self.min_k !=0:
          size_nnz=size_nnz+1
          m_row = (np.append(m_row,0)).astype(np.uint16)
          m_col = (np.append(m_col,0)).astype(np.uint16)
          m_value = (np.append(m_value,0)).astype(np.float32)
      ind = np.lexsort((m_row,m_col))
      m_row=m_row[ind]
      m_col=m_col[ind]
      m_value=m_value[ind]
      row_size_padded,col_size_padded = self.get_padded_shape([row_size,col_size], self.min_m, self.min_m)
      return row_size_padded,col_size_padded,size_nnz,m_row,m_col,m_value

    def predict ( self, inp):
      self.out_dim = (inp.shape[0],self.out_dim[1])
      inp = self.format_for_fpga(inp, 1, 1)
      B = inp.astype(np.float32)
      gemx.sendMat(B)            
      C = np.zeros ((inp.shape[0], self.sizes[0][0]), dtype=np.float32)
      gemx.sendMat(C)
      gemx.addUSPMVOp(self.A_list[0],B,C,inp.shape[0])  
      gemx.execute()
      gemx.getMat(C)
      gemx.clearInstrBuf()
      result = C        
      return result[:self.out_dim[0],:self.out_dim[1]]
 

