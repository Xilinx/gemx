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
from gemx_uspmv_rt import GemxUspmvRT

class KerasUspmvRT(GemxUspmvRT): 
    """
    runtime class to use USPMV in the keras example\n
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
      wgt = keras_model.get_weights()[0::2]
      activation_list=[]
      for i,wi in enumerate(wgt):
            l=keras_model.layers[i]
            activation = 0 if l.get_config()['activation'] == 'relu' else 1
            activation_list.append(activation) 
      GemxUspmvRT.__init__(self,wgt,activation_list,xclbin_opts)
      self.out_dim = ( batch_sz, keras_model.layers[-1].output_shape[1] )

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
      stage_size=int(self.xclbin_opts["GEMX_uspmvStages"])
      layer_size=len(self._qw)
      if stage_size==1: 
          inp = self.format_for_fpga(inp, 1, self.min_m)
          C_list = [inp.astype(np.float32)]
          gemx.sendMat(C_list[0])            
          for i in range(layer_size):   
            C_list.append(np.zeros ((inp.shape[0], self.sizes[i][0]), dtype=np.float32))
            gemx.sendMat(C_list[i+1])
            gemx.addUSPMVOp(self.A_list[i],C_list[i],C_list[i+1],inp.shape[0])
      else:
          inp = self.format_for_fpga(inp, 1, self.min_m)
          C_list = [inp.astype(np.float32)]
          gemx.sendMat(C_list[0])
          C_end = np.zeros ((inp.shape[0], self.sizes[-1][0]), dtype=np.float32)
          gemx.sendMat(C_end)
          gemx.addUSPMVOp(self.A_list[0],C_list[0],C_list[-1],inp.shape[0])  
      gemx.execute()
      gemx.getMat(C_list[-1]) 
      gemx.clearInstrBuf()
      result = C_list[-1]        
      return result[:self.out_dim[0],:self.out_dim[1]]
 
