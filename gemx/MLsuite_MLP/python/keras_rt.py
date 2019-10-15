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
import numpy as np
import gemx
from gemx_rt import GemxRT

        
class KerasRT(GemxRT):
    """
    runtime class to use FCN in the keras example\n
    It will send all the weights matrices loaded from the model to kernel for future usage
    
    Parameters
    ----------
    xclbin_opts: dictionary 
                 information read from config_info.dat used to build the xclbin
    keras_model: model 
                 pre-trained model
    wgt_scale:   list
                 Quantization parameters multiple with weight matrices
    bias_scale:  list
                 Quantization parameters multiple with bias matrices
    post_scale:  list
                 Quantization parameters multiple with output matrices 
    """
    def __init__(self, keras_model, xclbin_opts, wgt_scale, bias_scale, post_scale):
      keras_w = keras_model.get_weights()[0::2]
      keras_b = keras_model.get_weights()[1::2]
      GemxRT.__init__(self, xclbin_opts, keras_w, keras_b, wgt_scale, bias_scale, post_scale)
      self.kmodel = keras_model
       
    def loadInstr(self):
      gemx.clearInstrBuf()
      for i,l in enumerate(self.kmodel.layers):
          act = l.get_config()['activation']
          if self._qw[0].dtype == np.float32:
            if act == 'relu':
              gemx.addFCNOp( self._qw[i], self.fpga_buf[i], self.fpga_buf[i+1], self._qb[i], 1, 0, 0, 0)
            else:
              gemx.addGEMMOp( self._qw[i], self.fpga_buf[i], self.fpga_buf[i+1], self._qb[i], 1, 0)            
          else:
            if act == 'relu':
              gemx.addFCNOp( self._qw[i], self.fpga_buf[i], self.fpga_buf[i+1], self._qb[i], self.post_scale[i][0], self.post_scale[i][1], 0, 0)
            else:
              gemx.addGEMMOp( self._qw[i], self.fpga_buf[i], self.fpga_buf[i+1], self._qb[i], self.post_scale[i][0], self.post_scale[i][1])
