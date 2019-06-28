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

#!/usr/bin/env python
#
import numpy as np
from gemx_uspmv_rt import GemxUspmvRT
from knn_rt import KNNRT

class KNNUspmvRT(KNNRT):
  def __init__(self, X, Y, xclbin_opt):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = Y
    self.train_sum = np.sum(np.square(self.X_train), axis=1)
    self.GemxUspmvRT = GemxUspmvRT ([X.T], [1], xclbin_opt)
       
  def compute_dist_fpga(self, X):
    fpga_out = self.GemxUspmvRT.predict(X)
    test_sum = np.sum(np.square(X), axis=1)
    #print ("predict fpga", test_sum.shape, train_sum.shape, fpga_out.shape)
    dists = np.sqrt(-2 * fpga_out + test_sum.reshape(-1, 1) + self.train_sum)
    return dists  
