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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import gemx
from knn_rt import KNNRT
from knn_uspmv_rt import KNNUspmvRT

parser = gemx.default_args()
parser.add_argument('--engine', default = 'gemm', choices=['gemm','uspmv'],help='choose gemm or uspmv engine')
args = parser.parse_args()
xclbin_opt = gemx.parse_cfg ( args.cfg )

if args.engine == 'gemm':
    gemx.createGEMMHandle(args, xclbin_opt)
else:
    gemx.createUSPMVHandle(args,xclbin_opt)

# define column names
num_neighbor = 3
# loading training data
iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Quantization of floating point input should be applied for better accuracy
#Cast and round input data to int16 for brevity
X_train_int = np.ascontiguousarray(np.rint(X_train), dtype=np.int16)
X_test_int = np.ascontiguousarray(np.rint(X_test), dtype=np.int16)

knn = KNeighborsClassifier(n_neighbors=num_neighbor)
# fitting the model
knn.fit(X_train_int, y_train)
# predict the response
pred_sklearn = knn.predict(X_test)

pred_cpu = []
pred_fpga = []
if args.engine == 'gemm':
    knnInst = KNNRT(X_train_int, y_train , X_test.shape, xclbin_opt)
else:
    knnInst = KNNUspmvRT(X_train_int, y_train , xclbin_opt)
pred_cpu = knnInst.predict_cpu(X_test_int, num_neighbor)
pred_fpga = knnInst.predict_fpga(X_test_int, num_neighbor)

# evaluate accuracy
acc_sklearn = accuracy_score(y_test, pred_sklearn) * 100
print('\nsklearn classifier accuracy: %d%%' % acc_sklearn)

acc_cpu = accuracy_score(y_test, pred_cpu) * 100
print('\nCPU classifier accuracy: %d%%' % acc_cpu)

acc_fpga = accuracy_score(y_test, pred_fpga) * 100
print('\nFPGA classifier accuracy: %d%%' % acc_fpga)
