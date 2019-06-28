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
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cross_validation import cross_val_score
import gemx
from knn_rt import KNNRT
from gemx_rt import GemxRT
from knn_uspmv_rt import KNNUspmvRT
import time

parser = gemx.default_args()
parser.add_argument('--numiter', type = int, default = 100, help='number of iterations to run')
parser.add_argument('--engine', default = 'gemm', choices=['gemm','uspmv'],help='choose gemm or uspmv engine')
args = parser.parse_args()
xclbin_opt = gemx.parse_cfg ( args.cfg )

if args.engine == 'gemm':
    gemx.createGEMMHandle(args, xclbin_opt)
else:
    gemx.createUSPMVHandle( args, xclbin_opt )

# define column names
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
num_neighbor = 3
# loading training data
df = pd.read_csv('./examples/knn/iris.data', header=None, names=names)
print(df.head())

# create design matrix X and target vector y
X = np.array(df.ix[:, 0:4])
y = np.array(df['class']) 

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
start = time.time()
for i in range(args.numiter):
    knn.predict(X_test)

print("\nsklearn classifier performance: ", (time.time() - start) / args.numiter, " ms")


if args.engine == 'gemm':
    knnInst = KNNRT(X_train_int, y_train , X_test.shape, xclbin_opt)
else:
    knnInst = KNNUspmvRT(X_train_int, y_train , xclbin_opt)
    
start = time.time()
for i in range(args.numiter):
    knnInst.predict_cpu(X_test_int, num_neighbor)

print("\nCPU classifier performance: ", (time.time() - start) / args.numiter, " ms")

start = time.time()
for i in range(args.numiter):
    knnInst.predict_fpga(X_test_int, num_neighbor)

print("\nFPGA classifier performance: ", (time.time() - start) / args.numiter, " ms")
