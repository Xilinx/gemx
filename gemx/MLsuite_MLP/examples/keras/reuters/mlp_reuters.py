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

from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
import argparse
import gemx
import sys
sys.path.append("./examples/keras")
import mlp_common

#Quantization parameters to bring fp32 ranges to fit into int16; parameters are derived offline ( see quantize.py)
#This parameters won't be used if using fp32 xclbin
g_in_scale = 1
g_wgt_scale = [1024, 1024]
g_bias_scale = [1024.0, 262144.0]
g_post_scale = [[1, 2], [1, 10]] # 256/1024, 1/1024


def train(model, x_train, y_train, x_test, y_test):
    num_classes = np.max(y_train) + 1

    tokenizer = Tokenizer(num_words=1000)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    modelcheckpoint_callback = ModelCheckpoint("./best_reuters_model.h5", monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=5,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[modelcheckpoint_callback])
    score = model.evaluate(x_test, y_test,
                       batch_size=32, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def create_keras_model(num_classes):
    '''
    Generate a simple Keras model.
    '''  
    model = Sequential()
    model.add(Dense(512, input_shape=(1000,),activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model

if  __name__ == '__main__':
    np.random.seed(27)
    parser = argparse.ArgumentParser(description='GEMX')
    parser.add_argument('--model', required = True, help='model')
    parser.add_argument('--xclbin', required = True, help='file path to FPGA bitstream')
    parser.add_argument('--cfg', required = True, help='file describing properties of .xclbin')
    parser.add_argument('--gemxlib', required = True, help='file path to GEMX host code shared library')
    parser.add_argument('--engine', default = 'fcn', choices=['fcn','uspmv'],help='choose fcn, uspmv engine')
    parser.add_argument('--train', default = False, help='set to True if retrain the model')
    args = parser.parse_args()
    xclbin_prop = gemx.parse_cfg(args.cfg)

    #load xclbin 
    if args.engine == 'fcn':
        gemx.createFCNHandle( args, xclbin_prop )
    else:
        gemx.createUSPMVHandle( args, xclbin_prop )
        
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)
    tokenizer = Tokenizer(num_words=1000)   
    num_classes = np.max(y_train) + 1
    
    model = create_keras_model(num_classes)
    model.load_weights(args.model)
    
    if args.train:
        train(model, x_train, y_train, x_test, y_test)
    
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    cpu_out = mlp_common.predict_cpu( model, x_test)

    if args.engine == 'fcn':
        fpga_out = mlp_common.predict_fpga( model, x_test, xclbin_prop, g_in_scale, g_wgt_scale, g_bias_scale, g_post_scale)
    else:
        fpga_out = mlp_common.predict_uspmv_fpga(model, x_test, xclbin_prop)
        
        
    print("compare real data with cpu")
    mlp_common.compare_real_results( y_test, np.argmax(cpu_out,axis=1))
    print("compare real data with fpga")
    mlp_common.compare_real_results( y_test, np.argmax(fpga_out,axis=1))
    print("compare cpu with fpga:")
    mlp_common.compare_real_results( np.argmax(cpu_out,axis=1), np.argmax(fpga_out,axis=1))


