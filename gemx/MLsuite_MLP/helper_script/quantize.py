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
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
import argparse
from keras.datasets import reuters
from keras.datasets import mnist
from keras.preprocessing.text import Tokenizer

# Usage:
# compute_quantize_scale(numpy array, model.get_weights()):
# Give model's weight and numpy array of input values, the function will calculate the input scale, weight scale and post scale based on 3 different methods
#  
# python quantize.py --data examples/keras/data/SansEC_Train_Data.csv --model examples/keras/best_model.h5 --default_test local
# Give input data from csv file, a h5 file for the model, the script calculate the input scale, weight scale and post scale for the default example
# compute_quantize_scale_16 returns the best scale values
# 
# python quantize.py --model best_mnist_model.h5 --default_test mnist
# Quantization for mnist mlp example. compute_quantize_scale_8 returns the best scale values
# 
# python quantize.py  --model best_reuters_model.h5 --default_test reuters
# Quantization for reuters mlp example. common_quantize returns the best scale values
#
# python quantize.py --model ./XXX.h5
# Using your own function to create keras model from the h5 file, the script will calculate the input scale, weight scale and post scale for this case
# set --data if data is from csv file

class Quantization():
  def compute_quantize_scale( self, inp, wb, int_max):
    #assume Relu activation for each layer other than the final layer
    print ("    Min    ","    Max    ")
    print ("inp (", inp.min(), ",", inp.max(),")")
    weights = wb[0::2]
    bias=wb[1::2]
    for i,w in enumerate(weights):
        print ( "w", i, ": ", np.min(w), ", ", np.max(w))
    for i,b in enumerate(bias):
        print ( "b", i, ": ", np.min(b), ", ", np.max(b))   
    number_of_layers = len(weights)
    output_range = np.zeros(number_of_layers)
    C=[inp]
    for i in range(number_of_layers):
      o = np.matmul (C[i], weights[i])
      o = o + bias[i]
      print ("o" + str(i) +" (", np.min(o), ",", np.max(o),")")
      output_range[i] = 2 * max(abs(np.min(o)),abs(np.max(o)))
      o[o<0] = 0
      C.append(o)   
    input_range = 2 * max(abs(inp.min()),abs(inp.max()))
    input_scale = int_max/input_range
    weight_range = np.array([2 * max(abs(np.min(x)),abs(np.max(x))) for x in weights])
    weight_scale = int_max/weight_range
    output_scale = int_max/output_range
    print ("weight_range:",weight_range,"weight_scale:",weight_scale)
    print ("input_range:",input_range,"input_scale:",input_scale)
    print ("output_range:",output_range,"output_scale:",output_scale)
    
    output_scale_new = np.zeros(number_of_layers)
    
    output_scale_new[0] = output_scale[0]/(weight_scale[0]*input_scale)
    
    for i in range(1,number_of_layers-1):
      output_scale_new[i] = output_scale[i]/(weight_scale[i]*output_scale[i-1])
      
    # if using output_scale[-1]/(weight_scale[-1]*output_scale[-2]), then final result needs to be divided by output_scale[-1]
    output_scale_new[-1] = 1/(weight_scale[-1]*output_scale[-2]) 
    
    print('output_scale_new:', output_scale_new)
    
    my_dict = self.build_dictory()
    my_keys  = list(my_dict.keys())
    g_post_scale = []
    #try to find the closet key
    for i in range(number_of_layers):
      difference = [abs(x-output_scale_new[i]) for x in my_keys]
      idx = difference.index(min(difference))
      print(my_keys[idx])
      print(my_dict[my_keys[idx]])
      g_post_scale.append(my_dict[my_keys[idx]])
    print("======Copy the following information to mlp.py======")
    print("g_in_scale =", input_scale)
    print("g_wgt_scale =", "[" + ", ".join(str(i) for i in weight_scale)+ "]")
    print("g_post_scale =", g_post_scale)
    print("====================================================")
    return input_scale, weight_scale, g_post_scale
  
  def compute_quantize_scale_16( self, inp, wb):
    return self.compute_quantize_scale(inp, wb, pow(2,16))
  
  def compute_quantize_scale_8( self, inp, wb):
    return self.compute_quantize_scale(inp, wb, pow(2,8))
  
  def common_quantize( self, length, inp_scale, p_weight, p_output):
    #weights = wb[0::2]
    number_of_layers = length
    weight_scale=[]
    output_scale=[]
    for i in range(number_of_layers):
      weight_scale.append(pow(2,p_weight))
      output_scale.append(pow(2,p_output))    
    output_scale_new = np.zeros(number_of_layers)
    output_scale_new[0] = output_scale[0]/(weight_scale[0]*inp_scale)
    for i in range(1, number_of_layers-1):
      output_scale_new[i] = output_scale[i]/(weight_scale[i]*output_scale[i-1])
    output_scale_new[-1] = 1/(weight_scale[-1])
    my_dict = self.build_dictory()
    my_keys  = list(my_dict.keys())
    g_post_scale = []
    #try to find the closet key
    for i in range(number_of_layers):
      difference = [abs(x - output_scale_new[i]) for x in my_keys]
      idx = difference.index(min(difference))
      print(my_keys[idx])
      print(my_dict[my_keys[idx]])
      g_post_scale.append(my_dict[my_keys[idx]])
    print("==================Common Quantize===================")
    print("======Copy the following information to mlp.py======")
    print("g_in_scale =", inp_scale)
    print("g_wgt_scale =", "[" + ", ".join(str(i) for i in weight_scale)+ "]")
    print("g_post_scale =", g_post_scale)
    print("====================================================")
    return inp_scale, weight_scale, g_post_scale
  
  def build_dictory(self):
    my_dict={}
    for i in range(6,0,-1):
      for j in range(32,0,-1):
          my_key= i * pow(2,-j) 
          my_dict.update({my_key:[i,j]})         
    #print (my_dict)
    return my_dict

# local functions
def default_test_build ( weights, test_data, num_classes):
    model = create_keras_model(test_data.values.shape[1], num_classes )
    model.load_weights(weights)
    return model

def create_keras_model(in_dims, num_classes):  
    model = Sequential()
    model.add(Dense(100, input_dim=in_dims, activation='relu', name='d1'))
    model.add(Dense(25, activation='relu', name='d2'))
    model.add(Dense(num_classes, activation='softmax', name='d3'))
    model.summary()
    return model
  
def load_train_data(data):
    train_fd = pd.read_csv(args.data) # Load training data.
    IDcol = 'Run' # A column used to identified the run for data collection; not an independent variable.
    target = 'Class' # The column name for our dependent variable.
    predictors = [x for x in train_fd.columns if x not in [target, IDcol]] # Define column names to use as independent variables.
    return train_fd,predictors,len(train_fd[target].unique())

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEMX')
    parser.add_argument('--data', required = False, help='inference data file')
    parser.add_argument('--model', required = True, help='model')
    parser.add_argument('--default_test', required = False, default = 'No', choices = ['reuters', 'mnist', 'local', 'No'] ,help='set this argument if you just want to run the default test')
    args = parser.parse_args()
    
    if args.default_test == 'local':
      train_fd, predictors, num_classes = load_train_data(args.data)
      model = default_test_build( args.model, train_fd[predictors], num_classes)
      data = train_fd[predictors].values     
      Quantization().compute_quantize_scale_16(data,  model.get_weights())
    elif args.default_test == 'reuters':
      (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)
      tokenizer = Tokenizer(num_words=1000)
      x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
      data=x_test
      print(x_test)
      num_classes = np.max(y_train) + 1      
      model = Sequential()
      model.add(Dense(512, input_shape=(1000,),activation='relu'))
      model.add(Dense(num_classes, activation='softmax'))
      model.load_weights(args.model)
      Quantization().common_quantize(len(model.get_weights()[0::2]), 1, 10, 8) # input_scale = 1, weight_scale = pow(2, 10), output_scale = pow(2, 8)
    elif args.default_test == 'mnist':
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      x_train = x_train.reshape(60000, 784)
      x_test = x_test.reshape(10000, 784) 
      x_train = x_train.astype('float32')
      x_test = x_test.astype('float32')
      x_train /= 255
      x_test /= 255
      num_classes = 10
      y_train = keras.utils.to_categorical(y_train, num_classes)
      y_test = keras.utils.to_categorical(y_test, num_classes)
      model = Sequential()
      model.add(Dense(512, input_shape=(784,),activation='relu'))
      model.add(Dense(512, activation='relu'))
      model.add(Dense(num_classes, activation='softmax'))
      model.load_weights(args.model) 
      data=x_test 
      Quantization().compute_quantize_scale_8(data,  model.get_weights())
    else:
      print ('INFO: The script is not running the default test now, please create your own keras model')

