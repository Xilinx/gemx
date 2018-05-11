GEMX PYTHON API USER GUIDE
======================

This user guide contains the following sections:

1. OVERVIEW
2. SOFTWARE AND SYSTEM REQUIREMENTS
3. A USAGE EXAMPLE
4. PYTHON API LIST
5. SUPPORT
6. LICENSE AND CONTRIBUTING TO THE REPOSITORY
7. ACKNOWLEDGEMENTS
8. REVISION HISTORY

## 1. OVERVIEW
The Python APIs presented here allow users to use Python function calls for offloading matrix operations to FPGA-based GEMX engines. This Python binding support is implemented by a Python wrapper, which wraps c++ functions defined in a shared library to Python functions. A set of test programs have been provided to demonstrate the usage of these APIs. Please refer to GEMX_ENGINE_UG for detailed information about GEMX engine design.

* supported engines: GEMM, FCN, SPMV
* supported datatypes: short for GEMM and FCN, float and int for SPMV

## 2. SOFTWARE AND SYSTEM REQUIREMENTS
* Pre-built FPGA image and its paired configration data (This data is generated automaticly while building the FPGA image)
* Supported device: kcu1500, vcu1525, ku115 (Please refer to Makefile for more details on device support)

## 3. A USAGE EXAMPLE
Here is an example for using Python API to offload general matrix matrix multiplications to an FPGA card.

### 3.1 BUILDING A SINGLE KERNEL FPGA IMAGE
Navigate to the gemx/ directory, and then run command:
  
```
make run_hw GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=ku115 GEN_BIN_PROGRAM="gemm 512 512 512 512 512 512 512 1 0 A05 B05 C05 X05"
```
This command builds an image that includes one GEMM engine on ku115 platform and config_info.dat for configration data.

### 3.2 BUILDING SHARED LIBRARY

```
make host_lib
``` 
This command creates libgemxhost.so used for Python API. libgemxhost.so only needs to be built once and it could be used for FPGA images with different settings (the config_info.dat).

### 3.3 RUNNING THE EXAMPLE
Refer to the following environment variables setting:
```
export PYTHONPATH=./src/python
export XILINX_OPENCL=$1/xbinst
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$XILINX_OPENCL/runtime/lib/x86_64
export PATH=$XILINX_OPENCL/runtime/bin:$PATH
```
Launch the application via following command to offload GEMM operations:
```
python -u tests/test_gemm.py --xclbin gemx.xclbin --gemxlib libgemxhost.so --cfg config_info.dat
```
If you see similar outputs, that means your run is successful. Here bias Matrix refers to the X matrix in the GEMM engine (see GEMX_ENGINE_UG.md). 
```
test_basic(PE=0): 512 512 128 16 17
('A: ', 32766, -32768, 73.187705993652344)
('B: ', 32766, -32768, 5.9087066650390625)
('bias: ', 2147467977, -2147332159, -1853396.6735229492)
Success!

test_basic(PE=0): 256 512 128 2 18
('A: ', 32764, -32766, -47.010353088378906)
('B: ', 32766, -32765, 4.309112548828125)
('bias: ', 2147373750, -2147460991, 11723457.787384033)
Success!

test_basic(PE=0): 2048 512 128 4 18
('A: ', 32766, -32768, 15.935216903686523)
('B: ', 32766, -32767, 119.099609375)
('bias: ', 2147465722, -2147476258, -2109845.3656768799)
Success!

test_basic(PE=0): 2048 512 128 128 17
('A: ', 32766, -32768, 1.1315183639526367)
('B: ', 32766, -32768, 6.0645751953125)
('bias: ', 2147481777, -2147469048, 764635.85624313354)
Success!
```

### 3.4 EXAMPLE CODE STRUCTURE
The main function in test_gemm.py takes the steps below to offload GEMM operations:
* Read command line options information to args and config_info.dat information to xclbin_opts
* Create GEMM handle using the above information 
* Run test function hard coded in test_gemm.py 
 * Users could add more testcases with different parameters in main function to run them
 * Common test functions in test.py could be used as examples to create customised test functions

In test.py, test_basic_randint randomly initializes input matrices with given matrix sizes.
```
def test_basic_randint (self,PE, m, k, n, post_scale):
  int16_max = np.iinfo(np.int16).max
  int16_min = np.iinfo(np.int16).min
  int32_max = np.iinfo(np.int32).max
  int32_min = np.iinfo(np.int32).min      
  mat_A = np.random.randint(low=int16_min, high=int16_max, size=(m, k), dtype=np.int16)
  mat_B = np.random.randint(low=int16_min, high=int16_max, size=(k, n), dtype=np.int16)  
  bias = np.random.randint(low=int32_min, high=int32_max, size=(m, n), dtype=np.int32)      
  self.test_basic(PE,mat_A, mat_B, bias, post_scale)
```
test_basic function takes input matrices, sends matrices and operation instructions to the engine/kernel running on and FPGA card, launches the kernel and then reads the results back.
It also calls multiply_and_cmp function to calculate golden results locally and compares golden results to the results from the FPGA.
```
  gemx.sendMat(mat_A,PE)
  gemx.sendMat(mat_B,PE)
  gemx.sendMat(C_fpga,PE)    
  gemx.sendMat(bias, PE)
  gemx.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], PE) # default test_basic will call addGEMMOp
  gemx.execute(PE)
  gemx.getMat(C_fpga,PE)
  self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
```

## 4. PYTHON API LIST
Files related to Python APIs are mainly in gemx/tests, gemx/src/host and gemx/src/python.
gemx/src/host contains c++ codes to create libgemxhost.so. For more information about the coding details, users could read the source codes under that directory. 

### gemx/src/python

File | Function | Parameters | Description
-----|----------|-----------|-----------
gemx.py | createManager | *libFile*: path to the shared library | load shared library and then specify the required argument and return types for each functions in the shared library
gemx.py | createFCNHandle | *args*: includes path to the given image, board which the given image is built for and number of kernels in the given image <br> *xclbin_opts*: config_info.dat information | create FCN handle
gemx.py | createGEMMHandle | *args*: includes path to the given image, board which the given image is built for and number of kernels in the given image <br> *xclbin_opts*: config_info.dat information | create GEMM handle
gemx.py | createSPMVHandle | *args*: includes path to the given image, board which the given image is built for and number of kernels in the given image <br> *xclbin_opts*: config_info.dat information | create SPMV handle
gemx.py | addFCNOp | *A, B, C, bias*: pointers point to matrices <br> *postScale, postShift, PReLUScale, PReLUAlpha*: <br> *PE*: number of kernels | send FCN operation to kernel 
gemx.py | addGEMMOp | *A, B, C, bias*: pointers point to matrices <br> *postScale, postShift*: <br> *PE*: number of kernels | send GEMM operation to kernel
gemx.py | addSPMVOp | *A, B, C*: pointers point to matrices <br> *nnz*: number of non-zero elements in the sparse matrix <br> *PE*: number of kernels | send SPMV operation to kernel
gemx.py | execute | *PE*: number of kernels | start kernels
gemx.py | wait | *PE*: number of kernels |
gemx.py | sendMat | *A*: pointer points to matrix that sends to kernel <br> *PE*: number of kernels | send matrix to kernel
gemx.py | sendSpMat | *row,col,data*: pointers point to row, col and data array of input sparse matrix <br> *nnz*: number of non-zero elements in the sparse matrix <br> *dtype*: matrix type <br> *PE*: number of kernels | send sparse matrix to kernel
gemx.py | getMat | *A*: pointer points to matrix <br> *PE*: number of kernels | get result back from kernel
gemx.py | printStats |  | print time taken by functions in c++ side
gemx.py | getFreq |  | return frequency of the given image
gemx.py | create_fpga_buf | *shape, np_type* | see gemx/src/python/keras_rt.py for detail usage
gemx.py | load_buf | *np_list* | see gemx/src/python/keras_rt.py for detail usage
gemx.py | parse_cfg | *filename*: path to the configration data file | read configration data filename
gemx.py | default_args | | create a default parser
gemx.py | processCommandLine | | read command line options information to args and config_info.dat information to xclbin_opts

### gemx/tests

File | Class | Function | Parameters | Description
-----|-------|----------|-----------|-----------
test.py | Test | cmp | *A, B*: pointers point to matrices | compare if two arrays are totally the same
test.py | Test | cmpWithinTolerance | *A, B*: pointers point to matrices | compare if two arrays are equal within tolerance
test.py | Test |  multiply_and_cmp | *C, A, B, X*: pointers point to output matrix C, input matrices A, B, X <br> *m, n, post_scale*: matrix size and [postScaleVal, postScaleShift] | calculate GEMM or FCN golden result matrix locally and compare golden result to FPGA result 
test.py | Test | test_basic_randint | *PE*: number of kernels <br> *m, k, n*: matrices size <br> *post_scale*: | randomly initializes input matrices with given matrix sizes
test.py | Test |  test_basic_randint_range | *PE*: number of kernels <br> *A_range, B_range, bias_range*: range of random generated matrices <br> *m, k, n, post_scale*: matrix size and [postScaleVal, postScaleShift] |randomly initializes input matrices with given matrix sizes and given range
test.py | Test |  test_basic_randint_shift | *PE*: number of kernels <br> *A_range, B_range, bias_range*: range of random generated matrices <br> *A_shift, B_shift, bias_shift*: shift matrices <br> *m, k, n, post_scale*: matrix size and post scale |
test.py | Test |  test_rand_basic | *PE*: number of kernels <br> *xclbin_opts*: config_info.dat information <br> *post_scale*: <br> *max_dim*: control max of random matrix size   | ranmdom generate matirx sizes with given range
test.py | Test |  test_basic | *PE*: number of kernels <br> *mat_A, mat_B, bias*: input matrices <br> *post_scale*: | takes input matrices, send matrices and GEMM operations to kernel, execute it and then get the results back
test.py | Test |  test_perf | *timePointKernel*: total time <br> *total_operations*: number of operations that needed <br> *total_parallel_operations*: number of parallel operations that needed <br> *freq*: frequency of the given image <br> *m, k, n*: matrices size | calculate kernel performance 
test.py | Test |  check_input |  *m_size, k_size, n_size*: matrices size <br> *xclbin_opts*: config_info.dat information | check if given matrix sizes are within the range that GEMM and FCN engine could take 
test.py | FcnTest | test_basic | *PE*: number of kernels <br> *mat_A, mat_B, bias*: input matrices <br> *post_scale*: [postScaleVal, postScaleShift] | takes input matrices, send matrices and FCN operations to kernel, execute it and then get the results back
test.py | SpmvTest | multiply_and_cmp_spmv | *row,col,data*: pointers point to row, col and data array of input sparse matrix <br> *m,k*: matrix size <br> *nnz*: number of non-zero elements in the sparse matrix <br> *B, C*: pointers point to matrices | calculate SPMV golden result matrix locally and compare golden result to FPGA result 
test.py | SpmvTest | fillMod | *B*: pointer to matrix <br> *size*: matrix size <br> *Max*: max value | fill matirx

File | Function | Parameters | Description
-----|----------|-----------|-----------
test_fcn.py | test_multiInstrv1 | *int_range*: range of random generated matrices <br> *m, k, n*: matrices size <br> *add_bias*: check add bias or not | send multi and consecutive instructions to test FCN
test_fcn.py | test_perf_fcn | *A_range, B_range, bias_range*: range of random generated matrices <br> *m, k, n*: matrices size <br> *post_scale*: [postScaleVal, postScaleShift] | test FCN and print kernel performance 
test_fcn.py | test_perf_multi_fcn | *ins_count*: number of instructions <br> *m_size, k_size, n_size*: matrices size <br> *A_range, B_range*: range of random generated matrices <br> *post_scale*: [postScaleVal, postScaleShift] | test multi FCN instructions and print kernel performance
test_gemm.py | test_multiInstrv1 | *int_range*: range of random generated matrices <br> *m, k, n*: matrices size <br> *add_bias*: check add bias or not | send multi and consecutive instructions to test GEMM
test_spmv.py | common_spmv | *row,col,data*: pointers point to row, col and data array of input sparse matrix <br> *nnz*: number of non-zero elements in the sparse matrix <br> *vector_range*: range of random generated vector <br> *dtype*: matrix type | take input matrices, send matrices and SPMV operations to kernel, execute it and then get the results back
test_spmv.py | test_spmv_mtxfile | *mtxpath*: path to the mtx file for sparse matrix <br> *vector_range*: range of random generated vector <br> *dtype*: matrix type | create inputs for SPMV by given sparse matirx and random vector
test_spmv.py | test_spmv | *m,k*: matrix size <br> *nnz*: number of non-zero elements in the sparse matrix <br> *vector_range*: range of random generated vector <br> *dtype*: matrix type | create inputs for SPMV by random sparse matirx and vector

## 5. SUPPORT
For more information about SDAccel check the [SDAccel User Guides][]

For questions and to get help on this project or your own projects, visit the [SDAccel Forums][].


## 6. LICENSE AND CONTRIBUTING TO THE REPOSITORY
The sources for this project is licensed under the [3-Clause BSD License][]

To contribute to this project, follow the guidelines in the [Repository Contribution README][]

## 7. ACKNOWLEDGEMENTS
This example is written by developers at
- [Xilinx](http://www.xilinx.com)

## 8. REVISION HISTORY
Date | README Version | Description
-----|----------------|------------
May2018|1.0|Initial Xilinx Release

[3-Clause BSD License]: https://github.com/Xilinx/SDAccel_Examples/blob/master/LICENSE.txt
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[SDAccel User Guides]: http://www.xilinx.com/support/documentation-navigation/development-tools/software-development/sdaccel.html?resultsTablePreSelect=documenttype:SeeAll#documentation
