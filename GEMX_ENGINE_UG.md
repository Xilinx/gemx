GEMX ENGINE LIBRARY USER GUIDE
======================

This user guide contains the following sections:

1. OVERVIEW
2. GEMX ENGINES IN DETAILS  
3. BUILDING FPGA IMAGES WITH GEMX ENGINES
4. USING GEMX ENGINES IN SOFTWARE
5. SUPPORT
6. LICENSE AND CONTRIBUTING TO THE REPOSITORY
7. ACKNOWLEDGEMENTS
8. REVISION HISTORY


## 1. OVERVIEW
The GEMX engine library provides building blocks for constructing matrix operation accelerators on FPGAs. Each engine is implemented as a C++ template class and used to accelerate one type of matrix operations, e.g. matrix matrix multiplication, matrix vector multiplication and etc. To use this engine library to build an FPGA image, a SDAccel supported platform is required. The Makefile provided in gemx/ folder automates the flow for generating the FPGA image with the selected engines and SDAccel platforms.
 
## 2. GEMX ENGINES IN DETAILS
Each GEMX engine applies block-wise strategy to tile the matrices into small blocks, buffer them locally and process them with fully pipelined logic. The template parameters of each engine class allow users to configure the matrix block size, hence the local buffer size. The top function of each engine has BLAS-like interfaces, which normally include memory pointers for input and output matrices and size and lead dimension information of the matrices and vectors. The architecture of each engine includes three components connected by FIFOS. They are *load*, *compute* and *store/write* functions. The load function retrieve matrices and vectores from device memory, in this case DDR, tiles them into small blocks and transmit the block data to the compute function via a FIFO. The compute function retrieves the block data from its FIFO and performs required matrix operations and passes the results to the write function via a FIFO. The store or write function reads the block data from its FIFO, assembles them if necessary and writes the results back to the device memory.  

### 2.1 GEMM ENGINE
The GEMM engine is implemented by class Gemm in file gemx_gemm.h. The top function *runGemm* offloads matrix matrix multiplications on an FGPA. The items below list the supported operations, the template parameters, the functions and the features of the Gemm class

* Supported operations
```
  C = A * B
where
  A, B, C are matrices. In matrix multiplication, the sizes of A, B and C are normall referred to as M x K, K x N, and M x N;
```

* Template parameters

Parameter definition | Description | Configuration in Makefile
---------------------|-------------|--------------------------
typedef t_FloatType | matrix element type.<br> supported types: uint16, int16, short, unsigned short.<br> default type is short. | configured by GEMX_dataType
unsigned int t_DdrWidth | number of matrix elements in a memory word retrieved from DDR.<br> given DDR memory word is 64 bytes, t_DdrWidth = 64/sizeof(t_FloatType).<br> default value is 4. | configured by GEMX_ddrWidth
unsigned int t_aColMemWords | number of memory words that form the columns of the locl block buffer for matrix A.<br> default value is 1. | configured by GEMX_gemmMBlocks
unsigned int t_aRowMemWords | number of memory words that form the rows of the local block buffer for matrix A.<br> default value is 2. | configured by GEMX_gemmKBlocks
unsigned int t_bColMemWord  | number of memory words that form the colums of the local block buffer for matrix B.<br> default value is 1. | configured by GEMX_gemmNBlocks
The template parameters are configured at compile time and define the sizes of local block buffers used in the implementation. The matrix A local block buffer size is (t_DdrWidth * t_aRowMemWords) x (t_DdrWidth * t_aColMemWords). The matrix B local block buffer size is (t_DdrWidth * t_aColMemWords) x (t_DdrWidth * t_bColMemWords). The matrix C local block buffer size is (t_DdrWidth * t_aRowMemWords) x (t_DdrWidth * t_bColMemWords). In the matrix size representation given here, the first number always represents the height or rows in terms of matrix elements, the second number represents the width or columns in terms of matrix elements.

* Functions

Function name | Parameters | Description
--------------|------------|------------
runGemm | *DdrWideType \*p_DdrRd*: memory pointer used to read matrices from the device memory;<br> *DdrWideType \*p_DdrWr*: memory pointer used to write result matrix C back to the device memory;<br> *GemmArgsType &p_Args*: a record that contains the matrices' sizes and lead dimensions information. | it implements the matrix matrix multiplication on an FPGA.

* Features
  * supported matrix format
    * row-major
    * column-major, requires to run the transposer engine first
  * minimum matrix size
    * A : (t_DdrWidth) x (t_DdrWidth * 2)
    * B : (t_DdrWidth * 2) x (t_DdrWidth)
  * maximum matrix size
    * A : 2^31, or as big as the device memory can store
    * B : 2^31, or as big as the device memory can store
  * minimum block buffer size
    * A : (t_DdrWidth) x (t_DdrWidth * 2)
    * B : (t_DdrWidth*2) x (t_DdrWidth)
    * C : (t_DdrWidth) x (t_DdrWidth)
  * maximum block buffer size
    * for t_DdrWidth == 32, the configuration with t_aColMemWords=8, t_aRowMemWords=8 and t_bColMemWords=8 provides enough buffering to achieve 99% compute efficiency, meaning no overhead caused by accessing the device memory.
    * the maximul block buffer size is limited by the number of BRAMs you have on an FPGA platform.
  * matrix element type
    * only 16-bit integer type is supported
    

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
Oct2017|1.0|Initial Xilinx Release

[3-Clause BSD License]: https://github.com/Xilinx/SDAccel_Examples/blob/master/LICENSE.txt
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[SDAccel User Guides]: http://www.xilinx.com/support/documentation-navigation/development-tools/software-development/sdaccel.html?resultsTablePreSelect=documenttype:SeeAll#documentation
[Nimbix Getting Started Guide]: http://www.xilinx.com/support/documentation/sw_manuals/xilinx2016_2/ug1240-sdaccel-nimbix-getting-started.pdf
[Walkthrough Video]: http://bcove.me/6pp0o482
[Nimbix Application Submission README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/utility/nimbix/README.md
[Repository Contribution README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/CONTRIBUTING.md
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
