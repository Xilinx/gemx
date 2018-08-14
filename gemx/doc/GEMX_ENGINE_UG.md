GEMX ENGINE LIBRARY USER GUIDE
======================

This user guide contains the following sections:

1. OVERVIEW
2. GEMX ENGINES IN DETAILS  
3. BUILDING FPGA IMAGES WITH GEMX ENGINES
4. SUPPORT
5. LICENSE AND CONTRIBUTING TO THE REPOSITORY
6. ACKNOWLEDGEMENTS
7. REVISION HISTORY


## 1. OVERVIEW
The GEMX engine library provides building blocks for constructing matrix operation accelerators on FPGAs. Each engine is implemented as a C++ template class and used to accelerate one type of matrix operations, e.g. matrix matrix multiplication, matrix vector multiplication and etc. To use this engine library to build an FPGA image, a SDAccel supported platform is required. The Makefile provided in gemx/ folder automates the flow for generating the FPGA image with the selected engines and SDAccel platforms.
 
## 2. GEMX ENGINES IN DETAILS
Each GEMX dense matrix engine (GEMM, GEMV, TRANSP) applies block-wise matrix computation algorithm to tile the matrices into small blocks, buffer them locally and process them with fully pipelined logic. The template parameters of each engine class allow users to configure the matrix block size, hence the local buffer size. The top function of each engine has BLAS-like interfaces, which normally include memory pointers for input and output matrices and size and lead dimension information of the matrices and vectors. The architecture of each engine normally includes three components connected by FIFOS. They are *load*, *compute* and *store/write* functions. The load function retrieves matrices and vectores from device memory, in this case DDR, tiles them into small blocks and transmits the block data to the compute function via a FIFO. The compute function retrieves the block data from its FIFO and performs required matrix operations and passes the results to the write function via a FIFO. The store or write function reads the block data from its FIFO, assembles them if necessary and writes the results back to the device memory.  

### 2.1 GEMM ENGINE
The GEMM engine is implemented by class Gemm in file gemx_gemm.h. The top function *runGemm* implements matrix matrix multiplications on an FGPA. The items below list the supported operations, the template parameters, the functions and the features of the Gemm class

* Supported operations
```
  C = ((A * B + X) * alpha ) >> beta_bits
where
  A, B, X and C are dense matrices. In matrix multiplication, the sizes of A, B, X and C are normall referred to as M x K, K x N, M x N and M x N;
```

* Template parameters

Parameter definition | Description | Configuration in Makefile
---------------------|-------------|--------------------------
typename t_FloatType | A, B and C matrix element type. | configured by GEMX_dataType.<br> default value is short.
typename t_FloatEqIntType | the integer type that has the same bit width as the t_FloatType. | configured by GEMX_dataEqIntType. <br> default value is short.
typename t_XdataType | matrix X element type. | configured by GEMX_XdataType.<br> default value is int32_t. 
unsigned int t_DdrWidth | number of matrix A, B or C elements in a memory word retrieved from one DDR access.<br> given DDR memory word is 64 bytes, t_DdrWidth = 64/sizeof(t_FloatType). | configured by GEMX_ddrWidth.<br> default value is 32.
unsigned in t_XDdrWidth | number of matrix X elements in a memory word retrieved from one DDR access.<br> given DDR memory word is 64 bytes, t_XDdrWidth = 64/sizeof(t_XDataType). | configured by GEMX_XddrWidth.<br> default value is 16.
unsigned int t_aColMemWords | number of memory words that form the columns of the local block buffer for matrix A. | configured by GEMX_gemmKBlocks.<br> default value is 1.
unsigned int t_aRowMemWords | number of memory words that form the rows of the local block buffer for matrix A. | configured by GEMX_gemmMBlocks.<br> default value is 2.
unsigned int t_bColMemWord  | number of memory words that form the colums of the local block buffer for matrix B. | configured by GEMX_gemmNBlocks.<br> default value is 1.
unsigned int t_MacBits | number of bits preserved after each MAC step, maximum 48 bits. | configured by GEMX_macBits.<br> default value is 48.<br> This value is effective only when GEMX_keepMacBits=1. By default, GEMX_keepMacBits=0.

The template parameters are configured at compile time and used to define the sizes of local block buffers in the implementation. The matrix A local block buffer size is (t_aRowMemWords * t_DdrWidth) x (t_aColMemWords * t_DdrWidth). The matrix B local block buffer size is (t_aColMemWords * t_DdrWidth) x (t_bColMemWords * t_DdrWidth). The matrix C local block buffer size is (t_aRowMemWords * t_DdrWidth) x (t_bColMemWords * t_DdrWidth). The matrix X local block buffer size is same as matrix C local block buffer size. In the matrix size representation given here, the first number always represents the height or rows in terms of matrix elements, the second number represents the width or columns in terms of matrix elements.

* Functions

Function name | Parameters | Description
--------------|------------|------------
runGemm | *DdrWideType \*p_DdrRd*: memory pointer used to read matrices from the device memory;<br> *DdrWideType \*p_DdrWr*: memory pointer used to write result matrix C back to the device memory;<br> *GemmArgsType &p_Args*: a record that contains the matrices' sizes and lead dimensions' information. | it implements the matrix matrix multiplication on an FPGA.

* Features
  * supported matrix format
    * row-major
    * column-major, requires to run the transposer engine first
	* alpha, beta value
		* alpha is a 16-bit unsigned integer
		* beta  is a 8-bit unsigned integer
  * minimum matrix size
    * A : (t_DdrWidth) x (t_DdrWidth * 2)
    * B : (t_DdrWidth * 2) x (t_DdrWidth)
	* X : (t_DdrWidth) x (t_DdrWidth)
  * maximum matrix size
    * A : 2^31 bytes 
    * B : 2^31 bytes
	* X : 2^31 bytes
  * minimum block buffer size
    * A : (t_DdrWidth) x (t_DdrWidth * 2)
    * B : (t_DdrWidth*2) x (t_DdrWidth)
    * C : (t_DdrWidth) x (t_DdrWidth)
	* X : (t_DdrWidth) x (t_DdrWidth)
  * optimal block buffer size
    * for t_DdrWidth == 32, the configuration with t_aColMemWords=4, t_aRowMemWords=4 and t_bColMemWords=4 provides enough buffering to achieve 99% compute efficiency, meaning almost no overhead for accessing the device memory.
    * the maximum block buffer size is limited by the number of BRAMs you have on an FPGA platform.
  * matrix element type
    * 16-bit integer
    * 8-bit integer
    * fp32

### 2.2 GEMV ENGINE
The GEMV engine is implemented by class Gemv in file gemx_gemv.h. The top function *runGemv* implements matrix vector multiplication on an FPGA. The items below list the supported operations, the template parameters, the functions and the features of the Gemv class.

* Supported operations
```
  C += A * B
where
  A is a matrix, B and C are vectors. 
```

* Template parameters

Parameter definition | Description | Configuration in Makefile
---------------------|-------------|--------------------------
typename t_FloatType | Matrix and vector element type | configured by GEMX_dataType
unsigned int t_DdrWidth | number of matrix elements in one memory word retrieved from DDR | configured by GEMX_ddrWidth
unsigned int t_colMemWords | number of matrix elements that form the columns of the local block buffer for matrix A | configured by GEMX_transpBlocks.<br> default value is 1.
unsigned int t_rowMemWords | number of matrix elements that form the rows of the local block buffer for matrix A | configured by GEMX_gemvmGroups.<br> default value is 1.
unsigned int t_kVectorBlocks | maximum number of blocks allocated for vector B. | configured by GEMX_gemvkVectorBlocks/GEMX_transpBlocks.<br> default value for GEMX_gemvkVectorBlocks is 512. 
unsigned int t_mVectorBlocks | maximum number of blocks allocated for vector C. | configured by GEMX_gemvmVectorBlocks/GEMX_gemvmGroups.<br> default value for GEMX_gemvmVectorBlocks is 512.

The template parameters t_colMemWords and t_rowMemWords together define the local block buffer size for matrix A, which is (t_rowMemWords * t_DdrWidth) x (t_colMemWords * t_DdrWidth). The t_kVectorBlocks defines the maximum size of vector B, which has t_kVectorBlock * t_colMemWords * t_DdrWidth elements. The t_mVectorBlocks defines the maximum size of vector C, which has t_mVectorBlocks * t_rowMemWords * t_DdrWidth elements.

* Functions

Function name | Parameters | Description
--------------|------------|-------------
runGemv |*DdrWideType \*p_DdrRd*: memory pointer used to read matrices from the device memory;<br> *DdrWideType \*p_DdrWr*: memory pointer used to write result vector to the device memory;<br> *GemvArgsType &p_Args*: a record that contains the matrix' and vectors' sizes and lead dimensions information. | it implements the matrix vector multiplication C += A * B

* Features
  * supported matrix format
    * row-major
    * column-major, requires to run the transposer engine first
  * supported type
    * 16-bit integer
    * fp32
    * 8-bit integer
  * minimum matrix and vector size
    * A: t_DdrWidth x t_DdrWidth
    * B: t_DdrWidth
    * C: t_DdrWidth
  * maximum matrix and vector size
    * A: (t_mVectorBlocks * t_rowMemWords * t_DdrWidth) x (t_kVectorBlocks * t_colMemWords * t_DdrWidth)
    * B: t_kVectorBlocks * t_colMemWords * t_DdrWidth
    * C: t_mVectorBlocks * t_rowMemWords * t_DdrWidth
  * legal matrix and vector inputs' sizes
    * A: multiple of (t_rowMemWords * t_DdrWidth) x (t_colMemWords * t_DdrWidth)
    * B: multiple of (t_colMemWords * t_DdrWidth)

### 2.3 TRANSPOSER ENGINE
The TRANSPOSER engine is implemented by class Transp in file gemx_transp.h. The top function *runTransp* implements matrix transposition on an FPGA. The items below list the supported operations, the template parameters, the functions and the features of the Transp class.

* Supported operations

<code>
 B = A<sup>T</sup>
where
 A and B are matrices.
</code>  

* Template parameters

Parameter definition | Description | Configuration in Makefile
---------------------|-------------|-------------------------
typename t_FloatType | matrix element type | configured by GEMX_dataType.<br> default value is short.
unsigned int t_DdrWidth | number of matrix elements in one memory word retrieved from DDR | configured by GEMX_ddrWidth.<br> default value is 4.
unsigned int t_colMemWords | number of matrix elements that form the columns of the local block buffer for matrix A | configured by GEMX_transpBlocks.<br> default value is 1.
unsigned int t_rowMemWords | number of matrix elements that form the rows of the local block buffer for matrix A | configured by GEMX_gemvmGroups.<br> default value is 1.

The template parameters t_colMemWords and t_rowMemWords together define the local block buffer size for matrix A and B. For matrix A, the local block buffer size is (t_rowMemWords * t_DdrWidth) x (t_colMemWords * t_DdrWidth). For matrix B, the local block buffer size is (t_colMemWords * t_DdrWidth) x (t_rowMemWords * t_DdrWidth).

* Functions

Function name | Parameters | Description
--------------|------------|-------------
runTransp | *DdrWideType \*p_DdrRd*: memory pointer used to read matrices from the device memory;<br> *DdrWideType \*p_DdrWr*: memory pointer used to write the result matrix to the device memory;<br> *TranspArgsType &p_Args*: a record that contains the matrices' sizes and lead dimensions information. | it implements the matrix transposition operation.

* Features
  * supported matrix format
    * row-major
    * column-major
  * supported type
    * 16-bit integer
    * 8-bit integer
    * fp32
  * minimum matrix size
    * A: t_DdrWidth x t_DdrWidth
  * maximum matrix size
    * A: 2^31 bytes 
  * legal matrix inputs' size
    * A: multiple of (t_rowMemWords * t_DdrWidth) x (t_colMemWords * t_DdrWidth)

### 2.4 SPMV ENGINE
The SPMV engine has two, BRAM-based and URAM-based, implementations. Here BRAM or URAM refers to the on-chip memory used to store the vectors. BRAM-based implementation can be found in file gemx_spmv.h. URAM-based implementation can be found in file gemx_spmv_coo.h. Definition GEMX_useURAM is used to select one of these implementations at compile time. When GEMX_useURAM=0, BRAM-based implementation is used, otherwise URAM-based implementation is used to build an FPGA image.

* Supported operations
```
  C += A * B
where
  A is sparse matrix, B and C are dense vectors;
```
#### 2.4.1 BRAM-based SPMV implementation (see class Spmv in gemx_spmv.h)
* Storage
	* Sparse matrix format: coordinate formate
	* Sparse matrix representation: each sparse matrix entry is encoded as a three-element tuple <val, row, col>, and stored in DDR as a sequence of those tuples.
	* Sparse matrix index bits: each row or col index in the tuple is encoded with 16-bit unsigned integer.
	* Vector B: vector B is completely loaded from DDR into BRAM
	* Vector C: vector C is block-wise loaded from DDR into BRAM. A sparse matrix description struct SpmvAdesc is created and stored in DDR for storing the corresponding sparse matrix block information.

* Template parameters

Parameter definition | Description | Configuration in Makefile
---------------------|-------------|--------------------------
typename t_FloatType | A, B, C element value type. | configured by GEMX_dataType.<br> default value is short.
typename t_FloatEqIntType | the integer type that has the same bit width as the t_FloatType. | configured by GEMX_dataEqIntType. <br> default value is short.
unsigned int t_DdrWidth | number of B or C elements in a memory word retrieved from one DDR access.<br> given DDR memory word is 64 bytes, t_DdrWidth = 64/sizeof(t_FloatType). | configured by GEMX_ddrWidth.<br> default value is 32.
unsigned int t_SpmvWidth | number of sparse matrix element tuples, <val, row, col>,  retrieved in one DDR access. | configured by GEMX_spmvWidth.<br> default value is 8.
unsigned int t_kVectorBlocks | max length of the B vector. | configured by GEMX_spmvkVectorBlocks.<br> default value is 2048.
unsigned int t_mVectorBlocks | max length of the C vector. | configured by GEMX_spmvmVectorBlocks.<br> GEMX_spmvmVectorBlocks=((1 << (16 - GEMX_spmvColAddIdxBits)) / GEMX_spmvWidth / GEMX_spmvMacGroups / GEMX_ddrWidth) 
unsigned int t_MacGroups | number of accumulation engines to achieve fully pipelined MAC operations. | configured by GEMX_spmvMacGroups.<br> default value is 12.<br> Default value is decided by the latency of accumulation operation.
unsigned int t_ColAddIdxBits | number of bits borrowed from row index representation for encoding col index. | configured by GEMX_spmvColAddIdxBits.<br> default value is 2.
unsigned int t_NumCblocks | maximum number of C blocks. | configured by GEMX_spmvNumCblocks.<br> default value is 1024
unsigned int t_FloatPerDesc | number of t_FloatType elements each SpmvAdesc struct takes | configured by GEMX_spmvFloatPerDesc.<br> default value is 4.

* Functions

Function name | Parameters | Description
--------------|------------|------------
runSpmv | *DdrWideType \*p_DdrRd*: memory pointer used to read A, B and C  from the device memory;<br> *DdrWideType \*p_DdrWr*: memory pointer used to write result vector C back to the device memory;<br> *SpmvArgsType &p_Args*: a record that contains the matrices' sizes, the number of non-zero elements, the number of C blocks and A descriptors' information. | it implements the sparse matrix dense vector multiplication on an FPGA.

* Features
  * supported sparse matrix format
    * coordinate format
  * minimum matrix size
    * row (M) : (t_SpmvWidth) x (t_MacGroups)
    * nnz : t_spmvWidth
  * value type for matrix and vector element
    * 16-bit integer
    * fp32

#### 2.4.2 URAM-based SPMV implementation (see class SpmvCoo in gemx_spmv_coo.h)
* Storage
	* Sparse matrix format: coordinate formate
	* Sparse matrix representation: sparse matrix entries are stored in DDR as a sequece of 64-byte (or t_DdrWidth number of) values  and t_DdrWidth number of <row,col> index pairs for the 64-byte values.
	* Sparse matrix index bits: configurable via template parameters.
	* Vector B: vector B is completely loaded from DDR into URAM.
	* Vector C: vector C is completely loaded from DDR into URAM.

* Template parameters

Parameter definition | Description | Configuration in Makefile
---------------------|-------------|--------------------------
typename t_FloatType | A, B, C element value type. | configured by GEMX_dataType.<br> default value is short.
typename t_IdxType | data type used for storing the col and row indices of each sparse matrix entry. | configured by GEMX_idexType.<br> default value is int32_t.
unsigned int t_DdrWidth | number of B or C elements in a memory word retrieved from one DDR access.<br> given DDR memory word is 64 bytes, t_DdrWidth = 64/sizeof(t_FloatType). | configured by GEMX_ddrWidth.<br> default value is 32.
unsigned int t_NnzWords | number DDR bursts, with each burst retrieve t_DdrWidth matrix values. | configured by GEMX_nnzBlocks.<br> default value is 8.
unsigned int t_kVectorBlocks | max length of the B vector. | configured by GEMX_spmvKmaxBlocks.<br> default value is 32768.
unsigned int t_mVectorBlocks | max length of the C vector. | configured by GEMX_spmvMmaxBlocks.<br> default value is 5462 
unsigned int t_UramGroups | number of MAC groups to achieve fully pipelined MAC operations. | configured by GEMX_spmvUramGroups.<br> default value is 6.<br> t_UramGroups * number of t_FloatType values in each Uram unit = accumulation operation latency.

* Functions

Function name | Parameters | Description
--------------|------------|------------
runSpmv | *DdrWideType \*p_DdrRd*: memory pointer used to read A, B and C  from the device memory;<br> *DdrWideType \*p_DdrWr*: memory pointer used to write result vector C back to the device memory;<br> *SpmvArgsType &p_Args*: a record that contains the matrices' sizes, the number of non-zero elements' information. | it implements the sparse matrix dense vector multiplication on an FPGA.

* Features
  * supported sparse matrix format
    * coordinate format
  * matrix size
    * row (M) : multiple of (t_UramGroups*number_of_t_FloatType_values_per_UramUnit) x (t_DdrWidth / number_of_t_FloatType_values_per_UramUnit)
    * nnz : multiple of t_NnzWords * (t_DdrWidth / number_of_t_FloatType_values_per_UramUnit)
  * value type for matrix and vector element
    * fp32
    * 16/32-bit integer

## 3. BUILDING FPGA IMAGES WITH GEMX ENGINES
The Makefile under the gemx/ directory allows users to configure the template parameters of GEMX engine classes. It also provides the flexibility for building a multi-kernel FPGA image. As shown in the figure below, each kernel has a dedicated DDR bank and contains one or multiple GEMX engines.

![](./GEMX_kernel.png)

Apart from GEMX engines, each kernel also contains an *instruction decoder* to decode the instructions passed from host code to the device memory and use them to trigger different engines, one engine at a time. The configuration variables used in the Makefile are summarized in the table below.

Variable name | Description | Allowed values |Default value
--------------|-------------|----------------|--------------
GEMX_numKernels | number of kernels built in the FPGA image. | 1,2,3,4 | 1
GEMX_runGemm | whether each kernel contains a GEMM engine? | 0: not include; 1: include | 1
GEMX_runGemv | whether each kernel contains a GEMV engine? | 0: not include; 1: include | 1
GEMX_runTransp | whether each kernel contains a TRANSPOSER engine? | 0: not include; 1: include | 1
GEMX_part | which SDAccel platform is used to build the FPGA image.<br> please refer to section 3 in [GEMX_README] for supported platforms and SDAccel versions | ku115,vu9pf1 | ku115
GEMX_kernelHlsFreq | clock frequency in MHz that Vivado HLS is targeting at. | any reasonable frequency | DATA frequency provided by SDAccel platform 
GEMX_kernelVivadoFreq | clock frequency in MHz that Vivado is targeting at. | any reasonable frequency | DATA frequency provided by SDAccel platform
GEMX_useURAM  | use UltraRam to implement some buffers.<br> **only supported for vu9p platform** | 0: use; 1: do not use | 0
GEMX_splitMesh | split the systolic array into 4 smaller ones and connect them via FIFOs to improve achievable clock frequency | 0: not split; 1: split | 0
GEMX_vivadoFlow | configure Vivado to use more timing consuming placement and routing strategies to achieve higher clock frequency for the final FPGA image.<br> **only supported for vu9pf1 platform** | "", "EXP" | ""
GEMX_argInstrWidth | number of instructions in one 64-byte memory word | 64 / (GEMX_ddrWidth * sizeof(GEMX_dataType)) | 1 
GEMX_dataEqIntType | a type compatible with ap_uint<> of same size as t_FloatType | C, C++ fundamental types. | short 
GEMX_dataType<br> GEMX_ddrWidth<br> GEMX_transpBlocks<br> GEMX_gemvmGroups<br> GEMX_gemvkVectorBlocks<br> GEMX_gemvmVectorBlocks<br> GEMX_gemmMBlocks<br> GEMX_gemmKBlocks<br> GEMX_gemmNBlocks<br> | refer to the **GEMX ENGINES IN DETAILS** section for their usage and default values. | |
GEMX_BIN_PROGRAM | a string of instructions that are executed on the kernel.<br> refer to gemx_gen_bin.cpp for the example usage | |

### 3.1 EXAMPLE USAGES

* building a single kernel FPGA image

  * building a kernel that includes one GEMM engine on ku115 platform
  ```

	make run_hw GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=ku115 GEN_BIN_PROGRAM="gemm 512 512 512  512 512 512 512 1 0 A05 B05 C05 X05  gemm 1024 1024 1024  1024 1024 1024 1024 1 0 A1k B1k C1k X1K   gemm 1024 1024 1024  1536 2048 2560 1024 1 0 A1kld B1kld C1kld X1kld"
  where
     * "run_hw" can be changed to "run_cpu_em" or "run_hw_em" to run the design in SDAccel cpu or hw emulation mode. When the design is running at the hw emulation mode, user can use HWEMUGUI=1 to lunch the Vivado GUI and check the waveforms of the signals. 
     * for fast cpu emulation to check the correctness of the design, user can set GEMX_ddrWidth to small numbers, e.g. 4. 
     * the instruction string defined in the GEN_BIN_PROGRAM is compiled and transmitted to the kernel via the device memory. 
     * a Vivado HLS project can also be created from the cpu emulation results. Please refer to the run-hls.tcl to find steps of doing so.
  ``` 
 
  * building a kernel that includes one GEMM engine, one GEMV engine and one TRANSPOSER engine on vu9pf1 platform
 
 ```
 make run_hw SDA_FLOW=hw  GEMX_numKernels=1 GEMX_runGemv=1 GEMX_runGemm=1 GEMX_runTransp=1 GEMX_runSpmv=0 GEMX_splitMesh=1 GEMX_part=vu9pf1
 ```
 
* building multi-kernel FPGA image
  * building a 3 kernels fpga image with each kernel only contains one GEMM engine

  ```
  make run_hw SDA_FLOW=hw GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_numKernels=3 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_part=vcu1525 GEMX_splitMesh=1
  ``` 

### 3.2 Limitations

* The existing Makefile only support building idential kernels, meaning each kerneal has same engines.
* For multi-kenel design, the cpu emulation is not supported.
* For 4 GEMM kernel design, user can run hw_emu, but has to use run_multiGemm_hw_em target.

## 4. SUPPORT
For more information about SDAccel check the [SDAccel User Guides][]

For questions and to get help on this project or your own projects, visit the [SDAccel Forums][].


## 5. LICENSE AND CONTRIBUTING TO THE REPOSITORY
The sources for this project is licensed under the [3-Clause BSD License][]

To contribute to this project, follow the guidelines in the [Repository Contribution README][]

## 6. ACKNOWLEDGEMENTS
This example is written by developers at
- [Xilinx](http://www.xilinx.com)

## 7. REVISION HISTORY
Date | README Version | Description
-----|----------------|------------
Oct2017|1.0|Initial Xilinx Release
May2018|2.0|Added SPMV engine and extended GEMM engine

[3-Clause BSD License]: https://github.com/Xilinx/SDAccel_Examples/blob/master/LICENSE.txt
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[SDAccel User Guides]: http://www.xilinx.com/support/documentation-navigation/development-tools/software-development/sdaccel.html?resultsTablePreSelect=documenttype:SeeAll#documentation
[Nimbix Getting Started Guide]: http://www.xilinx.com/support/documentation/sw_manuals/xilinx2016_2/ug1240-sdaccel-nimbix-getting-started.pdf
[Walkthrough Video]: http://bcove.me/6pp0o482
[Nimbix Application Submission README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/utility/nimbix/README.md
[Repository Contribution README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/CONTRIBUTING.md
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
