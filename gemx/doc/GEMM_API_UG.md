GEMM API User Guide
======================

This user guide contains the following sections:

1. OVERVIEW
2. SOFTWARE TOOLS AND SYSTEM REQUIREMENTS
3. A USAGE EXAMPLE
4. GEMM API LIST
5. SUPPORT
6. LICENSE AND CONTRIBUTING TO THE REPOSITORY
7. ACKNOWLEDGEMENTS
8. REVISION HISTORY


## 1. OVERVIEW
THE GEMM APIs allow users to offload matrix multiplications (also called GEMM operations here) to the Amazon cloud with GEMM FPGA accelerator cards. These APIs are designed to work with the GEMM engine library realized on an FPGA. 

![](./GEMM4_arch.png)
*<center>Fig. 1: GEMM Accelerator Architecture</center>*

As shown in Figure 1, the GEMM enigne library implementation includes 4 identical kernels, with each kernel having its dedicated global memory, in this case, a DDR bank. Inside a kernel, an *instruction decoder* and a *GEMM engine* are built to decode and execute matrix multiplication instructions given by the client or host code software. Each instruction contains the opcode (in this case, GEMM), the DDR addresses of input and output matrices and the size and lead dimension information of the matrices. Once the instruction is decoded by the *instruction decoder*, the matrices' information is passed to the *GEMM engine* to trigger the engine. Once the engine is triggered, it will start to read input matrices from the DDR, carry out block-wise matrix multiplication and finally write the results block-by-block back to the DDR. This instruction decoding and execution step can be repeated again and again. In the current implementation, each kernel can execute a maximum of **16** instructions with the last instruction being a specific *lastOp* instruction to indicate the end of the kernel and to inform the host to read the results back from the DDR. Given this limitation, the host code can offload a maximum of **15** matrix multiplications to one kernel in one shot or without any interruption, meaning no data transfer back to the host memory. This is extremely efficient if the offloaded operation is a chain of matrix multiplications, for example, C1 = A1 * B1; C2 = C1 * B2; C3 = C1 * C2. With 4 kernels, the maximum number of matrix multiplications in one shot can reach **60**. The APIs presented here cover all aspects of using this accelerator library.

### 1.1 FEATURES
* supported operation: C = A * B, where A, B and C are matrices.
* supported matrix element types: int16_t, uint16_t, short, unsigned short
* supported matrix sizes: the number of rows and columns of the input matrices has to be multiple of 256. Currently, the tested maximum matrix size is 16384 x 16384. But it can support a maximum of 2^31 x 2^31 size matrices, or the maximum matrices that can be stored on the device's memory.

### 1.2 LIMITATIONS
* Floating point: currently fp32 type is not supported
* number of kernels: maximum 4 kernels can run in parallel to offload the GEMM operations.
## 2. SOFTWARE AND SYSTEM REQUIREMENTS
* AWS F1 instance that has GEMM engine library card installed
* gcc 6.2.0
* local version of the repository

## 3. A USAGE EXAMPLE
File src/gemx_api_gemm.cpp gives an example of using GEMM APIs to measure the performance of the GEMM engine library.

### 3.1. COMPILING AND RUNNING THE EXAMPLE ON AWS F1
 To compile this example, please follow the steps below:
1. navigate to the gemx/ directory, and change the path of gcc to point to the location of gcc 6.2.0 installed on your local machine, and then run command:

```
make GEMX_ddrWidth=32 GEMX_gemmMBlocks=8 GEMX_gemmKBlocks=8 GEMX_gemmNBlocks=8 GEMX_numKernels=4 out_host/gemx_api_gemm.exe
``` 
2. copy the generated gemx_api_gemm.exe to the AWS F1 instance, and set up the SDAccel environment on F1 by following the steps listed on: 
* [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]
3. launch the application via the following command:
```
gemx_api_gemm.exe gemx.awsxclbin 512 512 512
```
make sure the gemx_api_gemm.exe is under the same directory as the gemx.awsxclbin

If you see following output, that means your run is successful.

```
INFO: Calculating gold values on host ...
  Compared 262144 values:  exact match 262144  within tolerance 0  mismatch 0
INFO: accelerator kernel 0 result MATCHES the reference
INFO: status PASS
  Compared 262144 values:  exact match 262144  within tolerance 0  mismatch 0
INFO: accelerator kernel 1 result MATCHES the reference
INFO: status PASS
  Compared 262144 values:  exact match 262144  within tolerance 0  mismatch 0
INFO: accelerator kernel 2 result MATCHES the reference
INFO: status PASS
  Compared 262144 values:  exact match 262144  within tolerance 0  mismatch 0
INFO: accelerator kernel 3 result MATCHES the reference
INFO: status PASS
```
### 3.2 COMPILING AND RUNNING THE EXAMPLE IN HW_EMU MODE
You can also run the example in the SDx HW_EMU mode with the following command.
```
make run_multiGemm_hw_em SDA_FLOW=hw GEMX_ddrWidth=32 GEMX_gemmMBlocks=8 GEMX_gemmKBlocks=8 GEMX_gemmNBlocks=8 GEMX_numKernels=4 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_part=vu9pf1 GEMX_kernelHlsFreq=250 GEMX_kernelVivadoFreq=300 GEMX_useURAM=1

```

### 3.3. EXAMPLE CODE STRUCTURE
The main() function of gemx_api_gemm.cpp takes the steps below to offload matrix multiplications to the FPGA accelerator.
* compiling user's command line GEMM operations into instructions and allocate host memory for the input and output matrices
* initialize input matrices
* running the 4 GEMM FPGA accelerator
* reading the results returned from the accelerator, checking the correctness and reporting the performance.

The code for compiling GEMM operation commands, allocating host memory is listed below.
```
//define a ProgramType variable for  each GEMM kernel. It's used to control memory allocation and store instructions
ProgramType l_program[GEMX_numKernels];
...
//allocate all pages for matrices
//l_handleA, l_handleB, l_handleC: strings for the matrices' name
//l_newAllocA, l_newAllocB, l_newAllocC: boolean array to store the returen value (success/fail) of allocPages()
//l_M*l_LdA, l_K * l_LdB, l_M*l_LdC: the sizes of matrices
l_pageA[i] = l_program[i].allocPages(l_handleA[i], l_newAllocA[i], l_M * l_LdA);
l_pageB[i] = l_program[i].allocPages(l_handleB[i], l_newAllocB[i], l_K * l_LdB);
l_pageA[i] = l_program[i].allocPages(l_handleC[i], l_newAllocC[i], l_M * l_LdC);

//compile GEMM operations into instructions and store them into the host memory defined in the l_program variable
//l_pageA, l_pageB, l_pageC: store the number of pages allocated for matrix A, B and C
l_GEMMArgs[i].init(
  l_pageA[i], l_pageB[i], l_pageC[i],
  l_M, l_K, l_N,
  l_LdA, l_LdB, l_LdC
 );
 l_kargs[i].setGemmArgs(l_GEMMArgs[i]);
 l_kargs[i].store(l_program[i].addInstr(), 0);

 //retrive the host memory objects from l_program
  for (int i=0; i<GEMX_numKernels; ++i) { 
  	l_memDesc[i] = l_program[i].getMemDesc();
  }
```
The code for initializing input matrices is listed below.
```
// Get addresses where matrices are stored
l_matA[i].init(l_M, l_K, l_LdA, l_program[i].getPageAddr(l_pageA[i]));
l_matB[i].init(l_K, l_N, l_LdB, l_program[i].getPageAddr(l_pageB[i]));
l_matC[i].init(l_M, l_N, l_LdC, l_program[i].getPageAddr(l_pageC[i]));

// Fill inputs with random data
if (l_newAllocA[i]) {
  l_matA[i].fillMod(67, 1);
}
if (l_newAllocB[i]) {
  l_matB[i].fillMod(129, 65);
}

```
The code for running the 4 GEMM FPGA accelerator is listed below.
```
  // Init FPGA
  gemx::Fpga l_fpga;

  for (int i=0; i<GEMX_numKernels; ++i){
	kernelNames[i] = "gemxKernel_" + std::to_string(i);
  }
  if (l_fpga.loadXclbin(l_xclbinFile, kernelNames)) {
    std::cout << "INFO: created kernels" << std::endl;
  } else {
    std::cerr << "ERROR: failed to load " + l_xclbinFile + "\n";
    return EXIT_FAILURE;
  }
  showTimeData("loadXclbin", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

  //create buffers for transferring data to FPGA
  if (!l_fpga.createBuffers(l_memDesc)) {
    std::cerr << "ERROR: failed to create buffers for transffering data to FPGA DDR\n";
    return EXIT_FAILURE;
  }
  showTimeData("created buffers", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
  
  // Transfer data to FPGA
  if (l_fpga.copyToFpga()) {
    (VERBOSE > 0) && std::cout << "INFO: transferred data to FPGA" << std::endl;
  } else {
    std::cerr << "ERROR: failed to copy data to FPGA DDR\n";
    return EXIT_FAILURE;
  }
  showTimeData("copyToFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

  // Gemx kernel ops
  if (l_fpga.callKernels()) {
    (VERBOSE > 0) && std::cout << "INFO: Executed kernel" << std::endl;
  } else {
    std::cerr << "ERROR: failed to call kernels ";
	for (int i=0; i<GEMX_numKernels; ++i) {
		std::cerr << kernelNames[i] << " ";
	}
	std::cerr << "\n";
    return EXIT_FAILURE;
  }
  showTimeData("callKernel", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

  // Transfer data back to host 
  if (l_fpga.copyFromFpga()) {
    (VERBOSE > 0) && std::cout << "INFO: Transferred data from FPGA" << std::endl;
  } else {
    std::cerr << "ERROR: failed to copy data from FPGA DDR\n";
    return EXIT_FAILURE;
  }
```
### PERFORMANCE CALCULATION
At the end of the main() function of the gemx_api_gemm.cpp, the following performance figures are calculated.
1. cycle counts of each kernel's running time, meaning the time from when the kernel starts running until the kernel has done the entire operation.
```
  l_cycleCount[i] = l_instrRes[i].getDuration();
```
2. each kernel's running time in ms, l_timeKernelInMs.
```
The equation for calculating l_timeKernelInMs is:
  l_timeKernelInMs = l_cycleCount / kernel_frequence
where
  kernel_frequence is achieved kernel frequency, in this case, 246.6MHz.
```
3. each kernel's performance in Tops, l_perfKernelInTops.
```
  The equation for calculating l_perfKernelInTops is:
    l_perfKernelInTops = l_Ops / l_timeKernelInMs
  where
    Given the input matrices size is NxN, l_Ops = N * N * N *2
```
4. each kernel's best theoretical running time in ms, l_timeMsAt100pcEff.
```
  Because in each GEMM kernel, a systolic array is implemented to carry out one multiplication and acuumulation in one cycle, the equation for calculating l_timeMsAt100pcEff is:
    l_timeMsAt100pcEff = (l_Ops/(systolic_array_size *2))/kernel_frequency
  where
    systolic_array_size = GEMX_ddrWidth * GEMX_ddrWidth
``` 
5. kernel's efficiency from the pure kernel running time point of view, l_effKernelPct.
```
  The equation for calculating l_effKernelPct is:
    l_effKernelPct = l_timeMsAt100pcEff / l_timeKernelInMs
```
6. accelerator's efficiency from the api call point of view, l_effApiPct.
```
  The equation for calculating l_effApiPct is:
    l_effApiPct = l_timeMsAt100pcEff / l_timeApiInMs
  where
    l_timeApiInMs includes the time to transfer data between host memory and device memory.
```

## 4. GEMM API LIST
File | Class | Function | Parameters | Description
-----|-------|----------|-----------|-----------
gemx_gen_bin.h | Program | allocPages | *std::string p_Handle*: a string used to describe the allocated pages;<br> *bool &p_NewAlloc*: a return boolean variable indicating if the allocation is successful or not;<br> *size_t p_NumElements*: number of elements with type defined by GEMX_dataType | allocate pages for the given number of elements, return the index of the first allocated page.
gemx_gen_bin.h | Program | getPageAddr | *unsigned int p_PageIdx*: the index of the allocated page | return the pointer to the allocated paged indexed by the p_PageIdx.
gemx_gen_bin.h | Program | addInstr | | increase the number of instructions stored by m_NumInstr in class Program and return the pointer of the last added instruction.
gemx_gen_bin.h | Program | getMemDesc | | retrieve the MemDesc object stored in the Program instance. The MemDesc object contains a pointer to the allocated memory and the size in terms of number of elements with type GEMX_dataType.
gemx_gen_bin.h | Program | getBaseResAddr | | return the address of the memory that stores the returned the results from the accelerator.
gemx_gen_bin.h | Mat | init | *unsigned int p_Rows*: number of rows;<br> *unsigned int p_Cols*: number of columns;<br> *unsigned int p_Ld*: matrix lead dimension; <br> *T \*p_Addr*: address of allocated memory for storing the matrix | initialize a Mat instance with the given argumentss.
gemx_gen_bin.h | Mat | fillMode | *T p_Max*: the maximum value of the matrix elements;<br> *T p_First*: the starting value of the matrix elements | fill the memory used to store the matrix with random values that are equal to or greater than p_First, and smaller than p_Max.
gemx_kargs | GemmArgs | init | *unsigned int p_Aoffset*: memory offset for input matrix A;<br> *unsigned int p_Boffset*: memory offset for input matrix B;<br> *unsigned int p_Coffset*: memory offset for input matrix C;<br> *unsigned int p_M*: number of rows in matrix A;<br> *unsigned int p_K*: number of columns in matrix A;<br> *unsigned int p_N*: number of columns in matrix B;<br> *unsigned int p_Lda*: lead dimenstion of matrix A;<br> *unsigned int p_Ldb*: lead dimension of matrix B;<br> *unsigned int p_Ldc*: lead dimension of matrix C;| instantiate a GemmArgs instance with the given arguments.
gemx_kargs.h | Kargs | setGemmArgs | *GemmArgs p_args*: GemmArgs object that stores all required arguments for the GEMM operation. | set the arguments for GEMM operation.
gemx_kargs.h | Kargs | store | *DdrFloatType \*p_Addr*: base address of the allocated memory for storing instructions;<br> *unsigned int p_Pc*: memory offset for storing the current instruction.
gemx_kargs.h | Kargs | load | *DdrFloatType \*p_Addr*: base address of the memory for storing instructions;<br> *unsigned int p_Pc*: memory offset of the current instruction | retrieve the instruction from the given memory address and offset and return the decoded operation.
gemx_kargs.h | Kargs | getInstrResArgs | | retrieve the instruction exeuction results returned by the accelerator. The results include the kernel staring and finishing time in clock cycles.
gemx_fpga.h | Fpga | loadXclbin | *std::string p_XclbinFile*: xclbin file name;<br> *std::string p_KernelName[GEMX_numKernels]*: array of kernel names | instantiate accelerator with the .xclbin file and kernel names.
gemx_fpga.h | Fpga | createBuffers | *MemDesc p_MemDesc[GEMX_numKernels]*: array of MemDesc objects stored in each kernels' Program instances | create global buffers for the kernels.
gemx_fpga.h | Fpga | copyToFpga | | copy data from host memory to the global memories of the kernels.
gemx_fpga.h | Fpga | callKernels | | start kernels in parallel.
gemx_fpga.h | Fpga | copyFromFpga | | copy the results from kernels' global memory back to host memory.

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
