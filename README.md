General Matrix Operation
======================

This README file contains the following sections:

1. OVERVIEW
2. HOW TO DOWLOAD THE REPOSITORY
3. SOFTWARE TOOLS AND SYSTEM REQUIREMENTS
4. DESIGN FILE HIERARCHY
5. COMPILATION
6. EXECUTION IN CLOUD ENVIRONMENTS
7. SUPPORT
8. LICENSE AND CONTRIBUTING TO THE REPOSITORY
9. ACKNOWLEDGEMENTS
10. REVISION HISTORY


## 1. OVERVIEW
GEMX is a General Matrix Operation library, which is used for accelerating BLAS-like matrix operations on SDAccel supported FPGA cards. This library includes three components: an engine library, a host code compiler and an application or system building environment. The engine library consists of a set of C++ templates with BLAS-like function interfaces for realizing matrix operations on FPGAs. The host code compiler compiles the host code matrix function calls into a sequence of instructions for triggering matrix operations on FPGAs. The building environment utilizes GNU make flow to automate the FPGA and host code image generation process. It also allows users to configure different aspects of the system, e.g. FPGA platform, number of engines implemented in the FPGA image and etc. In summary:
* The accelerator/FPGA image has one or more kernels, each attached to a DDR. Kernels can be heterogeneous though the current simple Makefile limits the kernel content to being identical.
* One or more functional blocks (engines) are compiled into each kernel, controlled by the Makefile's run\<Engine\>=1
* The gemx_gen_bin.exe is a compiler: it generates data file (.bin file) for the matrix operations and matrices' inputs. It can also "dis-assemble" a .bin file (the -read option).
* The gemx_host.exe is a thin layer to load the program onto the FPGA DDR, execute, get the results back, and write to disk.
* The gemx_api_gemm.exe is used to measure the 4 GEMM kernel performance on F1. It also provides an example for using the GEMM accelerator from a C++ application.       
* All functionality is supported across cpu emulation, hw emulation, and running on board, as well as debugging and analyzing in HLS GUI.
* Code base: each engine is a separate .h file, gemx_<engine>.h. Each can be used as a standalone templatized HLS library (most depend on shared gemx_types.h, and HLS ap_*.h). The GEMX building environment far simplifies its compilation and verification.
* [GEMM_API_UG] explains the host code development details for ofloading matrix matrix multiplications to a GEMM FPGA accelerator card in AWS cloud.
* [GEMX_ENGINE_UG] lists the GEMX engine features, configuration parameters and example usages.    

## 2. HOW TO DOWNLOAD THE REPOSITORY
To get a local copy of the GEMX repository, clone this repository to the local system with the following command:
```
git clone https://github.com/Xilinx/gemx.git
```
This command needs to be executed only once. The only required software is a local installation of git.

## 3. SOFTWARE AND SYSTEM REQUIREMENTS
Board | Device Name | Software Version
------|-------------|-----------------
AWS VU9P F1|xilinx:aws-vu9p-f1:4ddr-xpr-2pr|SDAccel 2017.1
Xilinx KU115|xilinx:xil-accel-rd-ku115:4ddr-xpr|SDAccel 2017.2

## 4. DESIGN FILE HIERARCHY
Source code for building FPGA and host code images is located in the gemx/src directory. boost/ directory provides implementation for OpenCL functions used to instantiate an accelerator, trasmit data between the host and the accelerator and etc. Please refer to gemx_api_gemm.cpp to see its usage. gemx/Makefile is used to build FPGA and host images with different configurations. gemx/hls_config.tcl is used to configure the hls compilation options. gemx/run-hls.tcl is used to create vivado_hls project from cpu emulation results. Refer to line 36 in run-hls.tcl to see a usage example. gemx/post_opt.tcl, gemx/pre_route.tcl and gemx/post_route.tcl are used to build a 4-kernel design with each kernel containing one GEMM engine on AWS VU9P F1. gemx/data includs input sparse matrices' data.

## 5. COMPILATION
### Compiling an FPGA image with 4 GEMM kernels for AWS VU9P F1
Before compiling and building FPGA and host images, make sure SDAccel 2017.1 envioronment variales are set up properly and navigate to gemx/ directory
* compiling and building the FPGA image
```
make run_hw SDA_FLOW=hw GEMX_ddrWidth=32 GEMX_gemmMBlocks=8 GEMX_gemmKBlocks=8 GEMX_gemmNBlocks=8 GEMX_numKernels=4 GEMX_runGemv=0 GEMX_runTransp=0 GEMX_runGemm=1 GEMX_part=vu9pf1 GEMX_kernelHlsFreq=250 GEMX_kernelVivadoFreq=300 GEMX_useURAM=1 GEMX_vivadoFlow=EXP
```
where
```
GEMX_ddrWidth: define the number of matrix elements that form one external memory word. The external memory word from DDR is always 64 bytes. By default, the matrix element type is short, hence GEMX_ddrWidth = 64/sizeof(short) = 32.
GEMX_argInstrWide: number of instructions in one 64-byte memory word.
GEMX_numKernels: number of kernels realized on the FPGA.
GEMX_gemmMBlocks, GEMX_gemmKBlocks, GEMX_gemmNBlocks: define the buffer size for matrices A, B and C. For C=A*B, the buffer size for A is GEMX_gemmMBlocks*GEMX_ddrWidth x GEMX_gemmKBlocks*GEMX_ddrWidth; the buffer size for B is GEMX_gemmKBlocks*GEMX_ddrWidth x GEMX_gemmNBlocks*GEMX_ddrWidth; the buffer size for C is GEMX_gemmMBlocks*GEMX_ddrWidth x GEMX_gemmNBlocks*GEMX_ddrWidth
```
at the end of this step, out_hw directory will be created under gemx/ and gemx.xclbin and xbinst/ will be created under gemx/out_hw/.

* compiling host code
```
make GEMX_ddrWidth=32 GEMX_gemmMBlocks=8 GEMX_gemmKBlocks=8 GEMX_gemmNBlocks=8 GEMX_numKernels=4 out_host/gemx_api_gemm.exe
```

## 6. Execution in Cloud Environments
FPGA acceleration boards have been deployed to the cloud. For information on how to execute the example within a specific cloud, take a look at the following guides.
* [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]

First, copy host image gemx_api_gemm.exe, FPGA image gemx.xclbin and the xbinst/ folder to F1, then follow the instruction above to convert gemx.xclbin to gemx.awsxclbin, take the steps below to launch the application to measure the 4 GEMM kernel performance for accelerating 4 512x512 matrix multiplications.
```
gemx_api_gemm.exe gemx.awsxclbin 512 512 512
``` 

## 7. SUPPORT
For more information about SDAccel check the [SDAccel User Guides][]

For questions and to get help on this project or your own projects, visit the [SDAccel Forums][].


## 8. LICENSE AND CONTRIBUTING TO THE REPOSITORY
The source for this project is licensed under the [3-Clause BSD License][]

To contribute to this project, follow the guidelines in the [Repository Contribution README][]

## 9. ACKNOWLEDGEMENTS
This example is written by developers at
- [Xilinx](http://www.xilinx.com)

## 10. REVISION HISTORY
Date | README Version | Description
-----|----------------|------------
Oct2017|1.0|Initial Xilinx Release

[GEMM_API_UG]: https://github.com/Xilinx/gemx/blob/master/GEMM_API_UG.md
[GEMX_ENGINE_UG]: https://github.com/Xilinx/gemx/blob/master/GEMX_ENGINE_UG.md
[3-Clause BSD License]: https://github.com/Xilinx/SDAccel_Examples/blob/master/LICENSE.txt
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[SDAccel User Guides]: http://www.xilinx.com/support/documentation-navigation/development-tools/software-development/sdaccel.html?resultsTablePreSelect=documenttype:SeeAll#documentation
[Nimbix Getting Started Guide]: http://www.xilinx.com/support/documentation/sw_manuals/xilinx2016_2/ug1240-sdaccel-nimbix-getting-started.pdf
[Walkthrough Video]: http://bcove.me/6pp0o482
[Nimbix Application Submission README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/utility/nimbix/README.md
[Repository Contribution README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/CONTRIBUTING.md
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
