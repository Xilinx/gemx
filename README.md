General Matrix Operation
======================

This README file contains the following sections:

1. OVERVIEW
2. SOFTWARE TOOLS AND SYSTEM REQUIREMENTS
3. DESIGN FILE HIERARCHY
4. BUILD GEMX-BASED EXAMPLE APPLICATIONS
5. BUILD GEMX IMAGE FOR AWS_F1
6. SUPPORT
7. LICENSE AND CONTRIBUTING TO THE REPOSITORY
8. ACKNOWLEDGEMENTS
9. REVISION HISTORY


## 1. OVERVIEW
GEMX is a General Matrix Operation library, which is used for accelerating BLAS-like matrix operations on SDAccel supported FPGA cards. This library includes three components: an engine library, a host code compiler and an application or system building environment. The engine library consists of a set of C++ templates with BLAS-like function interfaces for realizing matrix operations on FPGAs. The host code compiler compiles the host code matrix function calls into a sequence of instructions for triggering matrix operations on FPGAs. The building environment utilizes GNU make flow to automate the FPGA and host code image generation process. It also allows users to configure different aspects of the system, e.g. FPGA platform, number of engines implemented in the FPGA image and etc. For detailed information about GEMX engine design, please refer to [GEMX_ENGINE_UG]

## 2. SOFTWARE AND SYSTEM REQUIREMENTS
Board | DSA Name | Software Version
------|-------------|-----------------
Xilinx KCU1500|xilinx:kcu1500:dynamic:5_0|SDx 2017.4
Xilinx VU9P|xilinx:vcu1525:dynamic:5_0|SDx 2017.4
Xilinx AWS_VU9P_F1|xilinx:aws-vu9p-f1-04261818:dynamic:5_0|SDx 2017.4

## 3. DESIGN FILE HIERARCHY
Source code for building FPGA and host code images is located in the gemx/src directory. boost/ directory provides implementation for OpenCL functions used to instantiate an accelerator, trasmit data between the host and the accelerator and etc. Please refer to gemx_api_gemm.cpp to see its usage. gemx/Makefile is used to build FPGA and host images with different configurations. gemx/hls_config.tcl is used to configure the hls compilation options. gemx/run-hls.tcl is used to create vivado_hls project from cpu emulation results. gemx/data includs input sparse matrices' data.

## 4.BUILD GEMX-BASED EXAMPLE APPLICATIONS 
Following four GEMX-BASED applications are created for xilinx:vcu1525:dynamic:5_0 DSA to demonstrate the GEMX engine usage.
* gemm_perf: dense matrix matrix multiplication performance measurement
* spmv_perf: sparse matrix vector multiplication performance measurement
* gemm_test_python: python-based densen matrix matrix multiplication testing
* gemx_func_test: GEMX engine testing in sofware emulation

Before compiling and building FPGA and host images, make sure SDAccel 2017.4 envioronment variales are set up properly and navigate to gemx/ directory, and enter command:
  
```
./run_app.sh
```
enter one of the four application names when the command line prompts for input. 

## 5. BUILD GEMX IMAGE FOR AWS_F1
To build GEMX image (.xclbin) file for AWS_F1, please follow the steps below.

```
1. git clone https://github.com/aws/aws-fpga.git
2. cd aws_fpga
3. source sdaccel_setup.sh
4. cp -r SDAccel/aws_platform/xilinx_aws-vu9p-f1-04261818_dynamic_5_0 gemx/dsa_f1
5. comment out line 530 ${XBINST_PATH}/xbinst --platform_repo_paths=${PLATFORM_REPO_PATH} --platform ${XDEVICE} -d ${OUT_DIR} in your Makefile
6. run make command

```
## 6. SUPPORT
For more information about SDAccel check the [SDAccel User Guides][]

For questions and to get help on this project or your own projects, visit the [SDAccel Forums][].


## 7. LICENSE AND CONTRIBUTING TO THE REPOSITORY
The source for this project is licensed under the [3-Clause BSD License][]

To contribute to this project, follow the guidelines in the [Repository Contribution README][]

## 8. ACKNOWLEDGEMENTS
This example is written by developers at
- [Xilinx](http://www.xilinx.com)

## 9. REVISION HISTORY
Date | README Version | Description
-----|----------------|------------
Oct2017|1.0|Initial Xilinx Release
Mar2018|2.0|Updated to SDx 2017.4
Sep2018|2.1|Updated to include aws_f1 support

[GEMM_API_UG]: https://github.com/Xilinx/gemx/blob/master/gemx/doc/GEMM_API_UG.md
[GEMX_ENGINE_UG]: https://github.com/Xilinx/gemx/blob/master/gemx/doc/GEMX_ENGINE_UG.md
[3-Clause BSD License]: https://github.com/Xilinx/SDAccel_Examples/blob/master/LICENSE.txt
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[SDAccel User Guides]: http://www.xilinx.com/support/documentation-navigation/development-tools/software-development/sdaccel.html?resultsTablePreSelect=documenttype:SeeAll#documentation
[Nimbix Getting Started Guide]: http://www.xilinx.com/support/documentation/sw_manuals/xilinx2016_2/ug1240-sdaccel-nimbix-getting-started.pdf
[Walkthrough Video]: http://bcove.me/6pp0o482
[Nimbix Application Submission README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/utility/nimbix/README.md
[Repository Contribution README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/CONTRIBUTING.md
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
