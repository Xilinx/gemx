General Matrix Operation
======================

This README file contains the following sections:

1. OVERVIEW
2. SOFTWARE TOOLS AND SYSTEM REQUIREMENTS
3. DESIGN FILE HIERARCHY
4. BUILD GEMX-BASED EXAMPLE APPLICATIONS
5. SUPPORT
6. LICENSE AND CONTRIBUTING TO THE REPOSITORY
7. ACKNOWLEDGEMENTS
8. REVISION HISTORY


## 1. OVERVIEW
GEMX is a General Matrix Operation library, which is used for accelerating BLAS-like matrix operations on SDAccel supported FPGA cards. This library includes three components: an engine library, a host code compiler and an application or system building environment. The engine library consists of a set of C++ templates with BLAS-like function interfaces for realizing matrix operations on FPGAs. The host code compiler compiles the host code matrix function calls into a sequence of instructions for triggering matrix operations on FPGAs. The building environment utilizes GNU make flow to automate the FPGA and host code image generation process. It also allows users to configure different aspects of the system, e.g. FPGA platform, number of engines implemented in the FPGA image and etc. For detailed information about GEMX engine design, please refer to [GEMX_ENGINE_UG]

## 2. SOFTWARE AND SYSTEM REQUIREMENTS
Board | DSA Name | Software Version
------|-------------|-----------------
Xilinx VU9P|xilinx:vcu1525:dynamic:5_1|SDx 2018.2
Xilinx ALVEO|xilinx:u200:xdma:201830_2|SDx 2019.1

## 3. DESIGN FILE HIERARCHY
Source code for building FPGA and host code images is located in the gemx/src directory. gemx/Makefile is used to build FPGA and host images with different configurations. gemx/hls_config.tcl is used to configure the hls compilation options. gemx/run-hls.tcl is used to create vivado_hls project from cpu emulation results. gemx/MLsuite_MLP provides Python bindings for GEMX engines. Those python bindings allow users to offload Python-based Matrix operations to GEMX engines.

## 4. Running GEMX Python APIs on Nimbix Cloud
To run the GEMX Python APIs on Nimbix Cloud, please follow the steps below:
- run
```
git clone https://github.com/Xilinx/gemx
```
to clone the master branch of this repository
- follow the user guide [SDx On Nimbix] to login to your Nimbix account
- launch application "Xilinx SDAccel Development & Alveo FPGA 2018.3" and select "Desktop Mode with FPGA"
- choose machine type "16 core, 128 GB RAM, Xilinx Alveo U200 FPGA (nx5u_xdma_201830_1)"
- copy the gemx/MLsuite_MLP directory to the Nimbix machine, and navigate to the MLsuite_MLP directory
- following [GEMX Python APIs] to setup Python environment on the Nimbix machine and run GEMX Python APIs.

***Important update:***
- .xclbin and config_info.dat files with FP32 type FCN engine has been added to the repository
- the .xclbin and config_info.dat file can be found in gemx/MLsuite_MLP/xclbins/u200_201830_2
- to run them on Nimbix, pleaselaunch application "Xilinx SDAccel Development 2019.1" and select "Desktop Mode with FPGA"
- choose machine type "16 core, 128 GB RAM, Xilinx Alveo U200 FPGA (nx5u_xdma_201830_2)"
- following [GEMX Python APIs] to setup Python environment on the Nimbix machine and run GEMX Python APIs.

## 5.BUILD GEMX-BASED EXAMPLE APPLICATIONS 
A set of make commands are used in the verify.sh to demonstrate the GEMX engine usage with xilinx:u200:xdma:201830_2 DSA. Before compiling and building FPGA and host images, make sure SDAccel 2019.1 envioronment variales are set up properly and navigate to gemx/ directory, and enter command:
  
```
./verify.sh
```
enter one of the build process names (sw_em, hw_em or hw) and one of the four engine names (gemm, spmv or fcn) when the command line prompts for input. File gemx/set_env.sh provides an example about how to set up SDAccel 2019.1 environment variables. 

## 6. SUPPORT
For more information about SDAccel check the [SDAccel User Guides][]

For questions and to get help on this project or your own projects, visit the [SDAccel Forums][].


## 7. LICENSE AND CONTRIBUTING TO THE REPOSITORY
The source for this project is licensed under the [Apache 2.0 license][]

To contribute to this project, follow the guidelines in the [Repository Contribution README][]

## 8. ACKNOWLEDGEMENTS
This example is written by developers at
- [Xilinx](http://www.xilinx.com)

## 9. REVISION HISTORY
Date | README Version | Description
-----|----------------|------------
Oct2017|1.0|Initial Xilinx Release
Mar2018|2.0|Updated to SDx 2017.4
Sep2018|2.1|Updated to SDx 2018.2
May2019|2.2|Updated to SDx 2019.1
Oct2019|2.3|Added .xclbin with FP32 FCN engine

[GEMM_API_UG]: /docs/GEMM_API_UG.md
[GEMX_ENGINE_UG]: /docs/GEMX_ENGINE_UG.md
[Apache 2.0 license]: https://www.apache.org/licenses/LICENSE-2.0
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[SDAccel User Guides]: http://www.xilinx.com/support/documentation-navigation/development-tools/software-development/sdaccel.html?resultsTablePreSelect=documenttype:SeeAll#documentation
[Nimbix Getting Started Guide]: http://www.xilinx.com/support/documentation/sw_manuals/xilinx2016_2/ug1240-sdaccel-nimbix-getting-started.pdf
[Walkthrough Video]: http://bcove.me/6pp0o482
[Nimbix Application Submission README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/utility/nimbix/README.md
[Repository Contribution README]: https://github.com/Xilinx/SDAccel_Examples/blob/master/CONTRIBUTING.md
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
[GEMX Python APIs]: https://xilinx.github.io/gemx/guide/pyguide.html
[SDx On Nimbix]: https://www.xilinx.com/support/documentation/sw_manuals/xilinx2018_3/ug1240-sdaccel-nimbix-getting-started.pdf
