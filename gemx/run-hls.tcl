## Copyright (c) 2017
## Xilinx, Inc.
## All rights reserved.
## $Id: //Rodin/Proj/O/OPENCL_APPS_DEV/src/matrix_mult/sgemm/run-hls.tcl#5 $
## 
## No part of this document may be copied, transmitted or
## disclosed in any form or fashion without the express
## written consent of Xilinx, Inc.
##
##  @brief HLS TCL script to compile and synthesize GEMX testbench and kernel
##
##  @author Jindrich Zejda (jzejda@xilinx.com)
##

# Basic usage - run synthesis, no testbench or simulation
#   vivado_hls -f run-hls.tcl  "doCsim 0  doRTLsynth 1"
#   vivado_hls -p prj_hls_ku115_4x4 &

# Testbench files are built by xocc using the Makefile
#   make run_cpu_em

# Advanced usage - run csim in HLS, run HLS debugger etc
#   Buils default project, do not run synthesis
#     vivado_hls -f run-hls.tcl  "doCsim 1  doRTLsynth 0"
#     vivado_hls -p prj_hls_ku115_4x4 &
#   Build 32x32
#     vivado_hls -f run-hls.tcl "doCsim 0  doRTLsynth 1   ddrWidth 32  argInstrWidth 1  gemvVectorBlocks 32  gemmMeshRows 32  gemmMeshCols 32  gemmMeshDepth 32 transpBlocks 1"
#     vivado_hls -p prj_hls_ku115_32x32 &
#   Build 8x8
#     vivado_hls -f run-hls.tcl "doCsim 1  doRTLsynth 0   ddrWidth 8  argInstrWidth 4  gemvVectorBlocks 8  gemmMeshRows 8  gemmMeshCols 8  gemmMeshDepth 8 transpBlocks 1"
#     vivado_hls -p prj_hls_ku115_8x8 &
#   Build 16x16
#     vivado_hls -f run-hls.tcl "doCsim 1  doRTLsynth 0   ddrWidth 16  argInstrWidth 2  gemvVectorBlocks 16  gemmMeshRows 16  gemmMeshCols 16  gemmMeshDepth 16 transpBlocks 1"
#     vivado_hls -p prj_hls_ku115_16x16 &

#  A simple way to build and debug a specific configuration as seen in regressions/*/run.sh
#    make run_cpu_em ... with your options
#    grep GEMX_dataType log-run_cpu_em.txt | grep bin/xocc | sed 's/.*TEST_SDX=1//; s/-D GEMX_/ /g; s/=/ /g; s/ -Wno.*//'
#  and paste the resulting config string:
#    vivado_hls -f run-hls.tcl  "doCsim 1  doRTLsynth 0 ...paste here"
#  Open GUI and run csim in debugger:
#     vivado_hls -p prj_hls_...your_config &

set pwd [pwd]
set pid [pid]

set GCC_VERSION 6.2.0

# 4x4 int16
array set opt {
  dataType        short
  dataEqIntType   short
  ddrWidth        4
  argInstrWidth   8   
  numInstr       16
  numKernels      1
  runGemv         1
  runGemm         1
  runTransp       1
  runSpmv         0
  gemvkVectorBlocks 512
  gemvmVectorBlocks 512
  gemvmGroups      1
  gemmMeshRows     4
  gemmMeshCols     4
  gemmMeshDepth    4
  gemmMBlocks	   1
  gemmKBlocks	   2
  gemmNBlocks	   1
  splitMesh	   0 
  transpBlocks 1
  spmvWidth            1
  spmvkVectorBlocks  512
  spmvMacGroups        4
  spmvColAddIdxBits    0
  spmvPadA             1
  spmvNumCblocks    1024
  spmvFloatPerDesc     4
  argPipeline  2
  useURAM     0
  part        ku115
  doCsim      0
  doRTLsynth  1
  doRTLsim    0
}
# For all other configurations see the above "A simple way to ..."

foreach arg $::argv {
  #puts "ARG $arg"
  foreach o [lsort [array names opt]] {
    if {[regexp "$o +(\\w+)" $arg unused opt($o)]} {
      puts "  Setting CONFIG  $o  [set opt($o)]"
    }
  }
}

puts "Final CONFIG"
set OPT_FLAGS "-std=c++0x "
foreach o [lsort [array names opt]] {
  puts "  Using CONFIG  $o  [set opt($o)]"
  append OPT_FLAGS [format {-D GEMX_%s=%s } $o $opt($o)]
}
#quit

set BOOST_SRC $pwd/../boost/src
set BOOST_LIB $pwd/../boost/lib
set CFLAGS_K "-I $pwd/src  $OPT_FLAGS -D GEMX_kernelId=0 "
set CFLAGS_H "$CFLAGS_K -g -I $BOOST_SRC"


set proj_dir [format prj_hls_%s_%sx%s  $opt(part) $opt(gemmMeshRows) $opt(gemmMeshCols) ]
open_project $proj_dir -reset
set_top gemxKernel_0

add_files src/gemx_kernel.cpp         -cflags "$CFLAGS_K"

add_files -tb src/gemx_main.cpp        -cflags "$CFLAGS_H"

open_solution "solution1"
config_compile -ignore_long_run_time
#config_schedule -effort medium -verbose

if {$opt(part) == "ku115"} {
  set_part {xcku115-flvb2104-2-e} -tool vivado
} else {
  set_part {xcvu9p-flgb2104-2-i} -tool vivado
}

create_clock -period 3.333333 -name default
#config_core DSP48 -latency 2

# Use SDx  make host  to compile the bin files
set run_args "gemx.xclbin $pwd/out_host/app.bin $pwd/$proj_dir/app_out.bin"

if {$opt(doCsim)} {
  puts "***** C SIMULATION *****"
  csim_design -ldflags "-L$BOOST_LIB -lboost_iostreams -lz -lrt -L/usr/lib64 -lstdc++ -Wl,--rpath=${BOOST_LIB} -Wl,--rpath=/usr/lib64" -argv "$run_args"
}

if {$opt(doRTLsynth)} {
  puts "***** C/RTL SYNTHESIS *****"
  csynth_design
  if {$opt(doRTLsim)} {
    puts "***** C/RTL SIMULATION *****"
    cosim_design -trace_level all -ldflags "-L$BOOST_LIB -lboost_program_options -lrt" -argv "$run_args"
  }
}

quit
