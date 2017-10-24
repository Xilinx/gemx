# Examples:
#   make -j 2 run_cpu_em
#   make -j 2 run_hw_em
#   make -j 2 run_hw

# Build 2-core kernels for both cpu_em, hw em, 5 cpus (1 cpu for teh host build)
#   make -j 5  run_cpu_em  run_hw_em  GEMX_numKernels=2
# Build and run all cpu , hw em, hw on 4-core design
#    make -j 16 GEMX_numKernels=4

# Build

# Sample HW runs for widths 4, 8, 16, 32
ifeq (0, 1) 
  set s = 4;   \
  make clean;  \
  make run_hw  \
  GEMX_ddrWidth=$s  \
  GEMX_argInstrWidth=`expr 32 / $s`  \
  GEMX_gemmMeshRows=$s  \
  GEMX_gemmMeshCols=$s  \
  GEMX_gemmMeshDepth=$s \
  GEMX_transpBlocks=1 \
  GEMX_numKernels=1 >& log &
endif


GCC_VERSION=6.2.0

HWEMUGUI = 0

BOOST_SRC=${PWD}/../boost/src
BOOST_LIB=${PWD}/../boost/lib
export BOOST_COMPUTE_DEFAULT_VENDOR=Xilinx


KERNEL_NAME = gemxKernel

##############################
# 4x4  16b
GEMX_dataType      = short
GEMX_dataEqIntType = short
GEMX_ddrWidth      =   4
GEMX_argInstrWidth =   8
GEMX_numInstr      =  16

GEMX_gemvkVectorBlocks  = 512
GEMX_gemvmVectorBlocks  = 512
GEMX_gemvmGroups   =  1

GEMX_gemmMeshRows  =  4
GEMX_gemmMeshCols  =  4
GEMX_gemmMeshDepth =  4
GEMX_gemmMBlocks   = 1 
GEMX_gemmKBlocks   = 2
GEMX_gemmNBlocks   = 1

GEMX_transpBlocks  =  1

GEMX_spmvWidth   =  1
GEMX_spmvkVectorBlocks  = 2048
GEMX_spmvMacGroups      =   4
GEMX_spmvPadA       = 1
GEMX_spmvNumCblocks = 1024
GEMX_spmvFloatPerDesc = 4

# Correlated for IdxBits 2 => row idx < 2**14 so blocks 10 (2**14 / ddrw / spmvw / groups
GEMX_spmvColAddIdxBits  =    2
#GEMX_spmvmVectorBlocks  =  256


GEMX_argPipeline   =  2
GEMX_part          = ku115
GEMX_kernelHlsFreq	   = 250 
GEMX_kernelVivadoFreq  = 250
# How many kernels to replicate in the accelerator (use 1 to 4)
GEMX_numKernels    = 1
GEMX_useURAM	   = 0
# What engines get included in each accelerator kernel (use 0 or 1)
# The more engines you include the more catability you get but P&R
# P&R becomes more difficult thus you get lower Fmax
GEMX_runGemv = 1
GEMX_runGemm = 1
GEMX_runTransp = 1
GEMX_runSpmv = 0

# Explorer mode off - Placer failed in 2017.3_sdx on 2 sample designs
#   ERROR: [Common 17-49] Internal Data Exception: Error on line 13 of file placementStrategies.cfg: syntax error
#GEMX_vivadoFlow = EXP

##############################

ifeq (${GEMX_ddrWidth}, 4)
  GEMX_gemvVectorBlocks  = 512
endif

# Defauts for SPMV 32-wide
ifeq (${GEMX_ddrWidth}, 32)
  GEMX_spmvWidth   =  8
endif

ifeq (${GEMX_dataType}, float)
  GEMX_dataEqIntType = int
  
  GEMX_gemvmGroups = 16
  GEMX_gemvmVectorBlocks = 43
  
  GEMX_spmvPadA = 0
  GEMX_spmvFloatPerDesc = 2
  GEMX_spmvMacGroups    =   12
  ifeq (${GEMX_ddrWidth}, 16)
    GEMX_spmvWidth   =  8
    #GEMX_spmvmVectorBlocks  =  10
  endif
  
endif

ifeq (${GEMX_part},ku115)
  DSA=4_0
  XDEVICE=xilinx:xil-accel-rd-ku115:4ddr-xpr:$(subst _,.,${DSA})
  DSA_PLATFORM=xilinx_xil-accel-rd-ku115_4ddr-xpr_${DSA}
  XDEVICE_REPO_PATH=$(XILINX_SDX)/platforms/${DSA_PLATFORM}/hw
else ifeq (${GEMX_part},vu9p)
  # When you change DSA version here you also have to edit LSF
  # selection strings in regressions/gemx_L*vu9p/testinfo.yml
  DSA=5_0
  XDEVICE=xilinx:xil-accel-rd-vu9p:4ddr-xpr:$(subst _,.,${DSA})
  DSA_PLATFORM=xilinx_xil-accel-rd-vu9p_4ddr-xpr_${DSA}
  #XDEVICE_REPO_PATH=$(XILINX_SDX)/../../../../internal_platforms/${DSA_PLATFORM}/hw
  PLATFORM_REPO_PATHS=$(XILINX_SDX)/../../../../internal_platforms
else ifeq (${GEMX_part},vu9pf1)
  DSA=4_0
  XDEVICE=xilinx:aws-vu9p-f1:4ddr-xpr-2pr:$(subst _,.,${DSA})
  DSA_PLATFORM=xilinx_aws-vu9p-f1_4ddr-xpr-2pr_${DSA}
  XDEVICE_REPO_PATH=$(XILINX_SDX)/platforms/${DSA_PLATFORM}/hw
	#XDEVICE_REPO_PATH=$(XILINX_SDX)/../../../../internal_platforms/${DSA_PLATFORM}/hw
else
  $(error Unknown GEMX_part ${GEMX_part})
endif

ifeq ("$(wildcard $(XDEVICE_REPO_PATH))","")
  ifeq ("$(wildcard $(PLATFORM_REPO_PATHS))","")
    $(error Missing DSA or platform repo)
  endif
endif
ifeq ("$(XDEVICE))","")
  $(error XDEVICE)
endif


KERNEL_CU_OPTS = --nk gemxKernel_0:1:gemxKernel_0
ifeq ($(shell test $(GEMX_numKernels) -gt 1; echo $$?),0)
  KERNEL_CU_OPTS += --nk gemxKernel_1:1:gemxKernel_1
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 2; echo $$?),0)
  KERNEL_CU_OPTS += --nk gemxKernel_2:1:gemxKernel_2
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 3; echo $$?),0)
  KERNEL_CU_OPTS += --nk gemxKernel_3:1:gemxKernel_3
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 4; echo $$?),0)
  $(error Unknown GEMX_numKernels=${GEMX_numKernels})
endif

GMEM0_M  =0

##########  Nightly regression board run testcases ##########
# The GEN_BIN_PROGRAM is just a default "algebra computation graph" that
# is compiled on host (by  gemx_gen_bin.exe -write app.bin $GEN_BIN_PROGRAM).
# It does NOT affect what is compiled into the accelerator kernel.
# This default covers kernel width of 4 (for fast debug) and 32 (full
# DDR width) since each engine typically has minimum size or alignment
# requirements.

############ short
ifeq (${GEMX_dataType}, short)
  
  ########  short w=4
  ifeq (${GEMX_ddrWidth}, 4)
    ifeq (${GEMX_runTransp}, 1)
      GEN_BIN_PROGRAM += \
        transp   4   4   8  12 rm cm  T0 T1  \
        transp  64  96 128 144 rm cm  T2 T3
    endif
    ifeq (${GEMX_runGemv}, 1)
      GEN_BIN_PROGRAM += \
        gemv   4   4   8    Av1 Bv1 Cv1  \
        gemv   8  12   16   Av2 Bv2 Cv2  \
        gemv  64  96   128  Av3 Bv3 Cv3
    endif
    ifeq (${GEMX_runGemm}, 1)
      GEN_BIN_PROGRAM += \
        gemm  32 64 32   64 32 32  Am1 Bm1 Cm1  \
        gemm  32 64 32   64 64 64    Am2 Bm2 Cm2  
    endif

  ########  short w=32
  else
    # Setup for full size DDR width 32 (kernels of width 4, 8, 16 can run
    #   these sizes too it is just oo slow to debug)  
    ifeq (${GEMX_runTransp}, 1)
      GEN_BIN_PROGRAM += \
        transp   32  32   64   96 rm cm  T0 T1  \
        transp  512 768 1024 1152 rm cm  T2 T3
    endif
    ifeq (${GEMX_runGemv}, 1)
      GEN_BIN_PROGRAM += \
        gemv   32  32    64  Av1 Bv1 Cv1  \
        gemv   64  96   128  Av3 Bv3 Cv3  \
        gemv  512 768  1024  Av5 Bv5 Cv5
    endif
    ifeq (${GEMX_runGemm}, 1)
      GEN_BIN_PROGRAM += \
        gemm  512  512  512   512  512  512   Am1 Bm1 Cm1  \
        gemm  512  512  512  1024 1024 1024   Am2 Bm2 Cm2  \
        gemm 1024 1024 1024  1024 1024 1024   Am3 Bm3 Cm3
    endif
  endif

############ float
else ifeq (${GEMX_dataType}, float)
  # Setup for test width 16 (10 hangs in HW emu, likely bad logic; 8 had II=11)
  ifeq (${GEMX_runTransp}, 1)
    GEN_BIN_PROGRAM += \
      transp   32  32   64  96 rm cm  T0 T1  \
      transp 512 768 1024 1152 rm cm  T2 T3
  endif
  ifeq (${GEMX_runGemv}, 1)
    GEN_BIN_PROGRAM += \
      gemv  256 256  272 A1 B1 C1  \
      gemv  512 768 1024 A5 B5 C5
  endif
  ifeq (${GEMX_runSpmv}, 1)
    GEN_BIN_PROGRAM += \
      spmv 96 128 256 none A0 B0 C0  spmv  \
      spmv 0 0 0 data/spmv/diag16k.mtx.gz A7 B7 C7
  endif
endif


######### Custom development testcases #########

# Lisa fast transpose HW EMU benchmark
#GEN_BIN_PROGRAM =  transp   1024   1024   1024  1024 rm cm T0 T1

# GEMV performance evaluation
ifdef GEMV_PERF
  GEN_BIN_PROGRAM = \
    gemv  1024  1024   1024 A2 B2 C2 \
    gemv  2048  2048   2048 A4 B4 C4 \
    gemv  4096  4096   4096 A6 B6 C6 \
    gemv  8192  8192   8192 A8 B8 C8 \
    gemv 16384 16384   16384 A10 B10 C10
endif
ifdef GEMV_PERF1
  GEN_BIN_PROGRAM = \
   	gemv  8192  8192   8192 A8 B8 C8
endif

########################################################################## 

ifndef SDA_FLOW
  SDA_FLOW = cpu_emu
endif

HOST_SRCS = src/gemx_main.cpp

KERNEL_SRCS = src/gemx_kernel.cpp


GMEM_FLAGS = -D GMEM_M=$(GMEM0_M) 

CFLAGS_K =  $(GMEM_FLAGS) -I ./src \
           -D TEST_SDX=1 \
           -D GEMX_dataType=$(GEMX_dataType) \
           -D GEMX_dataEqIntType=$(GEMX_dataEqIntType) \
           -D GEMX_ddrWidth=$(GEMX_ddrWidth) \
           -D GEMX_argInstrWidth=$(GEMX_argInstrWidth) \
           -D GEMX_numInstr=$(GEMX_numInstr) \
           -D GEMX_gemvkVectorBlocks=$(GEMX_gemvkVectorBlocks) \
           -D GEMX_gemvmVectorBlocks=$(GEMX_gemvmVectorBlocks) \
           -D GEMX_gemvmGroups=$(GEMX_gemvmGroups) \
           -D GEMX_gemmMeshRows=$(GEMX_gemmMeshRows) \
           -D GEMX_gemmMeshCols=$(GEMX_gemmMeshCols) \
           -D GEMX_gemmMeshDepth=$(GEMX_gemmMeshDepth) \
		   -D GEMX_gemmMBlocks=${GEMX_gemmMBlocks} \
		   -D GEMX_gemmKBlocks=${GEMX_gemmKBlocks} \
		   -D GEMX_gemmNBlocks=${GEMX_gemmNBlocks} \
           -D GEMX_transpBlocks=$(GEMX_transpBlocks) \
           -D GEMX_spmvWidth=$(GEMX_spmvWidth) \
           -D GEMX_spmvkVectorBlocks=$(GEMX_spmvkVectorBlocks) \
           -D GEMX_spmvMacGroups=$(GEMX_spmvMacGroups) \
           -D GEMX_spmvColAddIdxBits=$(GEMX_spmvColAddIdxBits) \
           -D GEMX_spmvPadA=$(GEMX_spmvPadA) \
           -D GEMX_spmvNumCblocks=$(GEMX_spmvNumCblocks) \
           -D GEMX_spmvFloatPerDesc=$(GEMX_spmvFloatPerDesc) \
           -D GEMX_argPipeline=$(GEMX_argPipeline) \
           -D GEMX_part=$(GEMX_part) \
		   -D GEMX_useURAM=${GEMX_useURAM} \
	   	   -D GEMX_runGemv=$(GEMX_runGemv) \
	   	   -D GEMX_runGemm=$(GEMX_runGemm) \
	       -D GEMX_runTransp=$(GEMX_runTransp) \
	   	   -D GEMX_runSpmv=$(GEMX_runSpmv) \
		   -D GEMX_numKernels=${GEMX_numKernels} \
           -Wno-ignored-attributes \
           

KERNEL_DEFS += $(CFLAGS_K)


KEEP_TEMP=1
KERNEL_DEBUG=1

XP_VIVADO_PARAMS =--xp vivado_param:project.writeIntermediateCheckpoints=1
XP_VIVADO_PARAMS+=--xp vivado_param:place.hardVerbose=469538
XP_VIVADO_PARAMS+=--xp vivado_param:place.oldMsgVerbose=1
XP_VIVADO_PARAMS+=--xp vivado_param:route.flowDbg=1
XP_VIVADO_PARAMS+=--xp vivado_param:route.timingDbg=1

XP_VIVADO_PARAMS+=--xp param:compiler.fanoutLimit=0
#XP_VIVADO_PARAMS+=--xp param:compiler.fanoutLimit=16000

# This defines placer target slack for high-priority clocks
#  - when slack is highly negative, placer works in steps with milder target
#  - forcing 0 here will make placer more aggressive
#  - It likely has only minimal impact
#XP_VIVADO_PARAMS+=--xp vivado_param:place.highPriorityClkTargetSlack=0

# For DSA instances with slack worse than this placer locks their placement (after adding OCL cells)
#  - The 0.0 usually means noop - if DSA met timing.
#  - The higher the value the morr chance to close timing on sys clocks as well as less
#    area/optimization for the OCL block.
#XP_VIVADO_PARAMS+=--xp param:compiler.lockFlowCritSlackThreshold=0.0 

#CLCC_OPT: CLCC options for both compile and link
#CLCC_COMP_OPT: CLCC options only for compile mode
#CLCC_LINK_OPT: CLCC options only for link mode
ifeq (${SDA_FLOW},cpu_emu)
    CLCC_OPT += -t sw_emu
else ifeq (${SDA_FLOW},hw_emu)
    CLCC_OPT += -t hw_emu
    ifeq ($(HWEMUGUI),1)
        CLCC_OPT += --xp param:hw_em.debugLevel=GUI
		else
				CLCC_OPT += --xp param:compiler.preserveHlsOutput=1 --report system
    endif
else ifeq (${SDA_FLOW},hw)
    CLCC_OPT += -t hw
endif


ifndef USE_SDX_1604
  CLCC_LINK_OPT += --temp_dir ${OUT_DIR}
endif

CLCC_LINK_OPT += --kernel_frequency ${GEMX_kernelVivadoFreq}
# CR 974833
CLCC_OPT += --xp prop:solution.hls_pre_tcl=hls_config.tcl
#-D_GLIBCXX_USE_CXX11_ABI=0
HOST_CFLAGS = -g -O2 -std=c++11 \
              -I $(BOOST_SRC) \
               -DCL_USE_DEPRECATED_OPENCL_1_1_APIS \
               -DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION \
	      -DBOOST_COMPUTE_HAVE_THREAD_LOCAL \
	      -DBOOST_COMPUTE_THREAD_SAFE \
              -D FLOW_HLS_CSIM $(CFLAGS_K) \
              -I$(XILINX_SDX)/Vivado_HLS/include \
              -I$(XILINX_VIVADO)/include \
              -I${XILINX_SDX}/runtime/include/1_2 


HOST_EXE_DIR=.
HOST_LFLAGS = \
              -L$(BOOST_LIB) \
              -lboost_iostreams -lz \
              -L/tools/batonroot/rodin/devkits/lnx64/gcc-${GCC_VERSION}/lib64 \
              -lstdc++ \
              -L${XILINX_SDX}/runtime/lib/x86_64 -lxilinxopencl -lrt \
              -lOpenCL -pthread \
              -Wl,--rpath=$(BOOST_LIB) \
              -Wl,--rpath=/tools/batonroot/rodin/devkits/lnx64/gcc-${GCC_VERSION}/lib64 \
              -Wl,--rpath=${XILINX_SDX}/lib/lnx64.o \
              -Wl,--rpath=${XILINX_SDX}/runtime/lib/x86_64 \
              

#HOST_EXE_LDPATH = ${XILINX_SDX}/runtime/lib/x86_64:${XILINX_SDX}/lib/lnx64.o


# Margin of what SDX considers failed/passed routed design
# SDx flow stops if Vivado router has slack worse than this
# TO_DO: Not clear if it applies only to sys clocks or all (and thus prevents scaling)
# This is required if using AXI/DDR on for KU115 since its MIG has 10-level
# paths with almost no margin.
CLCC_OPT += --xp param:compiler.worstNegativeSlack=-0.1 

#ifneq ($(findstring 2016.2,$(XILINX_SDX)),2016.2)
#    # Temp flag to restore timing by unsetting the fanoutLimit
#    CLCC_OPT += --xp param:compiler.fanoutLimit=0
#endif

HOST_ARGS = ${XCLBIN} ${APP_BIN} ${APP_OUT_BIN}


##################################################################
#####    DDR mapping
##################################################################

#####  VU9P                        KU115

  ######################       ######################
  #                    #       #      D       D     #
  #              D     #       #      D       D     #
  #              D     #       #      R       R     #
  #              R     #       #      2       3     #
  #              3     #       #                    #
  #                    #       #                    #
  # SLR2               #       # SLR1               #
  ######################       ######################
  #          |   Static#       #              D     #
  #     D    |    D    #       #              D  S  #
  # A   D    |    D    #       #              R  t  #
  # P   R    |    R    #       #       D      1  a  #
  # M   2    |    1    #       #       D  A      t  #
  #          |         #       #       R  P      i  #
  # SLR1     |         #       # SLR0  0  M      c  #
  ######################       ######################
  #                    #
  #       D            #
  #       D            #
  #       R            #
  #       0            #
  #                    #
  # SLR0               #
  ######################


ifeq (${GEMX_part},ku115)
  K0_DDR = 3
  K1_DDR = 2
  K2_DDR = 0
  K3_DDR = 1
else ifeq (${GEMX_part},vu9p)
  K0_DDR = 0
  K1_DDR = 3
  K2_DDR = 2
  K3_DDR = 1
else ifeq (${GEMX_part},vu9pf1)
  K0_DDR = 3
  K1_DDR = 2
  K2_DDR = 0
  K3_DDR = 1
endif

CLCC_LINK_OPT += --xp misc:map_connect=add.kernel.gemxKernel_0.M_AXI_GMEMM.core.OCL_REGION_0.M0${K0_DDR}_AXI
ifeq ($(shell test $(GEMX_numKernels) -gt 1; echo $$?),0)
  CLCC_LINK_OPT += --xp misc:map_connect=add.kernel.gemxKernel_1.M_AXI_GMEMM.core.OCL_REGION_0.M0${K1_DDR}_AXI
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 2; echo $$?),0)
  CLCC_LINK_OPT += --xp misc:map_connect=add.kernel.gemxKernel_2.M_AXI_GMEMM.core.OCL_REGION_0.M0${K2_DDR}_AXI
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 3; echo $$?),0)
  CLCC_LINK_OPT += --xp misc:map_connect=add.kernel.gemxKernel_3.M_AXI_GMEMM.core.OCL_REGION_0.M0${K3_DDR}_AXI
endif

GEMX_fpgaDdrBanks = XCL_MEM_DDR_BANK${K0_DDR},XCL_MEM_DDR_BANK${K1_DDR},XCL_MEM_DDR_BANK${K2_DDR},XCL_MEM_DDR_BANK${K3_DDR}
HOST_CFLAGS += -D GEMX_fpgaDdrBanks=${GEMX_fpgaDdrBanks}

#CLCC_COMP_OPT += --xp param:compiler.enableAutoPipelining=false

#################################################################
XCLBIN_FREQ=xclbin_get_freq.pl

#################################################################

ifndef XILINX_SDX
$(error Environment variable XILINX_SDX is required and should point to SDAccel install area)
endif

SHELL = /bin/bash
VPATH = ${PWD}

#supported flow: cpu_emu, hw_emu, hw
#CC = $(XILINX_SDX)/lnx64/tools/gcc/bin/g++
CC = /tools/batonroot/rodin/devkits/lnx64/gcc-${GCC_VERSION}/bin/g++
#CC = $(XILINX_SDX)/Vivado_HLS/lnx64/tools/gcc/bin/g++
CLCC = $(XILINX_SDX)/bin/xocc

ifeq ($(XDEVICE_REPO_PATH),)
#no device repo path set. do nothing
    DEVICE_REPO_OPT = 
else
    DEVICE_REPO_OPT = --xp prop:solution.device_repo_paths=${XDEVICE_REPO_PATH} 
endif
ifeq ($(PLATFORM_REPO_PATHS),)
#no device repo path set. do nothing
    DEVICE_REPO_OPT = 
else
    DEVICE_REPO_OPT = --xp prop:solution.platform_repo_paths=${PLATFORM_REPO_PATHS} 
endif

CLCC_OPT += $(CLCC_OPT_LEVEL) ${DEVICE_REPO_OPT} --platform ${XDEVICE} 
CLCC_COMP_OPT =  ${CLCC_OPT} ${KERNEL_DEFS}
CLCC_COMP_OPT += --kernel_frequency ${GEMX_kernelHlsFreq}

ifeq (${KEEP_TEMP},1)
    CLCC_OPT += -s
endif

ifeq (${KERNEL_DEBUG},1)
    CLCC_OPT += -g
endif

CLCC_LINK_OPT += ${KERNEL_CU_OPTS}

# Vivado place and route setup
# This EXP flow is only setup for vu9pf1
ifeq (${GEMX_vivadoFlow},EXP)
  CLCC_OPT += -O3
  CLCC_OPT += -j 8
  XP_VIVADO_PROPS +=--xp vivado_prop:run.synth_1.STRATEGY=FLow_PerfOptimized_high
  XP_VIVADO_PROPS +=--xp vivado_prop:run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=Explore
  XP_VIVADO_PROPS += --xp 'vivado_prop:run.impl_1.{STEPS.PLACE_DESIGN.ARGS.MORE OPTIONS}={-fanout_opt}'
  #XP_VIVADO_PROPS +=--xp vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=AltSpreadLogic_high
  XP_VIVADO_PROPS +=--xp vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_BalanceSLLs
  XP_VIVADO_PROPS +=--xp vivado_prop:run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true
  XP_VIVADO_PROPS +=--xp vivado_prop:run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore
  #XP_VIVADO_PROPS +=--xp vivado_prop:run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveFanoutOpt
  XP_VIVADO_PROPS += --xp vivado_prop:run.impl_1.STEPS.OPT_DESIGN.TCL.POST=${PWD}/post_opt.tcl
  XP_VIVADO_PROPS +=--xp vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore
  XP_VIVADO_PROPS += --xp vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.TCL.PRE=${PWD}/pre_route.tcl
  XP_VIVADO_PROPS += --xp vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.TCL.POST=${PWD}/post_route.tcl
endif



ifeq (${SDA_FLOW},hw)
    CLCC_LINK_OPT += $(XP_VIVADO_PARAMS) $(XP_VIVADO_PROPS)
endif


###############################################################################
OUT_DIR = out_${SDA_FLOW}
XCLBIN = ${OUT_DIR}/gemx.xclbin
OUT_HOST_DIR = out_host
HOST_EXE = ${OUT_HOST_DIR}/gemx_host.exe
GEN_BIN_EXE = ${OUT_HOST_DIR}/gemx_gen_bin.exe
API_GEMM_EXE = ${OUT_HOST_DIR}/gemx_api_gemm.exe
API_GEMM_MULTI_INSTR_EXE = ${OUT_HOST_DIR}/gemx_api_gemm_multiInstr.exe

APP_BIN      = ${OUT_HOST_DIR}/app.bin
APP_GOLD_BIN = ${OUT_HOST_DIR}/app_gold.bin
APP_OUT_BIN  = ${OUT_DIR}/app_out.bin
APP_TXT      = ${OUT_HOST_DIR}/app.txt
APP_GOLD_TXT = ${OUT_HOST_DIR}/app_gold.txt
APP_OUT_TXT  = ${OUT_DIR}/app_out.txt
MAKE_EXIT_OK_HW_FILE = out_hw/gemx.xclbin

KERNEL_XOS = ${OUT_DIR}/k0dir/gemx.xo
ifeq ($(shell test $(GEMX_numKernels) -gt 1; echo $$?),0)
  KERNEL_XOS += ${OUT_DIR}/k1dir/gemx.xo
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 2; echo $$?),0)
  KERNEL_XOS += ${OUT_DIR}/k2dir/gemx.xo
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 3; echo $$?),0)
  KERNEL_XOS += ${OUT_DIR}/k3dir/gemx.xo
endif

###############################################################################


.PHONY: all

all: host run_cpu_em run_hw_em run_hw

run_cpu_em: host
	+make SDA_FLOW=cpu_emu run_em_int  2>&1 | tee log-run_cpu_em.txt

run_hw_em: host
	+make SDA_FLOW=hw_emu run_em_int  2>&1 | tee log-run_hw_em.txt

run_multiGemm_hw_em: api_gemm 
	+make SDA_FLOW=hw_emu run_multiGemm_em_int  2>&1 | tee log-run_multGemm_hw_em.txt

run_hw: host
	+make SDA_FLOW=hw run_hw_int  2>&1 | tee log-run_hw.txt; test -f ${MAKE_EXIT_OK_HW_FILE}

run_em_int: xconfig host xbin
	@echo INFO: kernel xclbin frequency is $(shell ${XCLBIN_FREQ} ${XCLBIN}) MHz
	XCL_EMULATION_MODE=true XILINX_OPENCL=${XILINX_SDX} ${HOST_EXE} ${HOST_ARGS}
	+make check

run_multiGemm_em_int: xconfig api_gemm xbin
	@echo INFO: kernel xclbin frequency is $(shell ${XCLBIN_FREQ} ${XCLBIN}) MHz
	XCL_EMULATION_MODE=true XILINX_OPENCL=${XILINX_SDX} ${API_GEMM_EXE} ${XCLBIN} 512 512 512

run_hw_int : host xbin xbinst_hw
	@echo INFO: kernel xclbin frequency is $(shell ${XCLBIN_FREQ} ${XCLBIN}) MHz
	@echo INFO: THE BOARD RUN WILL USE  ${HOST_EXE} ${HOST_ARGS}
	@echo INFO: AFTER THE BOARD RUN CHECK CORRECTNESS BY  cmp -i 8192 -l ${APP_GOLD_BIN} ${APP_OUT_BIN}
check: 
	${GEN_BIN_EXE} -read ${APP_OUT_BIN} > ${APP_OUT_TXT}
	cmp -i 8192 ${APP_GOLD_BIN} ${APP_OUT_BIN} || ${GEN_BIN_EXE} -compare 1e-3 3e-6 ${APP_GOLD_BIN} ${APP_OUT_BIN}

host : ${HOST_EXE} ${GEN_BIN_EXE} ${APP_GOLD_TXT} ${API_GEMM_EXE} 

api_gemm : ${API_GEMM_EXE} 

${HOST_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile host executable *****"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} src/gemx_main.cpp -o $@ 

${GEN_BIN_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile testcase generator executable *****"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} -fdata-sections -ffunction-sections -Wl,--gc-sections src/gemx_gen_bin.cpp -o $@

# API examples
${API_GEMM_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile testcase generator executable *****"
	@echo ${CC} ${HOST_CFLAGS} ${HOST_LFLAGS}
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} src/gemx_api_gemm.cpp -o $@

${API_GEMM_MULTI_INSTR_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile testcase generator executable *****"
	@echo ${CC} ${HOST_CFLAGS} ${HOST_LFLAGS}
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} src/gemx_api_gemm_multiInstr.cpp -o $@

${APP_GOLD_TXT} : ${GEN_BIN_EXE}
	${GEN_BIN_EXE} -write ${APP_BIN} ${GEN_BIN_PROGRAM}
	${GEN_BIN_EXE} -read ${APP_BIN} > ${APP_TXT}
	${GEN_BIN_EXE} -read ${APP_GOLD_BIN} > ${APP_GOLD_TXT}

xconfig : ${OUT_HOST_DIR}/emconfig.json | ${OUT_HOST_DIR}

${OUT_HOST_DIR}/emconfig.json :
	$(XILINX_SDX)/bin/emconfigutil --xdevice ${XDEVICE} ${DEVICE_REPO_OPT} --od ${OUT_HOST_DIR}

xbin: ${XCLBIN}

${XCLBIN}: ${KERNEL_XOS}
	@echo "************* Compile XCLBIN ${XCLBIN}  from  ${KERNEL_XOS} *************"
	${CLCC} -l ${CLCC_OPT} ${CLCC_LINK_OPT} $^ -o ${XCLBIN}

ifdef USE_SDX_1604
# Hacks for buggy P&R
  ifeq ($(shell test $(GEMX_numKernels) -gt 1; echo $$?),0)
    $(error 1604 cannot build more than 1 core)
  endif
${OUT_DIR}/k%dir/gemx.xo : src/* | ${OUT_DIR}/k%dir
	@echo "***** Compiling XO $@ *****"
	${CLCC}  -c ${CLCC_COMP_OPT} --xp prop:kernel.${KERNEL_NAME}_$(*F).kernel_flags=-std=c++0x -k ${KERNEL_NAME}_$(*F) -D GEMX_kernelId=$(*F) -o $@ ${KERNEL_SRCS}
else
# Normal flow
${OUT_DIR}/k%dir/gemx.xo : src/* | ${OUT_DIR}/k%dir
	@echo "***** Compiling XO $@ *****"
	${CLCC} --temp_dir ${OUT_DIR}/k$(*F)dir -c ${CLCC_COMP_OPT} --xp prop:kernel.${KERNEL_NAME}_$(*F).kernel_flags=-std=c++0x -k ${KERNEL_NAME}_$(*F) -D GEMX_kernelId=$(*F) -o $@ ${KERNEL_SRCS}
endif

xbinst_hw:  host
	@echo 'Running xbinst...'
	$(XILINX_SDX)/bin/xbinst --platform $(XDEVICE) -d ${OUT_DIR}
	#cp ${XILINX_SDX}/runtime/lib/x86_64/libOpenCL.so ${OUT_DIR}/xbinst/runtime/lib

clean :
	+make SDA_FLOW=cpu_emu clean_int
	+make SDA_FLOW=hw_emu clean_int
	+make SDA_FLOW=hw clean_int
	${RM} -rf ${OUT_HOST_DIR} sdaccel_profile* .Xil

clean_int:
	${RM} -rf ${OUT_DIR} log-run_${SDA_FLOW}.txt

${OUT_DIR} :
	@echo "************* Creating DIR $@ *************"
	mkdir $@

${OUT_HOST_DIR} :
	@echo "************* Creating DIR $@ *************"
	mkdir $@

${OUT_DIR}/k%dir : | ${OUT_DIR}
	@echo "************* Creating DIR $@ *************"
	mkdir $@
        
        
