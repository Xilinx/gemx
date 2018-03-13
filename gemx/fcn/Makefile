#Examples
#
#make run_cpu_em GEMX_keepMacBits=1 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=1 GEMX_splitMesh=1 GEMX_part=kcu1500 GEN_BIN_PROGRAM=" fcn 512 512 32 512 32 32 32 1 0 1 0 A0 B0 C0 Bias0 fcn 512 512 32 512 32 32 32 1 0 1 0 A1 C0 C1 Bias1 fcn 2048 512 32 512 32 32 32 1 0 1 0 A2 C1 C2 Bias2 fcn 256 2048 32 2048 32 32 32 1 0 1 0 A3 C2 C3 Bias3 fcn 2048 256 32 256 32 32 32 1 0 1 0 A4 C3 C4 Bias4 fcn 512 2048 32 2048 32 32 32 1 0 1 0 A5 C4 C5 Bias5 fcn 512 512 32 512 32 32 32 1 0 1 0  A6 C5 C6 Bias6 fcn 512 512 32 512 32 32 32 1 0 1 0 A7 C6 C7 Bias7"
#
#make run_hw GEMX_keepMacBits=1 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=1 GEMX_splitMesh=1 GEMX_part=kcu1500 GEN_BIN_PROGRAM=" fcn 512 512 32 512 32 32 32 1 0 1 0 A0 B0 C0 Bias0 fcn 512 512 32 512 32 32 32 1 0 1 0 A1 C0 C1 Bias1 fcn 2048 512 32 512 32 32 32 1 0 1 0 A2 C1 C2 Bias2 fcn 256 2048 32 2048 32 32 32 1 0 1 0 A3 C2 C3 Bias3 fcn 2048 256 32 256 32 32 32 1 0 1 0 A4 C3 C4 Bias4 fcn 512 2048 32 2048 32 32 32 1 0 1 0 A5 C4 C5 Bias5 fcn 512 512 32 512 32 32 32 1 0 1 0  A6 C5 C6 Bias6 fcn 512 512 32 512 32 32 32 1 0 1 0 A7 C6 C7 Bias7"
########

GCC_VERSION=6.2.0

HWEMUGUI = 0

#DSA_PATH=/proj/xbuilds/2017.2_sdx_released/installs/lin64/SDx/2017.2/platforms
DSA_PATH=${XILINX_SDX}/platforms
GCC_PATH=${XILINX_VIVADO}/tps/lnx64
BOOST_SRC=${PWD}/../../boost/src
BOOST_LIB=${PWD}/../../boost/lib
export BOOST_COMPUTE_DEFAULT_VENDOR=Xilinx


KERNEL_NAME = gemxKernel

##############################
# default settings 
GEMX_dataType      = short
GEMX_dataEqIntType = short
GEMX_ddrWidth      =  32 
GEMX_XdataType		 = int32_t 
GEMX_XddrWidth		 = 16 
GEMX_argInstrWidth =   1
GEMX_numInstr      =  16

GEMX_gemvkVectorBlocks  = 512
GEMX_gemvmVectorBlocks  = 512
GEMX_gemvmGroups   			=  1
 

GEMX_gemmMBlocks   			= 1 
GEMX_gemmKBlocks   			= 2
GEMX_gemmNBlocks   			= 1
GEMX_splitMesh	   			= 0

GEMX_keepMacBits				= 0
GEMX_macBits						= 48

GEMX_transpBlocks  			= 1

GEMX_spmvWidth   				= 8 
GEMX_spmvkVectorBlocks  = 2048
GEMX_spmvMacGroups      = 12 
GEMX_spmvPadA       		= 1
GEMX_spmvNumCblocks 		= 1024
GEMX_spmvFloatPerDesc 	= 4

# Correlated for IdxBits 2 => row idx < 2**14 so blocks 10 (2**14 / ddrw / spmvw / groups
GEMX_spmvColAddIdxBits  = 2


GEMX_argPipeline   			= 2
GEMX_part          			= vcu1525
GEMX_kernelHlsFreq	   	= 250 
GEMX_kernelVivadoFreq  	= 250
# How many kernels to replicate in the accelerator (use 1 to 4)
GEMX_numKernels    			= 1
# What engines get included in each accelerator kernel (use 0 or 1)
# The more engines you include the more catability you get but P&R
# P&R becomes more difficult thus you get lower Fmax
GEMX_runGemv 						= 0 
GEMX_runGemm 						= 1
GEMX_runTransp 					= 0
GEMX_runSpmv 						= 0

##############################

# Defauts for SPMV 32-wide
ifeq (${GEMX_ddrWidth}, 32)
  GEMX_spmvWidth   			=  8
endif

ifeq (${GEMX_dataType}, float)
  GEMX_dataEqIntType 		= int
  
  GEMX_gemvmVectorBlocks = 43
  
  GEMX_spmvPadA 				= 0
  GEMX_spmvFloatPerDesc = 2
  GEMX_spmvMacGroups    =   12
  ifeq (${GEMX_ddrWidth}, 16)
    GEMX_spmvWidth   		=  8
    #GEMX_spmvmVectorBlocks  =  10
  endif
  
endif

ifeq (${GEMX_part},kcu1500)
  DSA=5_0
  XDEVICE_COLON=xilinx:kcu1500:dynamic:$(subst _,.,${DSA})
  XDEVICE=xilinx_kcu1500_dynamic_${DSA}
  DSA_PLATFORM=xilinx_kcu1500_dynamic_${DSA}
  XDEVICE_REPO_PATH=$(XILINX_SDX)/platforms
  PLATFORM_REPO_PATH=$(XILINX_SDX)/platforms/${DSA_PLATFORM}
	XOPENCL_LIB_PATH=${PLATFORM_REPO_PATH}/sw/lib/x86_64
else ifeq (${GEMX_part},vcu1525)
  DSA=5_0
  XDEVICE_COLON=xilinx:vcu1525:dynamic:$(subst _,.,${DSA})
  XDEVICE=xilinx_vcu1525_dynamic_${DSA}
  DSA_PLATFORM=xilinx_vcu1525_dynamic_${DSA}
  XDEVICE_REPO_PATH=$(XILINX_SDX)/platforms
  PLATFORM_REPO_PATH=$(XILINX_SDX)/platforms/${DSA_PLATFORM}
  XOPENCL_LIB_PATH=${PLATFORM_REPO_PATH}/sw/lib/x86_64
else ifeq (${GEMX_part},ku115)
  DSA=5_0
  XDEVICE_COLON=xilinx:xil-accel-rd-ku115:dynamic:$(subst _,.,${DSA})
  XDEVICE=xilinx_xil-accel-rd-ku115_dynamic_${DSA}
  DSA_PLATFORM=xilinx_xil-accel-rd-ku115_dynamic_${DSA}
  XDEVICE_REPO_PATH=$(DSA_PATH)
  PLATFORM_REPO_PATH=$(DSA_PATH)/${DSA_PLATFORM}
  XOPENCL_LIB_PATH=${DSA_PATH}/../runtime/lib/x86_64
else
  $(error Unknown GEMX_part ${GEMX_part})
endif

ifeq ("$(wildcard $(XDEVICE_REPO_PATH))","")
  ifeq ("$(wildcard $(PLATFORM_REPO_PATH))","")
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


########################################################################## 

ifndef SDA_FLOW
  SDA_FLOW = hw
endif

HOST_SRCS = src/gemx_fcn_main.cpp

KERNEL_SRCS = src/gemx_fcn_kernel.cpp


GMEM_FLAGS = -D GMEM_M=$(GMEM0_M) 

CFLAGS_K = $(GMEM_FLAGS) -I ./src -I ../src \
          -D TEST_SDX=1 \
          -D GEMX_dataType=$(GEMX_dataType) \
          -D GEMX_XdataType=$(GEMX_XdataType) \
          -D GEMX_dataEqIntType=$(GEMX_dataEqIntType) \
          -D GEMX_ddrWidth=$(GEMX_ddrWidth) \
          -D GEMX_XddrWidth=$(GEMX_XddrWidth) \
          -D GEMX_argInstrWidth=$(GEMX_argInstrWidth) \
          -D GEMX_numInstr=$(GEMX_numInstr) \
          -D GEMX_gemvkVectorBlocks=$(GEMX_gemvkVectorBlocks) \
          -D GEMX_gemvmVectorBlocks=$(GEMX_gemvmVectorBlocks) \
          -D GEMX_gemvmGroups=$(GEMX_gemvmGroups) \
	  			-D GEMX_gemmMBlocks=${GEMX_gemmMBlocks} \
	  			-D GEMX_gemmKBlocks=${GEMX_gemmKBlocks} \
	  			-D GEMX_gemmNBlocks=${GEMX_gemmNBlocks} \
					-D GEMX_keepMacBits=${GEMX_keepMacBits} \
	  			-D GEMX_macBits=${GEMX_macBits} \
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
	  			-D GEMX_splitMesh=${GEMX_splitMesh} \
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

#CLCC_OPT: CLCC options for both compile and link
#CLCC_COMP_OPT: CLCC options only for compile mode
#CLCC_LINK_OPT: CLCC options only for link mode
CLCC_OPT += -t ${SDA_FLOW}
ifeq (${SDA_FLOW},hw_emu)
    ifeq ($(HWEMUGUI),1)
        CLCC_OPT += --xp param:hw_em.debugLevel=GUI
		else
				CLCC_OPT += --xp param:compiler.preserveHlsOutput=1 --report system
    endif
endif


CLCC_LINK_OPT += --kernel_frequency ${GEMX_kernelVivadoFreq}
CLCC_OPT += --xp prop:solution.hls_pre_tcl=../hls_config.tcl
HOST_CFLAGS = -g -O0 -std=c++11 \
              -I $(BOOST_SRC) \
              -DCL_USE_DEPRECATED_OPENCL_1_1_APIS \
              -DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION \
	      			-DBOOST_COMPUTE_HAVE_THREAD_LOCAL \
	      			-DBOOST_COMPUTE_THREAD_SAFE \
              -D FLOW_HLS_CSIM $(CFLAGS_K) \
	      			-D HLS_NO_XIL_FPO_LIB=1 \
              -I$(XILINX_SDX)/Vivado_HLS/include \
              -I$(XILINX_VIVADO)/include \
              -I${XILINX_SDX}/runtime/include/1_2 


HOST_EXE_DIR=.
HOST_LFLAGS = \
              -L$(BOOST_LIB) \
              -lboost_iostreams -lz \
              -L${GCC_PATH}/gcc-${GCC_VERSION}/lib64 \
							-lxilinxopencl \
							-lstdc++ \
							-L${XOPENCL_LIB_PATH} \
              -lrt \
              -pthread \
							-Wl,--rpath=${XOPENCL_LIB_PATH} \
              -Wl,--rpath=$(BOOST_LIB) \
              -Wl,--rpath=${GCC_PATH}/gcc-${GCC_VERSION}/lib64 \
              

CLCC_OPT += --xp param:compiler.worstNegativeSlack=-0.1 

HOST_ARGS = ${XCLBIN} ${APP_BIN} ${APP_OUT_BIN}

ifeq (${GEMX_part},ku115)
  K0_DDR = 0
  K1_DDR = 1
  K2_DDR = 2
  K3_DDR = 3
else ifeq (${GEMX_part},kcu1500)
  K0_DDR = 0
  K1_DDR = 1
  K2_DDR = 2
  K3_DDR = 3
else ifeq (${GEMX_part},vcu1525)
  K0_DDR = 0
  K1_DDR = 1
  K2_DDR = 2
  K3_DDR = 3
endif

CLCC_LINK_OPT += --sp gemxKernel_0.m_axi_gmemm:bank${K0_DDR}
ifeq ($(shell test $(GEMX_numKernels) -gt 1; echo $$?),0)
  CLCC_LINK_OPT += --sp gemxKernel_1.m_axi_gmemm:bank${K1_DDR}
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 2; echo $$?),0)
  CLCC_LINK_OPT += --sp gemxKernel_2.m_axi_gmemm:bank${K2_DDR}
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 3; echo $$?),0)
  CLCC_LINK_OPT += --sp gemxKernel_3.m_axi_gmemm:bank${K3_DDR}
endif
GEMX_fpgaDdrBanks = XCL_MEM_DDR_BANK${K0_DDR},XCL_MEM_DDR_BANK${K1_DDR},XCL_MEM_DDR_BANK${K2_DDR},XCL_MEM_DDR_BANK${K3_DDR}
HOST_CFLAGS += -D GEMX_fpgaDdrBanks=${GEMX_fpgaDdrBanks}


#################################################################
XCLBIN_FREQ=xclbin_get_freq.pl

#################################################################

ifndef XILINX_SDX
$(error Environment variable XILINX_SDX is required and should point to SDAccel install area)
endif

SHELL = /bin/bash
VPATH = ${PWD}

#supported flow: sw_emu, hw_emu, hw
CC = ${GCC_PATH}/gcc-${GCC_VERSION}/bin/g++
CLCC = $(XILINX_SDX)/bin/xocc

ifeq ($(XDEVICE_REPO_PATH),)
#no device repo path set. do nothing
    DEVICE_REPO_OPT = 
else
    DEVICE_REPO_OPT = --xp prop:solution.device_repo_paths=${XDEVICE_REPO_PATH} 
endif
ifeq ($(PLATFORM_REPO_PATH),)
#no device repo path set. do nothing
    PLATFORM_REPO_OPT = 
else
    PLATFORM_REPO_OPT = --xp prop:solution.platform_repo_paths=${PLATFORM_REPO_PATH} 
endif

CLCC_OPT += $(CLCC_OPT_LEVEL) ${PLATFORM_REPO_OPT} --platform ${DSA_PLATFORM} 
CLCC_COMP_OPT =  ${CLCC_OPT} ${KERNEL_DEFS}
CLCC_COMP_OPT += --kernel_frequency ${GEMX_kernelHlsFreq}

ifeq (${KEEP_TEMP},1)
    CLCC_OPT += -s
endif

ifeq (${KERNEL_DEBUG},1)
    CLCC_OPT += -g
endif

CLCC_LINK_OPT += ${KERNEL_CU_OPTS}

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

#all: host run_cpu_em run_hw_em run_hw
all: host run_cpu_em run_hw_em run_hw 

run_cpu_em: host
	+make SDA_FLOW=sw_emu run_em_int  2>&1 | tee log-run_cpu_em.txt

run_hw_em: host
	+make SDA_FLOW=hw_emu run_em_int  2>&1 | tee log-run_hw_em.txt

run_hw: host
	+make SDA_FLOW=hw run_hw_int  2>&1 | tee log-run_hw.txt; test -f ${MAKE_EXIT_OK_HW_FILE}

run_em_int: xconfig host xbin
	@echo INFO: kernel xclbin frequency is $(shell ${XCLBIN_FREQ} ${XCLBIN}) MHz
	XCL_EMULATION_MODE=true XILINX_OPENCL=${XILINX_SDX} ${HOST_EXE} ${HOST_ARGS}
	+make check

run_hw_int : xbin xbinst_hw
	@echo INFO: kernel xclbin frequency is $(shell ${XCLBIN_FREQ} ${XCLBIN}) MHz
	@echo INFO: THE BOARD RUN WILL USE  ${HOST_EXE} ${HOST_ARGS}

check: 
ifeq ($(shell test $(GEMX_numKernels) -gt 0; echo $$?),0)
	${GEN_BIN_EXE} -read ${OUT_DIR}/app_out0.bin  > ${OUT_DIR}/app_out0.txt
	cmp -i 8192 ${APP_GOLD_BIN} ${OUT_DIR}/app_out0.bin || ${GEN_BIN_EXE} -compare 1e-3 3e-6 ${APP_GOLD_BIN} ${OUT_DIR}/app_out0.bin
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 1; echo $$?),0)
	${GEN_BIN_EXE} -read ${OUT_DIR}/app_out1.bin  > ${OUT_DIR}/app_out1.txt
	cmp -i 8192 ${APP_GOLD_BIN} ${OUT_DIR}/app_out1.bin || ${GEN_BIN_EXE} -compare 1e-3 3e-6 ${APP_GOLD_BIN} ${OUT_DIR}/app_out1.bin
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 2; echo $$?),0)
	${GEN_BIN_EXE} -read ${OUT_DIR}/app_out2.bin  > ${OUT_DIR}/app_out2.txt
	cmp -i 8192 ${APP_GOLD_BIN} ${OUT_DIR}/app_out2.bin || ${GEN_BIN_EXE} -compare 1e-3 3e-6 ${APP_GOLD_BIN} ${OUT_DIR}/app_out2.bin
endif
ifeq ($(shell test $(GEMX_numKernels) -gt 3; echo $$?),0)
	
	cmp -i 8192 ${APP_GOLD_BIN} ${OUT_DIR}/app_out3.bin || ${GEN_BIN_EXE} -compare 1e-3 3e-6 ${APP_GOLD_BIN} ${OUT_DIR}/app_out3.bin
endif

host : ${HOST_EXE} ${GEN_BIN_EXE} ${APP_GOLD_TXT} 

api_gemm : ${API_GEMM_EXE}

${HOST_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile host executable *****"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} src/gemx_fcn_main.cpp -o $@ 

${GEN_BIN_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile testcase generator executable *****"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} -fdata-sections -ffunction-sections -Wl,--gc-sections src/gemx_fcn_gen_bin.cpp -o $@

${APP_GOLD_TXT} : ${GEN_BIN_EXE} 
	${GEN_BIN_EXE} -write ${APP_BIN} ${GEN_BIN_PROGRAM}
	${GEN_BIN_EXE} -read ${APP_BIN} > ${APP_TXT}
	${GEN_BIN_EXE} -read ${APP_GOLD_BIN} > ${APP_GOLD_TXT}

xconfig : ${OUT_HOST_DIR}/emconfig.json | ${OUT_HOST_DIR}

${OUT_HOST_DIR}/emconfig.json :
	$(XILINX_SDX)/bin/emconfigutil --xdevice ${XDEVICE_COLON} ${DEVICE_REPO_OPT} --od ${OUT_HOST_DIR}

xbin: ${XCLBIN}

${XCLBIN}: ${KERNEL_XOS}
	@echo "************* Compile XCLBIN ${XCLBIN}  from  ${KERNEL_XOS} *************"
	${CLCC} -l ${CLCC_OPT} ${CLCC_LINK_OPT} $^ -o ${XCLBIN}

${OUT_DIR}/k%dir/gemx.xo : src/* | ${OUT_DIR}/k%dir
	@echo "***** Compiling XO $@ *****"
	${CLCC} --temp_dir ${OUT_DIR}/k$(*F)dir -c ${CLCC_COMP_OPT} --xp prop:kernel.${KERNEL_NAME}_$(*F).kernel_flags=-std=c++0x -k ${KERNEL_NAME}_$(*F) -D GEMX_kernelId=$(*F) -o $@ ${KERNEL_SRCS}

xbinst_hw:  host
	@echo 'Running xbinst...'
	$(DSA_PATH)/../bin/xbinst --platform_repo_paths=${PLATFORM_REPO_PATH} --platform $(XDEVICE) -d ${OUT_DIR}

clean :
	+make SDA_FLOW=sw_emu clean_int
	+make SDA_FLOW=hw_emu clean_int
	+make SDA_FLOW=hw clean_int
	${RM} -rf ${OUT_HOST_DIR} sdaccel_profile* .Xil _sds iprepo bd.* *.bit *.ltx *.dat *.hpfm *.xml _new_clk_freq dr.bd.tcl

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
        
        
