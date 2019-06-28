# Copyright 2019 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################################################
# Common Makefile for environment setting and common make targets
# Usage: 
# include common.mk
############################################################

GCC_VERSION=6.2.0
GCC_PATH=${XILINX_VIVADO}/tps/lnx64
HWEMUGUI = 0 
KERNEL_NAME = gemxKernel

##############################
# default settings 

GEMX_argPipeline        = 2
GEMX_part               = u200
GEMX_kernelHlsFreq      = 250 
GEMX_kernelVivadoFreq   = 250
GEMX_instructionSizeBytes = 64
# How many kernels to replicate in the accelerator (use 1 to 4)
GEMX_numKernels         = 1

GEMX_dataType      = short
GEMX_dataEqIntType = short
GEMX_ddrWidth      =  32 
GEMX_XdataType     = int32_t 
GEMX_XddrWidth     = 16 
GEMX_argInstrWidth =   1
GEMX_numInstr      =  16
TEST_MEMCPY        = 0

############################################################

# To build xclbin with other shell 
# 1. PLATFORM_REPO_PATH set to directory contains xpfm file for that shell
# 2. DSA_PLATFORM set to platform name
# 3. XOPENCL_LIB_PATH set to $XILINX_XRT/lib or ${PLATFORM_REPO_PATH}/sw/lib/x86_64 accordingly
# 4. Then set K0_DDR, K1_DDR, K2_DDR, K3_DDR accordingly as well


# DSA path and lib path
ifeq (${GEMX_part},u200)
  DSA_PLATFORM=xilinx_u200_xdma_201830_2
  PLATFORM_REPO_PATH=${TA_PATH}/../../xbb/dsadev/opt/xilinx/platforms/${DSA_PLATFORM}
  XOPENCL_LIB_PATH=${XILINX_XRT}/lib
else
  $(error Unknown GEMX_part ${GEMX_part})
endif

#Check if path exists or not
ifeq ("$(wildcard $(PLATFORM_REPO_PATH))","")
  PLATFORM_REPO_PATH=${XILINX_SDX}/platforms/${DSA_PLATFORM}
endif

CL_LIB_PATH = ${XILINX_XRT}/include

ifndef XILINX_XRT
  XOPENCL_LIB_PATH=${XILINX_SDX}/runtime/lib/x86_64
  CL_LIB_PATH=${XILINX_SDX}/runtime/include/1_2
endif

############################################################

ifndef XILINX_SDX
$(error Environment variable XILINX_SDX is required and should point to SDAccel install area)
endif

ifndef SDA_FLOW
  SDA_FLOW = hw
endif

GMEM0_M  =0
GMEM_FLAGS = -D GMEM_M=$(GMEM0_M) 

ifeq (${GEMX_part},u200)
  K0_DDR = 0
  K1_DDR = 3
  K2_DDR = 1
  K3_DDR = 2
endif

##############################
# host options

HOST_INCLUDE_CFLAGS = -I$(XILINX_VIVADO)/include \
                      -I${CL_LIB_PATH}

HOST_INCLUDE_CFLAGS += -I $(BOOST_SRC)

HOST_CFLAGS = -g -O0 -std=c++11 \
              -D FLOW_HLS_CSIM $(CFLAGS_K) \
              -D HLS_NO_XIL_FPO_LIB=1 $(HOST_INCLUDE_CFLAGS)

HOST_LIB_LFLAGS = -L$(BOOST_LIB) \
                  -L${XOPENCL_LIB_PATH} \
                  -lboost_iostreams -lz \
                  -lxilinxopencl \
                  -lstdc++ \
                  -lrt \
                  -pthread
                  
HOST_LFLAGS =  $(HOST_LIB_LFLAGS) \
              -Wl,--rpath=${XOPENCL_LIB_PATH}\
              -Wl,--rpath=$(BOOST_LIB)

              
HOST_ARGS = ${XCLBIN} ${APP_BIN} ${APP_OUT_BIN}

GEMX_fpgaDdrBanks = XCL_MEM_DDR_BANK${K0_DDR},XCL_MEM_DDR_BANK${K1_DDR},XCL_MEM_DDR_BANK${K2_DDR},XCL_MEM_DDR_BANK${K3_DDR}
HOST_CFLAGS += -D GEMX_fpgaDdrBanks=${GEMX_fpgaDdrBanks}

##############################

#CLCC_OPT: CLCC options for both compile and link
#CLCC_COMP_OPT: CLCC options only for compile mode
#CLCC_LINK_OPT: CLCC options only for link mode

SHELL = /bin/bash

#supported flow: sw_emu, hw_emu, hw
CC = ${GCC_PATH}/gcc-${GCC_VERSION}/bin/g++
CLCC = $(XILINX_SDX)/bin/xocc

##############################
# XOCC options for both compile and link

CLCC_OPT += -t ${SDA_FLOW}

ifeq (${SDA_FLOW},hw_emu)
    ifeq ($(HWEMUGUI),1)
        CLCC_OPT += --xp param:hw_em.debugLevel=GUI
    else
        CLCC_OPT += --xp param:compiler.preserveHlsOutput=1 --report system
    endif
endif

CLCC_OPT += --xp param:compiler.worstNegativeSlack=-0.1 

KEEP_TEMP=1
KERNEL_DEBUG=1

ifeq (${KEEP_TEMP},1)
    CLCC_OPT += -s
endif

ifeq (${KERNEL_DEBUG},1)
    CLCC_OPT += -g
endif

CLCC_OPT += $(CLCC_OPT_LEVEL) -f ${PLATFORM_REPO_PATH}/$(DSA_PLATFORM).xpfm

##############################
# XOCC compile options

KERNEL_DEFS += $(CFLAGS_K)

CLCC_COMP_OPT =  ${CLCC_OPT} ${KERNEL_DEFS}
CLCC_COMP_OPT += --kernel_frequency ${GEMX_kernelHlsFreq}

##############################
# XOCC link options

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

XP_VIVADO_PARAMS =--xp vivado_param:project.writeIntermediateCheckpoints=1
XP_VIVADO_PARAMS+=--xp vivado_param:place.hardVerbose=469538
XP_VIVADO_PARAMS+=--xp vivado_param:place.oldMsgVerbose=1
XP_VIVADO_PARAMS+=--xp vivado_param:route.flowDbg=1
XP_VIVADO_PARAMS+=--xp vivado_param:route.timingDbg=1
XP_VIVADO_PARAMS+=--xp param:compiler.fanoutLimit=0

CLCC_LINK_OPT += ${KERNEL_CU_OPTS}

ifeq (${SDA_FLOW},hw)
    CLCC_LINK_OPT += $(XP_VIVADO_PARAMS)
endif

CLCC_LINK_OPT += --kernel_frequency ${GEMX_kernelVivadoFreq}

ifeq (${GEMX_part},u280)
  CLCC_LINK_OPT += --sp gemxKernel_0.m_axi_gmemm:HBM[${K0_DDR}]
  ifeq ($(shell test $(GEMX_numKernels) -gt 1; echo $$?),0)
    CLCC_LINK_OPT += --sp gemxKernel_1.m_axi_gmemm:HBM[${K1_DDR}]
  endif
  ifeq ($(shell test $(GEMX_numKernels) -gt 2; echo $$?),0)
    CLCC_LINK_OPT += --sp gemxKernel_2.m_axi_gmemm:HBM[${K2_DDR}]
  endif
  ifeq ($(shell test $(GEMX_numKernels) -gt 3; echo $$?),0)
    CLCC_LINK_OPT += --sp gemxKernel_3.m_axi_gmemm:HBM[${K3_DDR}]
  endif
else
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
endif

###############################################################################
# For Makefile target only 

CONFIG_INFO = $(shell echo ${CFLAGS_K} | sed 's/.*TEST_SDX=1//; s/-D //g; s/ -Wno.*//')

XCLBIN_FREQ=xclbin_get_freq.pl

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

##############################

OUT_DIR = out_${SDA_FLOW}
XCLBIN = ${OUT_DIR}/gemx.xclbin
OUT_HOST_DIR = out_host
HOST_EXE = ${OUT_HOST_DIR}/gemx_host.exe
GEN_BIN_EXE = ${OUT_HOST_DIR}/gemx_gen_bin.exe

APP_BIN      = ${OUT_HOST_DIR}/app.bin
APP_GOLD_BIN = ${OUT_HOST_DIR}/app_gold.bin
APP_OUT_BIN  = ${OUT_DIR}/app_out.bin
APP_TXT      = ${OUT_HOST_DIR}/app.txt
APP_GOLD_TXT = ${OUT_HOST_DIR}/app_gold.txt
APP_OUT_TXT  = ${OUT_DIR}/app_out.txt
MAKE_EXIT_OK_HW_FILE = out_hw/gemx.xclbin

###############################################################################

.PHONY: all

${APP_GOLD_TXT} : ${GEN_BIN_EXE} 
	${GEN_BIN_EXE} -write ${APP_BIN} ${GEN_BIN_PROGRAM}
	${GEN_BIN_EXE} -read ${APP_BIN} > ${APP_TXT}
	${GEN_BIN_EXE} -read ${APP_GOLD_BIN} > ${APP_GOLD_TXT}

xconfig : ${OUT_HOST_DIR}/emconfig.json | ${OUT_HOST_DIR}

${OUT_HOST_DIR}/emconfig.json :
	$(XILINX_SDX)/bin/emconfigutil -f ${PLATFORM_REPO_PATH}/$(DSA_PLATFORM).xpfm --od ${OUT_HOST_DIR}

xbin: ${XCLBIN}
	+make dump_config

${XCLBIN}: ${KERNEL_XOS}
	@echo "************* Compile XCLBIN ${XCLBIN}  from  ${KERNEL_XOS} *************"
	${CLCC} -l ${CLCC_OPT} ${CLCC_LINK_OPT} $^ -o ${XCLBIN}

${OUT_DIR}/k%dir/gemx.xo : src/* | ${OUT_DIR}/k%dir
	@echo "***** Compiling XO $@ *****"
	${CLCC} --temp_dir ${OUT_DIR}/k$(*F)dir -c ${CLCC_COMP_OPT} --xp prop:kernel.${KERNEL_NAME}_$(*F).kernel_flags=-std=c++0x -k ${KERNEL_NAME}_$(*F) -D GEMX_kernelId=$(*F) -o $@ ${KERNEL_SRCS}

run_sw_em: host
ifeq ($(GEMX_numKernels), 1)
	+make SDA_FLOW=sw_emu run_em_int  2>&1 | tee log-run_sw_emu.txt
else
	$(error sw emulation doesn't not support multi-kernels)
endif

run_hw_em: host
	+make SDA_FLOW=hw_emu run_em_int  2>&1 | tee log-run_hw_emu.txt

run_hw: host
	+make SDA_FLOW=hw run_hw_int  2>&1 | tee log-run_hw.txt; test -f ${MAKE_EXIT_OK_HW_FILE}

run_em_int: xconfig host xbin
	@echo INFO: kernel xclbin frequency is $(shell ${XCLBIN_FREQ} ${XCLBIN}) MHz
	XCL_EMULATION_MODE=${SDA_FLOW} XILINX_OPENCL=${XILINX_SDX} ${HOST_EXE} ${HOST_ARGS}
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

clean :
	+make SDA_FLOW=sw_emu clean_int
	+make SDA_FLOW=hw_emu clean_int
	+make SDA_FLOW=hw clean_int
	${RM} -rf ${OUT_HOST_DIR} _xocc_link_gemx_gemx.dir sdaccel_profile* .Xil _sds iprepo bd.* *.bit *.ltx *.dat *.hpfm *.xml _new_clk_freq dr.bd.tcl routed.dcp xocc_gemx* _x

clean_int:
	${RM} -rf ${OUT_DIR} log-run_${SDA_FLOW}.txt

xbinst_hw:  host
	@echo 'Running xbinst...'
	mkdir -p ${OUT_DIR}/xbinst | tee ${OUT_DIR}/xbinst.log

dump_config: ${OUT_DIR}
	@echo ${CONFIG_INFO} GEMX_fpgaDdrBanks=${GEMX_fpgaDdrBanks} | tr " " "\n" > ${OUT_DIR}/config_info.dat

${OUT_DIR} :
	@echo "************* Creating DIR $@ *************"
	mkdir $@

${OUT_HOST_DIR} :
	@echo "************* Creating DIR $@ *************"
	mkdir $@

${OUT_DIR}/k%dir : | ${OUT_DIR}
	@echo "************* Creating DIR $@ *************"
	mkdir $@
