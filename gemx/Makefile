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

include common.mk

# Examples
# make gemx_func_test GEMX_keepMacBits=1 GEMX_splitMesh=1  GEMX_spmvPadA=1 GEMX_spmvFloatPerDesc=4 GEMX_runGemm=1 GEMX_runGemv=1 GEMX_runSpmv=1 GEMX_runTransp=1 GEN_BIN_PROGRAM="gemm 256 256 256  256 256 256 256 1 0 A1 B1 C1 X1 gemv 256 256 288 A2 B2 C2 spmv 96 128 256 none A3 B3 C3 true transp 32 32 64 96 rm cm A4 B4"
# 
# make run_sw_em GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemm=1 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=u200 GEN_BIN_PROGRAM="gemm 512 512 512  512 512 512 512 1 0 A05 B05 C05 X05"
# 
# make run_hw_em GEMX_ddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=0 GEMX_runTransp=0 GEMX_runSpmv=1 GEMX_dataType=float GEMX_part=u200 GEN_BIN_PROGRAM="spmv 96 128 256 none A0 B0 C0 true spmv 0 0 0 data/spmv/diag16.mtx A1 B1 C1 true"


BOOST_SRC=$(XILINX_VIVADO)/tps/boost_1_64_0
BOOST_LIB=$(XILINX_VIVADO)/lib/lnx64.o

##############################
# default settings 

#GEMV variables
GEMX_gemvkVectorBlocks  = 512
GEMX_gemvmVectorBlocks  = 512
GEMX_gemvmGroups        =  1
 
#GEMM variables
GEMX_gemmMBlocks        = 1 
GEMX_gemmKBlocks        = 2
GEMX_gemmNBlocks        = 1
GEMX_splitMesh          = 0

GEMX_keepMacBits        = 0
GEMX_macBits            = 48

#TRANSP variables
GEMX_transpBlocks       = 1

#SPMV USPMV variables
GEMX_spmvWidth          = 8 
GEMX_spmvkVectorBlocks  = 2048
GEMX_spmvMacGroups      = 12 
GEMX_spmvPadA           = 0
GEMX_spmvNumCblocks     = 1024
GEMX_spmvFloatPerDesc   = 2 
# Correlated for IdxBits 2 => row idx < 2**14 so blocks 10 (2**14 / ddrw / spmvw / groups
GEMX_spmvColAddIdxBits  = 2

#spmv definitions used for URAM-based SPMV implementation
GEMX_idxType            = int32_t
GEMX_nnzBlocks          = 8
GEMX_spmvKmaxBlocks     = 32768 
GEMX_spmvMmaxBlocks     = 5462 
GEMX_spmvUramGroups     = 6 

#definitions used for streamlined Uspmv
GEMX_uspmvStages = 3
GEMX_uspmvInterleaves = 12
#GEMX_uspmvMvectoBlocks must be multiple of GEMX_uspmvInterleaves
GEMX_uspmvMvectorBlocks = 1200  
GEMX_uspmvNnzVectorBlocks = 2048

GEMX_useURAM            = 0

# What engines get included in each accelerator kernel (use 0 or 1)
# The more engines you include the more catability you get but P&R
# P&R becomes more difficult thus you get lower Fmax
GEMX_runGemv            = 0 
GEMX_runGemm            = 0
GEMX_runTransp          = 0
GEMX_runSpmv            = 0
GEMX_runUspmv           = 0
GEMX_runFcn             = 0

##############################

# Defauts for SPMV 32-wide
ifeq (${GEMX_ddrWidth}, 32)
  GEMX_spmvWidth         = 8
endif

ifeq (${GEMX_dataType}, float)
  GEMX_dataEqIntType     = int
  
  GEMX_gemvmVectorBlocks = 43
  
  GEMX_spmvPadA          = 0
  GEMX_spmvFloatPerDesc  = 2
  GEMX_spmvMacGroups     = 12
  ifeq (${GEMX_ddrWidth}, 16)
    GEMX_spmvWidth       =  8
  endif
  
endif

##############################

HOST_SRCS = src/host/gemx_main.cpp

KERNEL_SRCS = src/kernel/gemx_kernel.cpp

COMMON_CFLAGS_K = -D TEST_MEMCPY=$(TEST_MEMCPY) \
					-D GEMX_instructionSizeBytes=$(GEMX_instructionSizeBytes) \
          -D GEMX_dataType=$(GEMX_dataType) \
          -D GEMX_dataEqIntType=$(GEMX_dataEqIntType) \
          -D GEMX_ddrWidth=$(GEMX_ddrWidth) \
          -D GEMX_argInstrWidth=$(GEMX_argInstrWidth) \
          -D GEMX_numInstr=$(GEMX_numInstr) \
          -D GEMX_argPipeline=$(GEMX_argPipeline) \
          -D GEMX_part=$(GEMX_part) \
          -D GEMX_runTransp=$(GEMX_runTransp) \
          -D GEMX_runGemv=$(GEMX_runGemv) \
          -D GEMX_runGemm=$(GEMX_runGemm) \
          -D GEMX_runSpmv=$(GEMX_runSpmv) \
          -D GEMX_runUspmv=${GEMX_runUspmv} \
          -D GEMX_runFcn=${GEMX_runFcn} \
          -D GEMX_numKernels=${GEMX_numKernels}


CFLAGS_K = $(GMEM_FLAGS) -I ./src \
          -I ./src/host \
          -I ./src/kernel \
          -D TEST_SDX=1

ifeq (${GEMX_runGemm}, 1)
CFLAGS_K += -D GEMX_gemmMBlocks=${GEMX_gemmMBlocks} \
            -D GEMX_gemmKBlocks=${GEMX_gemmKBlocks} \
            -D GEMX_gemmNBlocks=${GEMX_gemmNBlocks} \
            -D GEMX_splitMesh=${GEMX_splitMesh} \
            -D GEMX_keepMacBits=${GEMX_keepMacBits} \
            -D GEMX_macBits=${GEMX_macBits} \
            -D GEMX_XdataType=$(GEMX_XdataType) \
            -D GEMX_XddrWidth=$(GEMX_XddrWidth)
endif

ifeq (${GEMX_runFcn}, 1)
CFLAGS_K += -D GEMX_gemmMBlocks=${GEMX_gemmMBlocks} \
            -D GEMX_gemmKBlocks=${GEMX_gemmKBlocks} \
            -D GEMX_gemmNBlocks=${GEMX_gemmNBlocks} \
            -D GEMX_splitMesh=${GEMX_splitMesh} \
            -D GEMX_keepMacBits=${GEMX_keepMacBits} \
            -D GEMX_macBits=${GEMX_macBits} \
            -D GEMX_XdataType=$(GEMX_XdataType) \
            -D GEMX_XddrWidth=$(GEMX_XddrWidth)
endif

ifeq (${GEMX_runGemv}, 1)
CFLAGS_K += -D GEMX_gemvkVectorBlocks=$(GEMX_gemvkVectorBlocks) \
            -D GEMX_gemvmVectorBlocks=$(GEMX_gemvmVectorBlocks) \
            -D GEMX_gemvmGroups=$(GEMX_gemvmGroups) \
            -D GEMX_transpBlocks=$(GEMX_transpBlocks)
endif

ifeq (${GEMX_runSpmv}, 1)
CFLAGS_K += -D GEMX_spmvWidth=$(GEMX_spmvWidth) \
            -D GEMX_spmvkVectorBlocks=$(GEMX_spmvkVectorBlocks) \
            -D GEMX_spmvMacGroups=$(GEMX_spmvMacGroups) \
            -D GEMX_spmvColAddIdxBits=$(GEMX_spmvColAddIdxBits) \
            -D GEMX_spmvPadA=$(GEMX_spmvPadA) \
            -D GEMX_spmvNumCblocks=$(GEMX_spmvNumCblocks) \
            -D GEMX_spmvFloatPerDesc=$(GEMX_spmvFloatPerDesc) \
            -D GEMX_idxType=${GEMX_idxType} \
            -D GEMX_nnzBlocks=${GEMX_nnzBlocks} \
            -D GEMX_spmvKmaxBlocks=${GEMX_spmvKmaxBlocks} \
            -D GEMX_spmvMmaxBlocks=${GEMX_spmvMmaxBlocks} \
            -D GEMX_spmvUramGroups=${GEMX_spmvUramGroups} \
            -D GEMX_useURAM=${GEMX_useURAM}
endif

ifeq (${GEMX_runUspmv}, 1)
CFLAGS_K += -D GEMX_uspmvStages=${GEMX_uspmvStages} \
            -D GEMX_XdataType=$(GEMX_XdataType) \
            -D GEMX_idxType=${GEMX_idxType} \
            -D GEMX_uspmvInterleaves=${GEMX_uspmvInterleaves} \
            -D GEMX_uspmvMvectorBlocks=${GEMX_uspmvMvectorBlocks} \
            -D GEMX_uspmvNnzVectorBlocks=${GEMX_uspmvNnzVectorBlocks}
endif

ifeq (${GEMX_runTransp}, 1)
CFLAGS_K += -D GEMX_transpBlocks=$(GEMX_transpBlocks) \
            -D GEMX_gemvmGroups=$(GEMX_gemvmGroups)      
endif

CFLAGS_K += $(COMMON_CFLAGS_K) \
          -Wno-ignored-attributes

           
CLCC_OPT += --xp prop:solution.hls_pre_tcl=hls_config.tcl

###############################################################################

API_GEMM_EXE = ${OUT_HOST_DIR}/gemx_api_gemm.exe
API_SPMV_EXE = ${OUT_HOST_DIR}/gemx_api_spmv.exe
API_GEMM_MULTI_INSTR_EXE = ${OUT_HOST_DIR}/gemx_api_gemm_multiInstr.exe
API_FCN_MULTI_INSTR_EXE = ${OUT_HOST_DIR}/gemx_api_fcn_multiInstr.exe

###############################################################################

#all: host run_sw_em run_hw_em run_hw
all: gemx_func_test

host_python_lib: 
	+make clean all -C MLsuite_MLP/C++

gemx_func_test: host
	+make SDA_FLOW=sw_emu run_em_int  2>&1 | tee log-run_sw_emu.txt

gemm_test_python: host_python_lib
	+make SDA_FLOW=hw run_hw_int  2>&1 | tee log-run_hw.txt; test -f ${MAKE_EXIT_OK_HW_FILE}

run_apiGemm_sw_em: api_gemm 
	+make SDA_FLOW=sw_emu run_apiGemm_em_int  2>&1 | tee log-run_apiGemm_sw_em.txt

run_apiGemm_hw_em: api_gemm 
	+make SDA_FLOW=hw_emu run_apiGemm_em_int  2>&1 | tee log-run_apiGemm_hw_em.txt

run_apiGemm_em_int: xconfig api_gemm xbin
	@echo INFO: kernel xclbin frequency is $(shell ${XCLBIN_FREQ} ${XCLBIN}) MHz
	XCL_EMULATION_MODE=${SDA_FLOW} XILINX_OPENCL=${XILINX_SDX} ${API_GEMM_EXE} ${XCLBIN} ${GEN_BIN_PROGRAM}

api_gemm : ${API_GEMM_EXE}
api_spmv : ${API_SPMV_EXE}

${HOST_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile host executable *****"
	@echo "INFO: HOST INCLUDE IS " ${HOST_INCLUDE_CFLAGS}
	@echo "INFO: HOST LIB IS" ${HOST_LIB_LFLAGS}
	@echo "***********************************"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} src/host/gemx_main.cpp src/host/xcl2/xcl2.cpp -o $@ 

${GEN_BIN_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile testcase generator executable *****"
	@echo "INFO: HOST INCLUDE IS " ${HOST_INCLUDE_CFLAGS}
	@echo "INFO: HOST LIB IS" ${HOST_LIB_LFLAGS}
	@echo "*************************************************"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} -fdata-sections -ffunction-sections -Wl,--gc-sections src/host/gemx_gen_bin.cpp -o $@

# API examples
 
${API_GEMM_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile testcase generator executable *****"
	@echo "INFO: HOST INCLUDE IS " ${HOST_INCLUDE_CFLAGS}
	@echo "INFO: HOST LIB IS" ${HOST_LIB_LFLAGS}
	@echo "*************************************************"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} src/gemx_api_gemm.cpp src/host/xcl2/xcl2.cpp -o $@

${API_SPMV_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile testcase generator executable *****"
	@echo "INFO: HOST INCLUDE IS " ${HOST_INCLUDE_CFLAGS}
	@echo "INFO: HOST LIB IS" ${HOST_LIB_LFLAGS}
	@echo "*************************************************"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} src/gemx_api_spmv.cpp src/host/xcl2/xcl2.cpp -o $@

${API_GEMM_MULTI_INSTR_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile testcase generator executable *****"
	@echo "INFO: HOST INCLUDE IS " ${HOST_INCLUDE_CFLAGS}
	@echo "INFO: HOST LIB IS" ${HOST_LIB_LFLAGS}
	@echo "*************************************************"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} src/gemx_api_gemm_multiInstr.cpp src/host/xcl2/xcl2.cpp -o $@
	
${API_FCN_MULTI_INSTR_EXE} : ./src/* | ${OUT_HOST_DIR}
	@echo "***** Compile testcase generator executable *****"
	@echo "INFO: HOST INCLUDE IS " ${HOST_INCLUDE_CFLAGS}
	@echo "INFO: HOST LIB IS" ${HOST_LIB_LFLAGS}
	@echo "*************************************************"
	${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} src/gemx_api_fcn_multiInstr.cpp src/host/xcl2/xcl2.cpp -o $@
