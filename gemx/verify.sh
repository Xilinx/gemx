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

#!/bin/bash

echo "Please make sure SDx 2019.1 environment is setup properly before running this script. For 2019.1, only vcu1525 platform is supported."
echo "Please enter the build process (sw_em, hw_em or hw):"
read build_proc
echo "Please enter the engine name (gemm, spmv, or fcn):"
read engine_name

function download(){
	fileName=$(basename $1)
	pathName=download/${fileName}

	if [ ! -f $pathName ]; then
		wget -P download $1
		tar -C download -xzvf $pathName
	fi
}

download https://sparse.tamu.edu/MM/Yoshiyasu/image_interp.tar.gz
download https://sparse.tamu.edu/MM/GHS_indef/mario001.tar.gz
download https://sparse.tamu.edu/MM/GHS_indef/dawson5.tar.gz
download https://sparse.tamu.edu/MM/HB/bcsstk16.tar.gz
download https://sparse.tamu.edu/MM/Garon/garon2.tar.gz
download https://sparse.tamu.edu/MM/Watson/chem_master1.tar.gz
download https://sparse.tamu.edu/MM/Schenk_IBMNA/c-67.tar.gz

if [ $engine_name = "gemm" ]; then
	#verify gemm
	echo "Verifying GEMM ..."
	make clean
	make run_${build_proc} GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=u200 GEN_BIN_PROGRAM="gemm 512 512 512  512 512 512 512 1 0 A05 B05 C05 X05  gemm 1024 1024 1024  1024 1024 1024 1024 1 0 A1k B1k C1k X1K   gemm 1024 1024 1024  1536 2048 2560 1024 1 0 A1kld B1kld C1kld X1kld"

elif [ $engine_name = "spmv" ]; then
	#verify spmv
	echo "Verifying SPMV ..."
	make clean
	make run_${build_proc} GEMX_ddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=0 GEMX_runTransp=0 GEMX_runSpmv=1 GEMX_dataType=float GEMX_part=u200 GEN_BIN_PROGRAM="spmv 96 128 256 none A0 B0 C0 false spmv 0 0 0 download/image_interp/image_interp.mtx A1 B1 C1 false spmv 0 0 0 download/mario001/mario001.mtx A2 B2 C2 false spmv 0 0 0 download/dawson5/dawson5.mtx A3 B3 C3 false spmv 0 0 0 download/bcsstk16/bcsstk16.mtx A4 B4 C4 false "

elif [ $engine_name = "fcn" ]; then
	#verify fcn
	echo "Verifying FCN ..."
	make clean
	make run_${build_proc} GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1	GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4	GEMX_runFcn=1 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=u200 GEN_BIN_PROGRAM=" fcn 512 512 128 512 128 128 128 1 0 1 0 A0 B0 C0 Bias0 fcn 128 128 128 128 128 128 128 1 0 1 0 A1 B1 C1 Bias1"

else
	echo "Invalid engine name, please enter gemm, spmv or fcn."
fi
echo "Finished verification."
