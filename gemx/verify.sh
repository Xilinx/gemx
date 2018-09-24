#!/bin/bash

echo "Please make sure SDx 2018.2 environment is setup properly before running this script. For 2018.2, only vcu1525 platform is supported."
echo "Please enter the build process (sw_em, hw_em or hw):"
read build_proc
echo "Please enter the engine name (gemm, spmv, or fcn):"
read engine_name

if [ ! -d "download" ]; then
	mkdir download 
	pushd download
	echo "Downloading sparse matrices"
	wget https://sparse.tamu.edu/MM/Yoshiyasu/image_interp.tar.gz
	gunzip image_interp.tar.gz
	tar -xf image_interp.tar
	wget https://sparse.tamu.edu/MM/GHS_indef/mario001.tar.gz
	gunzip mario001.tar.gz
	tar -xf mario001.tar
	wget https://sparse.tamu.edu/MM/GHS_indef/dawson5.tar.gz
	gunzip dawson5.tar.gz
	tar -xf dawson5.tar
	wget https://sparse.tamu.edu/MM/HB/bcsstk16.tar.gz
	gunzip bcsstk16.tar.gz
	tar -xf bcsstk16.tar
	wget https://sparse.tamu.edu/MM/Garon/garon2.tar.gz
	gunzip garon2.tar.gz
	tar -xf garon2.tar
	wget https://sparse.tamu.edu/MM/Watson/chem_master1.tar.gz
	gunzip chem_master1.tar.gz
	tar -xf chem_master1.tar
	wget https://sparse.tamu.edu/MM/Schenk_IBMNA/c-67.tar.gz
	gunzip c-67.tar.gz
	tar -xf c-67.tar
	popd
fi

if [ $engine_name = "gemm" ]; then
	#verify gemm
	echo "Verifying GEMM ..."
	make clean
	make run_${build_proc} GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=vcu1525 GEN_BIN_PROGRAM="gemm 512 512 512  512 512 512 512 1 0 A05 B05 C05 X05  gemm 1024 1024 1024  1024 1024 1024 1024 1 0 A1k B1k C1k X1K   gemm 1024 1024 1024  1536 2048 2560 1024 1 0 A1kld B1kld C1kld X1kld"

elif [ $engine_name = "spmv" ]; then
	#verify spmv
	echo "Verifying SPMV ..."
	make clean
	make run_${build_proc} GEMX_ddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=0 GEMX_runTransp=0 GEMX_runSpmv=1 GEMX_dataType=float GEMX_part=vcu1525 GEN_BIN_PROGRAM="spmv 96 128 256 none A0 B0 C0 spmv 0 0 0 download/image_interp/image_interp.mtx A1 B1 C1 spmv 0 0 0 download/mario001/mario001.mtx A2 B2 C2 spmv 0 0 0 download/dawson5/dawson5.mtx A3 B3 C3 spmv 0 0 0 download/bcsstk16/bcsstk16.mtx A4 B4 C4"

elif [ $engine_name = "fcn" ]; then
	#verify fcn
	echo "Verifying FCN ..."
	pushd fcn
	make clean
	make run_${build_proc} GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=vcu1525 GEN_BIN_PROGRAM=" fcn 512 512 128 512 128 128 128 1 0 1 0 A0 B0 C0 Bias0 fcn 128 128 128 128 128 128 128 1 0 1 0 A1 B1 C1 Bias1"
	popd fcn

else
	echo "Invalid engine name, please enter gemm, spmv or fcn."
fi
echo "Finished verification."
