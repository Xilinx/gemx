#script to run the sprite regression suite tests

#
echo "=============================="
echo "Launching regressions........."
#

#set environments
export XILINX_SDX=/proj/xbuilds/2017.1_sdx_daily_latest/installs/lin64/SDx/2017.1
source $XILINX_SDX/settings64.sh
export XCL_EMULATION_MODE=true

rm -f -rf out_cpu_emu log-run_cpu_emu.txt
rm -f -rf out_hw_emu log-run_hw_emu.txt
rm -f -rf out_hw log-run_hw.txt
rm -f -rf out_host sdaccel_profile* .Xil

#build kernels for hw emu
export s=32

#build a multi-kernels xclbin with GEMM engine in it
make run_multiGemm_hw_em SDA_FLOW=hw_emu GEMX_ddrWidth=$s GEMX_argInstrWidth=`expr 32 / $s` GEMX_gemmMBlocks=8 GEMX_gemmKBlocks=8 GEMX_gemmNBlocks=8 GEMX_numKernels=4 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_part=vu9pf1 GEMX_kernelHlsFreq=250 GEMX_vivadoFlow=EXP GEMX_kernelVivadoFreq=300 GEMX_useURAM=1 GEN_BIN_PROGRAM="gemm 512 512 512  512 512 512  A B C"

#build a GEMM API for multi instructions
make GEMX_ddrWidth=$s GEMX_argInstrWidth=`expr 32/$s` GEMX_gemmMBlocks=8 GEMX_gemmKBlocks=8 GEMX_gemmNBlocks=8 GEMX_numKernels=4 out_host/gemx_api_gemm_multiInstr.exe

GEMX_HOST_DIR=out_host
GEMX_HW_DIR=out_hw_emu

echo "test hw emu"
nice ${GEMX_HOST_DIR}/gemx_api_gemm.exe ${GEMX_HW_DIR}/gemx.xclbin 512 512 512 
nice ${GEMX_HOST_DIR}/gemx_api_gemm_multiInstr.exe ${GEMX_HW_DIR}/gemx.xclbin 512 512 512 512 512 512 A0 B0 C0 512 512 512 512 512 512 C0 B1 C1

echo "Launching regressions.....DONE"
echo "=============================="
