#script to run the sprite regression suite tests

#
echo "=============================="
echo "Launching regressions........."

#set environment
export XILINX_SDX=/proj/xbuilds/2017.1_sdx_daily_latest/installs/lin64/SDx/2017.1
source $XILINX_SDX/settings64.sh
export XCL_EMULATION_MODE=true

rm -f -rf out_cpu_emu log-run_cpu_emu.txt
rm -f -rf out_hw_emu log-run_hw_emu.txt
rm -f -rf out_hw log-run_hw.txt
rm -f -rf out_host sdaccel_profile* .Xil

#build kernel for cpu emu
export s=32

#build a one-kernel xclbin with GEMM engine in it
make run_cpu_em SDA_FLOW=cpu_emu GEMX_ddrWidth=$s GEMX_argInstrWidth=`expr 32 / $s` GEMX_gemmMeshRows=$s GEMX_gemmMeshCols=$s GEMX_gemmMeshDepth=$s GEMX_gemmMBlocks=8 GEMX_gemmKBlocks=8 GEMX_gemmNBlocks=8 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0  GEMX_part=vu9pf1 GEMX_kernelHlsFreq=250 GEMX_kernelVivadoFreq=300 GEMX_useURAM=1 GEMX_vivadoFlow=EXP GEN_BIN_PROGRAM="gemm 512 512 512  512 512 512  A B C gemm 1024 1024 1024  1024 1024 1024  A1 B1 C1"

GEMX_HOST_DIR=out_host
GEMX_CPU_DIR=out_cpu_emu

#launch the application
echo "test cpu emu" 
nice ${GEMX_HOST_DIR}/gemx_host.exe  ${GEMX_CPU_DIR}/gemx.xclbin  ${GEMX_HOST_DIR}/app.bin  ${GEMX_CPU_DIR}/app_out.bin
nice ${GEMX_HOST_DIR}/gemx_gen_bin.exe -read ${GEMX_CPU_DIR}/app_out.bin > ${GEMX_CPU_DIR}/app_out.txt
echo INFO: Board performance data
head -22 ${GEMX_CPU_DIR}/app_out.txt
echo INFO: Comparing app_gold.bin app_out.bin
cmp -i 8192 ${GEMX_HOST_DIR}/app_gold.bin ${GEMX_CPU_DIR}/app_out.bin || ${GEMX_HOST_DIR}/gemx_gen_bin.exe -compare 1e-3 3e-6  ${GEMX_HOST_DIR}/app_gold.bin ${GEMX_CPU_DIR}/app_out.bin && echo INFO: Host Testbench ended Correctness test Status PASS

echo "Launching regressions.....DONE"
echo "=============================="
