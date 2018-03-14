FROM xilinxatg/sdx:2017.4

WORKDIR /gemx

RUN make out_host/gemx_api_gemm.exe GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4
RUN cp /gemx/scripts/start /gemx
RUN chmod +x /gemx/start

