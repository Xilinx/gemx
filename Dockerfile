#FROM xilinxatg/sdx:2017.4
FROM xilinxatg/sdx_dev

ADD /gemx /data/gemx
ADD /boost /data/boost
WORKDIR /data/gemx
RUN make out_host/gemx_api_gemm.exe GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4
#RUN cp scripts/start ./ 
#RUN chmod +x start
#RUN popd
