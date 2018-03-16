FROM xilinxatg/sdx_dev:2017.4

ADD /gemx /data/gemx
ADD /boost /data/boost
WORKDIR /data/gemx
RUN make out_host/gemx_api_gemm.exe GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4
#RUN cp scripts/start ./ 
#RUN chmod +x start
#RUN popd
