FROM xilinxatg/sdx:2017.4

#WORKDIR /gemx
RUN ls /
RUN find / -type d -name 'gemx'
RUN find / -type d -name 'fcn'
#RUN pushd /tmp/gemx
#RUN pwd
#RUN make out_host/gemx_api_gemm.exe GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4
#RUN cp scripts/start ./ 
#RUN chmod +x start
#RUN popd
