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

current_dir := $(shell pwd)
GEMX_SRC := ./src/gemx_host_c_api.cpp
XCL2_SRC :=./src/xcl2/xcl2.cpp
OPENCL_LIB=${XILINX_XRT}/lib
OPENCL_INC=${XILINX_XRT}/include
ifndef XILINX_XRT
$(error XILINX_XRT not defined)
endif
GEMX_OUT := libgemxhost.so

CXX := g++

GEMX_OBJS := $(addprefix objs/,$(addsuffix .o,$(basename $(GEMX_SRC))))
XCL2_OBJ = objs/xcl2.o
GEMX_INCLUDE := -I./src -I$(OPENCL_INC)
GEMX_DEF = -DCL_VERSION_1_2 
GEMX_LIBDIR = -L$(OPENCL_LIB)
GEMX_CXXFLAGS = -O3 -std=c++11 -fPIC -Wextra -Wall -Wno-ignored-attributes -Wno-unused-parameter -Wno-unused-variable
GEMX_LDFLAGS = -fPIC -Wl,--rpath=$(OPENCL_LIB)
              
GEMX_LIB = -lxilinxopencl -lz -lstdc++ -lrt -pthread 
                             
.PHONY: all

all: $(GEMX_OUT)
	@echo Done: Built shared lib successfully.

objs/%.o: %.cpp
	@mkdir -p objs/$(dir $<)
	$(CXX) -shared $(GEMX_INCLUDE) $(iGEMX_DEF) $(GEMX_CXXFLAGS) -c -o $@ $<

###Rule to build object file from xcl2.cpp
$(XCL2_OBJ): $(XCL2_SRC)
	$(CXX) $(GEMX_INCLUDE) $(GEMX_DEF) $(GEMX_CXXFLAGS) -c -o $@ $<

$(GEMX_OUT): $(GEMX_OBJS) $(XCL2_OBJ)
	mkdir -p lib
	$(CXX) -shared -o lib/$@ $(XCL2_OBJ) $< $(GEMX_LIBDIR) $(GEMX_LDFLAGS) $(GEMX_LIB) 
	chmod a+rx lib

clean:
	rm -rf objs lib
	

