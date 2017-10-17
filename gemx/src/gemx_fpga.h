/**********
Copyright (c) 2017, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
/**
 *  @brief FPGA utilities
 *
 */

#ifndef GEMX_FPGA_H
#define GEMX_FPGA_H

#include "assert.h"
#include "gemx_kernel.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include "CL/cl.h"
#include "CL/cl_ext.h"
#include <boost/compute.hpp>
#include <iostream>
#include <iterator>


namespace gemx {

class Fpga
{
  private:
    std::string  m_XclbinFile;

    boost::compute::context        m_Context;
    boost::compute::command_queue  m_CommandQueue;
    boost::compute::program        m_Program;
    boost::compute::kernel         m_Kernel[GEMX_numKernels];
    boost::compute::buffer         m_Buffer[GEMX_numKernels];
	boost::compute::wait_list	   m_waitInput[GEMX_numKernels];
	boost::compute::wait_list	   m_waitOutput[GEMX_numKernels];
    size_t                         m_DataSize[GEMX_numKernels];
	unsigned int                   m_KernelId;
  private:

    ///////////////////////////////////////////////////////////////////////////
    std::vector<char>
    load_file_to_memory(std::string p_FileName) { 
        std::vector<char> l_vec;
        std::ifstream l_file(p_FileName.c_str());
        if (l_file.is_open() && !l_file.eof() && !l_file.fail()) {
          l_file.seekg(0, std::ios_base::end);
          std::streampos l_fileSize = l_file.tellg();
          l_vec.resize(l_fileSize);
          l_file.seekg(0, std::ios_base::beg);
          l_file.read(&l_vec[0], l_fileSize);
          if (!l_file) {
            l_vec.clear();
          }
        }
        return(l_vec);
      }


  public:
    Fpga()
      {}

    Fpga(unsigned int p_KernelId)
      : m_KernelId(p_KernelId)
      {}
    ///////////////////////////////////////////////////////////////////////////
    bool
    loadXclbin(std::string p_XclbinFile, std::string p_KernelName[GEMX_numKernels]) {
        bool ok = false;
        // https://gitenterprise.xilinx.com/rkeryell/heterogeneous_examples/blob/master/vector_add/SDAccel-Boost.Compute/vector_add.cpp
        
        // Create the OpenCL context to attach resources on the device
        m_Context = std::move(boost::compute::system::default_context());
        // Create the OpenCL command queue to control the device
        //m_CommandQueue = std::move(boost::compute::system::default_queue());
 		//boost::compute::command_queue queue(boost::compute::system::default_context(), boost::compute::system::default_device(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
 		boost::compute::command_queue queue(boost::compute::system::default_context(), boost::compute::system::default_device(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		m_CommandQueue = std::move(queue);
        // Construct an OpenCL program from the precompiled kernel file
        m_Program = std::move(
          boost::compute::program::create_with_binary_file(p_XclbinFile,
                                                           m_Context));
        m_Program.build();

        for (int i=0; i<GEMX_numKernels; ++i) {
			m_Kernel[i] = std::move(boost::compute::kernel(m_Program, p_KernelName[i]));
        }
        ok = true;
        return(ok);
      }


    ///////////////////////////////////////////////////////////////////////////
	bool createBuffers(MemDesc p_MemDesc[GEMX_numKernels]) {
        bool ok = false;
        
        //decltype of cl_mem_ext_ptr_t.flags
        unsigned l_k2bank[] = {GEMX_fpgaDdrBanks};
        
        cl_mem_ext_ptr_t l_bufExt;
        l_bufExt.obj = NULL;
        l_bufExt.param = 0;
		for (unsigned int kernelId=0; kernelId<GEMX_numKernels; ++kernelId){
        	l_bufExt.flags = l_k2bank[kernelId];
        
        	m_DataSize[kernelId] = p_MemDesc[kernelId].sizeBytes();
        	// Buffers
        	m_Buffer[kernelId] = boost::compute::buffer(m_Context, m_DataSize[kernelId],
                      CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,
                      &l_bufExt);
		}
		ok = true;
		return(ok);
	}	
		
    ///////////////////////////////////////////////////////////////////////////
    bool
    copyToFpga(MemDesc p_MemDesc[GEMX_numKernels]) {
        bool ok = false;
        
        //decltype of cl_mem_ext_ptr_t.flags
        //unsigned l_k2bank[] = {GEMX_fpgaDdrBanks};
        
        //cl_mem_ext_ptr_t l_bufExt;
        //l_bufExt.obj = NULL;
        //l_bufExt.param = 0;
        boost::compute::event l_event;
		for (unsigned int kernelId=0; kernelId<GEMX_numKernels; ++kernelId){
        	//l_bufExt.flags = l_k2bank[kernelId];
        
        	//m_DataSize[kernelId] = p_MemDesc[kernelId].sizeBytes();
        	// Buffers
        	//m_Buffer[kernelId] = boost::compute::buffer(m_Context, m_DataSize[kernelId],
            //          CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,
            //          &l_bufExt);

        	// Send the input data to the accelerator
       		l_event = m_CommandQueue.enqueue_write_buffer(m_Buffer[kernelId], 0 /* Offset */,
                                            m_DataSize[kernelId], p_MemDesc[kernelId].data());
			m_waitInput[kernelId].insert(l_event);
        	m_Kernel[kernelId].set_args(m_Buffer[kernelId], m_Buffer[kernelId]);
		}
//        for (int i=0; i<GEMX_numKernels; ++i) {
//			l_writeEvents[i].wait();
//		}
		ok = true;
        return(ok);
      }

    ///////////////////////////////////////////////////////////////////////////
    bool
    callKernels() {
        bool ok = false;
        
        boost::compute::extents<1> offset { 0 };
        boost::compute::extents<1> global { 1 };
        // Use only 1 CU
        boost::compute::extents<1> local { 1 };
        // Launch kernels
        boost::compute::event l_event;
        for (unsigned int kernelId=0; kernelId<GEMX_numKernels; ++kernelId){
        	l_event = m_CommandQueue.enqueue_nd_range_kernel(m_Kernel[kernelId], offset, global, local, m_waitInput[kernelId]);
			m_waitOutput[kernelId].insert(l_event);
		}
        ok = true;
        return(ok);
      }
    
    ///////////////////////////////////////////////////////////////////////////
    bool
    copyFromFpga(MemDesc p_MemDesc[GEMX_numKernels]) {
        bool ok = false;
		bool success = true;
       
		boost::compute::event l_readEvents[GEMX_numKernels]; 
        for (unsigned int kernelId=0; kernelId<GEMX_numKernels; ++kernelId) {
        	assert(p_MemDesc[kernelId].sizeBytes() == m_DataSize[kernelId]);
        	// Get the output data from the accelerator
        	l_readEvents[kernelId] = m_CommandQueue.enqueue_read_buffer_async(m_Buffer[kernelId], 0 /* Offset */,
                                         m_DataSize[kernelId], p_MemDesc[kernelId].data(), m_waitOutput[kernelId]);
        	ok = (p_MemDesc[kernelId].sizePages() > 0);
			if (!ok) {
				success = false;
			}
		}
		for (int i=0; i<GEMX_numKernels; ++i) {
			l_readEvents[i].wait();
		}
        return(success);
      }
    ///////////////////////////////////////////////////////////////////////////
    bool
    loadXclbinSingleKernel(std::string p_XclbinFile, std::string p_KernelName) {
        bool ok = false;
        // https://gitenterprise.xilinx.com/rkeryell/heterogeneous_examples/blob/master/vector_add/SDAccel-Boost.Compute/vector_add.cpp
        
        // Create the OpenCL context to attach resources on the device
        m_Context = std::move(boost::compute::system::default_context());
        // Create the OpenCL command queue to control the device
        m_CommandQueue = std::move(boost::compute::system::default_queue());

        // Construct an OpenCL program from the precompiled kernel file
        m_Program = std::move(
          boost::compute::program::create_with_binary_file(p_XclbinFile,
                                                           m_Context));
        m_Program.build();

		m_Kernel[0] = std::move(boost::compute::kernel(m_Program, p_KernelName));
        ok = true;
        return(ok);
      }
    ///////////////////////////////////////////////////////////////////////////
	bool
    copyToFpgaSingleKernel(MemDesc &p_MemDesc) {
        bool ok = false;
        
        //decltype of cl_mem_ext_ptr_t.flags
        unsigned l_k2bank[] = {GEMX_fpgaDdrBanks};
        
        cl_mem_ext_ptr_t l_bufExt;
        l_bufExt.obj = NULL;
        l_bufExt.param = 0;
		l_bufExt.flags = l_k2bank[m_KernelId];
	
		m_DataSize[0] = p_MemDesc.sizeBytes();
		// Buffers
		m_Buffer[0] = boost::compute::buffer(m_Context, m_DataSize[0],
				  CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,
				  &l_bufExt);

		// Send the input data to the accelerator
		 m_CommandQueue.enqueue_write_buffer(m_Buffer[0], 0 /* Offset */,
										m_DataSize[0], p_MemDesc.data());
		m_Kernel[0].set_args(m_Buffer[0], m_Buffer[0]);
        ok = true;
        return(ok);
      }
    ///////////////////////////////////////////////////////////////////////////
    bool
    callSingleKernel(std::string p_KarnelName) {
        bool ok = false;
        
        boost::compute::extents<1> offset { 0 };
        boost::compute::extents<1> global { 1 };
        // Use only 1 CU
        boost::compute::extents<1> local { 1 };
        // Launch kernel
        m_CommandQueue.enqueue_nd_range_kernel(m_Kernel[0], offset, global, local);
        ok = true;
        return(ok);
      }
    ///////////////////////////////////////////////////////////////////////////
    bool
    copyFromFpgaSingleKernel(MemDesc &p_MemDesc) {
        bool ok = false;
        
		assert(p_MemDesc.sizeBytes() == m_DataSize[0]);
		// Get the output data from the accelerator
		m_CommandQueue.enqueue_read_buffer(m_Buffer[0], 0 /* Offset */,
									   m_DataSize[0], p_MemDesc.data());
		ok = (p_MemDesc.sizePages() > 0);
		
        return(ok);
    }
};

} // namespace

#endif
