/**********
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
* **********/
#ifndef _XHOST_H_
#define _XHOST_H_
#include "assert.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include "gemx_util.h"
#include "xcl2/xcl2.hpp"

//#define GEMX_PERF_DBG
using namespace std;
namespace gemx
{
    typedef enum
    {
        OpControl, OpGemv, OpGemm, OpTransp, OpSpmv, OpUspmv, OpResult, OpFail, OpFcn
    } OpType;

    class kArgs {

        public:
            virtual ~kArgs() {
            }
            virtual size_t sizeInBytes() = 0;
            virtual char* asByteArray() = 0;
    };

    //Base address will be the instruction memory region
    class XStream 
    {
        private:
            cl::Kernel m_Kernel;


            vector<cl::Event>   _waitInput;//m_Mem2FpgaEvents;
            vector<cl::Event>   _waitOutput;//m_ExeKernelEvents;
        public:
            cl::Context m_Context;
            cl::CommandQueue m_CommandQueue;
            cl::Device m_Device;

            XStream() = delete;
            XStream(const string &xclbin, const string & kernelName)
            {
                const char* l_kernelName = kernelName.c_str();

                vector<cl::Device> l_devices = xcl::get_xil_devices();
                cl::Device l_device = l_devices[0];
                string l_deviceName = l_device.getInfo<CL_DEVICE_NAME>();
                cout << "INFO: device name is: " << l_deviceName << endl;
                cl_int l_err= CL_SUCCESS;
                // Create the OpenCL context, cmmandQueue and program 
                cl::Context l_context(l_device);
                m_Context = l_context;
                m_Device=l_device;
                cl::CommandQueue l_cmdQueue(m_Context, l_device,  CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &l_err);
                if (l_err != CL_SUCCESS) {
                    exit(EXIT_FAILURE);
                }
                m_CommandQueue = l_cmdQueue;
                vector<cl::Device> temp_devices;
                temp_devices.push_back(m_Device);
                cl::Program::Binaries l_bins = xcl::import_binary_file(xclbin);
                static cl::Program l_program(m_Context, temp_devices, l_bins, NULL, &l_err);
                if (l_err != CL_SUCCESS) {
                    exit(EXIT_FAILURE);
                }
                m_Kernel = move(cl::Kernel(l_program, l_kernelName));
            }

            ~XStream() 
            {
            }

            cl::Device getDevice() {
                return m_Device;
            }

            cl::Buffer createBuf(void *ptr, size_t sz_bytes)
            {
                return cl::Buffer(m_Context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,sz_bytes,ptr);
            }

            bool copyToFpga(const cl::Buffer & buf, bool sync_send)
            {
                cl::Event l_event;
                vector<cl::Memory> l_buff;
                l_buff.push_back(buf);
                // Send the input data to the accelerator
                m_CommandQueue.enqueueMigrateMemObjects(l_buff,0,NULL,&l_event);
                if (sync_send)
                {
                    l_event.wait();
                } else
                {
                    _waitInput.push_back(l_event);
                }
                return true;
            }

            cl::Buffer copyToFpga(void * buf, size_t sz_bytes,
                    bool sync_send = false)
            {
                cl::Buffer cl_buf = createBuf(buf, sz_bytes);
                copyToFpga(cl_buf, sync_send);
                return cl_buf;
            }

            void copyFromFpga(const cl::Buffer & buf, bool sync_exec = true)
            {
                //cout << "copyFromFPGA" << endl;
                XTimer t;
                cl::Event l_readEvents;

                m_CommandQueue.enqueueMigrateMemObjects({buf},CL_MIGRATE_MEM_OBJECT_HOST,&_waitOutput,&l_readEvents);
                if ( sync_exec ){
                    l_readEvents.wait();
                    _waitOutput.clear();
                } else{
                    _waitOutput.push_back(l_readEvents);
                }
#ifdef GEMX_PERF_DBG
                cout << "copyFromFpga: " << t.elapsed() << endl;
#endif
            }
            void execKernel(const cl::Buffer & instr_buf, bool sync_exec = true )
            {
                // Launch kernels
                m_Kernel.setArg(0,instr_buf);
                m_Kernel.setArg(1,instr_buf);

                XTimer t;
                cl::Event l_event;
                m_CommandQueue.enqueueTask(m_Kernel, &(_waitInput),&l_event);

                if ( sync_exec ) {
                    l_event.wait();
                } else{
                    _waitOutput.push_back(l_event);
                }
                _waitInput.clear();
#ifdef GEMX_PERF_DBG
                cout << "execKernel: " << t.elapsed() << endl;
#endif

            }

            void wait ()
            {
                m_CommandQueue.finish();
                _waitInput.clear();
                _waitOutput.clear();
            }
    };

    template<typename HType>
        class XHost
        {
            public:
                XHost() = delete;

                XHost ( const string & xclbin, const string & kernelName)
                {
                    _fpga_stream = shared_ptr<XStream>(new XStream(xclbin, kernelName));
                    void *aligned_mem = nullptr;
                    int mem_alloc_status;
                    mem_alloc_status=posix_memalign(&aligned_mem, PAGE_SIZE, INSTR_BUF_SIZE);
                    cout<<"The posix mem alloc returned value::"<<mem_alloc_status<<"\n";
                    assert(!mem_alloc_status);
                    _instrBuf = (char*) aligned_mem;
                    _progBuf = (char*)aligned_mem;
                    cout<<"@step ... 1\n";
                    memset(_instrBuf, 0, INSTR_BUF_SIZE);

                    cout<<"@step ... 2\n";
                    _instr_offset = 0;
                    cout<<"Stating copying instruction to FPGA.......\n";
                    this->_cl_instr_buf = this->_fpga_stream->copyToFpga(_instrBuf, INSTR_BUF_SIZE+KERN_DBG_BUF_SIZE,
                            true);

                    cout<<"Done copying instruction to FPGA\n";
                    xclGetMemObjDeviceAddress(this->_cl_instr_buf.get(),
                            XHost<HType>::_fpga_stream->m_Device.get(),
                            sizeof(unsigned long long), &this->_ddrDeviceBaseAddr);
                }

                bool AllocProgBuf(unsigned int buf_sz){
                    bool l_res = true;
                    if (_progBuf != nullptr) {
                        free(_progBuf);
                    }
                    _progBuf =(char*)aligned_alloc(PAGE_SIZE, buf_sz);
                    assert(_progBuf != nullptr);
                    memset(_progBuf, 0, INSTR_BUF_SIZE);
                    _instr_offset = 0;

                    _cl_prog_buf = this->_fpga_stream->createBuf(_progBuf, buf_sz);
                    _allocated_pages = 2;
                    _total_prog_pages = buf_sz / PAGE_SIZE;

                    cl_buffer_region l_region;
                    l_region.origin=0;
                    l_region.size = PAGE_SIZE;

                    cl_int l_err;
                    _cl_instr_buf = _cl_prog_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &l_region, &l_err);
                    if (l_err != CL_SUCCESS) {
                        cerr << "ERROR: failed to create instr sub buffer" << endl;
                        l_res = false;
                    }
                    l_region.origin = PAGE_SIZE;
                    _cl_stats_buf = _cl_prog_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &l_region, &l_err);
                    if (l_err != CL_SUCCESS) {
                        cerr << "ERROR: failed to create instr sub buffer" << endl;
                        l_res = false;
                    }
                    return l_res;
                }

                virtual ~XHost()
                {
                    if (_progBuf != nullptr) {
                        free(_progBuf);
                    }
                }

                const cl::Program* loadxclbin (const string & xclbin)
                {

                    vector<cl::Device> l_devices;
                    l_devices.push_back(_fpga_stream->m_Device);
                    cl::Program::Binaries l_bins = xcl::import_binary_file(xclbin);
                    cl_int l_err;
                    static cl::Program l_program(_fpga_stream->m_Context, l_devices, l_bins, NULL, &l_err);
                    if (l_err != CL_SUCCESS) {
                        exit(EXIT_FAILURE);
                    }
                    return &l_program;
                }

                virtual void Execute( bool sync_exec = true) = 0;

                bool AddMat(const HType & handle, void * mat_ptr, unsigned long long buf_sz) {
                    auto &h = _hostMat;   //auto: type inferred by the compiler
                    auto &hz = _hostMatSz;
                    if (h.find(handle) == h.end()) {
                        h[handle] = mat_ptr;
                        hz[handle] = buf_sz;
                        return true;
                    }
                    else if (hz[handle] != buf_sz ){
                        h[handle] = mat_ptr;
                        hz[handle] = buf_sz;
                        this->_devHandle.erase(handle);
                        return true;
                    }
                    return false;
                }

                void * GetMat(const HType & handle,
                        bool queryFPGA = false, bool sync_get = true)
                {
                    auto& h = _hostMat;
                    void * ret_ptr = nullptr;
                    if (h.find(handle) != h.end()) {
                        if (queryFPGA)
                            GetFromFPGA(handle, sync_get);
                        ret_ptr = h[handle];
                    }
                    return ret_ptr;
                }

                void Wait()
                {
                    _fpga_stream->wait();
                }

                void SendToFPGA(const HType & handle, void * mat_ptr, unsigned long long buf_sz,
                        bool sync_send = false) {
                    AddMat(handle, mat_ptr, buf_sz);
                    SendToFPGA(handle, sync_send);
                }

                void SendToFPGA(const HType & handle, bool sync_send = false) {
                    XTimer t;
                    auto &h = _hostMat;
                    auto &d = _devHandle;
                    assert(h.find(handle) != h.end());

                    if (d.find(handle) != d.end()) {
                        _fpga_stream->copyToFpga(d[handle], sync_send);
                    } else {
                        d[handle] = _fpga_stream->copyToFpga(h[handle], _hostMatSz[handle], sync_send);
                    }
                    #ifdef GEMX_PERF_DBG
                    cout << "SendToFPGA: " << t.elapsed() << endl;
                    #endif
                }

                void GetFromFPGA(const HType & handle, bool sync_get) {
                    XTimer t;
                    auto &d = _devHandle;
                    assert(d.find(handle) != d.end());
                    _fpga_stream->copyFromFpga(d[handle], sync_get);
                    #ifdef GEMX_PERF_DBG
                    cout << "GetFromFPGA: " << t.elapsed() << endl;
                    #endif
                }

                void* AddDevBuf(const HType & handle, unsigned long long buf_sz) {
                    auto &h = _hostMatPageOffset;   //auto: type inferred by the compiler
                    auto &hz = _hostMatSz;
                    auto &d = _devHandle;

                    unsigned int l_pages = (buf_sz + PAGE_SIZE-1) / PAGE_SIZE;
                    assert((_allocated_pages + l_pages)<=_total_prog_pages);


                    if (h.find(handle) == h.end()) {
                        h[handle] = _allocated_pages;
                        hz[handle] = l_pages * PAGE_SIZE;
                        cl_buffer_region l_region;
                        l_region.origin= _allocated_pages * PAGE_SIZE;
                        l_region.size = l_pages * PAGE_SIZE;

                        cl_int l_err;
                        d[handle] = _cl_prog_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &l_region, &l_err);
                        if (l_err != CL_SUCCESS) {
                            cerr << "ERROR: failed to create device buffer" << endl;
                        }
                        _allocated_pages += l_pages;
                    }
                    else if (hz[handle] < buf_sz ){
                        cerr << "ERROR: smaller bufer already allocated" << endl;
                    }
                    return &_progBuf[h[handle]*PAGE_SIZE];
                }


                void* GetDevBuf(const HType & handle, bool queryFPGA = false, bool sync_get = true)
                {
                    auto& h = _hostMatPageOffset;
                    auto &hz = _hostMatSz;
                    void* ret_ptr = nullptr;
                    if (h.find(handle) != h.end()) {
                        if (queryFPGA)
                            GetFromFPGA(handle, sync_get);
                        ret_ptr = &_progBuf[h[handle]*PAGE_SIZE];
                    }
                    return ret_ptr;
                }

                unsigned int GetMatOffset(const HType &handle) {
                    auto& h = _hostMatPageOffset;
                    assert(h.find(handle) != h.end());
                    return h[handle];
                }

                void SendDevBuf(const HType & handle, bool sync_send = false) {
                    XTimer t;
                    auto &h = _hostMatPageOffset;
                    auto &d = _devHandle;
                    assert(h.find(handle) != h.end());
                    assert(d.find(handle) != d.end());
                    _fpga_stream->copyToFpga(d[handle], sync_send);
                    #ifdef GEMX_PERF_DBG
                    cout << "SendToFPGA: " << t.elapsed() << endl;
                    #endif
                }

                void AddInstr  ( kArgs * args )
                {
                    char * instr = args->asByteArray();
                    char * curr_pos = &_progBuf[_instr_offset];
                    memcpy(curr_pos, instr, args->sizeInBytes());
                    _instr_offset += args->sizeInBytes();
                }

                void ClearInstrBuf()
                {
                    memset(this->_progBuf, 0, PAGE_SIZE);
                    this->_instr_offset = 0;
                }
            protected:
                static const unsigned int PAGE_SIZE = 4096;
                static const unsigned int INSTR_BUF_SIZE = PAGE_SIZE;
                static const unsigned int KERN_DBG_BUF_SIZE = PAGE_SIZE;
                unordered_map<HType, unsigned int> _hostMatPageOffset;
                unordered_map<HType, void*  > _hostMat;
                unordered_map<HType, unsigned long long > _hostMatSz;
                unordered_map<HType, cl::Buffer> _devHandle;
                shared_ptr<XStream> _fpga_stream;


                unsigned long long _ddrDeviceBaseAddr;
                char* _progBuf;
                char* _instrBuf;
                cl::Buffer _cl_prog_buf, _cl_instr_buf, _cl_stats_buf;
                unsigned int _allocated_pages;
                unsigned int _total_prog_pages;
                unsigned int _instr_offset;
        };


};


#endif
