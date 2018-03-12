/*
 * gemx_host.h
 *
 *  Created on: Jan 20, 2018
 *      Author: xteng
 */

#ifndef SRC_GEMX_HOST_H_
#define SRC_GEMX_HOST_H_

#include "assert.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include "CL/cl.h"
#include "CL/cl_ext.h"
#include <boost/compute.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/program.hpp>

#include <iostream>
#include <iterator>
#include <unordered_map>
#include <stdlib.h>
#include <cstring>
#include <iomanip>
//#define GEMX_PERF_DBG
using namespace std;
namespace gemx {
typedef enum {
    OpControl, OpGemv, OpGemm, OpTransp, OpSpmv, OpResult, OpFail, OpFcn
} OpType;

class XTimer
{
  public:
    XTimer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
      return std::chrono::duration_cast<second_>
        (clock_::now() - beg_).count(); }

  private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

class kArgs {
public:
    virtual ~kArgs() {
    }
    virtual size_t sizeInBytes() = 0;
    virtual char* asByteArray() = 0;
};

class SpMvArgs: public kArgs {
public:
    virtual ~SpMvArgs() {
    }
};

//////////////////////////// GEMM ////////////////////////////
class GemmArgs: public kArgs {
public:
    virtual ~GemmArgs() {
    }
    GemmArgs() = delete;
    GemmArgs(unsigned int p_Aoffset, unsigned int p_Boffset,
            unsigned int p_Coffset, unsigned int p_Xoffset, unsigned int p_M, unsigned int p_K,
            unsigned int p_N, unsigned int p_Lda, unsigned int p_Ldb,
            unsigned int p_Ldc, unsigned int p_Ldx, int post_scale, int post_shift) :
                m_gemm_args( { int(OpGemm),  p_Aoffset, p_Boffset, p_Coffset, p_Xoffset, p_M, p_K,
        p_N, p_Lda, p_Ldb, p_Ldc, p_Ldx, 0, 0, 0, 0 }) {	
        m_gemm_args.m_postScaleVal = (post_scale << 8) | (post_shift & 0x000000ff);
    }
    size_t sizeInBytes() {
        return sizeof(m_gemm_args);
    }
    char *asByteArray() {
        return reinterpret_cast<char*>(&m_gemm_args);
    }

protected:
    struct {
        int m_optype;
        unsigned int m_Aoffset, m_Boffset, m_Coffset, m_Xoffset, m_M, m_K, m_N,
        m_Lda, m_Ldb, m_Ldc, m_Ldx;
	int m_postScaleVal;
        int dummy[3];
    } m_gemm_args;
};

class FcnArgs: public kArgs {
public:
    virtual ~FcnArgs() {
    }
    FcnArgs() = delete;
    FcnArgs(unsigned int p_Aoffset, unsigned int p_Boffset,
            unsigned int p_Coffset, unsigned int p_Xoffset, unsigned int p_M, unsigned int p_K,
            unsigned int p_N, unsigned int p_Lda, unsigned int p_Ldb,
            unsigned int p_Ldc, unsigned int p_Ldx, int post_scale, int post_shift, short prelu_scale, short prelu_alpha) :
                m_fcn_args( { OpFcn, p_Aoffset, p_Boffset, p_Coffset, p_Xoffset, p_M, p_K,
        p_N, p_Lda, p_Ldb, p_Ldc, p_Ldx, 0, 0, 0, 0 }) {

        m_fcn_args.m_postScaleVal = (post_scale << 8) | (post_shift & 0x000000ff);
        m_fcn_args.m_PReLUVal = (prelu_scale << 6) | (prelu_alpha & 0x003f);

        //std::cout << "s_dummy: " << m_fcn_args.s_dummy << std::endl;
        //printf ("PReLUVal: %d\n", m_fcn_args.m_PReLUVal);
        //std::stringstream stream;
        //std::cout << "optype: " << optype << " p_Aoffset: " << p_Aoffset << std::endl;

        /*
         int * data = (int*)asByteArray();
         for (int i = 0; i < sizeInBytes()/4; i++){
             std::cout << "word " << i << ": " << data[i] << std::endl;
         }
         */

        //std::string result( stream.str() );
        //std::cout << "Hex: " << result << std::endl;
    }
    size_t sizeInBytes() {
        return sizeof(m_fcn_args);
    }
    char *asByteArray() {
        return reinterpret_cast<char*>(&m_fcn_args);
    }

protected:
    struct {
        int m_optype;
        unsigned int m_Aoffset, m_Boffset, m_Coffset, m_Xoffset, m_M, m_K, m_N,
        m_Lda, m_Ldb, m_Ldc, m_Ldx;
        int m_postScaleVal;
        short m_PReLUVal;
        short s_dummy;
        int dummy[2];
    } m_fcn_args;
};

// Matrix descriptor with data itself stored in caller's space
template<typename T>
class Mat {
private:
    unsigned int m_Rows, m_Cols, m_Ld, m_buf_sz;
    bool m_ownmem;
    T *m_Addr;
public:
    const static size_t GEMX_CMP_WIDTH = 11;
    Mat() = delete;
    ~Mat() {
        if (m_ownmem && m_Addr) {
            free(m_Addr);
        }
    }
    Mat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld) :
        m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_ownmem(true) {
        m_buf_sz = sizeof(T) * p_Rows * p_Ld;
        posix_memalign((void**) &m_Addr, 4096, m_buf_sz);
        //m_Addr = (T*)aligned_alloc ( 4096, sizeof(T) * p_Rows * p_Cols);
    }

    Mat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld, T *p_Addr) :
        m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_Addr(p_Addr), m_ownmem(
                false) {
        m_buf_sz = sizeof(T) * p_Rows * p_Ld;
    }
    Mat& operator=(const Mat& p_Src) {
        assert(p_Src.rows() == rows());
        assert(p_Src.cols() == cols());
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < m_Ld; ++col) {
                m_Addr[row][col] = p_Src.getVal(row, col);
            }
        }
        return *this;
    }

    unsigned int buf_sz(){
        return m_buf_sz;
    }
    T*& data() {
        return m_Addr;
    }

    inline T &getVal(unsigned int p_Row, unsigned int p_Col) {
        return m_Addr[p_Row * ld() + p_Col];
    }
    inline unsigned int rows() {
        return m_Rows;
    }
    inline unsigned int cols() {
        return m_Cols;
    }
    inline unsigned int ld() {
        return m_Ld;
    }

    void init(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld,
            T *p_Addr) {
        m_Rows = p_Rows;
        m_Cols = p_Cols;
        m_Ld = p_Ld;
        m_Addr = p_Addr;
    }

    void fillModRange(T p_Min, T p_Max) {
        T l_val = p_Min;
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < ld(); ++col) {
                getVal(row, col) = l_val++;
                if ( l_val > p_Max ) l_val = p_Min;
            }
        }
    }

    void fillMod(T p_Max, T p_First = 0) {
        T l_val = p_First;
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < ld(); ++col) {
                getVal(row, col) = l_val;
                l_val++;
                l_val %= p_Max;
            }
        }
    }

    void multiply(Mat & p_A, Mat & p_B) {
        T l_val = 0;
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                T l_val = 0;
                for (unsigned int k = 0; k < p_A.cols(); ++k) {
                    l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
                }
                //cout << "    DEBUG multiply setting row=" << row << " col=" << col << endl;
                getVal(row, col) = l_val;
            }
        }
    }
    
        void
    multiplyAddScale(Mat & p_A, Mat & p_B,  Mat<int> & p_X, int32_t p_postScale) {
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
				assert(p_X.rows() == rows());
				assert(p_X.cols() == cols());
        for (unsigned int row = 0; row < rows(); ++row) {
          for (unsigned int col = 0; col < cols(); ++col) {
            int64_t l_val = 0;
            for (unsigned int k = 0; k < p_A.cols(); ++k) {
              l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
            }
	    l_val += p_X.getVal(row, col);
	    l_val = (l_val >> (p_postScale & 0x00ff)) * (p_postScale >> 8);	
            T l_entry = (T)(l_val);
            getVal(row, col) = l_entry;
          }
        }
      }

    void matMultWithScaleAndPRelu(Mat & p_A, Mat & p_B, Mat<int> & p_X,  int32_t p_postScale, int16_t p_PReluVal) {
        cout << "A rows: " << p_A.rows() << " this rows: " << rows() << endl;
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        assert(p_X.rows() == rows());
        assert(p_X.cols() == cols());
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                int64_t l_val = 0;
                for (unsigned int k = 0; k < p_A.cols(); ++k) {
                    l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
                    //                          if ((row==2) && (col == 0)) {
                    //                                  if (p_B.getVal(k, col) != 0) {
                    //                                      std::cout << " A[2," << std::dec << k << "]= " << p_A.getVal(row, k) << std::hex << " 0x" << p_A.getVal(row, k);
                    //                                      std::cout << " B[" << std::dec << k << ",0]= " << p_B.getVal(k,col) <<  std::hex << " 0x" << p_B.getVal(k,col);
                    //                                      std::cout << " A*B+C = " << std::dec << l_val << std::hex << " 0x" << l_val << "\n";
                    //                                  }
                    //                          }
                }

                //                      if ((row == 2) && (col == 0)) {
                //                          std::bitset<64> l_bVal{l_val};
                //                          std::cout << "C[2,0]= " << l_bVal << "\n";
                //                      }
                l_val += p_X.getVal(row,col);
                unsigned int l_psShift = p_postScale & 0x00ff;
                unsigned int l_psVal = p_postScale >> 8;
                l_val = (l_val >> l_psShift) * l_psVal;
                T l_entry = (T)(l_val);
                if (l_entry < 0) {
                    l_entry = (l_entry  >> (p_PReluVal & 0x003f))* (T)(p_PReluVal >> 6);
                }
                getVal(row, col) = l_entry;
            }
        }
    }
    void matMultWithScaleAndPRelu(Mat & p_A, Mat & p_B, int32_t p_bias, int32_t p_postScale, int16_t p_PReluVal) {
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                int64_t l_val = 0;
                for (unsigned int k = 0; k < p_A.cols(); ++k) {
                    l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
                }

                l_val += p_bias;
                l_val = (l_val >> (p_postScale & 0x00ff)) * (p_postScale >> 8);
                //handle saturation
                if (l_val & 0x100000000){//negative number
                    if ((l_val & 0x0ffff0000) != 0xffff0000) {//underflow
                        l_val = 0x8000;
                    }
                }
                else {
                    if ((l_val & 0x0ffff0000) != 0) {
                        l_val = 0x7fff;
                    }
                }
                T l_entry = (T)(l_val);
                if (l_entry < 0) {
                    l_entry = (l_entry * (T)(p_PReluVal >> 6)) >> (p_PReluVal & 0x003f);
                }
                getVal(row, col) = l_entry;
            }
        }
    }

    void multiplyGf(Mat & p_A, Mat & p_B, unsigned int p_EdgeWidth) {
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        cout << "  DEBUG multiplyGf rows=" << rows() << "  cols=" << cols()
                                << "\n";
        for (unsigned int rowBlock = 0; rowBlock < rows() / p_EdgeWidth;
                ++rowBlock) {
            for (unsigned int colBlock = 0; colBlock < cols() / p_EdgeWidth;
                    ++colBlock) {
                for (unsigned int row = 0; row < rows(); ++row) {
                    for (unsigned int col = 0; col < cols(); ++col) {
                        T l_val = 0;
                        for (unsigned int k = 0; k < p_A.cols(); ++k) {
                            l_val += p_A.getVal(k + rowBlock * p_EdgeWidth,
                                    col + colBlock * p_EdgeWidth)
                                                    * p_B.getVal(k + rowBlock * p_EdgeWidth,
                                                            col + colBlock * p_EdgeWidth);
                        }
                        getVal(row + rowBlock * p_EdgeWidth,
                                col + colBlock * p_EdgeWidth) = l_val;
                        cout << "DEBUG multiplyGf after k-loop " << *this
                                << "\n";
                    }
                }
            }
        }
    }
    // Matrix A is in GvA format (also dimensions are wider and shorter)
    // The p_rowEdgeWidth just inficates the compute array intake edge to allow for matrix dimension adjustment
    void multiplyGemvGf(Mat & p_A, Mat & p_B, unsigned int p_rowEdgeWidth) {
        assert(p_A.rows() * p_rowEdgeWidth == rows());
        assert(p_A.cols() == p_B.rows() * p_rowEdgeWidth);
        assert(p_B.cols() == cols());
        cout << "  DEBUG multiplyGvA format rows=" << rows() << "  cols="
                << cols() << "\n";
        // Rows here are mblocks, cols are within the mblock
        for (unsigned int row = 0; row < p_A.rows(); ++row) { // A is already in block format
            for (unsigned int col = 0; col < p_A.cols(); ++col) {
                unsigned int k = col / p_rowEdgeWidth;
                unsigned int w = col % p_rowEdgeWidth;
                T l_a = p_A.getVal(row, col);
                T l_b = p_B.getVal(k, 0);
                getVal(w + row * p_rowEdgeWidth, 0) += l_a * l_b;
                //cout << "        += a * b  = " << l_a << " * " << l_b << "\n";
            }
            //cout << "    DEBUG multiplyGemvGf after k-loop " << *this << "\n";
        }
    }
#if 0
    void
    multiplySpmv(SpMat<T, TspD, Tsp> & p_A, Mat & p_B) {
        T l_val = 0;
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        vector<MtxRow> l_rows = p_A.getNnzVector();
        for (MtxRow &l_row : l_rows) {
            unsigned int row = l_row.getRow(),
                    col = l_row.getCol();
            double l_val = l_row.getVal();
            getVal(row, 0) += l_val * p_B.getVal(col, 0);
            //cout << "DEBUG multiplySpmv row=" << row << " col=" << col << "  "
            //          << l_val << " * " << p_B.getVal(col, 0)
            //          << " was added to " << getVal(row, 0) << "\n";
        }
    }
#endif

    void transpose(Mat & p_A) {
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                getVal(row, col) = p_A.getVal(col, row);
            }
        }
        swap(m_Rows, m_Cols);
    }
    void transposeGva(Mat & p_A, unsigned int p_rowEdgeWidth,
            unsigned int p_colEdgeWidth) {
        unsigned int l_pos = 0;
        for (unsigned int rowBlock = 0; rowBlock < p_A.rows() / p_rowEdgeWidth;
                ++rowBlock) {
            for (unsigned int colBlock = 0;
                    colBlock < p_A.cols() / p_colEdgeWidth; ++colBlock) {
                for (unsigned int col = 0; col < p_colEdgeWidth; ++col) {
                    for (unsigned int row = 0; row < p_rowEdgeWidth; ++row) {
                        getVal(l_pos / cols(), l_pos % cols()) = p_A.getVal(
                                row + rowBlock * p_rowEdgeWidth,
                                col + colBlock * p_colEdgeWidth);
                        l_pos++;
                    }
                    //cout << "DEBUG transposeGva step " << *this << "\n";
                }
            }
        }
        swap(m_Rows, m_Cols);
    }
    void print(ostream& os) {
        os << m_Rows << "x" << m_Cols << " Ld=" << m_Ld << "\n";
        unsigned int l_cols = cols(); // normal matrix
        //ld();; // parent matrix (within Ld
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < l_cols; ++col) {
                os << int(getVal(row, col)) << " ";
            }
            os << "\n";
        }
    }
    bool cmp(float p_TolRel, float p_TolAbs, Mat &p_Ref) {
        bool ok = true;
        unsigned int l_verbose = 1; // 0 none, 1 if not exactly equal, 2 if passed tolerance, 3 show all
        unsigned int l_numExactMatches = 0, l_numMismatches = 0;
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                string l_Prefix = "      row " + to_string(row) + " col "
                        + to_string(col);
                T v = getVal(row, col);
                T vRef = p_Ref.getVal(row, col);
                bool l_exactMatch = false;
                bool l_ok = cmpVal(p_TolRel, p_TolAbs, vRef, v, l_Prefix,
                        l_exactMatch, 1);
                ok = ok && l_ok;
                if (l_exactMatch) {
                    l_numExactMatches++;
                }
                if (!l_ok) {
                    l_numMismatches++;
                }
            }
        }
        unsigned int l_total = rows() * cols();
        unsigned int l_withinTolerance = l_total - l_numExactMatches
                - l_numMismatches;
        cout << "  Compared " << l_total << " values:" << "  exact match "
                << l_numExactMatches << "  within tolerance "
                << l_withinTolerance << "  mismatch " << l_numMismatches
                << "\n";
        return (ok);
    }

    bool cmpVal(float p_TolRel, float p_TolAbs, T vRef, T v,
            std::string p_Prefix, bool &p_exactMatch, unsigned int p_Verbose) {
        float l_diffAbs = abs(v - vRef);
        float l_diffRel = l_diffAbs;
        if (vRef != 0) {
            l_diffRel /= abs(vRef);
        }
        p_exactMatch = (vRef == v);
        bool l_status = p_exactMatch || (l_diffRel <= p_TolRel)
                                || (l_diffAbs <= p_TolAbs);
        if ((p_Verbose >= 3) || ((p_Verbose >= 2) && !p_exactMatch)
                || ((p_Verbose >= 1) && !l_status)) {
            std::cout << p_Prefix << "  ValRef " << std::left
                    << std::setw(GEMX_CMP_WIDTH) << vRef << " Val " << std::left
                    << std::setw(GEMX_CMP_WIDTH) << v << "  DifRel "
                    << std::left << std::setw(GEMX_CMP_WIDTH) << l_diffRel
                    << " DifAbs " << std::left << std::setw(GEMX_CMP_WIDTH)
            << l_diffAbs << "  Status " << l_status << "\n";
        }
        return (l_status);
    }

};

//Base address will be the instruction memory region
class XCL_FPGA {
public:
    XCL_FPGA() = delete;
    XCL_FPGA(const string & xclbin, const string & kernelName) {
        loadXclbin(xclbin, kernelName);
    }

    ~XCL_FPGA() {
    }

    void loadXclbin(string p_XclbinFile, string p_KernelName) {
        // https://gitenterprise.xilinx.com/rkeryell/heterogeneous_examples/blob/master/vector_add/SDAccel-Boost.Compute/vector_add.cpp

        // Create the OpenCL context to attach resources on the device
        m_Context = move(boost::compute::system::default_context());
        // Create the OpenCL command queue to control the device
        //m_CommandQueue = move(boost::compute::system::default_queue());
        //boost::compute::command_queue queue(boost::compute::system::default_context(), boost::compute::system::default_device(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
        boost::compute::command_queue queue(
                boost::compute::system::default_context(),
                boost::compute::system::default_device(),
                CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        m_CommandQueue = move(queue);
        // Construct an OpenCL program from the precompiled kernel file
        m_Program = move(
                boost::compute::program::create_with_binary_file(p_XclbinFile,
                        m_Context));
        m_Program.build();

        m_Kernel = move(boost::compute::kernel(m_Program, p_KernelName));
    }

    boost::compute::buffer createBuf(void *ptr, size_t sz_bytes) {
        //decltype of cl_mem_ext_ptr_t.flags

        //unsigned l_k2bank[] = {GEMX_fpgaDdrBanks};
        //std::cout << "l_k2bank: " << l_k2bank[0] << std::endl;

        cl_mem_ext_ptr_t l_bufExt;
        //l_bufExt.obj = NULL;
        l_bufExt.param = 0;
        l_bufExt.flags = GEMX_fpgaDdrBanks;
        l_bufExt.obj = ptr;
        // Buffers
        return boost::compute::buffer(m_Context, sz_bytes,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,
                &l_bufExt);
    }

    bool copyToFpga(const boost::compute::buffer & buf, bool sync_send) {
        boost::compute::event l_event;
        //cout << "copyToFPGA" << endl;
        // Send the input data to the accelerator
        l_event = m_CommandQueue.enqueue_migrate_memory_objects(1, &(buf.get()),
                0);

        if (sync_send){
            l_event.wait();
        } else{
            m_waitInput.insert(l_event);
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::compute::buffer copyToFpga(void * buf, size_t sz_bytes,
            bool sync_send = false) {
        boost::compute::buffer cl_buf = createBuf(buf, sz_bytes);
        copyToFpga(cl_buf, sync_send);
        return cl_buf;
    }

    void copyFromFpga(const boost::compute::buffer & buf, bool sync_exec = true) {
        //cout << "copyFromFPGA" << endl;
        XTimer t;
        boost::compute::event l_readEvents =
                m_CommandQueue.enqueue_migrate_memory_objects(1, &(buf.get()),
                        CL_MIGRATE_MEM_OBJECT_HOST, m_waitOutput);
        //l_readEvents.wait();
        if ( sync_exec ){
            l_readEvents.wait();
            m_waitOutput.clear();
        } else{
            m_waitOutput.insert(l_readEvents);
        }
#ifdef GEMX_PERF_DBG
        cout << "copyFromFpga: " << t.elapsed() << endl;
#endif
    }

    void execKernel(const boost::compute::buffer & instr_buf, bool sync_exec = false) {
        boost::compute::extents<1> offset { 0 };
        boost::compute::extents<1> global { 1 };
        // Use only 1 CU
        boost::compute::extents<1> local { 1 };
        // Launch kernels
        m_Kernel.set_args(instr_buf, instr_buf);

        XTimer t;
        boost::compute::event l_event = m_CommandQueue.enqueue_nd_range_kernel(
                m_Kernel, offset, global, local, m_waitInput);

        if ( 1 ) {
            l_event.wait();
        } else{
            m_waitOutput.insert(l_event);
        }
        m_waitInput.clear();

#ifdef GEMX_PERF_DBG
        cout << "execKernel: " << t.elapsed() << endl;
#endif

    }

    void wait ()
    {
        for (size_t i = 0; i < m_waitOutput.size(); i++){
            m_waitOutput[i].wait();
        }
        m_waitInput.clear();
        m_waitOutput.clear();
    }

private:
    boost::compute::program m_Program;
    boost::compute::kernel m_Kernel;
    boost::compute::context m_Context;
    boost::compute::command_queue m_CommandQueue;
    boost::compute::wait_list m_waitInput, m_waitOutput;
};

template<typename HType>
class GEMMHost {
public:
    GEMMHost() = delete;
    ~GEMMHost() {
    }
    GEMMHost(const GEMMHost<HType> &) = delete;
    GEMMHost(const string & xclbin, const string & kernelName) {
        _fpga = shared_ptr<XCL_FPGA>(new XCL_FPGA(xclbin, kernelName));
        void *aligned_mem = nullptr;
        assert(!posix_memalign(&aligned_mem, PAGE_SIZE, INSTR_BUF_SIZE));
        _instrBuf = shared_ptr<char>((char*) aligned_mem);
        memset(_instrBuf.get(), 0, INSTR_BUF_SIZE);
        _instr_offset = 0;
        _cl_instr_buf = _fpga->copyToFpga(_instrBuf.get(), INSTR_BUF_SIZE,
                true);
        xclGetMemObjDeviceAddress(_cl_instr_buf.get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &_ddrDeviceBaseAddr);

        assert(!posix_memalign(&aligned_mem, PAGE_SIZE, KERN_DBG_BUF_SIZE));
        _kernDbgBuf = shared_ptr<char>((char*) aligned_mem);
        _cl_kern_dbg_buf = _fpga->copyToFpga(_kernDbgBuf.get(),
                KERN_DBG_BUF_SIZE, true);
    }

    bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift) {
        return AddGEMMOp (A, B, C, bias, m, k, n, k, n, n, n, postScale, postShift);
    }

    bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, unsigned int lda, unsigned int ldb, unsigned int ldc, unsigned int ldx, int postScale, int postShift) {
        XTimer t;
        if (_hostMat.find(A) == _hostMat.end()
                || _hostMat.find(B) == _hostMat.end()
                || _hostMat.find(C) == _hostMat.end()
                || _hostMat.find(bias) == _hostMat.end()) {
            cerr << "Matrix not found!" << endl;
            return false;
        }
  unsigned long long A_off = 0, B_off = 0, C_off = 0, X_off = 0;

        xclGetMemObjDeviceAddress(_devHandle[A].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &A_off);
        xclGetMemObjDeviceAddress(_devHandle[B].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &B_off);
        xclGetMemObjDeviceAddress(_devHandle[C].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &C_off);

        if ( _devHandle.find(bias) != _devHandle.end()){
            xclGetMemObjDeviceAddress(_devHandle[bias].get(),
                    boost::compute::system::default_device().get(),
                    sizeof(unsigned long long), &X_off);
            assert(X_off > _ddrDeviceBaseAddr);
            X_off -= _ddrDeviceBaseAddr;
        }

       // cout << "A_dev_addr: " << A_off << " B_dev_addr: " << B_off << " C_dev_addr: " << C_off << " X_dev_addr: " << X_off << endl;
        assert(A_off > _ddrDeviceBaseAddr);
        assert(B_off > _ddrDeviceBaseAddr);
        assert(C_off > _ddrDeviceBaseAddr);
        A_off -= _ddrDeviceBaseAddr;
        B_off -= _ddrDeviceBaseAddr;
        C_off -= _ddrDeviceBaseAddr;

        assert(A_off % PAGE_SIZE == 0);
        assert(B_off % PAGE_SIZE == 0);
        assert(C_off % PAGE_SIZE == 0);
        assert(X_off % PAGE_SIZE == 0);

        A_off /= PAGE_SIZE;
        B_off /= PAGE_SIZE;
        C_off /= PAGE_SIZE;
        X_off /= PAGE_SIZE;
	
        GemmArgs gargs(A_off, B_off, C_off, X_off, m,
                k, n, lda, ldb, ldc, ldx, postScale, postShift);
	AddInstr ( &gargs);
        return true;
    }

    bool AddMat(const HType & handle, void * mat_ptr, unsigned long long buf_sz) {
        if (_hostMat.find(handle) == _hostMat.end()) {
            _hostMat[handle] = mat_ptr;
            _hostMatSz[handle] = buf_sz;
            return true;
        }
        else if (_hostMatSz[handle] != buf_sz ){
            _hostMat[handle] = mat_ptr;
            _hostMatSz[handle] = buf_sz;
            _devHandle.erase(handle);
            cout << "Erasing devhandle!" << endl;
            return true;
        }
        //cout << "Matrix " << handle << " already added!" << endl;
        return false;
    }

    /*
    bool AddMat(const HType & handle, short *p_Addr, unsigned int p_Rows,
            unsigned int p_Cols, unsigned int p_Ld) {
        return AddMat(handle,
                shared_ptr<ShortMat>(
                        new ShortMat(p_Rows, p_Cols, p_Ld, p_Addr)));
    }

    bool AddMat(const HType & handle, int *p_Addr, unsigned int p_Rows,
            unsigned int p_Cols, unsigned int p_Ld) {
        return AddMat(handle,
                shared_ptr<IntMat>(
                        new IntMat(p_Rows, p_Cols, p_Ld, p_Addr)));
    }
     */


    /*
    shared_ptr<BaseMat> AllocMat(const HType & handle, size_t n_rows,
            size_t n_cols, size_t ld, bool sendToFPGA = false, bool sync_send =
                    false) {
        if (_hostMat.find(handle) != _hostMat.end()) {
            cout << "Matrix " << handle << " already created!" << endl;
            return _hostMat[handle];
        }
        cout << "Matrix " << handle << " " << n_rows << " " << n_cols << " "
                << ld << endl;
        void* aligned_buffer = nullptr;
        // assert ( !posix_memalign ( &aligned_buffer, PAGE_SIZE, sizeof(T) * n_rows * n_cols ));
        assert(
                !posix_memalign(&aligned_buffer, PAGE_SIZE,
     * n_rows * ld));
        shared_ptr<Mat<DType> > mat_ptr = shared_ptr<Mat<DType> >(
                new Mat<DType>(n_rows, n_cols, ld, (DType*) aligned_buffer));
        assert(AddMat(handle, mat_ptr));

        if (sendToFPGA)
            SendToFPGA(handle, sync_send);
        return mat_ptr;
    }
     */

    void * GetMat(const HType & handle,
            bool queryFPGA = false, bool sync_get = true) {
        void * ret_ptr = nullptr;
        if (_hostMat.find(handle) != _hostMat.end()) {
            if (queryFPGA)
                GetFromFPGA(handle, sync_get);
            ret_ptr = _hostMat[handle];
        } else{
            cout << "GetMat: Matrix not found!" << endl;
            assert(0);
        }
        return ret_ptr;
    }

    void Execute() {
        XTimer t;
        _fpga->copyToFpga(_cl_instr_buf, false);
        _fpga->execKernel(_cl_instr_buf);
        memset(_instrBuf.get(), 0, PAGE_SIZE);
        _instr_offset = 0;
#ifdef GEMX_PERF_DBG
        cout << "Execute: " << t.elapsed() << endl;
#endif
    }

    void Wait(){
        _fpga->wait();
    }

    /*
    void SendToFPGA(const HType & handle, int * p_addr, int n_row, int n_col,
            int ld, bool sync_send = false) {
        SendToFPGA(handle,
                shared_ptr<Mat >(
                        new Mat<DType>(n_row, n_col, ld, p_addr)), sync_send);
    }
     */

    void SendToFPGA(const HType & handle, void * mat_ptr, unsigned long long buf_sz,
            bool sync_send = false) {
        AddMat(handle, mat_ptr, buf_sz);
        SendToFPGA(handle, sync_send);
    }

    void SendToFPGA(const HType & handle, bool sync_send = false) {
        XTimer t;
        assert(_hostMat.find(handle) != _hostMat.end());

        //shared_ptr < Mat<T> > mat = _hostMat[handle];
        if (_devHandle.find(handle) != _devHandle.end()) {
            _fpga->copyToFpga(_devHandle[handle], sync_send);
        } else {
            _devHandle[handle] = _fpga->copyToFpga(_hostMat[handle], _hostMatSz[handle], sync_send);
        }
#ifdef GEMX_PERF_DBG
        cout << "SendToFPGA: " << t.elapsed() << endl;
#endif
    }

    void GetFromFPGA(const HType & handle, bool sync_get) {
        XTimer t;
        assert(_devHandle.find(handle) != _devHandle.end());
        _fpga->copyFromFpga(_devHandle[handle], sync_get);
#ifdef GEMX_PERF_DBG
        cout << "GetFromFPGA: " << t.elapsed() << endl;
#endif
    }

protected:

    void AddInstr  ( kArgs * args )
    {
        char * instr = args->asByteArray();
        char * curr_pos = &_instrBuf.get()[_instr_offset];
        memcpy(curr_pos, instr, args->sizeInBytes());
        _instr_offset += args->sizeInBytes();
    }

    static const unsigned int PAGE_SIZE = 4096;
    static const unsigned int INSTR_BUF_SIZE = PAGE_SIZE;
    static const unsigned int KERN_DBG_BUF_SIZE = PAGE_SIZE;

    unsigned long long _ddrDeviceBaseAddr;
    shared_ptr<char> _instrBuf, _kernDbgBuf;
    boost::compute::buffer _cl_instr_buf, _cl_kern_dbg_buf;
    unsigned int _instr_offset;
    unordered_map<HType, void*  > _hostMat;
    unordered_map<HType, unsigned long long > _hostMatSz;
    unordered_map<HType, boost::compute::buffer> _devHandle;
    shared_ptr<XCL_FPGA> _fpga;
};

template<typename HType>
class FCNHost : public GEMMHost <HType>
{
public:
    FCNHost() = delete;
    virtual ~FCNHost(){}
    FCNHost ( const FCNHost<HType>&) = delete;
    FCNHost(const string & xclbin, const string & kernelName ) : GEMMHost<HType> ( xclbin, kernelName)
            {
            }

    bool AddFCNOp ( const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha)
    {
        return AddFCNOp ( A, B, C, bias, m, k, n, k, n, n, n,postScale, postShift, PReLUScale, PReLUAlpha);
    }

    bool AddFCNOp ( const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, unsigned int lda, unsigned int ldb, unsigned int ldc, unsigned int ldx, int postScale, int postShift, short PReLUScale, short PReLUAlpha)
    {
        XTimer t;
        if (this->_hostMat.find(A) == this->_hostMat.end()
                || this->_hostMat.find(B) == this->_hostMat.end()
                || this->_hostMat.find(C) == this->_hostMat.end()
                || this->_hostMat.find(bias) == this->_hostMat.end()) {
            cerr << "Matrix not found!" << endl;
            return false;
        }

        if ( m < MIN_M ){
            cerr << "m dimension (" << m << ") is less than minimum supported size " << MIN_M << endl;
            return false;
        }
        if ( !isPowerOf2(m) ){
            cerr << "m dimension (" << m << ") isn't a power of 2" << endl;
            return false;
        }

        if ( k < MIN_K ){
            cerr << "k dimension (" << k << ") is less than minimum supported size " << MIN_K << endl;
            return false;
        }

        if ( !isPowerOf2(k) ){
            cerr << "k dimension (" << k << ") isn't a power of 2" << endl;
            return false;
        }

        if ( n < MIN_N ){
            cerr << "n dimension (" << n << ") is less than minimum supported size " << MIN_N << endl;
            return false;
        }
        if ( !isPowerOf2(n) ){
            cerr << "n dimension (" << n << ") isn't a power of 2" << endl;
            return false;
        }
        if ( lda < MIN_M ){
            cerr << "lda dimension (" << lda << ") is less than minimum supported size " << MIN_M << endl;
            return false;
        }
        if ( !isPowerOf2(lda) ){
            cerr << "lda dimension (" << lda << ") isn't a power of 2" << endl;
            return false;
        }
        if ( ldb < MIN_N ){
            cerr << "ldb dimension (" << ldb << ") is less than minimum supported size " << MIN_N << endl;
            return false;
        }

        if ( !isPowerOf2(ldb) ){
            cerr << "ldb dimension (" << ldb << ") isn't a power of 2" << endl;
            return false;
        }

        if ( ldc < MIN_N ){
            cerr << "ldc dimension (" << ldc << ") is less than minimum supported size " << MIN_N << endl;
            return false;
        }

        if ( !isPowerOf2(ldc) ){
            cerr << "ldc dimension (" << ldc << ") isn't a power of 2" << endl;
            return false;
        }

        //if ( ldx != ldc )
        unsigned long long A_off = 0, B_off = 0, C_off = 0, X_off = 0;

        xclGetMemObjDeviceAddress(this->_devHandle[A].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &A_off);
        xclGetMemObjDeviceAddress(this->_devHandle[B].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &B_off);
        xclGetMemObjDeviceAddress(this->_devHandle[C].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &C_off);

        if ( this->_devHandle.find(bias) != this->_devHandle.end()){
            xclGetMemObjDeviceAddress(this->_devHandle[bias].get(),
                    boost::compute::system::default_device().get(),
                    sizeof(unsigned long long), &X_off);
            assert(X_off > this->_ddrDeviceBaseAddr);
            X_off -= this->_ddrDeviceBaseAddr;
        }

        //cout << "A_dev_addr: " << A_off << " B_dev_addr: " << B_off << " C_dev_addr: " << C_off << endl;
        assert(A_off > this->_ddrDeviceBaseAddr);
        assert(B_off > this->_ddrDeviceBaseAddr);
        assert(C_off > this->_ddrDeviceBaseAddr);
        A_off -= this->_ddrDeviceBaseAddr;
        B_off -= this->_ddrDeviceBaseAddr;
        C_off -= this->_ddrDeviceBaseAddr;

        assert(A_off % this->PAGE_SIZE == 0);
        assert(B_off % this->PAGE_SIZE == 0);
        assert(C_off % this->PAGE_SIZE == 0);
        assert(X_off % this->PAGE_SIZE == 0);

        A_off /= this->PAGE_SIZE;
        B_off /= this->PAGE_SIZE;
        C_off /= this->PAGE_SIZE;
        X_off /= this->PAGE_SIZE;

        FcnArgs args(A_off, B_off, C_off, X_off, m,
                k, n, lda, ldb, ldc, ldx, postScale, postShift,  PReLUScale, PReLUAlpha);
        this->AddInstr ( &args);
#ifdef GEMX_PERF_DBG
        cout << "AddFCNOp: " << t.elapsed() << endl;
#endif
        return true;
    }

protected:
    const int MIN_M = 256;
    const int MIN_K = 256;
    const int MIN_N = 32;

    bool isPowerOf2( int n )
    {
        return ( (n & (n-1)) == 0 );
    }

};

}
;
// namespace

extern "C" {
gemx::FCNHost<short*> * MakeFCNHost(char *xclbin, char * kernName);
void DestroyFCNHost(gemx::FCNHost<short*> * ptr);

void SendToFPGAShrt(gemx::FCNHost<short*> * gh, short *A,  unsigned long long num_elem, bool sync_send);
void SendToFPGAInt(gemx::FCNHost<short*> * gh, int *A,  unsigned long long num_elem, bool sync_send);
void SendToFPGAShrt_dbg(gemx::FCNHost<short*> * gh, char * name, short *A, int m, int n, bool sync_send);
void SendToFPGAInt_dbg(gemx::FCNHost<short*> * gh, char * name, int *A, int m, int n, bool sync_send);

void SendToFPGA(gemx::FCNHost<short*> * gh, void * A, unsigned long long buf_sz, bool sync_send);

void* GetFromFPGA(gemx::FCNHost<short*> * gh, short *A, bool sync_get);

void Wait ( gemx::FCNHost<short*> * gh );
bool AddFCNOp(gemx::FCNHost<short*> * gh, void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha );
void Execute (gemx::FCNHost<short*> * gh);

gemx::GEMMHost<short*> * MakeGEMMHost(char *xclbin, char * kernName);
void DestroyGEMMHost(gemx::GEMMHost<short*> * ptr);

void SendToFPGAShrt_GEMM(gemx::GEMMHost<short*> * gh, short *A,  unsigned long long num_elem, bool sync_send);
void SendToFPGAInt_GEMM(gemx::GEMMHost<short*> * gh, int *A,  unsigned long long num_elem, bool sync_send);
void SendToFPGAShrt_dbg_GEMM(gemx::GEMMHost<short*> * gh, char * name, short *A, int m, int n, bool sync_send);
void SendToFPGAInt_dbg_GEMM(gemx::GEMMHost<short*> * gh, char * name, int *A, int m, int n, bool sync_send);

void SendToFPGA_GEMM(gemx::GEMMHost<short*> * gh, void * A, unsigned long long buf_sz, bool sync_send);

void* GetFromFPGA_GEMM(gemx::GEMMHost<short*> * gh, short *A, bool sync_get);

void Wait_GEMM ( gemx::GEMMHost<short*> * gh );
//void AddGEMMOp(gemx::FCNHost<short*> * gh, void * A, void * B, void *C,  unsigned int m, unsigned int k, unsigned int n);

bool AddGEMMOp(gemx::GEMMHost<short*> * gh, void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift);

void Execute_GEMM (gemx::GEMMHost<short*> * gh);
}

#endif /* SRC_GEMX_HOST_H_ */
