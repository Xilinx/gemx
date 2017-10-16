// Copyright (c) 2017
// Xilinx, Inc.
// All rights reserved.
// $Id: //Rodin/Proj/O/OPENCL_APPS_DEV/src/matrix_mult/gemx/src/gemx_ddr.h#2 $
// 
// No part of this document may be copied, transmitted or
// disclosed in any form or fashion without the express
// written consent of Xilinx, Inc.

/**
 *  @brief DDR loader and accelerator flow control
 *
 *  $DateTime: 2017/05/04 12:40:36 $
 *  $Author: jzejda $
 *
 *  DDR access utilities
 */

#ifndef GEMX_DDR_H
#define GEMX_DDR_H

#include "assert.h"
#include "gemx_types.h"
//#include "gemx_gemm.h"
//#include "gemx_kargs.h"

namespace gemx {

template <
    typename t_FloatType,
    unsigned int t_DdrWidth,     // In t_FloatType
    unsigned int t_DdrWidthBits  // In bits; both must be defined and be consistent
  >
class DdrUtil
{
  private:
  public:
    typedef WideType<t_FloatType, t_DdrWidth> DdrWideType;    
    
  private:
  public:
    
    
};

} // namespace
#endif

