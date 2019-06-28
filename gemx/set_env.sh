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

#!/usr/bin/env bash

unset SDSOC_SDK
unset SDSOC_VIVADO
unset SDSOC_VIVADO_HLS
unset SDX_VIVADO
unset SDX_VIVADO_HLS
unset XILINX_VIVADO_HLS
unset XILINX_SDK
unset PLATFORM_REPO_PATHS

export TA_PATH=/proj/xbuilds/2019.1_daily_latest/installs/lin64
export XILINX_SDX=${TA_PATH}/SDx/2019.1
export XILINX_VIVADO=${TA_PATH}/Vivado/2019.1

export LD_LIBRARY_PATH=`$XILINX_SDX/bin/ldlibpath.sh $XILINX_SDX/lib/lnx64.o`:${XILINX_SDX}/lnx64/tools/opencv
source ${TA_PATH}/../../xbb/xrt/packages/setenv.sh 
source ${TA_PATH}/SDx/2019.1/settings64.sh
