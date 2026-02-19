#pragma once

#include "params.h"

namespace sm120 {

template<typename InputT>
void run_flash_splitkv_mla_kernel(DecodingParams &params, cudaStream_t stream);

}
