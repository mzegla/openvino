﻿// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pooling_kernel_base.h"

#include <vector>

namespace kernel_selector {
class PoolingKernelGPUInt8Ref : public PoolingKernelBase {
public:
    PoolingKernelGPUInt8Ref() : PoolingKernelBase("pooling_gpu_int8_ref") {}
    virtual ~PoolingKernelGPUInt8Ref() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    bool Validate(const Params&, const optional_params&) const override;
    JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
