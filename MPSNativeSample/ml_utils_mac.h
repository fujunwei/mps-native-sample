// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_ML_UTILS_MAC_H_
#define SERVICES_ML_ML_UTILS_MAC_H_

// The header file can't be included, otherwise the declaration of
// MPSCNNKernel will be used.
// #import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <map>
#include <memory>
#include <vector>
#import <Foundation/Foundation.h>

#include "common.h"

@class MPSCNNKernel;
@class MPSCNNBinaryKernel;

typedef enum LocalOperation {
  KBNNSFilter = 1,
  KReshape = 2,
  KConcatenation = 3,
  KAdd = 4,
  KMul = 5,
} LocalOperation;

namespace ml {

struct OperandMac : public Operand {
  OperandMac();
  explicit OperandMac(const OperandMac&);
  explicit OperandMac(const Operand&);
  ~OperandMac();
  uint32_t read_count;
};

struct OperationMac : public Operation {
  OperationMac();
  explicit OperationMac(const OperationMac&);
  explicit OperationMac(const Operation&);
  ~OperationMac();
  API_AVAILABLE(macos(10_13)) MPSCNNKernel* custom_cnn_kernel;
  LocalOperation local_operation;

  int fuse_code;
  int input_batch_size;
  uint32_t offset_x;
  uint32_t offset_y;
  std::vector<float*> extend_input;
};

bool ParameterExtracterForConv(const OperationMac&,
                               const std::vector<uint32_t>&,
                               const std::vector<uint32_t>&,
                               const std::map<uint32_t, ValueInfo>& values,
                               const std::unique_ptr<int8_t[]>& memory,
                               const std::vector<OperandMac>& operands,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               bool&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               int32_t&,
                               bool depthwise = false);

//void SetupOperandInfoForOperands(
//    std::vector<std::unique_ptr<OperandInfo>>& opearnd_info_array,
//    std::vector<OperandMac>& operands,
//    const std::vector<uint32_t>& operands_index_array,
//     const std::map<uint32_t, ValueInfo>& values,
//     const std::unique_ptr<int8_t[]>& memory);

  std::vector<float> LoadData(NSString *path);
}  // namespace ml

#endif  // SERVICES_ML_ML_UTILS_MAC_H_
