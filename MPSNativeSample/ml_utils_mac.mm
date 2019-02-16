// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ml_utils_mac.h"

#include <iostream>

namespace ml {

OperandMac::OperandMac() = default;
OperandMac::OperandMac(const OperandMac& operand) = default;
OperandMac::OperandMac(const Operand& operand)
    : Operand(operand), read_count(0) {}
OperandMac::~OperandMac() = default;

OperationMac::OperationMac() = default;
OperationMac::OperationMac(const OperationMac& operation) = default;
OperationMac::OperationMac(const Operation& operation)
    : Operation(operation), local_operation(KBNNSFilter) {}
OperationMac::~OperationMac() = default;

bool ParameterExtracterForConv(const OperationMac& operation,
                               const std::vector<uint32_t>& inputs,
                               const std::vector<uint32_t>& outputs,
                               const std::map<uint32_t, ValueInfo>& values,
                               const std::unique_ptr<int8_t[]>& memory,
                               const std::vector<OperandMac>& operands,
                               int32_t& input_batch_size,
                               int32_t& input_width,
                               int32_t& input_height,
                               int32_t& output_width,
                               int32_t& output_height,
                               bool& implicit_padding,
                               int32_t& padding_left,
                               int32_t& padding_right,
                               int32_t& padding_top,
                               int32_t& padding_bottom,
                               int32_t& stride_width,
                               int32_t& stride_height,
                               int32_t& padding_code,
                               int32_t& fuse_code,
                               int32_t& depth_out,
                               int32_t& filter_height,
                               int32_t& filter_width,
                               int32_t& depth_in,
                               int32_t& depthwise_multiplier,
                               bool depthwise) {
  uint32_t output_idx = outputs[0];
  const OperandMac& output = operands[output_idx];
  output_height = output.dimensions[1];
  output_width = output.dimensions[2];
  int32_t index = 0;
  int32_t input_idx = inputs[index++];
  const OperandMac& input = operands[input_idx];
  // depth_in is the fourth dimension of input that shape is
  // [batches, height, width, depth_in].
  input_batch_size = input.dimensions[0];
  input_height = input.dimensions[1];
  input_width = input.dimensions[2];
  depth_in = input.dimensions[3];

  const OperandMac& filter = operands[inputs[index++]];
  if (depthwise) {
    depth_out = filter.dimensions[3];
  } else {
    depth_out = filter.dimensions[0];
  }
  filter_height = filter.dimensions[1];
  filter_width = filter.dimensions[2];

  const OperandMac& bias = operands[inputs[index++]];

  if ((!depthwise && inputs.size() == 10) ||
      (depthwise && inputs.size() == 11)) {
    implicit_padding = false;
    padding_left = getScalarInt32(values, inputs[index++], memory.get());
    padding_right = getScalarInt32(values, inputs[index++], memory.get());
    padding_top = getScalarInt32(values, inputs[index++], memory.get());
    padding_bottom = getScalarInt32(values, inputs[index++], memory.get());
  } else if ((!depthwise && inputs.size() == 7) ||
             (depthwise && inputs.size() == 8)) {
    implicit_padding = true;
    padding_code = getScalarInt32(values, inputs[index++], memory.get());
  } else {
    std::cout << "  inputs size is incorrect";
    return false;
  }
  stride_width = getScalarInt32(values, inputs[index++], memory.get());
  stride_height = getScalarInt32(values, inputs[index++], memory.get());
  if (depthwise == true) {
    depthwise_multiplier =
        getScalarInt32(values, inputs[index++], memory.get());
    if (depthwise_multiplier != 1) {
      std::cout << "  depthwise_multiplier " << depthwise_multiplier
                  << " is not supported.";
      return false;
    }
  }
  fuse_code = getScalarInt32(values, inputs[index++], memory.get());
  return true;
}

//void SetupOperandInfoForOperands(
//    std::vector<std::unique_ptr<OperandInfo>>& opearnd_info_array,
//    std::vector<OperandMac>& operands,
//    const std::vector<uint32_t>& operands_index_array,
//     const std::map<uint32_t, ValueInfo>& values,
//     const std::unique_ptr<int8_t[]>& memory) {
//  for (size_t i = 0; i < operands_index_array.size(); ++i) {
//    const uint32_t length = operands[operands_index_array[i]].requiredSize();
//    ValueInfo value_info = values.at(operands_index_array[i]);
//    void* new_memory =
//        reinterpret_cast<void*>(memory.get() + value_info.offset);
//    std::unique_ptr<OperandInfo> info(
//        new OperandInfo(length, new_memory));
//    opearnd_info_array.push_back(std::move(info));
//  }
//}
  
std::vector<float> LoadData(NSString *path) {
  NSError *error = nil;
  NSString *str = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
  if (error != nil) {
    NSLog([error localizedDescription]);//将错误信息输出来
  }
  NSArray *arr = [str componentsSeparatedByString:@","];
  std::vector<float> data;
  for (int i = 0; i < [arr count]; ++i) {
    data.push_back([[arr objectAtIndex:i] floatValue]);
  }
  return data;
}

}
