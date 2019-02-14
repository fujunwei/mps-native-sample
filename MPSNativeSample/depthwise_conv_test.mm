// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "depthwise_conv_test.h"

#include <iostream>
#include <vector>

#include "constants.h"
#include "model_impl_mac.h"
#include "compilation_impl_mac.h"
#include "execution_impl_mac_mps.h"

namespace ml {

void DepthwiseConv2dFloatLarge() {
  std::cout << "\nBegin DepthwiseConv2dFloatLarge test cases.\n\n";
  auto model = std::make_unique<ModelImplMac>();
  uint32_t operandIndex = 0;
  
  std::vector<float> op1_value = {10, 21, 10, 22, 10, 23, 10, 24};
  std::vector<float> expected_data = {110, 246};
  
  std::vector<uint32_t> dimensions0 = {1, 2, 2, 2};
  std::vector<uint32_t> dimensions1 = {2};
  std::vector<uint32_t> dimensions3 = {1, 1, 1, 2};
  
  uint32_t op1 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t op2 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t op3 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  uint32_t pad0 = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t act = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t channelMultiplier = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t op4 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions3, 0, 0);
  
  model->SetOperandValue(op1, op1_value);
  model->SetOperandValue(op2, {0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1});
  model->SetOperandValue(op3, {100, 200});
  model->SetOperandValue(pad0, {0});
  model->SetOperandValue(act, {0});
  model->SetOperandValue(stride, {1});
  model->SetOperandValue(channelMultiplier, {1});
  model->AddOperation(DEPTHWISE_CONV_2D, {op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, channelMultiplier, act}, {op4});
  
  model->IdentifyInputsAndOutputs({op1}, {op4});
  
  auto compilation = std::make_unique<CompilationImplMac>(model.get());
  compilation->Finish(PREFER_SUSTAINED_SPEED);
  
  auto execution = std::make_unique<ExecutionImplMacMPS>(compilation.get());
  execution->StartCompute();
  
  std::cout << "\n=============================";
  if (execution->OutputData() == expected_data) {
    std::cout << "\nConvFloat test case passed.\n";
  } else {
    std::cout << "\nConvFloat test case doesn't pass.\n";
  }
  std::cout << "=============================\n";
}
  
void Depthwise28_28Conv5_5() {
  auto model = std::make_unique<ModelImplMac>();
  uint32_t operandIndex = 0;
  
  std::vector<float> op1_value(28*28);
  std::vector<float> op2_value(5*5);
  std::vector<float> expected_data = {110, 246};
  
  std::vector<uint32_t> dimensions0 = {1, 28, 28, 1};
  std::vector<uint32_t> dimensions1 = {1, 5, 5, 1};
  std::vector<uint32_t> dimensions2 = {1};
  
  uint32_t op0 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t op1 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  uint32_t op2 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions2, 0, 0);
  uint32_t pad_code = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride_width = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride_height = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t channelMultiplier = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t fuse_code = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t op3 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  
  model->SetOperandValue(op0, op1_value);
  model->SetOperandValue(op1, op2_value);
  model->SetOperandValue(op2, {100});
  model->SetOperandValue(pad_code, {1});
  model->SetOperandValue(stride_width, {1});
  model->SetOperandValue(stride_height, {1});
  model->SetOperandValue(channelMultiplier, {1});
  model->SetOperandValue(fuse_code, {0});
  model->AddOperation(DEPTHWISE_CONV_2D, {op0, op1, op2, pad_code, stride_width, stride_height, channelMultiplier, fuse_code}, {op3});
  
  model->IdentifyInputsAndOutputs({op0}, {op3});
  
  auto compilation = std::make_unique<CompilationImplMac>(model.get());
  compilation->Finish(PREFER_SUSTAINED_SPEED);
  
  auto execution = std::make_unique<ExecutionImplMacMPS>(compilation.get());
  execution->StartCompute();
  
  if (execution->OutputData() == expected_data) {
    std::cout << "\nConvFloat test case passed\n";
  } else {
    std::cout << "\nConvFloat test case doesn't pass\n";
  }
}


}  // namespace ml
