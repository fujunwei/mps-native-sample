//
//  main.cpp
//  MPSNativeSample
//
//  Created by mac-webgl-stable on 1/31/19.
//  Copyright © 2019 mac-webgl-stable. All rights reserved.
//

#include <iostream>
#include "model_impl_mac.h"
#include "constants.h"

int main(int argc, const char * argv[]) {
  // insert code here...
  std::cout << "Hello, World!\n";
  auto model = new ml::ModelImplMac();
  
  std::vector<uint32_t> dimensions0 = {1, 3, 3, 1};
  std::vector<uint32_t> dimensions1 = {1, 2, 2, 1};
  std::vector<uint32_t> dimensions2 = {1};
  
  uint32_t operandIndex = 0;
  uint32_t op1 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t op2 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  uint32_t op3 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions2, 0, 0);
  uint32_t pad0 = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t act = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t op4 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  
  model->SetOperandValue(op1, {1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0});
  model->SetOperandValue(op2, {0.25, 0.25, 0.25, 0.25});
  model->SetOperandValue(op3, {0});
  model->SetOperandValue(pad0, {0});
  model->SetOperandValue(act, {0});
  model->SetOperandValue(stride, {1});
  model->AddOperation(CONV_2D, {op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act}, {op4});
  model->SetOperandValue(op4, {0.25, 0.25, 0.25, 0.25});
  
  model->IdentifyInputsAndOutputs({op1}, {op4});
//  await model.finish();
  
  model->CreateCompilation();
  
  return 0;
}
