// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "average_pool_test.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <mach/mach_time.h>

#include "constants.h"
#include "model_impl_mac.h"
#include "compilation_impl_mac.h"
#include "execution_impl_mac_mps.h"

namespace ml {

static double ConvertTimeDeltaToSeconds( uint64_t delta )
{
  static double conversion = 0.0;
  
  if( 0.0 == conversion)
  {
    mach_timebase_info_data_t info;
    kern_return_t err = mach_timebase_info( &info );
    if( 0 == err )
      conversion = (1e-9 * info.numer) / info.denom;
  }
  
  return delta * conversion;
}

void AveragePool() {
  std::cout << "\nBegin AveragePool test cases.\n\n";
  auto model = std::make_unique<ModelImplMac>();
  uint32_t operandIndex = 0;
  
  std::vector<float> op1_value(28 * 28 * 448, 1);//(5* 52* 60* 3, 1);
  std::vector<float> expected_data(448, 0.5);//(96 * 86, 0.5);
  
  std::vector<uint32_t> dimensions0 = {1, 28, 28, 448};//{5, 52, 60, 3};
  std::vector<uint32_t> dimensions1 = {1, 1, 1, 448};//{5, 11, 13, 3};
  
  uint32_t input = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t stride = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t filter = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t padding = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t activation = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t output = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  
  model->SetOperandValue(stride, {28});
  model->SetOperandValue(filter, {28});
  model->SetOperandValue(padding, {0});
  model->SetOperandValue(activation, {0});
  model->SetOperandValue(input, op1_value);
  model->AddOperation(AVERAGE_POOL_2D, {input, padding, padding, padding, padding, stride, stride, filter, filter, activation}, {output});
  
  model->IdentifyInputsAndOutputs({input}, {output});
  
  uint64_t starting_time = mach_absolute_time();
  auto compilation = std::make_unique<CompilationImplMac>(model.get());
  compilation->Finish(PREFER_SUSTAINED_SPEED);
  uint64_t compilation_time = mach_absolute_time();
  printf( "======the timer of compiling is %f ms.\n", ConvertTimeDeltaToSeconds(compilation_time - starting_time) * 1000.0);
  
  auto execution = std::make_unique<ExecutionImplMacMPS>(compilation.get());
  execution->StartCompute();
  printf( "======the timer of executing is %f ms.\n", ConvertTimeDeltaToSeconds(mach_absolute_time() - compilation_time) * 1000.0);
}

}  // namespace ml
