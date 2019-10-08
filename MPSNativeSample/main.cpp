//
//  main.cpp
//  MPSNativeSample
//
//  Created by mac-webgl-stable on 1/31/19.
//  Copyright © 2019 mac-webgl-stable. All rights reserved.
//

#include <iostream>
#include "test_cases.h"
#include "depthwise_conv_test.h"
#include "resize_bilinear_test.h"
#include "average_pool_test.h"

int main(int argc, const char * argv[]) {
//  ml::ConvFloat();
  
//  ml::Depthwise28_28Conv5_5();
//
//  ml::Depthwise28_28_528Conv5_5_528();
  
//  ml::ResizeBilinear65_65To513_513();
//
//  ml::ResizeBilinear65_65_21To513_513_21();
  
  ml::AveragePool();
  
  return 0;
}
