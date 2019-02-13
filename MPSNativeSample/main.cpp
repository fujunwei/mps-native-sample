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

int main(int argc, const char * argv[]) {
  // insert code here...
  std::cout << "Hello, World!\n";
  
  ml::ConvFloat();
  
  ml::DepthwiseConv2dFloatLarge();
  
  return 0;
}
