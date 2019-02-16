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
//  ml::ConvFloat();
  
  ml::Depthwise28_28Conv5_5();
  
  ml::Depthwise28_28_528Conv5_5_528();
  
  return 0;
}
