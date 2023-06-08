#!/bin/bash
TF250_LIB=${HOME}/tf250_lib
#TF250_LIB=./lib
LD_LIBRARY_PATH=${TF250_LIB} ./classify model/mod.tpu.tflite ./model/label.txt ./images/bird_320x320.bmp 0.2 1 100


#LD_LIBRARY_PATH=./lib/ ./classify model/efficientdet_lite0_320_ptq_edgetpu.tflite ./model/label.txt ./images/bird_320x320.bmp 0.2 1
##LD_LIBRARY_PATH=./lib/ ./classify model/recompiled.tpu.tflite ./model/label.txt ./images/bird_320x320.bmp 0.2 1
#LD_LIBRARY_PATH=./lib/ ./classify model/mod.tflite ./model/label.txt ./images/bird_320x320.bmp 0.2 1
