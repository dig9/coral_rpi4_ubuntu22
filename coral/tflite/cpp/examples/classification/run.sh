#!/bin/bash
TF250_LIB=${HOME}/tf250_lib
LD_LIBRARY_PATH=${TF250_LIB} ./classify model/mod.tflite ./model/label.txt ./images/bird_320x320.bmp 0.2 1 100

