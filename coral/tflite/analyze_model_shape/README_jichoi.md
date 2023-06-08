This script simply extracts detail information from input tf lite model file. you may need it when to define input shape & parse output. it supports for both cpu & tpu type model.

run command:
python analyze_model_shape.py ../cpp/examples/classification/model/efficientdet_lite0_320_ptq_edgetpu.tflite

python analyze_model_skel.py ../cpp/examples/classification/model/efficientdet_lite0_320_ptq_edgetpu.tflite depth=1800 dims=6   grad=85 inv=3 out=eff_lite0_1800_6_85_3.skel





