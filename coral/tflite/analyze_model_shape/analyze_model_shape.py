import numpy as np
#import tensorflow as tf
import tflite_runtime.interpreter as tflite

#by jichoi.
import platform
import sys

EDGETPU_SHARED_LIB = {
   'Linux': 'libedgetpu.so.1',
   'Darwin': 'libedgetpu.1.dylib',
   'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                {'device': device[0]} if device else {})
       ])


if len(sys.argv) != 2 :
    print("error input path of model file")
    sys.exit()

model_file= sys.argv[1]

print("input model file = ", model_file)
print("___________________________")

# Load the TFLite model and allocate tensors.
#interpreter = tflite.Interpreter(model_path="/home/jichoi/w_coral/work_coral/jichoi/model/face.tflite")
#interpreter = tflite.Interpreter(model_path=model_file)

interpreter = make_interpreter(model_file)

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("\ninput details=", input_details)
print("\noutput details=", output_details)

print("\n___________________________")
# Test the model on random input data.
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
print("\ninput shape=", input_shape)
print("\noutput shape=", output_shape)

print("\n___________________________")
input_name = input_details[0]['name']
output_name = output_details[0]['name']
print("\ninput name=", input_name)
print("\noutput name=", output_name)

print("\n___________________________")
input_index = input_details[0]['index']
output_index = output_details[0]['index']
print("\ninput index=", input_index)
print("\noutput index=", output_index)

print("\n___________________________")
#print("signature --------------------------")
#signatures = interpreter.get_signature_list()
#print(signatures)

#print("tensors --------------------------")
#signatures = interpreter.get_tensor_details()
#print(signatures)

