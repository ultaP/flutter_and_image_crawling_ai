# import kagglehub

# # Download latest version
# path = kagglehub.model_download("google/mobilenet-v2/tensorFlow2/035-128-classification")
# print("Path to model files:", path)

import tensorflow as tf

# TensorFlow Lite 모델 로드
interpreter = tf.lite.Interpreter(model_path="C:\img_c\converted_model.tflite")
interpreter.allocate_tensors()

# 입력 텐서의 모양 확인
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
print("Input shape:", input_shape)
