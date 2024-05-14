import os
import numpy as np
import tensorflow as tf

# TFLite 모델 경로
model_path = 'converted_model.tflite'

# TensorFlow Lite 인터프리터 생성
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 입력 및 출력 텐서의 인덱스 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 이미지 폴더 경로
image_folder_path = "C:\img_c\downloaded_images\pizza"

# 이미지 파일 목록
image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 이미지 추론 함수
def predict_image(image_path):
    # 이미지 전처리 및 로드
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))  # 모델에 맞는 크기로 조정
    image = tf.cast(image, tf.float32) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가

    # 이미지를 입력 텐서에 설정
    interpreter.set_tensor(input_details[0]['index'], image)

    # 추론 실행
    interpreter.invoke()

    # 결과 텐서 가져오기
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

# 이미지 폴더의 각 이미지에 대한 추론 수행
for image_file in image_files:
    # 이미지 추론
    output_data = predict_image(image_file)
    
    # 추론 결과 출력
    print("Image:", image_file)
    print("Output:", output_data)
