import os
import cv2
import numpy as np
import tensorflow as tf

# 저장한 모델을 로드합니다.
loaded_model = tf.keras.models.load_model("converted_model.tflite")

# 예측할 이미지가 있는 폴더 경로를 지정합니다.
image_folder = "C:\\img_c\\asd\\burger"


# 예측할 이미지 파일 목록을 가져옵니다.
image_files = os.listdir(image_folder)
saved_model_dir = "saved_model"


# 예측 결과를 저장할 리스트를 초기화합니다.
predictions = []

# 각 이미지에 대해 예측을 수행합니다.
for image_file in image_files:
    # 이미지를 읽어와서 전처리합니다.
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # 모델 입력 크기로 조정
    image = image / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    
    # 모델을 사용하여 예측을 수행합니다.
    prediction = loaded_model.predict(image)
    
    # 예측 결과를 해석하여 필요한 형식으로 변환합니다.
    # 여기에서는 간단히 확률 값으로만 저장합니다.
    predictions.append(prediction)

# 모든 예측 결과를 출력합니다.
for i, prediction in enumerate(predictions):
    print(f"이미지 {image_files[i]}의 예측 결과: {prediction}")
