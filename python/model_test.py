import tensorflow as tf
import numpy as np
from PIL import Image

# 이미지 파일 경로
image_path = "C:\image_c\downloaded_images\whopper\image_13_base64.jpg"

# 이미지를 읽고 전처리하기
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # 모델 입력 크기에 맞게 조정
    img = np.array(img) / 255.0  # 이미지를 0에서 1 사이의 값으로 정규화
    return img

# 전처리된 이미지 가져오기
input_image = preprocess_image(image_path)

# 모델 로드
loaded_model = tf.keras.models.load_model('my_model.keras')

# 이미지를 입력으로 모델에 전달하여 예측 수행
prediction = loaded_model.predict(np.expand_dims(input_image, axis=0))

# 이진 분류일 경우 예측 결과 출력
class_label = np.argmax(prediction) + 1  # 클래스 인덱스를 레이블로 변환
if class_label == 1:
    print("치즈버거입니다.")
elif class_label == 2:
    print("와퍼입니다.")
else:
    print("예측 결과 없음")