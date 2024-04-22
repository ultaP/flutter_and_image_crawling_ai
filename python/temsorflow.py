import tensorflow as tf
import pandas as pd

# CSV 파일 경로
csv_file = 'data.csv'

# CSV 파일을 데이터프레임으로 읽어오기
data = pd.read_csv(csv_file)

# 이미지 파일 경로와 라벨을 각각 리스트로 추출
image_paths = data['image_path'].tolist()
labels = data['label'].tolist()

# 이미지 파일을 불러와서 TensorFlow Dataset으로 변환하는 함수 정의
def load_and_preprocess_image(image_path, label):
    # 이미지 읽기
    image = tf.io.read_file(image_path)
    # JPEG 이미지를 디코딩하고 크기를 조정
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # 이미지 크기 조정
    # 이미지를 0에서 1 사이의 값으로 정규화
    image = image / 255.0
    return image, label

# 이미지 파일 경로와 라벨을 매핑하여 Dataset 생성
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
# Dataset에 이미지 로드 및 전처리 함수를 적용
dataset = dataset.map(load_and_preprocess_image)

# 학습을 위한 배치로 나누고 섞기
BATCH_SIZE = 32
dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(BATCH_SIZE)

# 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(dataset, epochs=10)
model.save('my_model.keras')
