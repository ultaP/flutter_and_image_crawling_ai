import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 데이터 준비
# 이미지 파일 경로와 레이블을 담은 데이터프레임을 사용한다고 가정합니다.
# 이를 이미지 파일 경로 리스트와 레이블 리스트로 변환합니다.
image_paths = [...]  # 이미지 파일 경로 리스트
labels = [...]       # 레이블 리스트

# 데이터를 훈련, 검증, 테스트 세트로 나눕니다.
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=42)

# 데이터를 로드하고 전처리하는 함수 정의
def load_and_preprocess_image(image_path, label):
    # 이미지 로드 및 전처리 작업 수행
    ...

# TensorFlow Dataset 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_and_preprocess_image)
train_dataset = train_dataset.shuffle(buffer_size=len(train_paths)).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(load_and_preprocess_image)
val_dataset = val_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_dataset = test_dataset.map(load_and_preprocess_image)
test_dataset = test_dataset.batch(BATCH_SIZE)

# 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 훈련 및 검증
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset)

# 테스트
test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)
