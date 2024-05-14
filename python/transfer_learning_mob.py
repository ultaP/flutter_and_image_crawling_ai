import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
from keras.layers import Dense
from keras.models import Model
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
tf.keras.utils.to_categorical
tf.config.run_functions_eagerly(True)

# CSV 파일에서 데이터 읽어오기
df = pd.read_csv("food.csv")

# 이미지 경로와 라벨 추출
image_paths = df['image_path'].tolist()
labels = df['label'].tolist()

# 클래스별로 데이터를 그룹화
class_groups = {}
for image_path, label in zip(image_paths, labels):
    if label not in class_groups:
        class_groups[label] = []
    class_groups[label].append((image_path, label))

# 각 클래스별로 데이터를 섞음
for label, data_list in class_groups.items():
    class_groups[label] = shuffle(data_list, random_state=42)

# 각 클래스별로 학습 및 테스트 데이터셋으로 나눔
train_data = []
test_data = []
for label, data_list in class_groups.items():
    train_data_per_label, test_data_per_label = train_test_split(data_list, test_size=0.2, random_state=42)
    train_data.extend(train_data_per_label)
    test_data.extend(test_data_per_label)

# 이미지 경로와 라벨을 튜플로 묶은 리스트를 numpy 배열로 변환
train_data_array = np.array(train_data)
test_data_array = np.array(test_data)

# 라벨 인코딩
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_data_array[:, 1])
test_labels_encoded = label_encoder.transform(test_data_array[:, 1])

# 이미지 경로와 인코딩된 라벨을 추출
train_image_paths = train_data_array[:, 0]
train_labels = train_labels_encoded
test_image_paths = test_data_array[:, 0]
test_labels = test_labels_encoded

num_classes = len(class_groups)

# 이미지 데이터를 실제 이미지로 로드하여 적절한 크기로 조정
def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))  # CNN 모델에 맞는 크기로 조정
    image = tf.cast(image, tf.float32) / 255.0  # 정규화
    newlabel = tf.one_hot(label, depth=num_classes)
    return image, newlabel
def load_and_preprocess_image2(image_path, label):
    # 이미지 읽기
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))  # CNN 모델에 맞는 크기로 조정
    image = tf.cast(image, tf.float32) / 255.0  # 정규화
    
     # 학습 데이터인 경우에만 좌우 반전된 이미지를 추가
    flipped_image = tf.image.flip_left_right(image)
    
    # 원본 이미지와 반전된 이미지를 쌍으로 묶어서 반환
   # return image, label
    return (flipped_image), (label)

def load_and_preprocess_image3(image_path, label):
    # 이미지 읽기
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))  # CNN 모델에 맞는 크기로 조정
    image = tf.cast(image, tf.float32) / 255.0  # 정규화
    
    rotated_image = tf.image.rot90(image)
    
    # 원본 이미지와 반전된 이미지를 쌍으로 묶어서 반환
   # return image, label
    return (rotated_image), (label)

def load_and_preprocess_image4(image_path, label):
    # 이미지 읽기
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))  # CNN 모델에 맞는 크기로 조정
    image = tf.cast(image, tf.float32) / 255.0  # 정규화
    
    shifted_image = tf.image.crop_and_resize(tf.expand_dims(image, 0),
                                          boxes=[[0, 0.2, 1, 0.8]],
                                          box_indices=[0],  # 이미지의 인덱스
                                          crop_size=(224, 224))
    shifted_image = shifted_image[0]
    
    # 원본 이미지와 반전된 이미지를 쌍으로 묶어서 반환
   # return image, label
    return (shifted_image), (label)

def load_and_preprocess_image5(image_path, label):
    # 이미지 읽기
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))  # CNN 모델에 맞는 크기로 조정
    image = tf.cast(image, tf.float32) / 255.0  # 정규화
    
    scaled_image = tf.image.resize_with_pad(image, int(224*0.8), int(224*0.8))
    scaled_image = tf.image.resize_with_pad(scaled_image, 224, 224)
    
    # 원본 이미지와 반전된 이미지를 쌍으로 묶어서 반환
   # return image, label
    return (scaled_image), (label)

# 데이터셋 생성 및 전처리 적용
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset2 = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset3 = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset4 = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset5 = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))

train_dataset = train_dataset.map(load_and_preprocess_image)

train_dataset_flipped = train_dataset2.map(load_and_preprocess_image2)
train_dataset_flipped2 = train_dataset2.map(load_and_preprocess_image3)
train_dataset_flipped3 = train_dataset2.map(load_and_preprocess_image4)
train_dataset_flipped4 = train_dataset2.map(load_and_preprocess_image5)
# print(len(train_dataset))
# train_dataset = train_dataset.concatenate(train_dataset_flipped)
# print(len(train_dataset))
# train_dataset = train_dataset.concatenate(train_dataset_flipped2)
# print(len(train_dataset))
# train_dataset = train_dataset.concatenate(train_dataset_flipped3)
# print(len(train_dataset))
# train_dataset = train_dataset.concatenate(train_dataset_flipped4)

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(load_and_preprocess_image)
print(len(train_dataset))

# MobileNetV2 모델 불러오기 (사전 훈련된 가중치 포함)
base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# 새로운 Fully Connected 레이어 추가
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)

# 클래스 수 계산
num_classes = len(class_groups)

# 모델 컴파일 시 출력 레이어 수정
predictions = Dense(num_classes, activation='softmax')(x)

# 새로운 모델 생성 (사전 훈련된 부분은 고정)
model = Model(inputs=base_model.input, outputs=predictions)

# 사전 훈련된 부분은 학습되지 않도록 설정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
# 모델을 저장할 파일명 설정
#checkpoint_path = "mob_model.keras"
checkpoint_path = "mob_image_c.keras"
# ModelCheckpoint 콜백 정의
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                      monitor='val_accuracy',  # 정확도를 모니터링
                                      save_best_only=True,    # 최상의 모델만 저장
                                      save_weights_only=False,  # 모델의 전체 상태 저장
                                      verbose=1)              # 저장 시 메시지 출력

# 모델 학습 시 ModelCheckpoint 콜백 추가
history = model.fit(train_dataset.batch(32), 
                    validation_data=test_dataset.batch(32), 
                    epochs=30, 
                    callbacks=[checkpoint_callback])# model.save("my_model.h5")
# loaded_model = load_model("my_model.h5")
# # 예측 수행
# predictions = loaded_model.predict(test_dataset.batch(32))
# print(predictions)

# 모델을 TFLite 형식으로 변환하여 저장
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("mobilenetv2.tflite", "wb") as f:
#     f.write(tflite_model)