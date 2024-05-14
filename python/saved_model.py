import tensorflow as tf

loaded_model = tf.keras.models.load_model("C:\\img_c\\mob_image_c.keras")

# # TFLite 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

# # # TFLite 모델로 변환
tflite_model = converter.convert()

# # # TFLite 모델을 파일로 저장
with open("converted_model.tflite", "wb") as f:
     f.write(tflite_model)
