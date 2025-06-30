# --- 라이브러리 임포트 ---
import numpy as np  # 수치 계산용
import requests  # URL에서 이미지 다운로드용
from io import BytesIO  # 이미지 바이트 처리용
from PIL import Image  # 이미지 처리용

import tensorflow as tf  # 딥러닝 프레임워크
from tensorflow import keras  # Keras API

from tensorflow.keras.preprocessing.sequence import pad_sequences  # 텍스트 패딩용

import pickle  # tokenizer 저장/로드용


# --- 1. 모델 및 토크나이저 로드 ---
model_path = './sample_data/multimodal_model.keras'
tokenizer_path = './sample_data/tokenizer.pkl'
max_seq_len = 100  # 학습 때 사용한 최대 시퀀스 길이로 변경 필요

print("모델 로드 중...")
model = keras.models.load_model(model_path)
print("모델 로드 완료!")

print("토크나이저 로드 중...")
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
print("토크나이저 로드 완료!")


# --- 2. 이미지 URL에서 이미지 불러와 전처리 ---
def load_image_from_url(url, target_size=(224, 224)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array


# --- 3. 텍스트 캡션 토큰화 및 패딩 ---
def preprocess_caption(caption, tokenizer, max_seq_len):
    seq = tokenizer.texts_to_sequences([caption])
    padded_seq = pad_sequences(seq, maxlen=max_seq_len, padding='post')
    return padded_seq


# --- 4. 예측 함수 ---
def predict_from_url_and_caption(image_url, caption):
    img_array = load_image_from_url(image_url)
    padded_caption = preprocess_caption(caption, tokenizer, max_seq_len)
    pred_prob = model.predict([img_array, padded_caption])[0][0]
    print(f"입력된 이미지와 캡션에 대한 예측 확률 (Positive class): {pred_prob:.4f}")
    return pred_prob


# --- 5. 예측 테스트 ---
if __name__ == "__main__":
    test_image_url = 'https://example.com/sample_image.png'  # 실제 이미지 URL로 바꾸세요
    test_caption = "A descriptive caption about the image content."

    predict_from_url_and_caption(test_image_url, test_caption)
