# --- 라이브러리 임포트 ---
import numpy as np  # 수치 계산용
import requests  # URL에서 이미지 다운로드용
from io import BytesIO  # 이미지 바이트 처리용
from PIL import Image  # 이미지 처리용

import tensorflow as tf  # 딥러닝 프레임워크
from tensorflow import keras  # Keras API

import pickle  # tokenizer 저장/로드용

# pad_sequences 임포트 예외처리
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from keras.preprocessing.sequence import pad_sequences

# --- 1. 모델 및 토크나이저 로드 ---
model_path = './save_models/multimodal_model.keras'
tokenizer_path = './save_models/tokenizer.pkl'
max_seq_len = 64  # 학습 때 사용한 최대 시퀀스 길이

print("모델 로드 중...")
model = keras.models.load_model(model_path)
print("모델 로드 완료!")

print("토크나이저 로드 중...")
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
print("토크나이저 로드 완료!")

# --- 2. URL에서 이미지 불러와 전처리 ---
def load_image_from_url(url, target_size=(224, 224)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# --- 3. 로컬 이미지 불러와 전처리 ---
def load_image_from_local(path, target_size=(224, 224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# --- 4. 텍스트 캡션 토큰화 및 패딩 ---
def preprocess_caption(caption, tokenizer, max_seq_len):
    seq = tokenizer.texts_to_sequences([caption])
    padded_seq = pad_sequences(seq, maxlen=max_seq_len, padding='post')
    return padded_seq

# --- 5. URL 이미지 + 캡션 예측 함수 ---
def predict_from_url_and_caption(image_url, caption):
    img_array = load_image_from_url(image_url)
    padded_caption = preprocess_caption(caption, tokenizer, max_seq_len)
    pred_prob = model.predict([img_array, padded_caption])[0][0]

    print(f"Input image and caption predicted positive class probability: {pred_prob:.4f}")

    threshold = 0.5
    if pred_prob >= threshold:
        print(f"Result: Positive class (probability >= {threshold}) → Match likely.")
    else:
        print(f"Result: Negative class (probability < {threshold}) → Match unlikely.")

    return pred_prob

# --- 6. 로컬 이미지 + 캡션 예측 함수 ---
def predict_from_local_and_caption(image_path, caption):
    img_array = load_image_from_local(image_path)
    padded_caption = preprocess_caption(caption, tokenizer, max_seq_len)
    pred_prob = model.predict([img_array, padded_caption])[0][0]

    print(f"Input image and caption predicted positive class probability: {pred_prob:.4f}")

    threshold = 0.5
    if pred_prob >= threshold:
        print(f"Result: Positive class (probability >= {threshold}) → Match likely.")
    else:
        print(f"Result: Negative class (probability < {threshold}) → Match unlikely.")

    return pred_prob

# --- 7. 실행부 ---
if __name__ == "__main__":
    # 1) 인터넷 URL 이미지 테스트 (사용하고 싶으면)
    # test_image_url = "https://example.com/your_image.png"
    # test_caption = "A descriptive caption about the image content."
    # predict_from_url_and_caption(test_image_url, test_caption)

    # 2) 로컬 이미지 테스트 (경로와 캡션 적절히 변경)
    test_image_path = './sample_data/images/26.png'
    test_caption = "A descriptive caption about the image content."

    predict_from_local_and_caption(test_image_path, test_caption)
