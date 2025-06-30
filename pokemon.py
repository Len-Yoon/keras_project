import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle  # 토크나이저 저장/로드용

# --- 0단계 --- 이미지 경로 및 캡션 데이터 불러오기 및 전처리 ---
image_dir = './sample_data/images/'  # 이미지 폴더 경로
caption_csv_path = './sample_data/compressed_captions.csv'  # 캡션 CSV 파일 경로

abs_image_dir = os.path.abspath(image_dir)  # 절대 경로 변환
if not os.path.exists(abs_image_dir):       # 폴더 존재 확인
    raise FileNotFoundError(f"이미지 폴더가 존재하지 않습니다: {abs_image_dir}")

image_files = os.listdir(abs_image_dir)     # 이미지 파일 리스트

# CSV 파일에서 이미지 파일명과 캡션 불러오기 (utf-8-sig 인코딩)
df = pd.read_csv(caption_csv_path, encoding='utf-8-sig')

# 이미지 경로 보정 함수 정의
def fix_image_path(fname):
    fname = fname.strip()                # 문자열 공백 제거
    p = Path(fname)
    if not p.suffix.lower() == '.png':  # 확장자가 png가 아니면 강제 변경
        p = p.with_suffix('.png')
    if p.is_absolute():                  # 절대경로면 그대로 반환
        return str(p)
    return str(Path(abs_image_dir) / p.name)  # 아니면 이미지 폴더 경로와 합침

# 이미지 경로 컬럼 생성 및 유효한 이미지/캡션만 필터링
df['image_path'] = df['image'].apply(fix_image_path)            # 이미지 경로 생성
df = df[df['image_path'].apply(os.path.exists)]                  # 실제 존재하는 이미지만
df = df[df['caption'].notnull()]                                # 캡션 null 제거
df = df[df['caption'].str.strip() != '']                       # 빈 캡션 제외

# --- 1단계 --- 토크나이저 생성 및 캡션 텍스트 토큰화 ---
tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")  # 상위 5000단어 + OOV 처리
tokenizer.fit_on_texts(df['caption'])  # 캡션 문장에 맞춰 토크나이저 학습

# 토크나이저 저장용 폴더 생성 및 pickle로 저장
tokenizer_save_dir = './save_models'
os.makedirs(tokenizer_save_dir, exist_ok=True)
tokenizer_save_path = os.path.join(tokenizer_save_dir, 'tokenizer.pkl')

with open(tokenizer_save_path, 'wb') as f:
    pickle.dump(tokenizer, f)  # 토크나이저 객체 저장

print(f"토크나이저가 '{tokenizer_save_path}' 경로에 저장되었습니다.")

# 캡션 텍스트를 정수 인덱스 시퀀스로 변환
sequences = tokenizer.texts_to_sequences(df['caption'])
sequences = [seq for seq in sequences if len(seq) > 0]  # 빈 시퀀스 제거
max_seq_len = max(len(seq) for seq in sequences)       # 최대 시퀀스 길이 계산

# 시퀀스 길이 맞추기 위해 패딩 추가 (뒤쪽에 0으로 채움)
padded_seqs = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_len, padding='post')

# --- 2단계 --- 학습/검증 데이터 분리 ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)  # 80:20 분할
train_idx = train_df.index  # 학습 데이터 인덱스
val_idx = val_df.index      # 검증 데이터 인덱스

# --- 3단계 --- 데이터 증강 정의 ---
img_size = (224, 224)  # 이미지 크기 지정
batch_size = 32        # 배치 사이즈 설정

# 케라스 Sequential API로 이미지 증강 레이어 정의
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),  # 좌우 반전
    keras.layers.RandomRotation(0.1),       # ±10% 범위 내 무작위 회전
    keras.layers.RandomZoom(0.1),           # ±10% 범위 내 무작위 줌
])

# --- 4단계 --- 데이터 제너레이터 함수 정의 ---
def data_generator(df, padded_seqs, indices, augment=False):
    """
    인덱스 목록을 순회하며 이미지와 토큰화된 캡션 시퀀스 반환
    이미지 증강이 필요하면 적용
    """
    for i in indices:
        img_path = df.loc[i, 'image_path']
        try:
            # 이미지 로드 및 지정 크기로 리사이징
            img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0  # 0~1로 정규화

            # 증강 적용 여부 판단
            if augment:
                # 4D 텐서(batch 차원 추가)로 변환 후 증강 적용
                img_array = data_augmentation(tf.expand_dims(img_array, 0))
                # 다시 3D 텐서로 축소 후 넘파이 배열로 변환
                img_array = tf.squeeze(img_array, 0).numpy()
        except Exception as e:
            print(f"Warning: 이미지 로드 실패 - {img_path} ({e})")
            continue

        text_seq = padded_seqs[i]  # 캡션 시퀀스

        # 현재 라벨은 임시로 1.0 고정 (필요시 실제 라벨로 교체 가능)
        yield (img_array, text_seq), 1.0

# 학습 데이터셋 생성: 증강 적용, 배치 단위, 무한 반복, 성능 최적화
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(df, padded_seqs, train_idx, augment=True),
    output_signature=(
        (
            tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

# 검증 데이터셋 생성: 증강 없이 배치 단위
val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(df, padded_seqs, val_idx, augment=False),
    output_signature=(
        (
            tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

# --- 5단계 --- 멀티모달 모델 정의 ---

vocab_size = len(tokenizer.word_index) + 1  # 단어 집합 크기 (+1은 패딩 토큰 대비)

# 이미지 입력
img_input = keras.Input(shape=(*img_size, 3))

# EfficientNetB0 (사전학습된 가중치 사용)으로 이미지 특징 추출
base_cnn = keras.applications.EfficientNetB0(include_top=False, input_tensor=img_input, weights='imagenet')
x_img = keras.layers.GlobalAveragePooling2D()(base_cnn.output)  # 공간적 특징 벡터로 변환
x_img = keras.layers.BatchNormalization()(x_img)               # 정규화
x_img = keras.layers.Dropout(0.3)(x_img)                       # 과적합 방지

# 텍스트 입력
text_input = keras.Input(shape=(max_seq_len,))

# 임베딩 층: 단어를 128차원 벡터로 변환
x_text = keras.layers.Embedding(vocab_size, 128, mask_zero=True)(text_input)

# LSTM 층: 시퀀스 내 단어 관계 학습
x_text = keras.layers.LSTM(256)(x_text)

# 정규화 및 드롭아웃
x_text = keras.layers.BatchNormalization()(x_text)
x_text = keras.layers.Dropout(0.3)(x_text)

# 이미지와 텍스트 특징 연결
x = keras.layers.concatenate([x_img, x_text])

# 완전 연결층 및 활성화
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.3)(x)

# 이진 분류 출력층 (sigmoid 활성화)
output = keras.layers.Dense(1, activation='sigmoid')(x)

# 모델 생성
model = keras.Model(inputs=[img_input, text_input], outputs=output)

# 모델 컴파일: Adam 옵티마이저, 이진 크로스엔트로피 손실함수, 정확도 평가
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 요약 출력
model.summary()

# --- 6단계 --- 조기 종료 콜백 설정 ---
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

steps_per_epoch = len(train_idx) // batch_size       # 한 epoch당 학습 스텝 수
validation_steps = len(val_idx) // batch_size        # 검증 스텝 수

# --- 7단계 --- 모델 학습 ---
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    epochs=30,
    callbacks=[early_stop]  # 검증 손실 개선 없으면 학습 조기 종료
)

print("모델 학습 완료!")

# --- 8단계 --- 모델 저장 ---
model_save_dir = './save_models'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, 'multimodal_model.keras')

model.save(model_save_path)  # 모델 전체 저장 (구조 + 가중치 포함)
print(f"모델이 '{model_save_path}' 경로에 저장되었습니다.")

# --- 9단계 --- 학습 결과 시각화 ---

# 정확도 그래프
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')      # 훈련 정확도
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # 검증 정확도
plt.title('Training and Validation Accuracy')                         # 그래프 제목
plt.xlabel('Epochs')                                                  # x축 레이블
plt.ylabel('Accuracy')                                                # y축 레이블
plt.legend()
plt.grid(True)
plt.show()

# 손실 그래프
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')              # 훈련 손실
plt.plot(history.history['val_loss'], label='Validation Loss')        # 검증 손실
plt.title('Training and Validation Loss')                             # 그래프 제목
plt.xlabel('Epochs')                                                  # x축 레이블
plt.ylabel('Loss')                                                   # y축 레이블
plt.legend()
plt.grid(True)
plt.show()
