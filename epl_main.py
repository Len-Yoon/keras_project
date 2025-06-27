# --- 라이브러리 임포트 ---
# 기본 및 helper 라이브러리
import numpy as np  # 수치 계산, 배열 처리를 위한 라이브러리
import pandas as pd  # 데이터 분석 및 CSV 파일 로드를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화(그래프)를 위한 라이브러리
import seaborn as sns # 더 예쁜 시각화를 위한 라이브러리 (Confusion Matrix에 사용)

# TensorFlow 및 Keras 관련 라이브러리
import tensorflow as tf  # 딥러닝 프레임워크
from tensorflow import keras  # TensorFlow의 고수준 API로, 딥러닝 모델을 쉽게 만들 수 있게 도와줌

# Scikit-learn 관련 라이브러리 (데이터 전처리 및 모델 평가용)
from sklearn.model_selection import train_test_split  # 데이터를 훈련용과 검증용으로 나누는 함수
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 수치 데이터 스케일링, 문자열 라벨을 숫자로 변환하는 도구
from sklearn.metrics import confusion_matrix, classification_report  # 모델 성능 평가를 위한 함수


# --- 1. 데이터 로드 및 초기 확인 ---
epl_data = "./sample_data/epl_player_stats_24_25.csv"  # 데이터 파일 경로 수정 (사용자 파일명 기준)
print("="*50)
print("[1단계] 데이터 로드 및 초기 확인")
print("="*50)
print(f"'{epl_data}' 파일 로드를 시작합니다...")
# pandas의 read_csv 함수를 사용해 CSV 파일을 DataFrame 형태로 불러옴
epl_df = pd.read_csv(epl_data)
print("데이터 로드 완료!")

print(epl_df.columns.tolist())

# .head(): 데이터의 첫 5줄을 샘플로 보여줌으로써 데이터 구조를 파악
print("\n--- 원본 데이터 샘플 (상위 5개) ---")
print(epl_df.head())

# .info(): 각 컬럼의 데이터 타입, 누락되지 않은 값의 개수 등 요약 정보를 보여줌
print("\n--- 원본 데이터 정보 ---")
print("데이터 형태:", epl_df.shape)
epl_df.info()

# .columns.tolist(): 모든 컬럼의 이름을 리스트 형태로 보여줌
print("\n--- 원본 데이터 컬럼 목록 ---")
print(epl_df.columns.tolist())


# --- 2. 목표(Target) 및 피처(Feature) 설정 및 정제 ---
print("\n" + "="*50)
print("[2단계] 목표 및 피처 설정, 데이터 정제")
print("="*50)

# 2-1. 골키퍼(GK) 데이터 제외
# 포지션 예측 문제를 단순화하기 위해, 스탯이 다른 필드 플레이어와 크게 다른 골키퍼는 일단 제외합니다.
df_field_players = epl_df[epl_df['Position'] != 'GK'].copy()
print(f"골키퍼 제외 후 데이터 수: {len(df_field_players)}")

# 2-2. 목표 및 피처 컬럼 정의 (EPL 데이터에 맞게 수정)
# 우리가 예측하고 싶은 목표 변수(y)를 'Position'으로 설정
TARGET_COLUMN = 'Position'
# 예측에 사용할 입력 변수(X), 즉 피처들을 리스트로 설정
FEATURE_COLUMNS = ['Appearances', 'Minutes', 'Goals', 'Assists', 'Shots', 'Shots On Target']

print(f"\n타겟 변수 (y): '{TARGET_COLUMN}'")
print(f"피처 변수 (X): {FEATURE_COLUMNS}")

# 2-3. 결측값(NaN)이 있는 행 제거
# 선택한 피처나 타겟에 결측값이 있는 선수는 학습에서 제외
df_final = df_field_players.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS)
print(f"최종 사용할 데이터 수: {len(df_final)}")

# 2-4. 타겟 변수 분포 확인
# .value_counts(): 포지션별 선수 인원수를 확인 (데이터 불균형 확인)
print("\n--- 타겟 변수의 분포 확인 (포지션별 선수 수) ---")
print(df_final[TARGET_COLUMN].value_counts())


# --- 3. 데이터 전처리 (분류 모델용) ---
print("\n" + "="*50)
print("[3단계] 데이터 전처리 (문자 -> 숫자)")
print("="*50)

# 3-1. 타겟 변수(y)를 숫자로 변환 (Label Encoding)
# LabelEncoder: 'DF', 'MF', 'FW' 같은 문자열 라벨을 0, 1, 2... 와 같은 숫자로 변환
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_final[TARGET_COLUMN])
num_classes = len(label_encoder.classes_) # 전체 클래스의 개수(포지션 종류 수)를 저장

print("\n--- 3-1. 타겟(Position) 레이블 인코딩 결과 ---")
print(f"'{TARGET_COLUMN}' 컬럼이 {num_classes}개의 숫자 클래스로 변환되었습니다.")
# .classes_: 어떤 포지션이 어떤 숫자에 매핑되었는지 보여줌
print("숫자-클래스 매핑:", list(enumerate(label_encoder.classes_)))


# 3-2. 피처(X) 전처리 (EPL 데이터에 맞게 수정)
print("\n--- 3-2. 피처 전처리 결과 ---")
# 이번에는 선택한 피처들이 모두 수치형 데이터이므로, 스케일링만 진행합니다.
# (원-핫 인코딩은 범주형 피처가 있을 때 사용)
X_raw = df_final[FEATURE_COLUMNS]

# StandardScaler: 모든 수치 피처들의 단위를 통일시켜 모델 학습을 안정화
scaler = StandardScaler()
X = scaler.fit_transform(X_raw) # 최종 피처 데이터셋 X 완성
print("모든 피처에 대해 스케일링 완료.")
print("최종 피처 데이터셋(X) 형태:", X.shape)


# --- 4. 훈련 / 검증 데이터 분리 ---
print("\n" + "="*50)
print("[4단계] 훈련 / 검증 데이터 분리")
print("="*50)
# train_test_split: 데이터를 훈련용(80%)과 검증용(20%)으로 나눔
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"훈련 데이터 (X_train): {X_train.shape}")
print(f"검증 데이터 (X_val): {X_val.shape}")


# --- 5. Keras 모델 생성 ---
print("\n" + "="*50)
print("[5단계] Keras 모델 생성")
print("="*50)
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=[X_train.shape[1]]), # 입력 피처 개수에 맞게 input_shape 설정
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax') # 출력층은 포지션 종류 수만큼 뉴런 설정
])
model.summary()


# --- 6. 모델 컴파일 ---
print("\n" + "="*50)
print("[6단계] 모델 컴파일")
print("="*50)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy', # 타겟(y)이 정수 형태이므로 sparse_... 사용
    metrics=['accuracy']
)
print("컴파일 완료!")

# --- 7. 모델 학습 ---
print("\n" + "="*50)
print("[7단계] 모델 학습 시작")
print("="*50)

history = model.fit(
    X_train, y_train,
    epochs=100, # 최대 100번 학습하되, EarlyStopping 조건이 되면 멈춤
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)
print("모델 학습 완료!")


# --- 8. 학습 결과 시각화 ---
print("\n" + "="*50)
print("[8단계] 학습 결과 시각화")
print("="*50)
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# --- 9. 최종 모델 평가 ---
print("\n" + "="*50)
print("[9단계] 최종 모델 평가")
print("="*50)
final_loss, final_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"검증 세트에 대한 최종 손실(Loss)    : {final_loss:.4f}")
print(f"검증 세트에 대한 최종 정확도(Accuracy): {final_accuracy:.4f} ({final_accuracy:.2%})")

# Confusion Matrix 및 Classification Report
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
class_names = label_encoder.classes_

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual (True) Position')
plt.xlabel('Predicted Position')
plt.show()

# 분류 리포트 출력
print("\n--- Classification Report ---")
print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))