# --- 라이브러리 임포트 ---
# 기본 및 helper 라이브러리
import numpy as np  # 수치 계산, 배열 처리를 위한 라이브러리
import pandas as pd  # 데이터 분석 및 CSV 파일 로드를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화(그래프)를 위한 라이브러리

# TensorFlow 및 Keras 관련 라이브러리
import tensorflow as tf  # 딥러닝 프레임워크
from tensorflow import keras  # TensorFlow의 고수준 API로, 딥러닝 모델을 쉽게 만들 수 있게 도와줌

# Scikit-learn 관련 라이브러리 (데이터 전처리 및 모델 평가용)
from sklearn.model_selection import train_test_split  # 데이터를 훈련용과 검증용으로 나누는 함수
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 수치 데이터 스케일링, 문자열 라벨을 숫자로 변환하는 도구


# --- 1. 데이터 로드 및 초기 확인 ---
pocket_tcg_data = "./sample_data/pocket-tcg-dataset.csv"  # 데이터 파일 경로 지정
print("="*50)
print("[1단계] 데이터 로드 및 초기 확인")
print("="*50)
print(f"'{pocket_tcg_data}' 파일 로드를 시작합니다...")
# pandas의 read_csv 함수를 사용해 CSV 파일을 DataFrame 형태로 불러옴
pocket_df = pd.read_csv(pocket_tcg_data)
print("데이터 로드 완료!")

# .head(): 데이터의 첫 5줄을 샘플로 보여줌으로써 데이터 구조를 파악
print("\n--- 원본 데이터 샘플 (상위 5개) ---")
print(pocket_df.head())

# .shape: 데이터의 행과 열 개수(차원)를 보여줌
# .info(): 각 컬럼의 데이터 타입, 누락되지 않은 값의 개수 등 요약 정보를 보여줌
print("\n--- 원본 데이터 정보 ---")
print("데이터 형태:", pocket_df.shape)
pocket_df.info()

# .columns.tolist(): 모든 컬럼의 이름을 리스트 형태로 보여줌
print("\n--- 원본 데이터 컬럼 목록 ---")
print(pocket_df.columns.tolist())


# --- 2. 목표(Target) 및 피처(Feature) 설정 및 정제 ---
print("\n" + "="*50)
print("[2단계] 목표 및 피처 설정, 데이터 정제")
print("="*50)
# 우리가 예측하고 싶은 목표 변수(y)를 'Rarity'로 설정
TARGET_COLUMN = 'Rarity'
# 예측에 사용할 입력 변수(X), 즉 피처들을 리스트로 설정
FEATURE_COLUMNS = ['Set Name', 'Total Cards', 'Card Name']

print(f"타겟 변수 (y): '{TARGET_COLUMN}'")
print(f"피처 변수 (X): {FEATURE_COLUMNS}")

# .dropna(): 특정 컬럼에 빈 값(NaN, 결측치)이 있는 행을 학습에서 제외하기 위해 제거함
# .copy(): 원본 DataFrame(pocket_df)을 보존하기 위해 복사본을 만들어 작업함
df_clean = pocket_df.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS).copy()
print(f"\n결측값 제거 전 데이터 수: {len(pocket_df)}")
print(f"결측값 제거 후 데이터 수: {len(df_clean)}")

# .value_counts(): 특정 컬럼에 있는 각 값들의 개수를 세어줌 (데이터 불균형 확인에 유용)
print("\n--- 타겟 변수의 분포 확인 (어떤 종류의 Rarity가 몇 개씩 있는지) ---")
print(df_clean[TARGET_COLUMN].value_counts())


# --- 3. 데이터 전처리 (분류 모델용) ---
print("\n" + "="*50)
print("[3단계] 데이터 전처리 (문자 -> 숫자)")
print("="*50)

# 3-1. 타겟 변수(y)를 숫자로 변환 (Label Encoding)
# LabelEncoder: 'Common', 'Rare' 같은 문자열 라벨을 0, 1, 2... 와 같은 컴퓨터가 이해할 수 있는 숫자로 변환
label_encoder = LabelEncoder()
# .fit_transform(): 어떤 문자열이 어떤 숫자에 해당하는지 규칙을 학습(fit)하고, 실제 변환(transform)까지 수행
y = label_encoder.fit_transform(df_clean[TARGET_COLUMN])

print("\n--- 3-1. 타겟(Rarity) 레이블 인코딩 결과 ---")
# np.unique(y): 중복을 제거한 고유한 클래스의 개수를 셈
print(f"'{TARGET_COLUMN}' 컬럼이 {len(np.unique(y))}개의 숫자 클래스로 변환되었습니다.")
print("변환된 숫자(레이블) 샘플 (상위 10개):", y[:10])
# .classes_: 어떤 클래스(문자열)가 어떤 숫자(인덱스)에 매핑되었는지 보여줌
print("숫자-클래스 매핑:", list(enumerate(label_encoder.classes_)))
num_classes = len(label_encoder.classes_)  # 전체 클래스의 개수를 저장


# 3-2. 피처(X) 전처리
print("\n--- 3-2. 피처 전처리 결과 ---")
# 범주형 피처 원-핫 인코딩
categorical_features = ['Set Name', 'Card Name']
# pd.get_dummies: 문자열로 된 범주형 데이터를 0과 1로 이루어진 여러 개의 컬럼으로 변환 (원-핫 인코딩)
# drop_first=True: 다중공선성 문제를 피하기 위해 첫 번째 카테고리 컬럼은 제거 (선택 사항)
X_categorical_encoded = pd.get_dummies(df_clean[categorical_features], drop_first=True)
print("범주형 피처 원-핫 인코딩 후 형태:", X_categorical_encoded.shape)
print("원-핫 인코딩 샘플 (상위 5개):")
print(X_categorical_encoded.head())

# 수치형 피처 스케일링
numerical_features = ['Total Cards']
# StandardScaler: 수치 데이터의 평균을 0, 표준편차를 1로 만들어 데이터의 단위를 통일시킴 (모델 학습 안정화에 도움)
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(df_clean[numerical_features])
X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features, index=X_categorical_encoded.index)
print("\n수치형 피처 스케일링 후 샘플 (상위 5개):")
print(X_numerical_scaled_df.head())

# 최종 피처 데이터셋 X 완성
# pd.concat: 전처리된 범주형 데이터와 수치형 데이터를 다시 하나의 DataFrame으로 합침
X = pd.concat([X_categorical_encoded, X_numerical_scaled_df], axis=1)
print("\n[완료] 최종 피처 데이터셋(X) 형태:", X.shape)
print("최종 피처 데이터셋 샘플 (상위 5개):")
print(X.head())


# --- 4. 훈련 / 검증 데이터 분리 ---
print("\n" + "="*50)
print("[4단계] 훈련 / 검증 데이터 분리")
print("="*50)
# train_test_split: 데이터를 훈련용과 검증(테스트)용으로 나눔
# test_size=0.2: 전체 데이터의 20%를 검증용으로 사용
# random_state=42: 매번 동일한 방식으로 데이터를 나누기 위해 시드(seed)를 고정 (결과 재현에 필요)
# stratify=y: 훈련/검증 데이터에 타겟(y)의 클래스 비율이 원본과 동일하게 유지되도록 함 (불균형 데이터에 중요)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"훈련 데이터 (X_train): {X_train.shape}")
print(f"훈련 타겟 (y_train): {y_train.shape}")
print(f"검증 데이터 (X_val): {X_val.shape}")
print(f"검증 타겟 (y_val): {y_val.shape}")


# --- 5. Keras 모델 생성 (Dropout 추가 버전) ---
print("\n" + "="*50)
print("[5단계] Keras 모델 생성 (Dropout 추가)")
print("="*50)
# keras.Sequential: 신경망 층을 순서대로 쌓아 모델을 만듦
model = keras.Sequential([
    # Dense: 가장 기본적인 완전 연결 신경망 층
    # 128: 이 층의 뉴런(노드) 개수
    # activation='relu': 활성화 함수로 'ReLU'를 사용 (음수는 0으로, 양수는 그대로 전달)
    # input_shape: 첫 번째 층에만 필요하며, 입력 데이터의 피처 개수를 알려줌
    keras.layers.Dense(128, activation='relu', input_shape=[X.shape[1]]),
    # Dropout: 과적합을 방지하기 위한 규제 기법. 훈련 시 지정된 비율(40%)만큼 뉴런을 무작위로 비활성화
    keras.layers.Dropout(0.3),

    # 두 번째 Dense 층
    keras.layers.Dense(64, activation='relu'),
    # Dropout 층 추가
    keras.layers.Dropout(0.3),

    # 출력층: 클래스의 개수(num_classes)만큼 뉴런을 가짐
    # activation='softmax': 모든 출력 뉴런의 값을 합하면 1이 되는 확률 값으로 변환 (다중 클래스 분류에 사용)
    keras.layers.Dense(num_classes, activation='softmax')
])
# model.summary(): 생성된 모델의 구조와 파라미터 수를 표로 요약하여 보여줌
model.summary()


# --- 6. 모델 컴파일 ---
print("\n" + "="*50)
print("[6단계] 모델 컴파일")
print("="*50)
# model.compile: 생성된 모델의 학습 방식(옵티마이저), 오차 측정 방식(손실 함수), 평가 지표를 설정
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # optimizer: 모델을 최적화하는 알고리즘. Adam에 학습률을 직접 지정
    loss='sparse_categorical_crossentropy',  # loss: 손실 함수. 모델의 예측이 얼마나 틀렸는지 측정. 타겟(y)이 정수(0,1,2..)일 때 사용
    metrics=['accuracy']  # metrics: 훈련 과정을 모니터링할 평가 지표 (정확도)
)
print("컴파일 완료!")

# --- 7. 모델 학습 ---
print("\n" + "="*50)
print("[7단계] 모델 학습 시작")
print("="*50)
# model.fit: 모델에 데이터를 주입하여 실제 학습을 시작
# history 객체에는 에포크별 훈련/검증 손실과 정확도가 저장됨
history = model.fit(
    X_train, y_train,  # 훈련 데이터와 훈련 타겟
    epochs=20,  # epochs: 전체 훈련 데이터를 몇 번 반복해서 학습할지 결정
    batch_size=32,  # batch_size: 한 번에 몇 개의 데이터를 보고 가중치를 업데이트할지 결정
    validation_data=(X_val, y_val),  # validation_data: 매 에포크가 끝날 때마다 모델 성능을 검증할 데이터
    verbose=1  # verbose: 학습 과정을 얼마나 자세히 출력할지 결정 (1: 진행 바 표시)
)
print("모델 학습 완료!")


# --- 8. 학습 결과 시각화 (정확도) ---
print("\n" + "="*50)
print("[8단계] 학습 결과 시각화")
print("="*50)
# matplotlib 라이브러리를 사용해 그래프를 그림
plt.figure(figsize=(10, 6))
# history에 저장된 훈련 정확도(파란선)와 검증 정확도(주황선)를 그래프로 그림
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')  # 그래프 제목
plt.xlabel('Epochs')  # x축 라벨
plt.ylabel('Accuracy')  # y축 라벨
plt.legend()  # 범례 표시
plt.grid(True)  # 격자 표시
plt.show()  # 그래프를 화면에 출력


# --- 9. 최종 모델 평가 ---
print("\n" + "="*50)
print("[9단계] 최종 모델 평가")
print("="*50)
# model.evaluate: 학습이 완료된 모델의 최종 성능을 검증 데이터로 평가하여 손실과 정확도를 반환
final_loss, final_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"검증 세트에 대한 최종 손실(Loss)    : {final_loss:.4f}")
print(f"검증 세트에 대한 최종 정확도(Accuracy): {final_accuracy:.4f} ({final_accuracy:.2%})")