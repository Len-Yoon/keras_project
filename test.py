import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

# GPU 장치 목록 가져오기
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # GPU가 발견되면, 세부 정보 출력
        for gpu in gpus:
            print("Found a GPU:", gpu)
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 오류 발생 시 출력
        print(e)
    print("GPU가 성공적으로 인식되었습니다. 학습 시 GPU가 사용됩니다.")
else:
    print("GPU를 찾을 수 없습니다. 학습 시 CPU가 사용됩니다.")