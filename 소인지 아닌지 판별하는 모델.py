import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


def extract_features(file_path, audio_data=None, sr=None):
    """오디오 파일에서 특성 추출"""
    try:
        # 오디오 로드 또는 제공된 데이터 사용
        if audio_data is None:
            # 파일에서 직접 로드 - 30초로 변경
            audio, sr = librosa.load(file_path, duration=30)
        else:
            # 제공된 오디오 데이터 사용
            audio = audio_data

        # 고정된 길이로 패딩 또는 잘라내기 (30초)
        target_length = sr * 30  # 30초
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            # 부족한 길이는 0으로 패딩
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        # mel-spectrogram 변환
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64,
            n_fft=2048,
            hop_length=512,
            fmax=8000
        )

        # dB 스케일 변환
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 정규화
        mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-6)
        mel_spec_db = np.clip(mel_spec_db, -3, 3)
        mel_spec_db = (mel_spec_db + 3) / 6

        # 고정된 크기로 리사이즈 (30초 기준으로 변경)
        target_time_steps = 1292  # 30초에 대한 시간 스텝 (hop_length 고려)
        if mel_spec_db.shape[1] > target_time_steps:
            mel_spec_db = mel_spec_db[:, :target_time_steps]
        else:
            padding_width = target_time_steps - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding_width)), mode='constant')

        return mel_spec_db
    except Exception as e:
        print(f"파일 처리 중 오류 발생 ({file_path}): {str(e)}")
        return None


def prepare_data(cow_sound_dir, non_cow_sound_dir):
    """소 울음소리와 비-소 울음소리 데이터 로드"""
    X = []  # 특성 데이터
    y = []  # 라벨 (1: 소 울음소리, 0: 비-소 울음소리)

    # 1. 소 울음소리 로드 (10000개)
    print("\n소 울음소리 데이터 로딩 중...")
    cow_count = 0
    for filename in os.listdir(cow_sound_dir):
        if filename.endswith(('.wav', '.WAV')):
            file_path = os.path.join(cow_sound_dir, filename)
            features = extract_features(file_path)

            if features is not None and features.shape == (64, 1292):
                X.append(features)
                y.append(1)  # 소 울음소리는 1
                cow_count += 1

                if cow_count % 500 == 0:
                    print(f"처리된 소 울음소리 파일 수: {cow_count}")

                if cow_count >= 10000:  # 10000개 제한
                    break

    # 2. 비-소 울음소리 로드 (3300개)
    print("\n비-소 울음소리 데이터 로딩 중...")
    non_cow_count = 0
    for filename in os.listdir(non_cow_sound_dir):
        if filename.endswith(('.wav', '.WAV')):
            file_path = os.path.join(non_cow_sound_dir, filename)
            features = extract_features(file_path)

            if features is not None and features.shape == (64, 1292):
                X.append(features)
                y.append(0)  # 비-소 울음소리는 0
                non_cow_count += 1

                if non_cow_count % 500 == 0:
                    print(f"처리된 비-소 울음소리 파일 수: {non_cow_count}")

                if non_cow_count >= 3300:  # 3300개 제한
                    break

    print(f"\n총 처리된 파일 수:")
    print(f"- 소 울음소리: {cow_count}개")
    print(f"- 비-소 울음소리: {non_cow_count}개")

    return np.array(X), np.array(y)


def create_model(input_shape):
    """CNN 분류 모델 생성"""
    model = models.Sequential([
        # 첫 번째 컨볼루션 블록
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # 두 번째 컨볼루션 블록
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # 세 번째 컨볼루션 블록
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Dense 레이어
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    print("=== 소 울음소리 분류 모델 훈련 시작 ===")

    # 데이터 디렉토리 설정
    cow_sound_dir = 'noise_remove_audio'  # 소 울음소리 디렉토리
    non_cow_sound_dir = 'noise_remove_audio(2)'  # 비-소 울음소리 디렉토리

    # 1. 데이터 준비
    X, y = prepare_data(cow_sound_dir, non_cow_sound_dir)
    if len(X) == 0:
        print("오류: 데이터를 찾을 수 없습니다!")
        exit()

    # 데이터 형태 변환
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    print(f"데이터 shape: {X.shape}")

    # 2. 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n데이터 분할:")
    print(f"- 훈련 데이터: {len(X_train)}개")
    print(f"- 검증 데이터: {len(X_val)}개")

    # 3. 모델 생성 및 훈련
    print("\n모델 훈련 시작...")
    model = create_model(X_train.shape[1:])

    # Early stopping 및 모델 체크포인트 설정
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]

    # 클래스 가중치 계산
    total_samples = len(y_train)
    neg_samples = np.sum(y_train == 0)
    pos_samples = np.sum(y_train == 1)
    weight_for_0 = (1 / neg_samples) * (total_samples / 2.0)
    weight_for_1 = (1 / pos_samples) * (total_samples / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    # 모델 훈련
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # 4. 최종 성능 평가
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n=== 최종 성능 ===")
    print(f"검증 정확도: {accuracy * 100:.1f}%")
    print(f"검증 손실: {loss:.4f}")

    # 5. 모델 저장
    model.save('cow_sound_classifier_final.h5')
    print("\n모델이 저장되었습니다.")