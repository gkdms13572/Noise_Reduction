import tensorflow as tf
import numpy as np
import librosa
from tensorflow.keras.models import load_model


def predict_audio(audio_path, model_path='cow_sound_classifier_final.h5'):
    """
    새로운 오디오 파일이 소 울음소리인지 예측

    Parameters:
    audio_path (str): 분석할 오디오 파일 경로
    model_path (str): 저장된 모델 파일 경로

    Returns:
    tuple: (예측 결과 (bool), 확률)
    """
    try:
        # 모델 로드
        model = load_model(model_path)

        # 특성 추출 (기존 extract_features 함수 사용)
        features = extract_features(audio_path)

        if features is None:
            return False, 0.0

        # 배치 차원 추가 및 채널 차원 추가
        features = features.reshape(1, features.shape[0], features.shape[1], 1)

        # 예측
        prediction = model.predict(features, verbose=0)
        probability = float(prediction[0][0])

        # 확률이 0.5 이상이면 소 울음소리로 판단
        is_cow = probability >= 0.5

        return is_cow, probability

    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
        return False, 0.0


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


# 사용 예시
if __name__ == "__main__":
    # 테스트할 오디오 파일 경로 설정
    test_audio_path = "새 폴더/FSD2024111300120241215112013.WAV"  # 여기에 실제 테스트할 파일 경로를 입력하세요

    # 예측 실행
    is_cow, probability = predict_audio(test_audio_path)

    # 결과 출력
    print("\n=== 예측 결과 ===")
    print(f"입력 파일: {test_audio_path}")
    print(f"소 울음소리일 확률: {probability * 100:.1f}%")
    print(f"판정 결과: {'소 울음소리' if is_cow else '다른 소리'}")