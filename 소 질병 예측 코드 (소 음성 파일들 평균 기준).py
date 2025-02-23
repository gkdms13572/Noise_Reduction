import os
import numpy as np
import librosa
import glob
import tensorflow as tf
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler


def extract_features(file_path, duration=30):
    """오디오 파일에서 특성 추출"""
    try:
        # 오디오 로드
        audio, sr = librosa.load(file_path, duration=duration)

        # 고정된 길이로 패딩 또는 잘라내기
        target_length = sr * duration
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
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

        # 고정된 크기로 리사이즈
        target_time_steps = 1292
        if mel_spec_db.shape[1] > target_time_steps:
            mel_spec_db = mel_spec_db[:, :target_time_steps]
        else:
            padding_width = target_time_steps - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding_width)), mode='constant')

        return mel_spec_db.flatten()  # 1차원 벡터로 변환
    except Exception as e:
        print(f"파일 처리 중 오류 발생 ({file_path}): {str(e)}")
        return None


def calculate_disease_probability(file_path, normal_features_mean, normal_features_std):
    """
    소리의 특징 벡터와 정상 소리의 평균, 표준편차를 비교하여
    질병 가능성 계산
    """
    # 특징 추출
    features = extract_features(file_path)
    if features is None:
        return None

    # 유클리디안 거리 계산
    distance = euclidean(features, normal_features_mean)

    # 표준편차 기반 거리 정규화
    normalized_distance = distance / np.linalg.norm(normal_features_std)

    # 거리를 기반으로 질병 확률 계산 (시그모이드 함수 사용)
    disease_probability = 1 / (1 + np.exp(-normalized_distance))

    return disease_probability * 100  # 백분율로 변환


def analyze_cow_sounds():
    # 정상 소 울음소리 경로
    normal_cow_paths = glob.glob("noise_remove_audio/*.[Ww][Aa][Vv]")

    # 모든 정상 소 울음소리의 특징 추출
    normal_features = []
    for path in normal_cow_paths:
        feature = extract_features(path)
        if feature is not None:
            normal_features.append(feature)

    # 정상 소 울음소리 특징의 평균과 표준편차 계산
    normal_features_mean = np.mean(normal_features, axis=0)
    normal_features_std = np.std(normal_features, axis=0)

    print("=== 소 울음소리 질병 예측 시스템 ===")
    print(f"분석된 정상 소 울음소리 샘플 수: {len(normal_features)}")

    # 테스트할 소리 파일 경로 (예시)
    test_paths = [
        "새 폴더/a93bc491-f50f-4ab7-9f78-df62e4970756.WAV",  # 테스트할 실제 파일 경로로 대체
        "새 폴더/ab0fbde8-016e-4b94-b853-11da36e03a02.WAV"  # 테스트할 실제 파일 경로로 대체
    ]

    print("\n개별 소리 분석:")
    for path in test_paths:
        try:
            probability = calculate_disease_probability(
                path,
                normal_features_mean,
                normal_features_std
            )

            if probability is not None:
                print(f"\n파일: {os.path.basename(path)}")
                print(f"질병 가능성: {probability:.2f}%")

                # 질병 위험도 해석
                if probability < 20:
                    print("해석: 건강할 가능성 높음")
                elif 20 <= probability < 50:
                    print("해석: 주의 필요")
                elif 50 <= probability < 80:
                    print("해석: 질병 의심")
                else:
                    print("해석: 즉시 수의사 진단 필요")

        except Exception as e:
            print(f"파일 분석 중 오류: {path}")
            print(e)


if __name__ == "__main__":
    analyze_cow_sounds()