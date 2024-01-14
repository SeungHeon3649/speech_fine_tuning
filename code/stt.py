# from google.cloud import speech
# import os

# # 인증 키 파일 설정
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "speech-to-text-403101-d7030ca7a9a6.json"

# # 클라이언트 초기화
# client = speech.SpeechClient()

# # 오디오 파일 설정 (로컬 파일 또는 Google Cloud Storage URI)
# gcs_uri = "gs://sh-stt/recorded_audio.wav"

# # RecognitionConfig 설정
# config = speech.RecognitionConfig(
#     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#     sample_rate_hertz=16000,  # 샘플링 속도 (Hz)
#     language_code="ko-KR",  # 언어 코드
# )

# # RecognitionAudio 설정
# audio = speech.RecognitionAudio(uri=gcs_uri)

# # 음성을 텍스트로 변환
# response = client.recognize(config=config, audio=audio)

# # 변환된 텍스트 출력
# for result in response.results:
#     print("Transcript: {}".format(result.alternatives[0].transcript))


# Imports the Google Cloud client library
from google.cloud import speech

# Instantiates a client
client = speech.SpeechClient()

# The name of the audio file to transcribe
gcs_uri = "gs://sh-stt/recorded_audio.wav"

audio = speech.RecognitionAudio(uri=gcs_uri)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="ko-KR",
)

# Detects speech in the audio file
response = client.recognize(config=config, audio=audio)

for result in response.results:
    print("Transcript: {}".format(result.alternatives[0].transcript))