import pyaudio
import wave
from math import log10
import numpy as np

# 오디오 녹음 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 샘플 속도 (16 kHz)
CHUNK = 1024
RECORD_SECONDS = 5  # 녹음할 시간(초)
OUTPUT_FILENAME = "recorded_audio.wav"

audio = pyaudio.PyAudio()

# 오디오 스트림 열기
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Recording...")

frames = []
silence_frames = 0
silence_threshold = 100  # 음성 활동을 판단하기 위한 임계값 (임의 설정)
cnt = 0

while True:
    data = stream.read(CHUNK)
    frames.append(data)
    
    # 데이터의 에너지 레벨 계산
    energy = sum([abs(int.from_bytes(frame, byteorder='big')) for frame in frames[-10:]])
    #print(energy)
    # 에너지 레벨이 임계값 미만인지 확인
    if energy < silence_threshold:
        silence_frames += 1
    else:
        silence_frames = 0
    
    # silence_frames가 일정 시간(예: 2초) 동안 지속되면 녹음 중지
    if silence_frames >= (RATE / CHUNK * 2):
        break
    
print("Finished recording.")

# 오디오 스트림 닫기
stream.stop_stream()
stream.close()
audio.terminate()

# WAV 파일로 저장
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Saved audio to {OUTPUT_FILENAME}")
