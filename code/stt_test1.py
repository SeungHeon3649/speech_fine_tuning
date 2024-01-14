# 바로바로 해석하는 코드

from __future__ import division

import re
import sys
import pororo
from google.cloud import speech
import pyaudio
from six.moves import queue
from tts_save import save_speech_file
from tts_play import play_speech_file
# 오디오 녹음에 대한 매개변수 설정
RATE = 48000
CHUNK = int(RATE / 10)  # 100ms

# MicrophoneStream 클래스 정의(오디오 스트림 열기, 오디오 청크 생성)
class MicrophoneStream(object):
    
    # 샘플링 속도, 청크 크기 설정, 오디오 데이터 저장 위한 버퍼 생성
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        self._buff = queue.Queue()
        self.closed = True

    # with 문으로 객체를 사용할 때 호출되는 메서드
    def __enter__(self):
        # 오디오 인터페이스 설정
        self._audio_interface = pyaudio.PyAudio()
        # 오디오 스트림 설정, 오디오 데이터를 채우는 콜백함수(fill_buffer) 정의
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        # 스트림이 열렸음을 나타내는 플래그 설정
        self.closed = False
        return self


    # with 문이 끝나고 enter문이 종료되면 호출되는 메서드
    def __exit__(self, type, value, traceback):
        # 오디오 스트림 중지하고 닫음
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        # 스트림이 닫혔음을 나타내는 플래그 설정
        self.closed = True

        # 버퍼에 None을 넣어서 스트림이 종료됨을 나타냄
        self._buff.put(None)
        # 오디오 인터페이스 종료
        self._audio_interface.terminate()

    # 오디오 스트림에 호출되는 콜백 함수
    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        # 오디오 데이터를 버퍼에 넣음
        self._buff.put(in_data)
        # None을 반환하고 스트림 계속 유지
        return None, pyaudio.paContinue

    # 오디오 데이터 생성하는 제너레이터 함수
    def generator(self):
        # 스트림이 열려 있으면 계속 실행
        while not self.closed:
            # 버퍼에서 청크를 가져옴
            chunk = self._buff.get()
            # 청크가 비어있으면 종료
            if chunk is None:
                return
            # 청크를 data 리스트에 넣음(초기 청크)
            data = [chunk]

            # 청크를 계속 가져와서 리스트에 추가
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    # 청크가 None이면 함수 종료
                    if chunk is None:
                        return
                    data.append(chunk)
                # 버퍼가 비어있으면 루프 종료
                except queue.Empty:
                    break
            # 데이터 리스트를 연결하고 바이트로 반환
            yield b''.join(data)

def listen_print_loop(responses):
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)
            
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0
            return transcript + overwrite_chars 
def main():
    language_code = 'ko-KR'  
    global text

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    while True:
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = client.streaming_recognize(streaming_config, requests)

            trans = listen_print_loop(responses)
            file_name = save_speech_file(pororo.return_answer_by_chatbot(trans))
            play_speech_file(file_name)
            

if __name__ == '__main__':
    main()

