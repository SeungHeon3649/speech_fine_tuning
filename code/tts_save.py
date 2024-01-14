import os
from google.cloud import texttospeech
from pydub import AudioSegment

def save_speech_file(tts_text):   
    # 환경 변수 설정
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "speech-test-project-403005-7aebfc0e2e5d.json"

    # 인증 정보 설정
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(
        text = tts_text) # 변환할 텍스트

    # 음성 설정
    voice = texttospeech.VoiceSelectionParams(
        language_code='ko-KR',  # 한국어
        name='ko-KR-Wavenet-A', # 성우 종류
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3)

    request = texttospeech.SynthesizeSpeechRequest(
        input=input_text,
        voice=voice,
        audio_config=audio_config
    )

    # 음성 합성 요청
    response = client.synthesize_speech(request)

    # 변환된 음성 파일 저장
    with open('output.mp3', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')
        return out.name




# audio = AudioSegment.from_mp3('output.mp3')

# with tempfile.TemporaryDirectory() as temp_dir:
#     temp_path = os.path.join(temp_dir, 'output.wav')
#     audio.export(temp_path, format="wav")
#     play(AudioSegment.from_wav(temp_path))