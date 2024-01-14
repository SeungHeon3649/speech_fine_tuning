import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM
import cv2
from mutagen.mp3 import MP3
import time



def return_answer_by_chatbot(user_text):
    model_name = 'path_to_save_model'  # 모델이 저장된 디렉토리 경로
    tokenizer = AutoTokenizer.from_pretrained(model_name, bos_token='</s>', eos_token='</s>', pad_token='<pad>')
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    sent = '<usr>' + user_text + '<sys>'
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])
    output = model.generate(input_ids, max_length=50, do_sample=True, top_k=20)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    chatbot_response = sentence.split('<sys>')[1].replace('</s>', '')
    return chatbot_response

def test(path, time_):
    video_file = path
    ck = False
    while True:
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print("동영상 파일을 열 수 없습니다.")
            exit()

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            if time.time() == time_:
                cv2.imshow('Video', frame)
            if cv2.waitKey(30) == ord('q'):
                ck = True
                break
        
        if ck:
            break

    cap.release()
    cv2.destroyAllWindows()

# # test('your_video.mp4')
# # print()

# from tts_save import save_speech_file
# from tts_play import play_speech_file



# file_name = save_speech_file(return_answer_by_chatbot('우리 뭐 하고 놀까?'))

# # audio = MP3(file_name)

# # # 재생 시간 가져오기 (초 단위)
# # playback_time = audio.info.length
# # print(playback_time)
# play_speech_file(file_name)
# # test('your_video.mp4', playback_time)