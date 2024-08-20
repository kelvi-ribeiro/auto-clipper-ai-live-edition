import os
from threading import Thread
import numpy as np
import speech_recognition as sr
import whisper
import torch
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
import cv2
import mediapipe as mp
from utils.string_utils import remove_special_chars_and_accents

SPECIAL_WORD = 'transição'
SPECIAL_GESTURE = 'rock' 
FONT_SCALE = 0.8
RECORD_TIMEOUT = 2
PHRASE_TIMEOUT = 3
TEXT_COLOR = (0, 255, 255) 
BACKGROUND_COLOR = (0, 0, 0) 
FONT_THICKNESS = 2

phrase_time = None
data_queue = Queue()
transcription = ['']
start_time = datetime.now()
special_word_found_description = None
special_gesture_found_description = None
special_color_found_description = None
mp_hands = mp.solutions.hands

def init_speech_recognition():
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)
    return recorder, source

def load_audio_model(model_name="base"):
    return whisper.load_model(model_name)

def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.9)
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing

def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return thumb_tip.y < thumb_ip.y < thumb_mcp.y and thumb_mcp.y < index_finger_mcp.y

def is_peace_sign(hand_landmarks):
    return (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y)

def is_rock_sign(hand_landmarks):
    return (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y)

def draw_text_with_background(frame, text, position, font_scale, color, thickness, background_color):

    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position


    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), background_color, cv2.FILLED)


    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def record_callback(_, audio: sr.AudioData) -> None:
    data_queue.put(audio.get_raw_data())

def start_microphone(audio_model):
    global phrase_time, transcription, special_word_found_description
    while True:
        try:
            now = datetime.now()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=PHRASE_TIMEOUT):
                    phrase_complete = True
                phrase_time = now
                
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                for line in transcription:
                    if remove_special_chars_and_accents(SPECIAL_WORD) in remove_special_chars_and_accents(line):
                        transcription = []
                        special_word_found_description = f"Corte detectado pela palavra {remove_special_chars_and_accents(SPECIAL_WORD)}"
                        break
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

def start_cam(hands, mp_drawing):
    global special_gesture_found_description, special_color_found_description
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if special_word_found_description:
            draw_text_with_background(frame, special_word_found_description or '', (100, 100), FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, BACKGROUND_COLOR)
        if special_gesture_found_description:
            draw_text_with_background(frame, special_gesture_found_description or '', (100, 250), FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, BACKGROUND_COLOR)
        if special_color_found_description:
            draw_text_with_background(frame, special_color_found_description or '', (100, 400), FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, BACKGROUND_COLOR)


        frame_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ## TODO MELHORAR E SEPARAR POR MÉTODOS Color recg
        fps = cap.get(cv2.CAP_PROP_FPS)
        threshold=10
        frame_interval = int(fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position + frame_interval)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        time_difference = datetime.now() - start_time
        if cv2.mean(gray_frame)[0] < threshold and time_difference >= timedelta(seconds=10):
            special_color_found_description = "Achei foc"

        ## End
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if is_peace_sign(hand_landmarks) and SPECIAL_GESTURE == 'peace':
                    special_gesture_found_description = 'Corte detectado pelo gesto de paz'
                elif is_thumb_up(hand_landmarks) and SPECIAL_GESTURE == 'thumb_up':
                    special_gesture_found_description = 'Corte detectado pelo gesto de joinha'
                elif is_rock_sign(hand_landmarks) and SPECIAL_GESTURE == 'rock':
                    special_gesture_found_description = 'Corte detectado pelo gesto de rock'

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Recognition live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    global phrase_time, transcription, special_word_found_description, special_gesture_found_description, special_color_found_description

    recorder, source = init_speech_recognition()
    audio_model = load_audio_model()
    hands, mp_drawing = init_mediapipe()
    recorder.listen_in_background(source, record_callback, phrase_time_limit=RECORD_TIMEOUT)

    cam_thread = Thread(target=start_cam, args=(hands, mp_drawing))
    cam_thread.start()

    start_microphone(audio_model)

if __name__ == '__main__':
    main()
