#! python3.7

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

# Configurações gerais
SPECIAL_WORD = 'transição'
SPECIAL_GESTURE = 'rock'  # peace, thumb_up, rock
FONT_SCALE = 0.8
RECORD_TIMEOUT = 2
PHRASE_TIMEOUT = 3
TEXT_COLOR = (0, 255, 255)  # Amarelo em BGR
BACKGROUND_COLOR = (0, 0, 0)  # Preto em BGR
FONT_THICKNESS = 2

# Variáveis globais
phrase_time = None
data_queue = Queue()
transcription = ['']
special_word_found_description = None
special_gesture_found_description = None
mp_hands = mp.solutions.hands

# Inicializa o reconhecimento de fala
def init_speech_recognition():
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)
    return recorder, source

# Carrega o modelo Whisper
def load_audio_model(model_name="base"):
    return whisper.load_model(model_name)

# Inicializa os módulos do Mediapipe
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.9)
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing

# Verifica se o gesto é "thumb up"
def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return thumb_tip.y < thumb_ip.y < thumb_mcp.y and thumb_mcp.y < index_finger_mcp.y

# Verifica se o gesto é "peace"
def is_peace_sign(hand_landmarks):
    return (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y)

# Verifica se o gesto é "rock"
def is_rock_sign(hand_landmarks):
    return (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y)

# Desenha texto com fundo no frame
def draw_text_with_background(frame, text, position, font_scale, color, thickness, background_color):
    # Obtem o tamanho do texto
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position

    # Desenha o fundo
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), background_color, cv2.FILLED)

    # Desenha o texto
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# Função de callback para gravação de áudio
def record_callback(_, audio: sr.AudioData) -> None:
    data_queue.put(audio.get_raw_data())

# Processa o áudio e atualiza as transcrições
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

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                    if remove_special_chars_and_accents(SPECIAL_WORD) in remove_special_chars_and_accents(line):
                        transcription = []
                        special_word_found_description = f"Corte detectado pela palavra {remove_special_chars_and_accents(SPECIAL_WORD)}"
                        break
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

# Processa o vídeo e detecta gestos
def start_cam(hands, mp_drawing):
    global special_gesture_found_description
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if special_word_found_description:
            draw_text_with_background(frame, special_word_found_description or '', (100, 50), FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, BACKGROUND_COLOR)
        if special_gesture_found_description:
            draw_text_with_background(frame, special_gesture_found_description or '', (100, 200), FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, BACKGROUND_COLOR)

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

# Inicializa os componentes e threads
def main():
    global phrase_time, transcription, special_word_found_description, special_gesture_found_description

    recorder, source = init_speech_recognition()
    audio_model = load_audio_model()
    hands, mp_drawing = init_mediapipe()

    recorder.listen_in_background(source, record_callback, phrase_time_limit=RECORD_TIMEOUT)

    print("Model loaded.\n")

    cam_thread = Thread(target=start_cam, args=(hands, mp_drawing))
    cam_thread.start()

    start_microphone(audio_model)

if __name__ == '__main__':
    main()
