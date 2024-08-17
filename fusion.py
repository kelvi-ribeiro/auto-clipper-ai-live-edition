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


# The last time a recording was retrieved from the queue.
phrase_time = None
# Thread safe Queue for passing data from the threaded recording callback.
data_queue = Queue()
# We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
recorder = sr.Recognizer()
recorder.energy_threshold = 1000
# Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = False
source = sr.Microphone(sample_rate=16000)

# Load / Download model
model = "base"
audio_model = whisper.load_model(model)

record_timeout = 2
phrase_timeout = 3

transcription = ['']

with source:
    recorder.adjust_for_ambient_noise(source)


# Initialize Mediapipe solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
prev_hand_landmarks = None

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

def is_arms_crossed(pose_landmarks):
    # Pontos de referência para os braços e ombros
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Verifica se a mão esquerda está próxima ao ombro direito
    # e se a mão direita está próxima ao ombro esquerdo
    threshold = 0.2  # Ajuste conforme necessário

    left_hand_near_right_shoulder = abs(left_wrist.x - right_shoulder.x) < threshold and abs(left_wrist.y - right_shoulder.y) < threshold
    right_hand_near_left_shoulder = abs(right_wrist.x - left_shoulder.x) < threshold and abs(right_wrist.y - left_shoulder.y) < threshold

    return left_hand_near_right_shoulder and right_hand_near_left_shoulder

def is_y_pose(pose_landmarks):
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hand = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_hand = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    return (left_hand.y < left_shoulder.y and right_hand.y < right_shoulder.y and
            left_hand.x < left_shoulder.x and right_hand.x > right_shoulder.x)

def start_cam(mp_hands, hands, mp_pose, pose, mp_drawing, is_thumb_up, is_peace_sign, is_rock_sign, is_arms_crossed, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

    # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if is_peace_sign(hand_landmarks):
                    print("Peace Sign Gesture")
                elif is_thumb_up(hand_landmarks):
                    print("Thumb Up Gesture")
                elif is_rock_sign(hand_landmarks):
                    print("Rock Sign Gesture")

                prev_hand_landmarks = hand_landmarks

    # Process pose landmarks
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks
            if is_arms_crossed(pose_landmarks):
                print("Arms crossed")

    # Draw landmarks on the image
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Recognition live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def record_callback(_, audio:sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)

# Create a background thread that will pass us raw audio bytes.
# We could do this manually but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

# Cue the user that we're ready to go.
print("Model loaded.\n")
audio_thread = Thread(target=start_cam)
audio_thread.start()
while True:
    try:
        now = datetime.utcnow()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                phrase_complete = True
            # This is the last time we received new audio data from the queue.
            phrase_time = now
            
            # Combine audio data from queue
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()
            
            # Convert in-ram buffer to something the model can use directly without needing a temp file.
            # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
            # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Read the transcription.
            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result['text'].strip()

            # If we detected a pause between recordings, add a new item to our transcription.
            # Otherwise edit the existing one.
            if phrase_complete:
                transcription.append(text)
            else:
                transcription[-1] = text

            # Clear the console to reprint the updated transcription.
            os.system('cls' if os.name=='nt' else 'clear')
            for line in transcription:
                print(line)
            # Flush stdout.
            print('', end='', flush=True)
        else:
            # Infinite loops are bad for processors, must sleep.
            sleep(0.25)
    except KeyboardInterrupt:
        break

print("\n\nTranscription:")
for line in transcription:
    print(line)