import cv2
import mediapipe as mp

# Initialize Mediapipe solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Functions to check specific gestures
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

# Initialize video capture
cap = cv2.VideoCapture(0)
prev_hand_landmarks = None

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

cap.release()
cv2.destroyAllWindows()