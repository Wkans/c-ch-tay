import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Khởi tạo âm lượng hệ thống
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_control = cast(interface, POINTER(IAudioEndpointVolume))

minVol, maxVol = volume_control.GetVolumeRange()[:2]

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy tọa độ ngón cái và trỏ
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            thumb_pos = (int(thumb.x * w), int(thumb.y * h))
            index_pos = (int(index.x * w), int(index.y * h))

            # Tính khoảng cách
            dist = get_distance(thumb_pos, index_pos)
            vol = np.interp(dist, [30, 200], [minVol, maxVol])
            volume_control.SetMasterVolumeLevel(vol, None)

            # Vẽ vòng tròn
            cv2.circle(frame, thumb_pos, 10, (255, 0, 0), -1)
            cv2.circle(frame, index_pos, 10, (255, 0, 0), -1)
            cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 3)

            # Hiển thị mức âm lượng (tùy chọn)
            vol_bar = np.interp(dist, [30, 200], [400, 150])
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 255), 3)
            cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 0, 255), -1)

    cv2.imshow("Volume Control by Hand Gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
