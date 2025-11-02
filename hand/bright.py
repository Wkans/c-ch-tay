import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy vị trí ngón cái và trỏ
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            thumb_pos = (int(thumb.x * w), int(thumb.y * h))
            index_pos = (int(index.x * w), int(index.y * h))

            # Vẽ vòng tròn và đường thẳng
            cv2.circle(frame, thumb_pos, 10, (255, 0, 0), -1)
            cv2.circle(frame, index_pos, 10, (255, 0, 0), -1)
            cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 3)

            # Tính khoảng cách và điều chỉnh độ sáng
            dist = get_distance(thumb_pos, index_pos)
            brightness = np.interp(dist, [30, 200], [0, 100])
            sbc.set_brightness(int(brightness))

            # Hiển thị thanh độ sáng
            cv2.rectangle(frame, (50, 150), (85, 400), (255, 255, 0), 2)
            bright_bar = np.interp(brightness, [0, 100], [400, 150])
            cv2.rectangle(frame, (50, int(bright_bar)), (85, 400), (255, 255, 0), -1)
            cv2.putText(frame, f'{int(brightness)} %', (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Brightness Control by Hand", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
