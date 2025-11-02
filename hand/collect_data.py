import cv2
import mediapipe as mp
import numpy as np
import csv
import string
import time

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

labels = list(string.ascii_uppercase) + list(string.digits)
current_label = ''

print("Nhấn SPACE để nhập nhãn (A-Z, 0-9), ESC để thoát.")
print("Tự động lưu khi tay/mặt đứng yên trong 0.5 giây.")

# Biến lưu trước đó để kiểm tra chuyển động
prev_landmarks = None
last_check_time = time.time()
auto_save_interval = 0.5  # giây
movement_threshold = 0.01  # sai lệch nhỏ được phép

with open('hand_face_landmarks.csv', mode='a', newline='') as f:
    writer = csv.writer(f)
    saved_landmarks = None

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = hands.process(rgb)
        results_face = face_mesh.process(rgb)

        all_landmarks_px = []  # (x, y, z)
        x_list, y_list = [], []

        # Thu thập landmark tay
        if results_hands.multi_hand_landmarks:
            hand_landmarks_list = results_hands.multi_hand_landmarks or []
            for _ in range(2 - len(hand_landmarks_list)):
                # thêm tay giả nếu thiếu
                for _ in range(21):
                    all_landmarks_px.append((0, 0, 0))
            for handLms in results_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                for lm in handLms.landmark:
                    x_px = int(lm.x * img_w)
                    y_px = int(lm.y * img_h)
                    x_list.append(x_px)
                    y_list.append(y_px)
                    all_landmarks_px.append((x_px, y_px, lm.z))

        # Thu thập landmark khuôn mặt
        if results_face.multi_face_landmarks:
            for faceLms in results_face.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, faceLms, mp_face_mesh.FACEMESH_TESSELATION,
                                       landmark_drawing_spec=None,
                                       connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                for lm in faceLms.landmark:
                    x_px = int(lm.x * img_w)
                    y_px = int(lm.y * img_h)
                    x_list.append(x_px)
                    y_list.append(y_px)
                    all_landmarks_px.append((x_px, y_px, lm.z))

        # Tính bounding box và chuẩn hóa
        if x_list and y_list:
            x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, img_w)
            y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, img_h)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            crop_w = x_max - x_min
            crop_h = y_max - y_min
            normalized_landmarks = []
            for x, y, z in all_landmarks_px:
                norm_x = (x - x_min) / crop_w
                norm_y = (y - y_min) / crop_h
                normalized_landmarks.extend([norm_x, norm_y, z])

            saved_landmarks = normalized_landmarks

        # Tự động lưu nếu landmark ổn định
        now = time.time()
        if saved_landmarks and current_label and now - last_check_time >= auto_save_interval:
            if prev_landmarks and len(prev_landmarks) == len(saved_landmarks):
                diff = np.array(saved_landmarks) - np.array(prev_landmarks)
                error = np.sum(diff ** 2)
                if error < movement_threshold:
                    writer.writerow([current_label] + saved_landmarks)
                    print(f"✅ Tự động lưu nhãn '{current_label}' (độ sai lệch: {error:.6f})")
                    time.sleep(1)  # tránh lưu liên tục nhiều lần
            prev_landmarks = saved_landmarks
            last_check_time = now

        # Giao diện
        cv2.putText(frame, f'Label: {current_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE → nhập nhãn
            print("⏳ Nhập chuỗi nhãn mới:")
            current_label = input("Label: ").strip()

cap.release()
cv2.destroyAllWindows()
