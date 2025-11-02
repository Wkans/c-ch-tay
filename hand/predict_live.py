import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

# Tải mô hình, encoder và scaler
model = joblib.load('gesture_model_svm.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

#Biến theo dõi độ ổn định
last_stable_time = 0
last_label =""
saved_sentence = ""
stable_threshold = 2  # giây
position_tolerance = 10  # pixel
last_positions = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(rgb)
    results_face = face_mesh.process(rgb)

    all_landmarks_px = []
    x_list, y_list = [], []

    # Tay
    if results_hands.multi_hand_landmarks:
        hand_landmarks_list = results_hands.multi_hand_landmarks or []
        for _ in range(2 - len(hand_landmarks_list)):
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

    # Mặt
    if results_face.multi_face_landmarks:
        for faceLms in results_face.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, faceLms, mp_face_mesh.FACEMESH_TESSELATION,
                                   landmark_drawing_spec=None,
                                   connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1))
            for lm in faceLms.landmark:
                x_px = int(lm.x * img_w)
                y_px = int(lm.y * img_h)
                x_list.append(x_px)
                y_list.append(y_px)
                all_landmarks_px.append((x_px, y_px, lm.z))

    if x_list and y_list:
        x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, img_w)
        y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, img_h)
        crop_w, crop_h = x_max - x_min, y_max - y_min

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        # Chuẩn hóa
        normalized_landmarks = []
        for x, y, z in all_landmarks_px:
            norm_x = (x - x_min) / crop_w
            norm_y = (y - y_min) / crop_h
            normalized_landmarks.extend([norm_x, norm_y, z])

        # Dự đoán nếu đủ số lượng điểm (ví dụ ~500 điểm trở lên)
        if len(normalized_landmarks) == model.n_features_in_:
            X_input = np.array(normalized_landmarks).reshape(1, -1)
            X_input_scaled = scaler.transform(X_input)
            y_pred = model.predict(X_input_scaled)[0]
            label = label_encoder.inverse_transform([y_pred])[0]

            cv2.putText(frame, f'Prediction: {label}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Theo dõi sự ổn định
            current_positions = np.array(all_landmarks_px)

            if last_positions is not None and current_positions.shape == last_positions.shape:
                diff = np.abs(current_positions - last_positions)
                if np.all(diff < position_tolerance):
                    if label == last_label:
                        if time.time() - last_stable_time >= stable_threshold:
                            if len(label) == 1:
                                saved_sentence += label  # Không thêm khoảng trắng
                            else:
                                saved_sentence += label + " "
                            print("Saved:", saved_sentence)
                            last_stable_time = time.time() + 1  # tránh lưu trùng
                    else:
                        last_label = label
                        last_stable_time = time.time()
                else:
                    last_stable_time = time.time()
                    last_label = label
            else:
                last_stable_time = time.time()
                last_label = label

            last_positions = current_positions
        else:
            cv2.putText(frame, f'Dang xu ly...', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Hiển thị câu đã lưu
    cv2.putText(frame, f'Text: {saved_sentence}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Real-time Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
